from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from src.utils.analysis import load_jsonl
from src.utils.cec_style import CEC_DARK, PAPER_MODEL_COLORS
from src.utils.journal_figures import apply_journal_style, double_column_figsize, save_figure
from src.utils.stats import mean_t_ci95


OUTPUT_DEFAULT = Path("outputs/runs/paper_figures_journal/Figures")

MODELS_8B = [
    "Qwen/Qwen3-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Ministral-3-8B-Instruct-2512",
]
MODELS_SCALE_ONLY = [
    "mistralai/Ministral-3-14B-Instruct-2512",
    "microsoft/Phi-4-mini-instruct",
]
LABELS = {
    "Qwen/Qwen3-8B": "Qwen3-8B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "mistralai/Ministral-3-8B-Instruct-2512": "Ministral-3-8B",
    "mistralai/Ministral-3-14B-Instruct-2512": "Ministral-3-14B",
    "microsoft/Phi-4-mini-instruct": "Phi-4-mini",
}

CONSENSUS_WITH_PI_RUNS = [
    "20260109_102923_consensus_4x4_4llms",
    "20260116_110026_consensus_ministral14B",
    "20260120_003407_consensus",
]
DIFFUSION_WITH_PI_RUNS = [
    "20260115_073051_diffusion_4x4_4llms",
    "20260116_110026_diffusion_ministral14B",
    "20260120_003407_diffusion",
]
CONSENSUS_NO_PI_RUNS = [
    "20260117_174520_consensus_5seeds",
    "20260124_010246_consensus_add5nopi",
]
DIFFUSION_NO_PI_RUNS = [
    "20260117_174520_diffusion_5seeds",
    "20260124_010246_diffusion_add5nopi",
]


apply_journal_style(font_size=8.5, title_size=9, label_size=8.5, tick_size=8, legend_size=8)


def _load_trials(outputs_root: Path, run_ids: list[str]) -> list[dict]:
    trials: list[dict] = []
    for run_id in run_ids:
        run_trials = load_jsonl(outputs_root / run_id / "trial_summary.jsonl")
        for record in run_trials:
            record.setdefault("run_id", run_id)
        trials.extend(run_trials)
    return trials


def _filter_trials(trials: list[dict], model_id: str, prompt_mode: str) -> list[dict]:
    filtered = []
    for record in trials:
        if record.get("model_id") != model_id:
            continue
        mode = record.get("prompt_mode") or "with_pi"
        if prompt_mode == "with_pi" and mode == "no_pi":
            continue
        if prompt_mode == "no_pi" and mode != "no_pi":
            continue
        if int(record.get("grid_size", 4)) != 4:
            continue
        if int(record.get("steps", 10)) != 10:
            continue
        filtered.append(record)
    return filtered


def _metric_by_mu(trials: list[dict], metric_key: str) -> dict[float, list[float]]:
    grouped: dict[float, list[float]] = defaultdict(list)
    for record in trials:
        mu = record.get("mu")
        metric = record.get(metric_key)
        if mu is None or not isinstance(metric, (int, float)):
            continue
        grouped[float(mu)].append(float(metric))
    return dict(grouped)


def _fmt_tick(value: int) -> str:
    text = str(value)
    return "−" + text[1:] if text.startswith("-") else text


def _plot_with_pi(ax, trials: list[dict], metric_key: str, model_id: str) -> list[float]:
    grouped = _metric_by_mu(_filter_trials(trials, model_id, "with_pi"), metric_key)
    mus = sorted(grouped.keys())
    means = []
    cis = []
    for mu in mus:
        mean, _std, ci = mean_t_ci95(grouped[mu])
        means.append(mean)
        cis.append(ci)
    color = PAPER_MODEL_COLORS[model_id]
    x = np.asarray(mus, dtype=float)
    y = np.asarray(means, dtype=float)
    ci_arr = np.asarray(cis, dtype=float)
    ax.plot(x, y, color=color, linestyle="-", marker="o", markersize=2.5, linewidth=1.0)
    ax.fill_between(x, y - ci_arr, y + ci_arr, color=color, alpha=0.14, linewidth=0.0)
    return mus


def _plot_no_pi(ax, trials: list[dict], metric_key: str, model_id: str, x_min: float, x_max: float) -> None:
    no_trials = _filter_trials(trials, model_id, "no_pi")
    values = [float(record[metric_key]) for record in no_trials if isinstance(record.get(metric_key), (int, float))]
    if not values:
        return
    mean, _std, ci = mean_t_ci95(values)
    xs = np.asarray([x_min, x_max], dtype=float)
    ax.plot(xs, [mean, mean], color=CEC_DARK, linestyle="--", linewidth=0.9)
    ax.fill_between(xs, [mean - ci, mean - ci], [mean + ci, mean + ci], color=CEC_DARK, alpha=0.10, linewidth=0.0)


def _plot_model_panel(
    ax,
    *,
    with_trials: list[dict],
    no_trials: list[dict],
    metric_key: str,
    ylabel: str | None,
    model_id: str,
) -> None:
    mus = _plot_with_pi(ax, with_trials, metric_key, model_id)
    x_min = min(mus) if mus else -5.0
    x_max = max(mus) if mus else 5.0
    _plot_no_pi(ax, no_trials, metric_key, model_id, x_min, x_max)
    ax.set_title(LABELS[model_id], pad=2.0)
    ax.set_xlabel("μ")
    if ylabel:
        ax.set_ylabel(ylabel)
    ticks = [t for t in [-5, 0, 5] if x_min <= t <= x_max]
    ax.set_xticks(ticks)
    ax.set_xticklabels([_fmt_tick(t) for t in ticks])
    ax.tick_params(axis="both", length=2, width=0.3)


def _make_task_figure(
    *,
    with_trials: list[dict],
    no_trials: list[dict],
    metric_key: str,
    ylabel: str,
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=double_column_figsize(92))
    outer = fig.add_gridspec(
        nrows=2,
        ncols=6,
        left=0.060,
        right=0.990,
        bottom=0.100,
        top=0.865,
        wspace=0.34,
        hspace=0.62,
    )

    for idx, model_id in enumerate(MODELS_8B):
        ax = fig.add_subplot(outer[0, idx * 2 : idx * 2 + 2])
        _plot_model_panel(
            ax,
            with_trials=with_trials,
            no_trials=no_trials,
            metric_key=metric_key,
            ylabel=ylabel if idx == 0 else None,
            model_id=model_id,
        )
    for idx, model_id in enumerate(MODELS_SCALE_ONLY):
        start = 1 + idx * 2
        ax = fig.add_subplot(outer[1, start : start + 2])
        _plot_model_panel(
            ax,
            with_trials=with_trials,
            no_trials=no_trials,
            metric_key=metric_key,
            ylabel=ylabel if idx == 0 else None,
            model_id=model_id,
        )

    handles = [
        Line2D([0], [0], color=CEC_DARK, linestyle="--", linewidth=0.9, label=r"w/o $P_i$"),
        Line2D([0], [0], color="black", linestyle="-", marker="o", markersize=2.5, linewidth=1.0, label=r"w/ $P_i$"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.52, 0.985))
    save_figure(fig, out_path, pad_inches=0.0, bbox_inches=None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create journal Fig6/Fig7: P_i ablation split by task.")
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    _make_task_figure(
        with_trials=_load_trials(args.outputs_root, CONSENSUS_WITH_PI_RUNS),
        no_trials=_load_trials(args.outputs_root, CONSENSUS_NO_PI_RUNS),
        metric_key="final_dispersion",
        ylabel="MAD",
        out_path=args.out_dir / "Fig6.pdf",
    )
    _make_task_figure(
        with_trials=_load_trials(args.outputs_root, DIFFUSION_WITH_PI_RUNS),
        no_trials=_load_trials(args.outputs_root, DIFFUSION_NO_PI_RUNS),
        metric_key="roughness_final",
        ylabel="roughness",
        out_path=args.out_dir / "Fig7.pdf",
    )


if __name__ == "__main__":
    main()
