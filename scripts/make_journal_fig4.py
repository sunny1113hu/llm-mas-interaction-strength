from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.analysis import load_jsonl
from src.utils.cec_style import PAPER_MODEL_COLORS
from src.utils.journal_figures import apply_journal_style, double_column_figsize, save_figure
from src.utils.stats import mean_t_ci95


OUTPUT_DEFAULT = Path("outputs/runs/paper_figures_journal/Figures")

MODELS_8B = [
    "Qwen/Qwen3-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Ministral-3-8B-Instruct-2512",
]
LABELS = {
    "Qwen/Qwen3-8B": "Qwen3-8B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "mistralai/Ministral-3-8B-Instruct-2512": "Ministral-3-8B",
}
LINE_STYLES = {
    "Qwen/Qwen3-8B": ("-", "o"),
    "meta-llama/Llama-3.1-8B-Instruct": ("--", "s"),
    "mistralai/Ministral-3-8B-Instruct-2512": ("-.", "^"),
}

CONSENSUS_RUNS = [
    "20260109_102923_consensus_4x4_4llms",
    "20260120_003407_consensus",
]
DIFFUSION_RUNS = [
    "20260115_073051_diffusion_4x4_4llms",
    "20260120_003407_diffusion",
]


apply_journal_style(font_size=9, title_size=10, label_size=9, tick_size=8, legend_size=8)


def _load_trials(outputs_root: Path, run_ids: list[str]) -> list[dict]:
    trials: list[dict] = []
    for run_id in run_ids:
        run_trials = load_jsonl(outputs_root / run_id / "trial_summary.jsonl")
        for record in run_trials:
            record.setdefault("run_id", run_id)
        trials.extend(run_trials)
    return trials


def _filter_trials(trials: list[dict], model_id: str) -> list[dict]:
    filtered = []
    for record in trials:
        if record.get("model_id") != model_id:
            continue
        if (record.get("prompt_mode") or "with_pi") == "no_pi":
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


def _set_mu_ticks(ax, mus: list[float]) -> None:
    if not mus:
        return
    ticks = [t for t in range(-5, 6) if min(mus) <= t <= max(mus)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([_fmt_tick(t) for t in ticks])


def _plot_model_series(ax, grouped: dict[float, list[float]], *, model_id: str) -> list[float]:
    mus = sorted(grouped.keys())
    means = []
    cis = []
    for mu in mus:
        mean, _std, ci = mean_t_ci95(grouped[mu])
        means.append(mean)
        cis.append(ci)
    color = PAPER_MODEL_COLORS[model_id]
    linestyle, marker = LINE_STYLES[model_id]
    x = np.asarray(mus, dtype=float)
    y = np.asarray(means, dtype=float)
    ci_arr = np.asarray(cis, dtype=float)
    ax.plot(
        x,
        y,
        color=color,
        linestyle=linestyle,
        marker=marker,
        linewidth=1.2,
        markersize=3.0,
        label=LABELS[model_id],
    )
    ax.fill_between(x, y - ci_arr, y + ci_arr, color=color, alpha=0.14, linewidth=0.0)
    return mus


def _plot_panel(
    ax,
    *,
    trials: list[dict],
    metric_key: str,
    ylabel: str,
    panel_label: str,
) -> None:
    mus_all: list[float] = []
    for model_id in MODELS_8B:
        grouped = _metric_by_mu(_filter_trials(trials, model_id), metric_key)
        if not grouped:
            continue
        mus_all.extend(_plot_model_series(ax, grouped, model_id=model_id))

    _set_mu_ticks(ax, sorted(set(mus_all)))
    ax.set_xlabel("μ")
    ax.set_ylabel(ylabel)
    ax.text(0.5, -0.25, f"({panel_label})", transform=ax.transAxes, ha="center", va="top", fontsize=9)
    ax.legend(frameon=False, loc="upper left", handlelength=2.2, borderaxespad=0.2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create journal Fig4: 8B model-dependence panels.")
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    consensus_trials = _load_trials(args.outputs_root, CONSENSUS_RUNS)
    diffusion_trials = _load_trials(args.outputs_root, DIFFUSION_RUNS)

    fig, axes = plt.subplots(1, 2, figsize=double_column_figsize(60), constrained_layout=False)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.24, top=0.97, wspace=0.34)
    _plot_panel(
        axes[0],
        trials=consensus_trials,
        metric_key="final_dispersion",
        ylabel="MAD",
        panel_label="a",
    )
    _plot_panel(
        axes[1],
        trials=diffusion_trials,
        metric_key="roughness_final",
        ylabel="roughness",
        panel_label="b",
    )
    save_figure(fig, args.out_dir / "Fig4.pdf", pad_inches=0.0, bbox_inches=None)


if __name__ == "__main__":
    main()
