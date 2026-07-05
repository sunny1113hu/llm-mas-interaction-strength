from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.analysis import load_jsonl
from src.utils.cec_style import CEC_BLUE, CEC_RED
from src.utils.journal_figures import apply_journal_style, double_column_figsize, save_figure
from src.utils.stats import mean_t_ci95


OUTPUT_DEFAULT = Path("outputs/runs/paper_figures_journal/Figures")
QWEN_MODEL = "Qwen/Qwen3-8B"
MOBILITY_MODEL = "baseline/mobility_scaled"

TASK_SPECS = {
    "consensus": {
        "metric": "final_dispersion",
        "ylabel": "MAD",
        "qwen_runs": ["20260109_102923_consensus_4x4_4llms", "20260120_003407_consensus"],
        "mobility_runs": ["baseline_v2_consensus_mobility_scaled"],
    },
    "diffusion": {
        "metric": "roughness_final",
        "ylabel": "roughness",
        "qwen_runs": ["20260115_073051_diffusion_4x4_4llms", "20260120_003407_diffusion"],
        "mobility_runs": ["baseline_v2_diffusion_mobility_scaled"],
    },
}


apply_journal_style(font_size=9, title_size=10, label_size=9, tick_size=8, legend_size=8)


def _load_trials(outputs_root: Path, run_ids: list[str]) -> list[dict]:
    trials: list[dict] = []
    for run_id in run_ids:
        run_trials = load_jsonl(outputs_root / run_id / "trial_summary.jsonl")
        for record in run_trials:
            record.setdefault("run_id", run_id)
        trials.extend(run_trials)
    return trials


def _filter_trials(trials: list[dict], *, model_id: str) -> list[dict]:
    filtered = []
    for record in trials:
        if record.get("model_id") != model_id:
            continue
        if (record.get("prompt_mode") or "with_pi") != "with_pi":
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


def _plot_series(
    ax,
    grouped: dict[float, list[float]],
    *,
    color: str,
    label: str,
    linestyle: str,
    marker: str,
) -> list[float]:
    mus = sorted(grouped.keys())
    means = []
    cis = []
    for mu in mus:
        mean, _std, ci = mean_t_ci95(grouped[mu])
        means.append(mean)
        cis.append(ci)
    x = np.asarray(mus, dtype=float)
    y = np.asarray(means, dtype=float)
    ci_arr = np.asarray(cis, dtype=float)
    ax.plot(x, y, color=color, linestyle=linestyle, marker=marker, linewidth=1.2, markersize=3.0, label=label)
    ax.fill_between(x, y - ci_arr, y + ci_arr, color=color, alpha=0.14, linewidth=0.0)
    return mus


def _plot_panel(ax, *, task: str, panel_label: str, outputs_root: Path) -> None:
    spec = TASK_SPECS[task]
    qwen_trials = _filter_trials(_load_trials(outputs_root, spec["qwen_runs"]), model_id=QWEN_MODEL)
    mobility_trials = _filter_trials(_load_trials(outputs_root, spec["mobility_runs"]), model_id=MOBILITY_MODEL)
    qwen_grouped = _metric_by_mu(qwen_trials, spec["metric"])
    mobility_grouped = _metric_by_mu(mobility_trials, spec["metric"])
    mus_all = sorted(set(qwen_grouped.keys()) | set(mobility_grouped.keys()))

    _plot_series(ax, qwen_grouped, color=CEC_BLUE, label=r"Qwen3-8B w/ $P_i$", linestyle="-", marker="o")
    _plot_series(ax, mobility_grouped, color=CEC_RED, label=r"RSP rule w/ $P_i$", linestyle="--", marker="s")
    _set_mu_ticks(ax, mus_all)
    ax.set_xlabel("μ")
    ax.set_ylabel(spec["ylabel"])
    ax.text(0.5, -0.25, f"({panel_label})", transform=ax.transAxes, ha="center", va="top", fontsize=9)
    ax.legend(frameon=False, loc="best", handlelength=2.2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create journal Fig10: Qwen vs RSP rule.")
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=double_column_figsize(60), constrained_layout=False)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.24, top=0.97, wspace=0.34)
    _plot_panel(axes[0], task="consensus", panel_label="a", outputs_root=args.outputs_root)
    _plot_panel(axes[1], task="diffusion", panel_label="b", outputs_root=args.outputs_root)
    save_figure(fig, args.out_dir / "Fig10.pdf", pad_inches=0.0, bbox_inches=None)


if __name__ == "__main__":
    main()
