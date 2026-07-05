from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.analysis import load_jsonl
from src.utils.cec_style import CEC_BLUE, CEC_GREEN, CEC_ORANGE
from src.utils.journal_figures import (
    apply_journal_style,
    double_column_figsize,
    save_figure,
)
from src.utils.stats import mean_t_ci95


OUTPUT_DEFAULT = Path("outputs/runs/paper_figures_journal/Figures")
MAIN_MODEL = "Qwen/Qwen3-8B"

CONSENSUS_RUNS = [
    "20260109_102923_consensus_4x4_4llms",
    "20260116_110026_consensus_ministral14B",
    "20260120_003407_consensus",
    "20260125_111339_consensus_T=30",
    "20260204_082317_consensus_6x6",
    "20260206_103341_consensus_8x8",
]
DIFFUSION_RUNS = [
    "20260115_073051_diffusion_4x4_4llms",
    "20260116_110026_diffusion_ministral14B",
    "20260120_003407_diffusion",
    "20260125_111339_diffusionT=30",
    "20260317_diffusion_6x6_diffusion",
    "20260317_diffusion_8x8_diffusion",
]


apply_journal_style(font_size=9, title_size=10, label_size=9, tick_size=8, legend_size=8)


def _trial_key(record: dict) -> tuple[str, int]:
    return str(record.get("run_id", "")), int(record.get("trial_id", -1))


def _load_runs(outputs_root: Path, run_ids: list[str]) -> tuple[list[dict], list[dict]]:
    trials: list[dict] = []
    steps: list[dict] = []
    for run_id in run_ids:
        run_dir = outputs_root / run_id
        run_trials = load_jsonl(run_dir / "trial_summary.jsonl")
        run_steps = load_jsonl(run_dir / "steps.jsonl")
        for record in run_trials:
            record.setdefault("run_id", run_id)
        for record in run_steps:
            record.setdefault("run_id", run_id)
        trials.extend(run_trials)
        steps.extend(run_steps)
    return trials, steps


def _filter_main_trials(trials: list[dict]) -> list[dict]:
    out = []
    for record in trials:
        if record.get("model_id") != MAIN_MODEL:
            continue
        if (record.get("prompt_mode") or "with_pi") != "with_pi":
            continue
        if int(record.get("grid_size", 4)) != 4:
            continue
        if int(record.get("steps", 10)) != 10:
            continue
        out.append(record)
    return out


def _filter_steps_for_trials(steps: list[dict], trials: list[dict]) -> list[dict]:
    keys = {_trial_key(record) for record in trials}
    return [record for record in steps if _trial_key(record) in keys and record.get("model_id") == MAIN_MODEL]


def _metric_by_mu(trials: list[dict], metric_key: str) -> dict[float, list[float]]:
    grouped: dict[float, list[float]] = defaultdict(list)
    for record in trials:
        metric = record.get(metric_key)
        mu = record.get("mu")
        if mu is None or not isinstance(metric, (int, float)):
            continue
        grouped[float(mu)].append(float(metric))
    return dict(grouped)


def _mean_step_change_by_trial(steps: list[dict]) -> dict[tuple[str, int], float]:
    by_trial_step: dict[tuple[tuple[str, int], int], dict[int, dict]] = defaultdict(dict)
    for record in steps:
        trial_key = _trial_key(record)
        step = int(record.get("step", 0))
        agent_id = int(record.get("agent_id", -1))
        by_trial_step[(trial_key, step)][agent_id] = record

    by_trial: dict[tuple[str, int], list[float]] = defaultdict(list)
    for (trial_key, step), curr in by_trial_step.items():
        if step == 0:
            continue
        prev = by_trial_step.get((trial_key, step - 1), {})
        for agent_id, record in curr.items():
            prev_record = prev.get(agent_id)
            if prev_record is None:
                continue
            try:
                by_trial[trial_key].append(abs(int(record["value"]) - int(prev_record["value"])))
            except Exception:
                continue
    return {key: float(np.mean(vals)) if vals else 0.0 for key, vals in by_trial.items()}


def _masc_by_mu(trials: list[dict], steps: list[dict]) -> dict[float, list[float]]:
    trial_meta = {_trial_key(record): float(record["mu"]) for record in trials}
    step_change = _mean_step_change_by_trial(steps)
    grouped: dict[float, list[float]] = defaultdict(list)
    for trial_key, value in step_change.items():
        if trial_key not in trial_meta:
            continue
        grouped[trial_meta[trial_key]].append(float(value))
    return dict(grouped)


def _set_mu_ticks(ax, mus: list[float]) -> None:
    ticks = [t for t in range(-5, 6) if min(mus) <= t <= max(mus)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t).replace("-", "−") for t in ticks])


def _plot_series(
    ax,
    grouped: dict[float, list[float]],
    *,
    color: str,
    label: str,
    linestyle: str,
    marker: str,
    linewidth: float = 1.2,
    alpha: float = 0.16,
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
    ax.plot(x, y, color=color, linestyle=linestyle, marker=marker, linewidth=linewidth, markersize=3.2, label=label)
    ax.fill_between(x, y - ci_arr, y + ci_arr, color=color, alpha=alpha, linewidth=0.0)
    return mus


def _plot_panel(
    ax,
    *,
    panel_label: str,
    task_grouped: dict[float, list[float]],
    masc_grouped: dict[float, list[float]],
    task_color: str,
    task_label: str,
) -> None:
    mus = _plot_series(
        ax,
        task_grouped,
        color=task_color,
        label=task_label,
        linestyle="-",
        marker="o",
        linewidth=1.4,
        alpha=0.16,
    )
    ax_right = ax.twinx()
    _plot_series(
        ax_right,
        masc_grouped,
        color=CEC_ORANGE,
        label="MASC",
        linestyle="--",
        marker="s",
        linewidth=1.0,
        alpha=0.10,
    )
    _set_mu_ticks(ax, mus)
    ax.set_xlabel("μ")
    ax.set_ylabel(task_label)
    ax_right.set_ylabel("MASC")
    ax.text(
        0.5,
        -0.25,
        f"({panel_label})",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )
    lines0, labels0 = ax.get_legend_handles_labels()
    lines1, labels1 = ax_right.get_legend_handles_labels()
    ax.legend(lines0 + lines1, labels0 + labels1, frameon=False, loc="lower right")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create journal Fig1: phase-transition panels.")
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    cons_trials_all, cons_steps_all = _load_runs(args.outputs_root, CONSENSUS_RUNS)
    diff_trials_all, diff_steps_all = _load_runs(args.outputs_root, DIFFUSION_RUNS)

    cons_trials = _filter_main_trials(cons_trials_all)
    diff_trials = _filter_main_trials(diff_trials_all)
    cons_steps = _filter_steps_for_trials(cons_steps_all, cons_trials)
    diff_steps = _filter_steps_for_trials(diff_steps_all, diff_trials)

    fig, axes = plt.subplots(1, 2, figsize=double_column_figsize(60), constrained_layout=False)
    fig.subplots_adjust(left=0.065, right=0.92, bottom=0.24, top=0.97, wspace=0.45)
    _plot_panel(
        axes[0],
        panel_label="a",
        task_grouped=_metric_by_mu(cons_trials, "final_dispersion"),
        masc_grouped=_masc_by_mu(cons_trials, cons_steps),
        task_color=CEC_BLUE,
        task_label="MAD",
    )
    _plot_panel(
        axes[1],
        panel_label="b",
        task_grouped=_metric_by_mu(diff_trials, "roughness_final"),
        masc_grouped=_masc_by_mu(diff_trials, diff_steps),
        task_color=CEC_GREEN,
        task_label="roughness",
    )
    save_figure(fig, args.out_dir / "Fig1.pdf", pad_inches=0.0, bbox_inches=None)


if __name__ == "__main__":
    main()
