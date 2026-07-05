from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.analysis import load_jsonl
from src.utils.cec_style import CEC_BLUE, CEC_RED
from src.utils.journal_figures import apply_journal_style, double_column_figsize, save_figure


OUTPUT_DEFAULT = Path("outputs/runs/paper_figures_journal/Figures")
QWEN_MODEL = "Qwen/Qwen3-8B"
MOBILITY_MODEL = "baseline/mobility_scaled"

TASK_SPECS = {
    "consensus": {
        "metric": "final_dispersion",
        "qwen_runs": ["20260109_102923_consensus_4x4_4llms", "20260120_003407_consensus"],
        "mobility_runs": ["baseline_v2_consensus_mobility_scaled"],
    },
    "diffusion": {
        "metric": "roughness_final",
        "qwen_runs": ["20260115_073051_diffusion_4x4_4llms", "20260120_003407_diffusion"],
        "mobility_runs": ["baseline_v2_diffusion_mobility_scaled"],
    },
}


apply_journal_style(font_size=9, title_size=10, label_size=9, tick_size=8, legend_size=8)


def _trial_key(record: dict) -> tuple[str, int]:
    return str(record.get("run_id", "")), int(record.get("trial_id", -1))


def _load_runs(outputs_root: Path, run_ids: list[str]) -> tuple[list[dict], list[dict]]:
    trials: list[dict] = []
    steps: list[dict] = []
    for run_id in run_ids:
        run_trials = load_jsonl(outputs_root / run_id / "trial_summary.jsonl")
        run_steps = load_jsonl(outputs_root / run_id / "steps.jsonl")
        for record in run_trials:
            record.setdefault("run_id", run_id)
        for record in run_steps:
            record.setdefault("run_id", run_id)
        trials.extend(run_trials)
        steps.extend(run_steps)
    return trials, steps


def _filter_trials(
    trials: list[dict],
    *,
    model_id: str,
    prompt_mode: str = "with_pi",
    grid_size: int = 4,
    steps: int = 10,
) -> list[dict]:
    filtered = []
    for record in trials:
        if record.get("model_id") != model_id:
            continue
        if (record.get("prompt_mode") or "with_pi") != prompt_mode:
            continue
        if int(record.get("grid_size", grid_size)) != grid_size:
            continue
        if int(record.get("steps", steps)) != steps:
            continue
        filtered.append(record)
    return filtered


def _filter_steps_for_trials(step_records: list[dict], trial_records: list[dict]) -> list[dict]:
    keys = {_trial_key(record) for record in trial_records}
    return [record for record in step_records if _trial_key(record) in keys]


def _best_mu(trials: list[dict], metric_key: str) -> float:
    grouped: dict[float, list[float]] = defaultdict(list)
    for record in trials:
        mu = record.get("mu")
        metric = record.get(metric_key)
        if mu is None or not isinstance(metric, (int, float)):
            continue
        grouped[float(mu)].append(float(metric))
    means = {mu: float(np.mean(vals)) for mu, vals in grouped.items()}
    return min(means, key=means.get)


def _response_records(step_records: list[dict], mu: float) -> list[dict]:
    records: list[dict] = []
    for record in step_records:
        if int(record.get("step", 0)) <= 0:
            continue
        if abs(float(record.get("mu")) - mu) > 1e-9:
            continue
        if record.get("role") == "anchor":
            continue
        records.append(
            {
                "pressure": int(record.get("neighbors_higher", 0)) - int(record.get("neighbors_lower", 0)),
                "actual_step": int(record.get("delta") or 0),
            }
        )
    return records


def _plot_panel(ax, rows: list[dict], *, color: str, ymin: float, ymax: float) -> None:
    pressures = sorted(set(int(row["pressure"]) for row in rows))
    grouped = [[row["actual_step"] for row in rows if row["pressure"] == p] for p in pressures]
    violin = ax.violinplot(
        grouped,
        positions=pressures,
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in violin["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_linewidth(0.6)
        body.set_alpha(0.34)

    means = [float(np.mean(vals)) for vals in grouped]
    ax.plot(
        pressures,
        means,
        color="#222222",
        marker="o",
        linewidth=1.8,
        markersize=4.0,
    )
    ax.axhline(0.0, color="#888888", linewidth=0.8, alpha=0.7)
    ax.set_xticks(range(-4, 5))
    ax.set_ylim(ymin, ymax)


def _add_panel_label(ax, label: str, *, y: float) -> None:
    ax.text(
        0.5,
        y,
        label,
        transform=ax.transAxes,
        ha="center",
        va="top",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create journal Fig12: zoomed step-response comparison.")
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    cache: dict[str, dict[str, object]] = {}

    for task, spec in TASK_SPECS.items():
        qwen_trials_all, qwen_steps_all = _load_runs(args.outputs_root, spec["qwen_runs"])
        mobility_trials_all, mobility_steps_all = _load_runs(args.outputs_root, spec["mobility_runs"])

        qwen_trials = _filter_trials(qwen_trials_all, model_id=QWEN_MODEL)
        mobility_trials = _filter_trials(mobility_trials_all, model_id=MOBILITY_MODEL)
        qwen_steps = _filter_steps_for_trials(qwen_steps_all, qwen_trials)
        mobility_steps = _filter_steps_for_trials(mobility_steps_all, mobility_trials)

        qwen_mu = _best_mu(qwen_trials, spec["metric"])
        mobility_mu = _best_mu(mobility_trials, spec["metric"])

        qwen_records = _response_records(qwen_steps, qwen_mu)
        mobility_records = _response_records(mobility_steps, mobility_mu)

        cache[task] = {
            "qwen_records": qwen_records,
            "mobility_records": mobility_records,
        }

    ymin, ymax = -8, 8

    fig, axes = plt.subplots(2, 2, figsize=double_column_figsize(96), constrained_layout=False, sharex=True, sharey=True)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.27, top=0.945, wspace=0.12, hspace=0.36)

    _plot_panel(axes[0, 0], cache["consensus"]["qwen_records"], color=CEC_BLUE, ymin=ymin, ymax=ymax)
    _plot_panel(axes[0, 1], cache["consensus"]["mobility_records"], color=CEC_RED, ymin=ymin, ymax=ymax)
    _plot_panel(axes[1, 0], cache["diffusion"]["qwen_records"], color=CEC_BLUE, ymin=ymin, ymax=ymax)
    _plot_panel(axes[1, 1], cache["diffusion"]["mobility_records"], color=CEC_RED, ymin=ymin, ymax=ymax)

    _add_panel_label(axes[0, 0], "(a) Qwen3-8B (consensus)", y=-0.12)
    _add_panel_label(axes[0, 1], "(b) RSP rule (consensus)", y=-0.12)
    _add_panel_label(axes[1, 0], "(c) Qwen3-8B (diffusion)", y=-0.18)
    _add_panel_label(axes[1, 1], "(d) RSP rule (diffusion)", y=-0.18)

    axes[0, 0].set_ylabel("Actual signed step")
    axes[1, 0].set_ylabel("Actual signed step")

    for ax in axes[0, :]:
        ax.tick_params(labelbottom=False)
    for ax in axes[:, 1]:
        ax.tick_params(labelleft=False)

    fig.supxlabel(r"Local pressure $\Delta = n_i^{>} - n_i^{<}$", y=0.13)

    legend_handles = [
        plt.Line2D([], [], color=CEC_BLUE, linewidth=6, alpha=0.34, label="Qwen3-8B distribution"),
        plt.Line2D([], [], color=CEC_RED, linewidth=6, alpha=0.34, label="RSP rule distribution"),
        plt.Line2D([], [], color="#222222", marker="o", linewidth=1.8, markersize=4.0, label="mean step"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=3,
        frameon=False,
        columnspacing=1.8,
        handlelength=2.2,
    )

    save_figure(fig, args.out_dir / "Fig12.pdf", pad_inches=0.0, bbox_inches=None)


if __name__ == "__main__":
    main()
