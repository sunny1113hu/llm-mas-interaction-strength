from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, patches

from src.utils.analysis import load_jsonl
from src.utils.journal_figures import apply_journal_style, double_column_figsize, save_figure


OUTPUT_DEFAULT = Path("outputs/runs/paper_figures_journal/Figures")
MAIN_MODEL = "Qwen/Qwen3-8B"
MU_TARGETS = [-5.0, -0.25, 5.0]
SIGMA_TARGET = 0.5

CONSENSUS_RUNS = [
    "20260109_102923_consensus_4x4_4llms",
    "20260116_110026_consensus_ministral14B",
    "20260120_003407_consensus",
]
DIFFUSION_RUNS = [
    "20260115_073051_diffusion_4x4_4llms",
    "20260116_110026_diffusion_ministral14B",
    "20260120_003407_diffusion",
]


apply_journal_style(font_size=8.5, title_size=9, label_size=8.5, tick_size=8, legend_size=8)


def _fmt_mu(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return "−" + text[1:] if text.startswith("-") else text


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
    filtered = []
    for record in trials:
        if record.get("model_id") != MAIN_MODEL:
            continue
        if (record.get("prompt_mode") or "with_pi") != "with_pi":
            continue
        if int(record.get("grid_size", 4)) != 4:
            continue
        if int(record.get("steps", 10)) != 10:
            continue
        filtered.append(record)
    return filtered


def _filter_steps_for_trials(steps: list[dict], trials: list[dict]) -> list[dict]:
    keys = {_trial_key(record) for record in trials}
    return [record for record in steps if _trial_key(record) in keys and record.get("model_id") == MAIN_MODEL]


def _find_trial(trials: list[dict], *, mu: float, seed: int) -> dict:
    for record in trials:
        if abs(float(record.get("mu", np.nan)) - mu) > 1e-9:
            continue
        if abs(float(record.get("sigma", np.nan)) - SIGMA_TARGET) > 1e-9:
            continue
        if int(record.get("seed", -1)) != seed:
            continue
        return record
    raise RuntimeError(f"Representative trial not found: mu={mu}, sigma={SIGMA_TARGET}, seed={seed}")


def _index_steps(steps: list[dict]) -> dict[tuple[tuple[str, int], int], dict[int, dict]]:
    indexed: dict[tuple[tuple[str, int], int], dict[int, dict]] = defaultdict(dict)
    for record in steps:
        trial_key = _trial_key(record)
        step = int(record.get("step", 0))
        agent_id = int(record.get("agent_id", -1))
        indexed[(trial_key, step)][agent_id] = record
    return indexed


def _grid_from_index(
    indexed_steps: dict[tuple[tuple[str, int], int], dict[int, dict]],
    *,
    trial_key: tuple[str, int],
    step: int,
    grid_size: int,
) -> np.ndarray:
    grid = np.full((grid_size, grid_size), np.nan)
    for agent_id, record in indexed_steps.get((trial_key, step), {}).items():
        value = record.get("value")
        if not isinstance(value, int):
            continue
        row, col = divmod(agent_id, grid_size)
        grid[row, col] = value
    return grid


def _draw_grid(
    ax,
    grid: np.ndarray,
    *,
    norm: colors.Normalize,
    cmap,
    anchor_cells: list[tuple[int, int]] | None = None,
) -> None:
    size = grid.shape[0]
    x = np.arange(size + 1)
    y = np.arange(size + 1)
    ax.pcolormesh(
        x,
        y,
        np.flipud(grid),
        cmap=cmap,
        norm=norm,
        edgecolors="white",
        linewidth=0.3,
        shading="flat",
    )
    if anchor_cells:
        for row, col in anchor_cells:
            ax.add_patch(
                patches.Rectangle(
                    (col, size - 1 - row),
                    1.0,
                    1.0,
                    fill=False,
                    edgecolor="red",
                    linewidth=0.8,
                    clip_on=False,
                    zorder=5,
                )
            )
    ax.set_aspect("equal")
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.3)


def _draw_snapshot_panel(
    fig,
    subgrid,
    *,
    trials: list[dict],
    steps: list[dict],
    seed: int,
    panel_label: str,
    mark_anchors: bool = False,
) -> None:
    indexed_steps = _index_steps(steps)
    step_values = list(range(0, 11))
    cmap = plt.get_cmap("viridis")
    norm = colors.Normalize(vmin=0, vmax=25)

    for row_idx, mu in enumerate(MU_TARGETS):
        trial = _find_trial(trials, mu=mu, seed=seed)
        trial_key = _trial_key(trial)
        grid_size = int(trial.get("grid_size", 4))
        label_ax = plt.subplot(subgrid[row_idx, 0])
        label_ax.set_axis_off()
        label_ax.text(0.95, 0.5, f"μ={_fmt_mu(mu)}", ha="right", va="center", fontsize=8)
        anchor_cells = [(0, 0), (grid_size - 1, grid_size - 1)] if mark_anchors else None
        for col_idx, step in enumerate(step_values):
            ax = plt.subplot(subgrid[row_idx, col_idx + 1])
            grid = _grid_from_index(indexed_steps, trial_key=trial_key, step=step, grid_size=grid_size)
            _draw_grid(ax, grid, norm=norm, cmap=cmap, anchor_cells=anchor_cells)
            if row_idx == 0:
                ax.set_title(f"t={step}", fontsize=8, pad=3.0)

    cax = plt.subplot(subgrid[:, -1])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label("value", fontsize=8)
    cbar.ax.tick_params(labelsize=8, length=2, width=0.3)
    cbar.outline.set_linewidth(0.3)

    label_ax = plt.subplot(subgrid[3, 1:-1])
    label_ax.set_axis_off()
    label_ax.text(0.5, 0.40, f"({panel_label})", ha="center", va="center", fontsize=9)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create journal Fig2: snapshots redrawn from step logs.")
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

    fig = plt.figure(figsize=double_column_figsize(112))
    outer = fig.add_gridspec(
        nrows=2,
        ncols=1,
        left=0.018,
        right=0.935,
        bottom=0.03,
        top=0.955,
        hspace=0.24,
    )
    width_ratios = [0.95] + [1.0] * 11 + [0.12]
    height_ratios = [1.0, 1.0, 1.0, 0.18]
    cons_grid = outer[0].subgridspec(4, 13, width_ratios=width_ratios, height_ratios=height_ratios, wspace=0.07, hspace=0.12)
    diff_grid = outer[1].subgridspec(4, 13, width_ratios=width_ratios, height_ratios=height_ratios, wspace=0.07, hspace=0.12)

    _draw_snapshot_panel(fig, cons_grid, trials=cons_trials, steps=cons_steps, seed=11, panel_label="a")
    # Anchor outlines are intentionally disabled: the red frames were visually
    # distracting and the anchored gradient is evident from the values themselves.
    _draw_snapshot_panel(fig, diff_grid, trials=diff_trials, steps=diff_steps, seed=42, panel_label="b", mark_anchors=False)

    save_figure(fig, args.out_dir / "Fig2.pdf", pad_inches=0.0, bbox_inches=None)


if __name__ == "__main__":
    main()
