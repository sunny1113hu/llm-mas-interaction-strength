from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MM_PER_INCH = 25.4
SINGLE_COLUMN_MM = 84.0
DOUBLE_COLUMN_MM = 174.0
MAX_HEIGHT_MM = 234.0
JOURNAL_DPI = 1200
MIN_LINEWIDTH_PT = 0.3
JOURNAL_FONT_STACK = [
    "Arial",
    "Liberation Sans",
    "Helvetica",
    "Nimbus Sans",
    "DejaVu Sans",
]


def mm_to_inches(mm: float) -> float:
    return float(mm) / MM_PER_INCH


def journal_figsize(width_mm: float = DOUBLE_COLUMN_MM, height_mm: float = 90.0) -> tuple[float, float]:
    height_mm = min(float(height_mm), MAX_HEIGHT_MM)
    return mm_to_inches(width_mm), mm_to_inches(height_mm)


def single_column_figsize(height_mm: float = 65.0) -> tuple[float, float]:
    return journal_figsize(SINGLE_COLUMN_MM, height_mm)


def double_column_figsize(height_mm: float = 90.0) -> tuple[float, float]:
    return journal_figsize(DOUBLE_COLUMN_MM, height_mm)


def apply_journal_style(
    *,
    font_size: float = 10.0,
    title_size: float = 11.0,
    label_size: float = 10.0,
    tick_size: float = 9.0,
    legend_size: float = 9.0,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": JOURNAL_FONT_STACK,
            "font.size": font_size,
            "axes.titlesize": title_size,
            "axes.labelsize": label_size,
            "xtick.labelsize": tick_size,
            "ytick.labelsize": tick_size,
            "legend.fontsize": legend_size,
            "axes.linewidth": MIN_LINEWIDTH_PT,
            "grid.linewidth": MIN_LINEWIDTH_PT,
            "lines.linewidth": 1.0,
            "lines.markersize": 4.0,
            "patch.linewidth": MIN_LINEWIDTH_PT,
            "hatch.linewidth": MIN_LINEWIDTH_PT,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(
    fig,
    path: Path,
    *,
    pad_inches: float = 0.05,
    dpi: int = JOURNAL_DPI,
    bbox_inches: str | None = "tight",
) -> None:
    path = Path(path)
    stem = path.with_suffix("")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches=bbox_inches, pad_inches=pad_inches)
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
    plt.close(fig)


def add_panel_labels(
    axes,
    *,
    xpos: float = -0.10,
    ypos: float = 1.02,
    fontsize: float = 10.0,
    weight: str = "bold",
) -> None:
    flat_axes = np.atleast_1d(axes).ravel()
    for idx, ax in enumerate(flat_axes):
        label = f"({chr(ord('a') + idx)})"
        ax.text(
            xpos,
            ypos,
            label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=fontsize,
            fontweight=weight,
        )
