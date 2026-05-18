"""Sports betting charts: CLV distribution, CLV over time, pattern flags."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from numpy.typing import NDArray

from ..themes import (
    FIG_FACECOLOR, AXES_FACECOLOR, AXIS_COLOR, GRID_COLOR, TEXT_PRIMARY,
    TEXT_SECONDARY, TEXT_MUTED, EV_POSITIVE, EV_NEGATIVE, EV_WARNING,
    MEDIAN_COLOR, HISTOGRAM_COLOR, LINE_WEIGHT_MAIN, LINE_WEIGHT_SECONDARY,
    LINE_WEIGHT_GRID, FONT_TITLE, FONT_LABEL, FONT_TICK, FONT_ANNOTATION, FIG_DPI,
)
from ..formatters import fmt_pct, fmt_prob


def clv_distribution(
    clv_values: NDArray[np.float64],
    mean_clv: float,
    median_clv: float,
    title: str = "CLV Distribution",
) -> Figure:
    """Histogram of CLV values with mean, median, and zero marked explicitly.

    Args:
        clv_values: 1-D array of CLV values (fraction, not percentage).
        mean_clv: Mean CLV.
        median_clv: Median CLV.
        title: Chart title.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5), dpi=FIG_DPI)
    fig.patch.set_facecolor(FIG_FACECOLOR)
    ax.set_facecolor(AXES_FACECOLOR)

    ax.hist(clv_values, bins=40, color=HISTOGRAM_COLOR, edgecolor=AXIS_COLOR, linewidth=0.5, alpha=0.85)

    ax.axvline(0, color=AXIS_COLOR, linewidth=LINE_WEIGHT_MAIN, linestyle="-", label="Zero")
    ax.axvline(mean_clv, color=EV_POSITIVE, linewidth=LINE_WEIGHT_SECONDARY, linestyle="--",
               label=f"Mean = {fmt_pct(mean_clv)}")
    ax.axvline(median_clv, color=MEDIAN_COLOR, linewidth=LINE_WEIGHT_SECONDARY, linestyle=":",
               label=f"Median = {fmt_pct(median_clv)}")

    ax.set_xlabel("CLV", color=TEXT_SECONDARY, fontsize=FONT_LABEL)
    ax.set_ylabel("Count", color=TEXT_SECONDARY, fontsize=FONT_LABEL)
    ax.tick_params(colors=AXIS_COLOR, labelsize=FONT_TICK)
    ax.grid(True, color=GRID_COLOR, linewidth=LINE_WEIGHT_GRID, axis="y")
    ax.legend(fontsize=FONT_ANNOTATION, facecolor=AXES_FACECOLOR, edgecolor=AXIS_COLOR, labelcolor=TEXT_SECONDARY)

    ax.set_title(title, color=TEXT_PRIMARY, fontsize=FONT_TITLE, pad=12)
    fig.tight_layout()
    return fig


def clv_over_time(
    timestamps: list[str],
    clv_values: NDArray[np.float64],
    loess_x: NDArray[np.float64] | None = None,
    loess_y: NDArray[np.float64] | None = None,
    title: str = "CLV Over Time",
) -> Figure:
    """Scatter of CLV over time with optional LOESS smoother and prominent zero line.

    Args:
        timestamps: ISO timestamp strings for x-axis.
        clv_values: 1-D array of CLV values.
        loess_x: Optional LOESS smoother x values.
        loess_y: Optional LOESS smoother y values.
        title: Chart title.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 5), dpi=FIG_DPI)
    fig.patch.set_facecolor(FIG_FACECOLOR)
    ax.set_facecolor(AXES_FACECOLOR)

    # Color points by sign
    colors = np.where(clv_values > 0, EV_POSITIVE, EV_NEGATIVE)
    ax.scatter(range(len(clv_values)), clv_values, c=colors, s=8, alpha=0.6, edgecolors="none")

    if loess_x is not None and loess_y is not None:
        ax.plot(loess_x, loess_y, color=MEDIAN_COLOR, linewidth=LINE_WEIGHT_MAIN, label="LOESS")

    ax.axhline(0, color=AXIS_COLOR, linewidth=LINE_WEIGHT_MAIN, linestyle="-", label="Zero CLV")

    # Sparse x-axis labels
    step = max(1, len(timestamps) // 8)
    tick_positions = list(range(0, len(timestamps), step))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([timestamps[i][:10] for i in tick_positions], fontsize=FONT_TICK, color=TEXT_MUTED, rotation=30, ha="right")

    ax.set_xlabel("Date", color=TEXT_SECONDARY, fontsize=FONT_LABEL)
    ax.set_ylabel("CLV", color=TEXT_SECONDARY, fontsize=FONT_LABEL)
    ax.tick_params(colors=AXIS_COLOR, labelsize=FONT_TICK)
    ax.grid(True, color=GRID_COLOR, linewidth=LINE_WEIGHT_GRID)
    ax.legend(fontsize=FONT_ANNOTATION, facecolor=AXES_FACECOLOR, edgecolor=AXIS_COLOR, labelcolor=TEXT_SECONDARY)

    ax.set_title(title, color=TEXT_PRIMARY, fontsize=FONT_TITLE, pad=12)
    fig.tight_layout()
    return fig


def pattern_flags_small_multiples(
    flag_data: dict[str, list[dict[str, object]]],
    metric_key: str = "mean_clv",
    title: str = "CLV Pattern Flags",
) -> Figure:
    """Small multiples by sport/book/tier with consistent axes across panels.

    Args:
        flag_data: Dict mapping group name (e.g. sport name) to list of flag dicts.
            Each flag dict has at least the metric_key and a "value" key for the sub-group name.
        metric_key: Key in each flag dict for the metric to display.
        title: Chart title.

    Returns:
        Matplotlib Figure.
    """
    groups = list(flag_data.keys())
    n = len(groups)
    if n == 0:
        fig, ax = plt.subplots(figsize=(4, 3), dpi=FIG_DPI)
        fig.patch.set_facecolor(FIG_FACECOLOR)
        ax.set_facecolor(AXES_FACECOLOR)
        ax.text(0.5, 0.5, "No pattern flags", ha="center", va="center", color=TEXT_MUTED, fontsize=FONT_LABEL)
        ax.set_axis_off()
        return fig

    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3), dpi=FIG_DPI, squeeze=False)
    fig.patch.set_facecolor(FIG_FACECOLOR)

    # Compute global range for consistent axes
    all_vals = [float(f[metric_key]) for flags in flag_data.values() for f in flags]
    global_min = min(all_vals) if all_vals else -0.05
    global_max = max(all_vals) if all_vals else 0.05
    margin = (global_max - global_min) * 0.1
    global_min -= margin
    global_max += margin

    for idx, (group_name, flags) in enumerate(flag_data.items()):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        ax.set_facecolor(AXES_FACECOLOR)

        values = [float(f[metric_key]) for f in flags]
        names = [str(f.get("value", f"item{i}")) for i, f in enumerate(flags)]
        colors = [EV_POSITIVE if v > 0 else EV_NEGATIVE for v in values]

        ax.barh(range(len(values)), values, color=colors, edgecolor=AXIS_COLOR, linewidth=0.5)
        ax.axvline(0, color=AXIS_COLOR, linewidth=LINE_WEIGHT_GRID, linestyle="--")
        ax.set_xlim(global_min, global_max)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=FONT_TICK - 1, color=TEXT_SECONDARY)
        ax.tick_params(colors=AXIS_COLOR, labelsize=FONT_TICK - 1)
        ax.set_title(group_name, color=TEXT_PRIMARY, fontsize=FONT_ANNOTATION + 1, pad=4)
        ax.grid(True, color=GRID_COLOR, linewidth=LINE_WEIGHT_GRID, axis="x")

    # Hide empty panels
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_axis_off()

    fig.suptitle(title, color=TEXT_PRIMARY, fontsize=FONT_TITLE, y=1.01)
    fig.tight_layout()
    return fig
