"""ICM charts: equity by stack, bubble factor heatmap, push/fold range grid."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
from numpy.typing import NDArray

from ..themes import (
    FIG_FACECOLOR, AXES_FACECOLOR, AXIS_COLOR, GRID_COLOR, TEXT_PRIMARY,
    TEXT_SECONDARY, EV_POSITIVE, EV_NEGATIVE, EV_WARNING, MEDIAN_COLOR,
    BAND_COLOR, HISTOGRAM_COLOR, LINE_WEIGHT_MAIN, LINE_WEIGHT_SECONDARY,
    LINE_WEIGHT_GRID, FONT_TITLE, FONT_LABEL, FONT_TICK, FONT_ANNOTATION, FIG_DPI,
)
from ..formatters import fmt_dollar, fmt_prob


def icm_equity_by_stack(
    equities: NDArray[np.float64],
    stacks: NDArray[np.float64],
    total_prize_pool: float,
    title: str = "ICM Equity by Stack",
) -> Figure:
    """Horizontal bar chart of ICM equity, stacks ordered by size, prize pool as reference.

    Args:
        equities: Dollar equity per stack.
        stacks: Chip stack sizes (same length as equities).
        total_prize_pool: Sum of all payouts (for reference line).
        title: Chart title.

    Returns:
        Matplotlib Figure.
    """
    n = len(stacks)
    # Sort by stack size descending
    order = np.argsort(-stacks)
    sorted_stacks = stacks[order]
    sorted_equities = equities[order]
    labels = [f"Stack {i+1} ({int(sorted_stacks[i]):,})" for i in range(n)]

    fig, ax = plt.subplots(figsize=(8, max(3, n * 0.5 + 1)), dpi=FIG_DPI)
    fig.patch.set_facecolor(FIG_FACECOLOR)
    ax.set_facecolor(AXES_FACECOLOR)

    colors = [EV_POSITIVE if eq > 0 else AXIS_COLOR for eq in sorted_equities]
    bars = ax.barh(range(n), sorted_equities, color=colors, edgecolor=AXIS_COLOR, linewidth=0.5)

    # Direct labels on bars
    for i, (bar, eq) in enumerate(zip(bars, sorted_equities)):
        ax.text(eq + total_prize_pool * 0.01, i, fmt_dollar(eq),
                va="center", color=TEXT_SECONDARY, fontsize=FONT_ANNOTATION)

    ax.axvline(total_prize_pool / n, color=AXIS_COLOR, linewidth=LINE_WEIGHT_GRID, linestyle="--",
               label=f"Even split = {fmt_dollar(total_prize_pool / n)}")

    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, color=TEXT_SECONDARY, fontsize=FONT_TICK)
    ax.set_xlabel("Dollar Equity", color=TEXT_SECONDARY, fontsize=FONT_LABEL)
    ax.tick_params(colors=AXIS_COLOR, labelsize=FONT_TICK)
    ax.invert_yaxis()
    ax.grid(True, color=GRID_COLOR, linewidth=LINE_WEIGHT_GRID, axis="x")
    ax.legend(fontsize=FONT_ANNOTATION, facecolor=AXES_FACECOLOR, edgecolor=AXIS_COLOR, labelcolor=TEXT_SECONDARY)

    ax.set_title(title, color=TEXT_PRIMARY, fontsize=FONT_TITLE, pad=12)
    fig.tight_layout()
    return fig


def bubble_factor_heatmap(
    bf_matrix: NDArray[np.float64],
    player_labels: list[str] | None = None,
    title: str = "Bubble Factor Matrix",
) -> Figure:
    """Heatmap of bubble factors: player × opponent scenario matrix.

    Args:
        bf_matrix: 2-D square matrix of bubble factors.
        player_labels: Labels for each player.
        title: Chart title.

    Returns:
        Matplotlib Figure.
    """
    n = bf_matrix.shape[0]
    if player_labels is None:
        player_labels = [f"P{i+1}" for i in range(n)]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=FIG_DPI)
    fig.patch.set_facecolor(FIG_FACECOLOR)
    ax.set_facecolor(AXES_FACECOLOR)

    vmin = max(0.5, float(np.min(bf_matrix)))
    vmax = min(3.0, float(np.max(bf_matrix)))
    im = ax.imshow(bf_matrix, cmap="YlOrRd", aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(player_labels, fontsize=FONT_TICK, color=TEXT_SECONDARY, rotation=45, ha="right")
    ax.set_yticklabels(player_labels, fontsize=FONT_TICK, color=TEXT_SECONDARY)
    ax.tick_params(colors=AXIS_COLOR)

    # Annotate — BF > 1 is the interesting case (amber/red)
    for i in range(n):
        for j in range(n):
            val = bf_matrix[i, j]
            color = TEXT_PRIMARY if val < 1.5 else AXES_FACECOLOR
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=FONT_ANNOTATION)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=AXIS_COLOR, labelsize=FONT_TICK)
    cbar.set_label("Bubble Factor", color=TEXT_SECONDARY, fontsize=FONT_LABEL)

    ax.set_title(title, color=TEXT_PRIMARY, fontsize=FONT_TITLE, pad=12)
    fig.tight_layout()
    return fig


def push_fold_range_grid(
    ev_matrix: NDArray[np.float64],
    hand_labels: tuple[list[str], list[str]] | None = None,
    title: str = "Push/Fold EV Range",
) -> Figure:
    """Color-encoded 13×13 poker hand matrix for push/fold EV.

    Args:
        ev_matrix: 13×13 array where rows=suited hands, cols=offsuit hands (or vice versa).
            Positive EV = push, negative = fold, zero = indifferent.
        hand_labels: Tuple of (row_labels, col_labels) — defaults to standard ranks A K Q J T 9 8 7 6 5 4 3 2.
        title: Chart title.

    Returns:
        Matplotlib Figure.
    """
    ranks = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
    if hand_labels is None:
        row_labels, col_labels = ranks, ranks
    else:
        row_labels, col_labels = hand_labels

    # Custom diverging colormap: red (fold) → black (indifferent) → green (push)
    cmap = ListedColormap([EV_NEGATIVE, AXES_FACECOLOR, EV_POSITIVE])

    fig, ax = plt.subplots(figsize=(7, 6), dpi=FIG_DPI)
    fig.patch.set_facecolor(FIG_FACECOLOR)
    ax.set_facecolor(AXES_FACECOLOR)

    vmax = max(0.001, float(np.max(np.abs(ev_matrix))))
    im = ax.imshow(ev_matrix, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="equal")

    ax.set_xticks(range(13))
    ax.set_yticks(range(13))
    ax.set_xticklabels(col_labels, fontsize=FONT_TICK, color=TEXT_SECONDARY)
    ax.set_yticklabels(row_labels, fontsize=FONT_TICK, color=TEXT_SECONDARY)
    ax.tick_params(colors=AXIS_COLOR)

    # Diagonal = pairs, upper triangle = suited, lower = offsuit (convention)
    ax.set_xlabel("Offsuit", color=TEXT_MUTED, fontsize=FONT_ANNOTATION)
    ax.set_ylabel("Suited / Pairs", color=TEXT_MUTED, fontsize=FONT_ANNOTATION)

    ax.set_title(title, color=TEXT_PRIMARY, fontsize=FONT_TITLE, pad=12)
    fig.tight_layout()
    return fig
