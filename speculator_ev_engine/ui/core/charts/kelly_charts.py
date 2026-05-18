"""Kelly criterion charts: fraction vs growth, correlated heatmap, uncertain edge distribution."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from numpy.typing import NDArray

from ..themes import (
    FIG_FACECOLOR, AXES_FACECOLOR, AXIS_COLOR, GRID_COLOR, TEXT_PRIMARY,
    TEXT_SECONDARY, EV_POSITIVE, EV_NEGATIVE, EV_WARNING, MEDIAN_COLOR,
    BAND_COLOR, LINE_WEIGHT_MAIN, LINE_WEIGHT_SECONDARY, LINE_WEIGHT_GRID,
    FONT_TITLE, FONT_LABEL, FONT_TICK, FONT_ANNOTATION, FIG_DPI,
)
from ..formatters import fmt_frac, fmt_ev, fmt_ruin, fmt_prob


def kelly_fraction_vs_growth(
    fractions: NDArray[np.float64],
    log_growths: NDArray[np.float64],
    ruin_probs: NDArray[np.float64],
    optimal_idx: int | None = None,
    title: str = "Kelly Fraction vs Expected Log Growth",
) -> Figure:
    """Line chart: Kelly fraction vs expected log growth rate with ruin probability on secondary axis.

    Args:
        fractions: 1-D array of Kelly fractions.
        log_growths: 1-D array of expected log growth rates.
        ruin_probs: 1-D array of ruin probabilities.
        optimal_idx: Index of the optimal (maximum growth) fraction.
        title: Chart title.

    Returns:
        Matplotlib Figure.
    """
    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=FIG_DPI)
    fig.patch.set_facecolor(FIG_FACECOLOR)
    ax1.set_facecolor(AXES_FACECOLOR)

    ax1.plot(fractions, log_growths, color=EV_POSITIVE, linewidth=LINE_WEIGHT_MAIN)
    ax1.set_xlabel("Kelly Fraction", color=TEXT_SECONDARY, fontsize=FONT_LABEL)
    ax1.set_ylabel("Expected Log Growth", color=EV_POSITIVE, fontsize=FONT_LABEL)
    ax1.tick_params(colors=AXIS_COLOR, labelsize=FONT_TICK)
    ax1.axhline(0, color=AXIS_COLOR, linewidth=LINE_WEIGHT_GRID, linestyle="--")
    ax1.grid(True, color=GRID_COLOR, linewidth=LINE_WEIGHT_GRID)

    # Mark optimal point
    if optimal_idx is not None:
        ax1.axvline(fractions[optimal_idx], color=EV_WARNING, linewidth=LINE_WEIGHT_SECONDARY, linestyle="--")
        ax1.annotate(
            f"opt f={fmt_frac(fractions[optimal_idx])}",
            xy=(fractions[optimal_idx], log_growths[optimal_idx]),
            xytext=(10, 10), textcoords="offset points",
            color=EV_WARNING, fontsize=FONT_ANNOTATION,
        )

    # Ruin probability on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(fractions, ruin_probs, color=EV_NEGATIVE, linewidth=LINE_WEIGHT_SECONDARY, linestyle=":")
    ax2.set_ylabel("Ruin Probability", color=EV_NEGATIVE, fontsize=FONT_LABEL)
    ax2.tick_params(colors=AXIS_COLOR, labelsize=FONT_TICK)
    ax2.set_ylim(bottom=0)

    ax1.set_title(title, color=TEXT_PRIMARY, fontsize=FONT_TITLE, pad=12)
    fig.tight_layout()
    return fig


def correlated_kelly_heatmap(
    fractions_matrix: NDArray[np.float64],
    asset_labels: list[str] | None = None,
    title: str = "Kelly Fractions Across Correlated Bets",
) -> Figure:
    """Heatmap of Kelly fractions for correlated bet pairs.

    Args:
        fractions_matrix: 2-D square matrix of Kelly fractions.
        asset_labels: Labels for each axis.
        title: Chart title.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(6, 5), dpi=FIG_DPI)
    fig.patch.set_facecolor(FIG_FACECOLOR)
    ax.set_facecolor(AXES_FACECOLOR)

    n = fractions_matrix.shape[0]
    if asset_labels is None:
        asset_labels = [f"Bet {i+1}" for i in range(n)]

    im = ax.imshow(fractions_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=max(0.25, float(np.max(fractions_matrix))))
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(asset_labels, fontsize=FONT_TICK, color=TEXT_SECONDARY, rotation=45, ha="right")
    ax.set_yticklabels(asset_labels, fontsize=FONT_TICK, color=TEXT_SECONDARY)
    ax.tick_params(colors=AXIS_COLOR)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = fractions_matrix[i, j]
            color = TEXT_PRIMARY if val < 0.12 else AXES_FACECOLOR
            ax.text(j, i, fmt_frac(val), ha="center", va="center", color=color, fontsize=FONT_ANNOTATION)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=AXIS_COLOR, labelsize=FONT_TICK)
    cbar.set_label("Kelly Fraction", color=TEXT_SECONDARY, fontsize=FONT_LABEL)

    ax.set_title(title, color=TEXT_PRIMARY, fontsize=FONT_TITLE, pad=12)
    fig.tight_layout()
    return fig


def uncertain_edge_kelly(
    fraction_samples: NDArray[np.float64],
    median_fraction: float,
    title: str = "Kelly Fraction Distribution Under Edge Uncertainty",
) -> Figure:
    """Shaded histogram showing Kelly fraction distribution given edge uncertainty.

    Args:
        fraction_samples: 1-D array of Kelly fractions from Monte Carlo edge draws.
        median_fraction: The median (recommended) fraction.
        title: Chart title.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(7, 4), dpi=FIG_DPI)
    fig.patch.set_facecolor(FIG_FACECOLOR)
    ax.set_facecolor(AXES_FACECOLOR)

    ax.hist(fraction_samples, bins=40, color=BAND_COLOR, edgecolor=AXIS_COLOR, linewidth=0.5, alpha=0.85)
    ax.axvline(median_fraction, color=EV_WARNING, linewidth=LINE_WEIGHT_MAIN, linestyle="--",
               label=f"median = {fmt_frac(median_fraction)}")
    ax.axvline(0, color=AXIS_COLOR, linewidth=LINE_WEIGHT_GRID, linestyle=":")

    ax.set_xlabel("Kelly Fraction", color=TEXT_SECONDARY, fontsize=FONT_LABEL)
    ax.set_ylabel("Count", color=TEXT_SECONDARY, fontsize=FONT_LABEL)
    ax.tick_params(colors=AXIS_COLOR, labelsize=FONT_TICK)
    ax.grid(True, color=GRID_COLOR, linewidth=LINE_WEIGHT_GRID, axis="y")
    ax.legend(fontsize=FONT_ANNOTATION, facecolor=AXES_FACECOLOR, edgecolor=AXIS_COLOR, labelcolor=TEXT_SECONDARY)

    ax.set_title(title, color=TEXT_PRIMARY, fontsize=FONT_TITLE, pad=12)
    fig.tight_layout()
    return fig
