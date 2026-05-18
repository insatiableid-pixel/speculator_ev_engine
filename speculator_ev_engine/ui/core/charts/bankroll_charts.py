"""Bankroll simulation charts: Monte Carlo paths, drawdown distribution, fraction sweep."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from numpy.typing import NDArray

from ..themes import (
    FIG_FACECOLOR, AXES_FACECOLOR, AXIS_COLOR, GRID_COLOR, TEXT_PRIMARY,
    TEXT_SECONDARY, EV_POSITIVE, EV_NEGATIVE, EV_WARNING, MEDIAN_COLOR,
    BAND_COLOR, HISTOGRAM_COLOR, LINE_WEIGHT_MAIN, LINE_WEIGHT_SECONDARY,
    LINE_WEIGHT_GRID, FONT_TITLE, FONT_LABEL, FONT_TICK, FONT_ANNOTATION, FIG_DPI,
)
from ..formatters import fmt_frac, fmt_ev, fmt_ruin, fmt_drawdown


def bankroll_monte_carlo(
    paths: NDArray[np.float64],
    title: str = "Bankroll Monte Carlo Simulation",
    n_highlight: int = 5,
) -> Figure:
    """Spaghetti plot of bankroll paths: median highlighted, 10th/90th band, individual paths in low alpha.

    Args:
        paths: 2-D array (n_paths, n_steps) of bankroll levels.
        title: Chart title.
        n_highlight: Number of individual paths to draw at low alpha.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=FIG_DPI)
    fig.patch.set_facecolor(FIG_FACECOLOR)
    ax.set_facecolor(AXES_FACECOLOR)

    n_steps = paths.shape[1]
    x = np.arange(n_steps)

    # Percentile bands
    p10 = np.percentile(paths, 10, axis=0)
    p90 = np.percentile(paths, 90, axis=0)
    median = np.median(paths, axis=0)

    ax.fill_between(x, p10, p90, color=BAND_COLOR, alpha=0.4, label="10th–90th percentile")
    ax.plot(x, median, color=MEDIAN_COLOR, linewidth=LINE_WEIGHT_MAIN, label="Median")

    # A few individual paths
    rng = np.random.default_rng(42)
    indices = rng.choice(paths.shape[0], size=min(n_highlight, paths.shape[0]), replace=False)
    for idx in indices:
        ax.plot(x, paths[idx], color=TEXT_SECONDARY, alpha=0.15, linewidth=0.6)

    ax.axhline(1.0, color=AXIS_COLOR, linewidth=LINE_WEIGHT_GRID, linestyle="--")
    ax.set_yscale("log")
    ax.set_xlabel("Steps", color=TEXT_SECONDARY, fontsize=FONT_LABEL)
    ax.set_ylabel("Bankroll (log scale)", color=TEXT_SECONDARY, fontsize=FONT_LABEL)
    ax.tick_params(colors=AXIS_COLOR, labelsize=FONT_TICK)
    ax.grid(True, color=GRID_COLOR, linewidth=LINE_WEIGHT_GRID, which="both")
    ax.legend(fontsize=FONT_ANNOTATION, facecolor=AXES_FACECOLOR, edgecolor=AXIS_COLOR, labelcolor=TEXT_SECONDARY)

    ax.set_title(title, color=TEXT_PRIMARY, fontsize=FONT_TITLE, pad=12)
    fig.tight_layout()
    return fig


def drawdown_distribution(
    drawdowns: NDArray[np.float64],
    max_drawdown: float,
    title: str = "Drawdown Distribution",
) -> Figure:
    """Histogram of drawdown values with empirical CDF overlay and max drawdown marked.

    Args:
        drawdowns: 1-D array of drawdown fractions from all paths and timesteps.
        max_drawdown: Maximum drawdown fraction observed.
        title: Chart title.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5), dpi=FIG_DPI)
    fig.patch.set_facecolor(FIG_FACECOLOR)
    ax.set_facecolor(AXES_FACECOLOR)

    # Filter out near-zero drawdowns for cleaner histogram
    nonzero = drawdowns[drawdowns > 1e-6]
    if len(nonzero) > 0:
        counts, bin_edges, _ = ax.hist(
            nonzero, bins=40, color=HISTOGRAM_COLOR, edgecolor=AXIS_COLOR,
            linewidth=0.5, alpha=0.8, density=False, label="Drawdown histogram",
        )

        # Empirical CDF on secondary axis
        ax2 = ax.twinx()
        sorted_dd = np.sort(nonzero)
        cdf = np.arange(1, len(sorted_dd) + 1) / len(sorted_dd)
        ax2.plot(sorted_dd, cdf, color=MEDIAN_COLOR, linewidth=LINE_WEIGHT_SECONDARY, label="Empirical CDF")
        ax2.set_ylabel("CDF", color=MEDIAN_COLOR, fontsize=FONT_LABEL)
        ax2.tick_params(colors=AXIS_COLOR, labelsize=FONT_TICK)
        ax2.set_ylim(0, 1.05)

    # Mark max drawdown
    ax.axvline(max_drawdown, color=EV_NEGATIVE, linewidth=LINE_WEIGHT_MAIN, linestyle="--",
               label=f"Max DD = {fmt_drawdown(max_drawdown)}")

    ax.set_xlabel("Drawdown Fraction", color=TEXT_SECONDARY, fontsize=FONT_LABEL)
    ax.set_ylabel("Count", color=TEXT_SECONDARY, fontsize=FONT_LABEL)
    ax.tick_params(colors=AXIS_COLOR, labelsize=FONT_TICK)
    ax.grid(True, color=GRID_COLOR, linewidth=LINE_WEIGHT_GRID, axis="y")
    ax.legend(fontsize=FONT_ANNOTATION, facecolor=AXES_FACECOLOR, edgecolor=AXIS_COLOR,
              labelcolor=TEXT_SECONDARY, loc="upper right")

    ax.set_title(title, color=TEXT_PRIMARY, fontsize=FONT_TITLE, pad=12)
    fig.tight_layout()
    return fig


def fraction_sweep_small_multiples(
    sweep_data: list[tuple[float, float, float]],
    title: str = "Kelly Fraction Sweep",
) -> Figure:
    """Small multiples: ruin probability and median terminal bankroll vs Kelly fraction.

    Args:
        sweep_data: List of (fraction, mean_log_growth, ruin_probability) tuples.
        title: Chart title.

    Returns:
        Matplotlib Figure with two subplots.
    """
    fractions = np.array([d[0] for d in sweep_data])
    log_growths = np.array([d[1] for d in sweep_data])
    ruin_probs = np.array([d[2] for d in sweep_data])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=FIG_DPI)
    fig.patch.set_facecolor(FIG_FACECOLOR)

    # Left: log growth vs fraction
    ax1.set_facecolor(AXES_FACECOLOR)
    ax1.plot(fractions, log_growths, color=EV_POSITIVE, linewidth=LINE_WEIGHT_MAIN)
    ax1.axhline(0, color=AXIS_COLOR, linewidth=LINE_WEIGHT_GRID, linestyle="--")
    opt_idx = int(np.argmax(log_growths))
    ax1.axvline(fractions[opt_idx], color=EV_WARNING, linewidth=LINE_WEIGHT_SECONDARY, linestyle="--")
    ax1.annotate(f"opt f={fmt_frac(fractions[opt_idx])}", xy=(fractions[opt_idx], log_growths[opt_idx]),
                 xytext=(10, -15), textcoords="offset points", color=EV_WARNING, fontsize=FONT_ANNOTATION)
    ax1.set_xlabel("Kelly Fraction", color=TEXT_SECONDARY, fontsize=FONT_LABEL)
    ax1.set_ylabel("Mean Log Growth", color=EV_POSITIVE, fontsize=FONT_LABEL)
    ax1.tick_params(colors=AXIS_COLOR, labelsize=FONT_TICK)
    ax1.grid(True, color=GRID_COLOR, linewidth=LINE_WEIGHT_GRID)

    # Right: ruin probability vs fraction
    ax2.set_facecolor(AXES_FACECOLOR)
    ax2.plot(fractions, ruin_probs, color=EV_NEGATIVE, linewidth=LINE_WEIGHT_MAIN)
    ax2.axvline(fractions[opt_idx], color=EV_WARNING, linewidth=LINE_WEIGHT_SECONDARY, linestyle="--")
    ax2.set_xlabel("Kelly Fraction", color=TEXT_SECONDARY, fontsize=FONT_LABEL)
    ax2.set_ylabel("Ruin Probability", color=EV_NEGATIVE, fontsize=FONT_LABEL)
    ax2.tick_params(colors=AXIS_COLOR, labelsize=FONT_TICK)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, color=GRID_COLOR, linewidth=LINE_WEIGHT_GRID)

    fig.suptitle(title, color=TEXT_PRIMARY, fontsize=FONT_TITLE, y=1.02)
    fig.tight_layout()
    return fig
