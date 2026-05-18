"""Decision log / leak / calibration charts: EV vs outcome scatter, tilt detection, calibration curve."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from numpy.typing import NDArray

from ..themes import (
    FIG_FACECOLOR, AXES_FACECOLOR, AXIS_COLOR, GRID_COLOR, TEXT_PRIMARY,
    TEXT_SECONDARY, TEXT_MUTED, EV_POSITIVE, EV_NEGATIVE, EV_WARNING,
    MEDIAN_COLOR, HISTOGRAM_COLOR, DIAGONAL_COLOR, LINE_WEIGHT_MAIN,
    LINE_WEIGHT_SECONDARY, LINE_WEIGHT_GRID, FONT_TITLE, FONT_LABEL,
    FONT_TICK, FONT_ANNOTATION, FIG_DPI,
)
from ..formatters import fmt_ev, fmt_prob, fmt_pct


def ev_vs_outcome_scatter(
    ev_estimates: NDArray[np.float64],
    outcomes: NDArray[np.float64],
    title: str = "EV Estimate vs Outcome",
) -> Figure:
    """Scatter: each point a decision. Quadrant lines at EV=0 and outcome=0.

    Upper-left: +EV decision that lost (variance masking quality).
    Lower-right: -EV decision that won (variance masking leak).
    These two quadrants are the interesting ones.

    Args:
        ev_estimates: 1-D array of estimated EVs.
        outcomes: 1-D array of actual outcomes.
        title: Chart title.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 7), dpi=FIG_DPI)
    fig.patch.set_facecolor(FIG_FACECOLOR)
    ax.set_facecolor(AXES_FACECOLOR)

    # Color by quadrant — the two interesting quadrants get semantic color
    colors = []
    for ev, out in zip(ev_estimates, outcomes):
        if ev > 0 and out < 0:
            colors.append(EV_WARNING)   # +EV lost — amber (variance, not a leak)
        elif ev < 0 and out > 0:
            colors.append(EV_NEGATIVE)  # -EV won — red (luck masking a leak)
        elif ev > 0:
            colors.append(EV_POSITIVE) # +EV won — green
        else:
            colors.append(TEXT_MUTED)   # -EV lost — neutral
    colors = np.array(colors)

    ax.scatter(ev_estimates, outcomes, c=colors, s=12, alpha=0.6, edgecolors="none")

    ax.axhline(0, color=AXIS_COLOR, linewidth=LINE_WEIGHT_MAIN)
    ax.axvline(0, color=AXIS_COLOR, linewidth=LINE_WEIGHT_MAIN)

    # Quadrant labels (subtle)
    span = max(abs(ev_estimates).max(), abs(outcomes).max(), 0.1)
    ax.text(span * 0.6, -span * 0.6, "+EV / lost", color=EV_WARNING, fontsize=FONT_ANNOTATION, alpha=0.7)
    ax.text(-span * 0.8, span * 0.6, "−EV / won", color=EV_NEGATIVE, fontsize=FONT_ANNOTATION, alpha=0.7)

    ax.set_xlabel("Estimated EV", color=TEXT_SECONDARY, fontsize=FONT_LABEL)
    ax.set_ylabel("Actual Outcome", color=TEXT_SECONDARY, fontsize=FONT_LABEL)
    ax.tick_params(colors=AXIS_COLOR, labelsize=FONT_TICK)
    ax.grid(True, color=GRID_COLOR, linewidth=LINE_WEIGHT_GRID)
    ax.set_aspect("equal", adjustable="datalim")

    ax.set_title(title, color=TEXT_PRIMARY, fontsize=FONT_TITLE, pad=12)
    fig.tight_layout()
    return fig


def tilt_detection_time_series(
    stakes: NDArray[np.float64],
    recent_loss_indicators: NDArray[np.float64],
    title: str = "Tilt Detection: Stake Size Over Time",
) -> Figure:
    """Stake size over time colored by recent outcome sequence — visual tilt signature.

    Args:
        stakes: 1-D array of stake amounts.
        recent_loss_indicators: 1-D array of same length, 0-1 indicating recent loss context.
        title: Chart title.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 4), dpi=FIG_DPI)
    fig.patch.set_facecolor(FIG_FACECOLOR)
    ax.set_facecolor(AXES_FACECOLOR)

    x = np.arange(len(stakes))

    # Color segments: after losses → amber/red intensity
    for i in range(len(stakes)):
        loss_frac = float(recent_loss_indicators[i])
        if loss_frac > 0.6:
            color = EV_NEGATIVE
        elif loss_frac > 0.3:
            color = EV_WARNING
        else:
            color = TEXT_MUTED
        ax.plot(x[i : i + 2], stakes[i : i + 2], color=color, linewidth=LINE_WEIGHT_MAIN)

    ax.set_xlabel("Decision #", color=TEXT_SECONDARY, fontsize=FONT_LABEL)
    ax.set_ylabel("Stake Size", color=TEXT_SECONDARY, fontsize=FONT_LABEL)
    ax.tick_params(colors=AXIS_COLOR, labelsize=FONT_TICK)
    ax.grid(True, color=GRID_COLOR, linewidth=LINE_WEIGHT_GRID, axis="y")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=TEXT_MUTED, linewidth=2, label="Normal"),
        Line2D([0], [0], color=EV_WARNING, linewidth=2, label="After some losses"),
        Line2D([0], [0], color=EV_NEGATIVE, linewidth=2, label="After heavy losses"),
    ]
    ax.legend(handles=legend_elements, fontsize=FONT_ANNOTATION, facecolor=AXES_FACECOLOR,
              edgecolor=AXIS_COLOR, labelcolor=TEXT_SECONDARY)

    ax.set_title(title, color=TEXT_PRIMARY, fontsize=FONT_TITLE, pad=12)
    fig.tight_layout()
    return fig


def calibration_curve(
    bin_centers: NDArray[np.float64],
    observed_frequencies: NDArray[np.float64],
    bin_counts: NDArray[np.int64],
    ece: float,
    title: str = "Calibration Curve",
) -> Figure:
    """Reliability diagram with diagonal reference, histogram of predicted probabilities below, ECE prominent.

    Args:
        bin_centers: Mean forecast probability per bin.
        observed_frequencies: Observed frequency per bin.
        bin_counts: Number of forecasts per bin.
        ece: Expected Calibration Error.
        title: Chart title.

    Returns:
        Matplotlib Figure.
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 6), dpi=FIG_DPI,
        gridspec_kw={"height_ratios": [3, 1]}, sharex=True,
    )
    fig.patch.set_facecolor(FIG_FACECOLOR)

    # Top: reliability diagram
    ax1.set_facecolor(AXES_FACECOLOR)
    ax1.plot([0, 1], [0, 1], color=DIAGONAL_COLOR, linewidth=LINE_WEIGHT_SECONDARY, linestyle="--", label="Perfect calibration")
    ax1.scatter(bin_centers, observed_frequencies, color=MEDIAN_COLOR, s=30, zorder=5)
    ax1.plot(bin_centers, observed_frequencies, color=MEDIAN_COLOR, linewidth=LINE_WEIGHT_MAIN, alpha=0.5)

    ax1.set_ylabel("Observed Frequency", color=TEXT_SECONDARY, fontsize=FONT_LABEL)
    ax1.tick_params(colors=AXIS_COLOR, labelsize=FONT_TICK)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, color=GRID_COLOR, linewidth=LINE_WEIGHT_GRID)

    # ECE annotation — prominent
    ax1.text(0.98, 0.02, f"ECE = {ece:.3f}", transform=ax1.transAxes,
             ha="right", va="bottom", color=EV_WARNING, fontsize=FONT_LABEL, fontweight="bold")

    ax1.legend(fontsize=FONT_ANNOTATION, facecolor=AXES_FACECOLOR, edgecolor=AXIS_COLOR, labelcolor=TEXT_SECONDARY)
    ax1.set_title(title, color=TEXT_PRIMARY, fontsize=FONT_TITLE, pad=12)

    # Bottom: histogram of predicted probabilities
    ax2.set_facecolor(AXES_FACECOLOR)
    ax2.bar(bin_centers, bin_counts, width=0.08, color=HISTOGRAM_COLOR, edgecolor=AXIS_COLOR, linewidth=0.5)
    ax2.set_xlabel("Predicted Probability", color=TEXT_SECONDARY, fontsize=FONT_LABEL)
    ax2.set_ylabel("Count", color=TEXT_SECONDARY, fontsize=FONT_LABEL)
    ax2.tick_params(colors=AXIS_COLOR, labelsize=FONT_TICK)
    ax2.grid(True, color=GRID_COLOR, linewidth=LINE_WEIGHT_GRID, axis="y")

    fig.tight_layout()
    return fig
