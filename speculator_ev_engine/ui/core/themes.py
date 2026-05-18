"""Visual tokens: colors, fonts, sizes. Single source of truth for all UI rendering."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Semantic color palette — color encodes meaning, never decoration
# Green/red = positive/negative EV. Amber = warning. All else is neutral.
# ---------------------------------------------------------------------------

# EV semantic colors (used in all three interfaces)
EV_POSITIVE = "#22c55e"       # green-500
EV_NEGATIVE = "#ef4444"       # red-500
EV_ZERO = "#94a3b8"           # slate-400
EV_WARNING = "#f59e0b"        # amber-500

# Dark theme surface colors (web + TUI)
BG_PRIMARY = "#0f172a"        # slate-900
BG_SECONDARY = "#1e293b"     # slate-800
BG_SURFACE = "#334155"        # slate-700
TEXT_PRIMARY = "#f8fafc"      # slate-50
TEXT_SECONDARY = "#94a3b8"    # slate-400
TEXT_MUTED = "#64748b"        # slate-500
BORDER = "#475569"            # slate-600

# Chart-specific
AXIS_COLOR = "#64748b"        # slate-500
GRID_COLOR = "#1e293b"       # slate-800 — minimal, not distracting
BAND_COLOR = "#334155"        # slate-700 for confidence bands
MEDIAN_COLOR = "#f8fafc"      # slate-50
PERCENTILE_10 = "#22c55e"     # green-500
PERCENTILE_90 = "#ef4444"    # red-500
DIAGONAL_COLOR = "#64748b"   # slate-500 for calibration diagonal
HISTOGRAM_COLOR = "#475569"  # slate-600

# Line weights
LINE_WEIGHT_MAIN = 2.0
LINE_WEIGHT_SECONDARY = 1.0
LINE_WEIGHT_GRID = 0.3

# Font sizes (points — matplotlib)
FONT_TITLE = 13
FONT_LABEL = 10
FONT_TICK = 8
FONT_ANNOTATION = 8

# Figure defaults
FIG_DPI = 120
FIG_FACECOLOR = BG_PRIMARY
AXES_FACECOLOR = BG_PRIMARY


def ev_color(value: float) -> str:
    """Return semantic color for an EV value."""
    if value > 0.001:
        return EV_POSITIVE
    elif value < -0.001:
        return EV_NEGATIVE
    return EV_ZERO


def warning_color(is_warning: bool) -> str:
    """Return amber if warning flag is true, else neutral."""
    return EV_WARNING if is_warning else TEXT_SECONDARY
