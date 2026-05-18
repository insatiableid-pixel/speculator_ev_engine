"""Number formatting: EV to 3dp, fractions to %, probabilities to %, dollars with commas."""

from __future__ import annotations


def fmt_ev(value: float) -> str:
    """Format EV to 3 decimal places with sign."""
    return f"{value:+.3f}"


def fmt_prob(value: float) -> str:
    """Format probability as percentage to 1 decimal."""
    return f"{value * 100:.1f}%"


def fmt_frac(value: float) -> str:
    """Format Kelly fraction as percentage to 1 decimal."""
    return f"{value * 100:.1f}%"


def fmt_dollar(value: float) -> str:
    """Format dollar amount with commas and sign."""
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value):,.2f}"


def fmt_pct(value: float) -> str:
    """Format generic percentage to 1 decimal with sign."""
    return f"{value * 100:+.1f}%"


def fmt_brier(value: float) -> str:
    """Format Brier score to 4 decimal places."""
    return f"{value:.4f}"


def fmt_sharp(value: float) -> str:
    """Format Sharpe ratio to 2 decimal places."""
    return f"{value:.2f}"


def fmt_ruin(value: float) -> str:
    """Format ruin probability as percentage to 1 decimal."""
    return f"{value * 100:.1f}%"


def fmt_drawdown(value: float) -> str:
    """Format drawdown as percentage to 1 decimal."""
    return f"{value * 100:.1f}%"
