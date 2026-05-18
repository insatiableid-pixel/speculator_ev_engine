"""Odds format conversions, implied probability, vig extraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class OddsConversion:
    """Odds in multiple formats for a single outcome.

    Attributes:
        american: American odds (e.g. +150, -110).
        decimal: Decimal odds (e.g. 2.5).
        fractional: Fractional odds string (e.g. "3/2").
        implied_prob: Implied probability (vig-inclusive).
        true_prob: Vig-free probability (if vig extracted).
    """
    american: int
    decimal: float
    fractional: str
    implied_prob: float
    true_prob: float | None = None


def american_to_decimal(odds: int) -> float:
    """Convert American odds to decimal odds.

    Args:
        odds: American odds (positive or negative).

    Returns:
        Decimal odds.
    """
    if odds >= 100:
        return odds / 100.0 + 1.0
    elif odds <= -100:
        return 100.0 / abs(odds) + 1.0
    else:
        raise ValueError(
            f"American odds must be >= 100 or <= -100, got {odds}"
        )


def decimal_to_american(decimal_odds: float) -> int:
    """Convert decimal odds to American odds.

    Args:
        decimal_odds: Decimal odds (must be >= 1.0).

    Returns:
        American odds.
    """
    if decimal_odds < 1.0:
        raise ValueError(f"Decimal odds must be >= 1.0, got {decimal_odds}")
    if decimal_odds >= 2.0:
        return int(round((decimal_odds - 1.0) * 100))
    else:
        return int(round(-100.0 / (decimal_odds - 1.0)))


def american_to_implied_prob(odds: int) -> float:
    """Convert American odds to implied probability (vig-inclusive).

    Args:
        odds: American odds.

    Returns:
        Implied probability in [0, 1].
    """
    if odds >= 100:
        return 100.0 / (odds + 100.0)
    elif odds <= -100:
        return abs(odds) / (abs(odds) + 100.0)
    else:
        raise ValueError(
            f"American odds must be >= 100 or <= -100, got {odds}"
        )


def decimal_to_implied_prob(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability.

    Args:
        decimal_odds: Decimal odds (>= 1.0).

    Returns:
        Implied probability.
    """
    if decimal_odds < 1.0:
        raise ValueError(f"Decimal odds must be >= 1.0, got {decimal_odds}")
    return 1.0 / decimal_odds


def implied_prob_to_decimal(prob: float) -> float:
    """Convert implied probability to decimal odds.

    Args:
        prob: Implied probability in (0, 1].

    Returns:
        Decimal odds.
    """
    if prob <= 0 or prob > 1:
        raise ValueError(f"Probability must be in (0, 1], got {prob}")
    return 1.0 / prob


def extract_vig(outcome_probs: NDArray[np.float64] | list[float]) -> float:
    """Extract the vig (overround) from a set of implied probabilities.

    Args:
        outcome_probs: Array of implied probabilities for all outcomes.

    Returns:
        Vig as a fraction (e.g. 0.05 for 5% overround).
    """
    probs = np.asarray(outcome_probs, dtype=np.float64)
    total = float(np.sum(probs))
    if total < 1.0:
        raise ValueError(f"Total implied probability {total:.4f} < 1.0 — no vig present")
    if abs(total - 1.0) < 1e-9:
        return 0.0
    return total - 1.0


def remove_vig(outcome_probs: NDArray[np.float64] | list[float]) -> NDArray[np.float64]:
    """Remove vig from a set of implied probabilities, normalizing to sum to 1.

    Args:
        outcome_probs: Array of implied probabilities (vig-inclusive).

    Returns:
        Vig-free probabilities summing to 1.0.
    """
    probs = np.asarray(outcome_probs, dtype=np.float64)
    total = float(np.sum(probs))
    if total <= 0:
        raise ValueError("Total implied probability must be positive")
    return probs / total


def convert_odds(american: int, other_outcomes: list[int] | None = None) -> OddsConversion:
    """Convert American odds into all formats.

    Args:
        american: American odds for the outcome.
        other_outcomes: American odds for other outcomes (used for vig extraction).

    Returns:
        OddsConversion with all representations.
    """
    dec = american_to_decimal(american)
    implied = american_to_implied_prob(american)

    # Fractional string
    if american >= 100:
        num = american
        den = 100
    else:
        num = 100
        den = abs(american)
    # Simplify
    from math import gcd
    g = gcd(num, den)
    fractional = f"{num // g}/{den // g}"

    true_prob = None
    if other_outcomes is not None:
        all_probs = [implied] + [american_to_implied_prob(o) for o in other_outcomes]
        vig_free = remove_vig(all_probs)
        true_prob = float(vig_free[0])

    return OddsConversion(
        american=american,
        decimal=dec,
        fractional=fractional,
        implied_prob=implied,
        true_prob=true_prob,
    )


def spread_to_moneyline(spread: float, total: float = 42.0) -> int:
    """Approximate moneyline conversion from point spread.

    Uses the standard approximation: each point of spread ≈ 20 cents of ML.

    Args:
        spread: Point spread (negative = favorite).
        total: Over/under total (affects conversion accuracy).

    Returns:
        Approximate American odds.
    """
    # Standard conversion formula
    win_prob = float(norm_cdf_approx(-spread / (total / 7.0)))
    if win_prob <= 0.5:
        ml = int(round(100 / win_prob - 100))
        return max(ml, 100)
    else:
        ml = int(round(-100 * win_prob / (1 - win_prob)))
        return min(ml, -100)


def norm_cdf_approx(x: float) -> float:
    """Fast normal CDF approximation (Abramowitz & Stegun).

    Args:
        x: Standard normal z-score.

    Returns:
        Approximate CDF value.
    """
    # Approximation 26.2.17 from Abramowitz & Stegun
    a1, a2, a3, a4, a5 = (
        0.254829592,
        -0.284496736,
        1.421413741,
        -1.453152027,
        1.061405429,
    )
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x = abs(x) / np.sqrt(2.0)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return 0.5 * (1.0 + sign * y)
