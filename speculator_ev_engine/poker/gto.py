"""GTO concepts: range construction, MDF, alpha, indifference frequencies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


# TODO: Implement full range-vs-range equity calculations
# TODO: Implement MDF (Minimum Defense Frequency) solver
# TODO: Implement alpha (bluff-to-value ratio) calculations
# TODO: Implement indifference frequency solver for multi-street games


@dataclass(frozen=True)
class RangeCombo:
    """A hand combo with its weight in a range.

    Attributes:
        hand: Hand identifier string (e.g. "AKs", "TT", "72o")
        weight: Fractional weight in [0, 1] representing how often this hand is in the range.
        equity: Pre-computed equity vs opponent range (0-1), if available.
    """
    hand: str
    weight: float
    equity: float | None = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.weight <= 1.0):
            raise ValueError(f"weight must be in [0,1], got {self.weight}")


class Range:
    """A weighted range of poker hands.

    TODO: Implement range construction, filtering, and equity computation.
    """

    def __init__(self, combos: list[RangeCombo] | None = None) -> None:
        self._combos = combos or []

    @property
    def combos(self) -> list[RangeCombo]:
        return self._combos

    def total_weight(self) -> float:
        """Sum of all combo weights."""
        return sum(c.weight for c in self._combos)

    # TODO: range_vs_range_equity(other: Range) -> float
    # TODO: remove_hand(hand: str) -> Range
    # TODO: filter_by_equity(min_eq: float, max_eq: float) -> Range


def minimum_defense_frequency(pot_size: float, bet_size: float) -> float:
    """Compute Minimum Defense Frequency.

    MDF = 1 - alpha, where alpha = bet_size / (pot_size + bet_size).
    This is the fraction of the defending range that must not fold
    to prevent the bettor from profiting with any two cards.

    Args:
        pot_size: Current pot size before the bet.
        bet_size: Size of the bet facing the defender.

    Returns:
        MDF as a fraction in [0, 1].
    """
    if pot_size < 0 or bet_size < 0:
        raise ValueError("pot_size and bet_size must be non-negative")
    alpha_val = alpha(pot_size, bet_size)
    return 1.0 - alpha_val


def alpha(pot_size: float, bet_size: float) -> float:
    """Compute alpha: the bluff-to-value ratio for a balanced betting range.

    alpha = bet_size / (pot_size + bet_size)

    A bettor's range should contain alpha fraction of bluffs and (1-alpha)
    fraction of value bets to make the opponent indifferent to calling.

    Args:
        pot_size: Current pot size.
        bet_size: Size of the bet.

    Returns:
        Alpha fraction in [0, 1].
    """
    if pot_size + bet_size <= 0:
        raise ValueError("pot_size + bet_size must be positive")
    return bet_size / (pot_size + bet_size)


def indifference_call_frequency(pot_size: float, bet_size: float) -> float:
    """Compute the call frequency that makes the bettor indifferent to bluffing.

    The bettor's bluff breaks even when:
    p_call * (-bet_size) + (1 - p_call) * pot_size = 0
    → p_call = pot_size / (pot_size + bet_size) = 1 - alpha

    Args:
        pot_size: Pot before the bet.
        bet_size: Bet size.

    Returns:
        Indifferent call frequency.
    """
    return 1.0 - alpha(pot_size, bet_size)


def pot_odds(pot_size: float, call_amount: float) -> float:
    """Compute pot odds as a fraction.

    Args:
        pot_size: Pot before calling.
        call_amount: Amount required to call.

    Returns:
        Required equity to call profitably.
    """
    if call_amount <= 0:
        raise ValueError(f"call_amount must be positive, got {call_amount}")
    return call_amount / (pot_size + call_amount)
