"""Arbitrage detection across correlated assets and books."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..sports.odds import american_to_implied_prob, remove_vig


# TODO: Implement cross-book arbitrage detection
# TODO: Implement correlated market arbitrage (same-game parlays, derivatives)
# TODO: Implement latency-aware arb detection for live markets


@dataclass(frozen=True)
class ArbitrageOpportunity:
    """A detected arbitrage opportunity.

    Attributes:
        legs: List of (book, outcome, odds_american) tuples.
        guaranteed_profit: Risk-free profit as fraction of total stake.
        total_implied: Sum of vig-free implied probabilities (< 1 means arb exists).
    """
    legs: list[tuple[str, str, int]]
    guaranteed_profit: float
    total_implied: float


def detect_arbitrage(
    books: dict[str, dict[str, int]],
) -> list[ArbitrageOpportunity]:
    """Detect arbitrage opportunities across sportsbooks.

    Args:
        books: Dict mapping book name → dict mapping outcome → American odds.
            Example: {"draftkings": {"win": -110, "lose": -110}, ...}

    Returns:
        List of ArbitrageOpportunity sorted by guaranteed_profit descending.
    """
    if not books:
        return []

    # Get all outcomes across books
    all_outcomes: set[str] = set()
    for book_odds in books.values():
        all_outcomes.update(book_odds.keys())

    outcomes = sorted(all_outcomes)

    # For each outcome, find the best odds across all books
    best_legs: list[tuple[str, str, int]] = []
    implied_probs: list[float] = []

    for outcome in outcomes:
        best_odds = -999999
        best_book = ""
        for book_name, book_odds in books.items():
            if outcome in book_odds:
                odds = book_odds[outcome]
                # Best odds: highest positive or least negative
                if odds > best_odds:
                    best_odds = odds
                    best_book = book_name

        if best_odds > -999999:
            best_legs.append((best_book, outcome, best_odds))
            implied_probs.append(american_to_implied_prob(best_odds))

    total_implied = sum(implied_probs)

    if total_implied >= 1.0:
        return []  # No arbitrage

    guaranteed_profit = 1.0 - total_implied

    return [ArbitrageOpportunity(
        legs=best_legs,
        guaranteed_profit=guaranteed_profit,
        total_implied=total_implied,
    )]
