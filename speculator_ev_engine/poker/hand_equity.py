"""Monte Carlo equity engine for poker hands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


# TODO: Implement full Monte Carlo equity simulation
# TODO: Support Texas Hold'em, Omaha, Short Deck
# TODO: Support board runouts with weighted opponent ranges
# TODO: Support multi-way equity calculations


@dataclass(frozen=True)
class EquityResult:
    """Result of a Monte Carlo equity simulation.

    Attributes:
        equities: Win equity per hand (0-1).
        ties: Tie equity per hand (0-1).
        n_simulations: Number of Monte Carlo trials run.
        std_error: Standard error of the equity estimate.
    """
    equities: NDArray[np.float64]
    ties: NDArray[np.float64]
    n_simulations: int
    std_error: float


def monte_carlo_equity(
    hero_hands: Sequence[str],
    board: str = "",
    dead_cards: str = "",
    n_simulations: int = 100_000,
    game: str = "holdem",
    seed: int | None = None,
) -> EquityResult:
    """Run Monte Carlo equity simulation.

    Args:
        hero_hands: Sequence of hand strings (e.g. ["AsKh", "QdQh"]).
        board: Community cards string (e.g. "AhKd7s").
        dead_cards: Cards to exclude from the deck.
        n_simulations: Number of simulation trials.
        game: Poker variant ("holdem", "omaha", "short_deck").
        seed: Random seed for reproducibility.

    Returns:
        EquityResult with win/tie equities and simulation metadata.

    TODO: Implement hand evaluation engine. This is a stub that returns
          uniform equities — the actual implementation requires a full
          hand evaluator (recommend using deuces or similar as backend).
    """
    n_hands = len(hero_hands)
    # Stub: return uniform equities
    equities = np.full(n_hands, 1.0 / n_hands, dtype=np.float64)
    ties = np.zeros(n_hands, dtype=np.float64)
    std_error = 1.0 / np.sqrt(n_simulations)

    return EquityResult(
        equities=equities,
        ties=ties,
        n_simulations=n_simulations,
        std_error=std_error,
    )
