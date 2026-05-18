"""ICM models: Malmuth-Harville, chip chop hybrid, bubble factor, Nash push/fold interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize


@dataclass(frozen=True)
class ICMResult:
    """Output of an ICM equity calculation.

    Attributes:
        equities: Dollar equity per stack, same length as stacks.
        total_prize_pool: Sum of payouts for verification.
    """
    equities: NDArray[np.float64]
    total_prize_pool: float

    def __post_init__(self) -> None:
        total = float(np.sum(self.equities))
        if abs(total - self.total_prize_pool) > 1e-4 * self.total_prize_pool:
            raise ValueError(
                f"ICM equities sum to {total:.4f}, expected {self.total_prize_pool:.4f}"
            )


@dataclass(frozen=True)
class BubbleFactorResult:
    """Bubble factor for a specific stack at a specific payout configuration.

    Attributes:
        bubble_factor: Ratio of equity lost when losing a chip to equity gained
            when winning that same chip. BF > 1 means the spot is bubble-weighted.
        equity_lost: ICM equity lost if the player loses `chips_at_risk` chips.
        equity_gained: ICM equity gained if the player wins `chips_at_risk` chips.
    """
    bubble_factor: float
    equity_lost: float
    equity_gained: float


class ICMModel(Protocol):
    """Protocol for any ICM equity model."""

    def calculate(self, stacks: NDArray[np.float64],
                  payouts: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return dollar equity per stack."""
        ...


def malmuth_harville_icm(stacks: NDArray[np.float64],
                         payouts: NDArray[np.float64]) -> ICMResult:
    """Standard Malmuth-Harville ICM calculation.

    Uses the Malmuth-Harville model: each player's probability of finishing
    in nth place is proportional to their stack divided by total remaining
    stacks, computed via the standard permutation-based recursive formula.

    For a set of stacks s_1, ..., s_n, player i's equity is:
        equity_i = sum over all finishing positions j: P(i finishes j) * payout_j

    Where P(i finishes first) = s_i / sum(s), and subsequent position
    probabilities are computed recursively by removing each player who
    finished ahead.

    Args:
        stacks: 1-D array of chip stacks (all must be positive).
        payouts: 1-D array of prize pool payouts by finishing position.

    Returns:
        ICMResult with dollar equity per stack.
    """
    stacks = np.asarray(stacks, dtype=np.float64)
    payouts = np.asarray(payouts, dtype=np.float64)

    if np.any(stacks <= 0):
        raise ValueError("All stacks must be positive")
    if np.any(payouts < 0):
        raise ValueError("Payouts must be non-negative")
    if len(stacks) < 2:
        raise ValueError("Need at least 2 stacks")
    if len(payouts) > len(stacks):
        raise ValueError("Cannot have more payout positions than players")

    total_prize = float(np.sum(payouts))
    n_players = len(stacks)

    # Compute probability of each player finishing in each position
    # using the Malmuth-Harville model
    equities = _mh_equities(stacks, payouts)

    return ICMResult(equities=equities, total_prize_pool=total_prize)


def _mh_equities(stacks: NDArray[np.float64],
                 payouts: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute Malmuth-Harville ICM equities via iterative DP.

    For each subset of remaining players (represented as a bitmask),
    compute each player's probability of finishing in each remaining
    position. Then multiply by payouts and sum.
    """
    n = len(stacks)
    n_paid = len(payouts)

    # prob[bitmask][player] = P(player finishes next among remaining)
    # Use a different approach: for each permutation prefix, accumulate
    # the product of conditional probabilities.
    #
    # More efficient: compute P(player i finishes in position j)
    # using the standard recursive formula.

    equities = np.zeros(n, dtype=np.float64)

    # We iterate over all orderings by building finishing order sequentially
    # State: (set of players already placed, position we're filling)
    # For each state, we know the probability of reaching that state
    # and we distribute the next position proportionally.

    # Use a recursive approach with memoization
    from functools import lru_cache

    total_chips = float(np.sum(stacks))

    # Cache: key = frozenset of players already assigned positions
    # Returns: dict mapping player -> probability of being in this state
    # Actually, we just need to track the probability of each
    # ordering prefix and accumulate equities.

    # Simpler direct approach: compute finish probabilities position by position
    # finish_probs[i][j] = P(player i finishes in position j)
    finish_probs = np.zeros((n, n_paid), dtype=np.float64)

    # Build finish probabilities using the standard ICM recursion
    # For each player i, P(i finishes first) = stacks[i] / total
    # For position j > 0, P(i finishes in position j | who finished before)
    # is computed by summing over all possible sets of j-1 players
    # finishing ahead of i.

    # Use the standard approach: enumerate all permutations of first n_paid players
    # and accumulate their probabilities.

    # Let's use the efficient recursive formulation:
    # For a set S of players already placed in positions 0..k-1,
    # the probability of this configuration is the product of
    # stacks[player] / sum(stacks of remaining) for each placement.
    # Then for each remaining player i, they get probability
    # stacks[i] / sum(remaining) for position k.

    def _compute() -> NDArray[np.float64]:
        eq = np.zeros(n, dtype=np.float64)
        # Generate all orderings of the first min(n, n_paid) positions
        # using recursion
        remaining_chips = float(np.sum(stacks))

        def recurse(placed: list[int], prob: float, remaining_stack: float) -> None:
            pos = len(placed)
            if pos >= n_paid or pos >= n:
                return
            placed_set = set(placed)
            for i in range(n):
                if i in placed_set:
                    continue
                p_i = stacks[i] / remaining_stack
                eq[i] += prob * p_i * payouts[pos]
                recurse(placed + [i], prob * p_i, remaining_stack - stacks[i])

        recurse([], 1.0, remaining_chips)
        return eq

    return _compute()


def chip_chop_icm(
    stacks: NDArray[np.float64],
    payouts: NDArray[np.float64],
    blend_weight: float = 0.5,
) -> ICMResult:
    """Chip chop hybrid: weighted blend of pure chip chop and ICM equity.

    equity = blend * ICM_equity + (1 - blend) * chip_chop_equity

    Args:
        stacks: 1-D array of chip stacks.
        payouts: 1-D array of payouts by position.
        blend_weight: Weight of ICM vs chip chop (0 = pure chip chop, 1 = pure ICM).

    Returns:
        ICMResult with blended equity.
    """
    if not (0.0 <= blend_weight <= 1.0):
        raise ValueError(f"blend_weight must be in [0,1], got {blend_weight}")

    total_prize = float(np.sum(payouts))
    total_chips = float(np.sum(stacks))

    icm_result = malmuth_harville_icm(stacks, payouts)
    chip_chop = stacks / total_chips * total_prize

    blended = blend_weight * icm_result.equities + (1.0 - blend_weight) * chip_chop

    return ICMResult(equities=blended, total_prize_pool=total_prize)


def bubble_factor(
    stacks: NDArray[np.float64],
    payouts: NDArray[np.float64],
    player_index: int,
    chips_at_risk: float | None = None,
    model: ICMModel | None = None,
) -> BubbleFactorResult:
    """Calculate bubble factor for a specific player at a given stack/payout config.

    Bubble factor = equity_lost / equity_gained when a chip is won/lost.
    BF > 1 means the player is on the bubble and should tighten up relative
    to chip-EV calculations.

    Args:
        stacks: Current chip stacks.
        payouts: Payout structure.
        player_index: Index of the player to calculate for.
        chips_at_risk: Number of chips at risk in the hand. Defaults to the
            player's entire stack (all-in scenario).
        model: Optional ICM model to use (defaults to Malmuth-Harville).

    Returns:
        BubbleFactorResult with bubble factor, equity lost, and equity gained.
    """
    stacks = np.asarray(stacks, dtype=np.float64).copy()
    payouts = np.asarray(payouts, dtype=np.float64)

    if not (0 <= player_index < len(stacks)):
        raise ValueError(
            f"player_index must be in [0, {len(stacks)}), got {player_index}"
        )

    if chips_at_risk is None:
        chips_at_risk = float(stacks[player_index])

    if chips_at_risk <= 0:
        raise ValueError(f"chips_at_risk must be positive, got {chips_at_risk}")

    # Calculate base equity
    calc = model.calculate if model else malmuth_harville_icm
    base_equity = calc(stacks, payouts)
    if isinstance(base_equity, ICMResult):
        base_equity = base_equity.equities

    base_ev = float(base_equity[player_index])

    # Win scenario: player gains chips_at_risk
    stacks_win = stacks.copy()
    stacks_win[player_index] += chips_at_risk
    # Deduct from a generic opponent (player 0 or next player)
    opponent = 0 if player_index != 0 else 1
    stacks_win[opponent] = max(stacks_win[opponent] - chips_at_risk, 1.0)

    win_equity = calc(stacks_win, payouts)
    if isinstance(win_equity, ICMResult):
        win_equity = win_equity.equities
    ev_win = float(win_equity[player_index])

    # Loss scenario: player loses chips_at_risk
    stacks_loss = stacks.copy()
    stacks_loss[player_index] = max(stacks_loss[player_index] - chips_at_risk, 1.0)
    stacks_loss[opponent] += chips_at_risk

    loss_equity = calc(stacks_loss, payouts)
    if isinstance(loss_equity, ICMResult):
        loss_equity = loss_equity.equities
    ev_loss = float(loss_equity[player_index])

    equity_gained = max(ev_win - base_ev, 1e-12)
    equity_lost = max(base_ev - ev_loss, 1e-12)

    return BubbleFactorResult(
        bubble_factor=equity_lost / equity_gained,
        equity_lost=equity_lost,
        equity_gained=equity_gained,
    )


def icm_push_fold_ev(
    stacks: NDArray[np.float64],
    payouts: NDArray[np.float64],
    hero_index: int,
    hero_hand_strength: float,
    caller_indices: list[int],
    caller_call_frequencies: NDArray[np.float64],
    caller_hand_ranges: NDArray[np.float64],
    big_blind: float = 1.0,
) -> dict[str, float]:
    """Compute EV of push/fold at an ICM spot.

    This provides a clean interface for computing push/fold EV. A full Nash
    solver is out of scope for this library, but this interface allows
    plugging in external solver outputs.

    Args:
        stacks: Current chip stacks.
        payouts: Payout structure.
        hero_index: Hero's seat index.
        hero_hand_strength: Hero's estimated equity against a calling range (0-1).
        caller_indices: Indices of players who may call.
        caller_call_frequencies: Probability each caller calls (same length as caller_indices).
        caller_hand_ranges: Hero's equity vs each caller's calling range (same length).
        big_blind: Size of the big blind (used for pot calculation).

    Returns:
        Dict with 'ev_push', 'ev_fold', 'ev_diff' (push - fold).
    """
    stacks = np.asarray(stacks, dtype=np.float64).copy()
    payouts = np.asarray(payouts, dtype=np.float64)
    caller_call_freq = np.asarray(caller_call_frequencies, dtype=np.float64)
    caller_ranges = np.asarray(caller_hand_ranges, dtype=np.float64)

    hero_stack = float(stacks[hero_index])

    # EV of folding: ICM equity with unchanged stacks
    fold_equity = malmuth_harville_icm(stacks, payouts)
    ev_fold = float(fold_equity.equities[hero_index])

    # EV of pushing: fold + scenarios where called
    # Everyone folds probability
    p_all_fold = float(np.prod(1.0 - caller_call_freq))

    # When all fold, hero wins the pot
    pot = big_blind * 1.5  # SB + BB approximation
    stacks_push_fold = stacks.copy()
    stacks_push_fold[hero_index] += pot
    push_fold_equity = malmuth_harville_icm(stacks_push_fold, payouts)
    ev_push_fold_scenario = float(push_fold_equity.equities[hero_index])

    ev_push = p_all_fold * ev_push_fold_scenario

    # When called, compute weighted EV across callers
    for i, caller_idx in enumerate(caller_indices):
        p_call = float(caller_call_freq[i])
        p_others_fold = float(np.prod(1.0 - caller_call_freq[np.arange(len(caller_call_freq)) != i]))

        # Hero wins vs this caller
        equity_vs_caller = float(caller_ranges[i])
        risk_amount = min(hero_stack, float(stacks[caller_idx]))

        # Win scenario
        stacks_win = stacks.copy()
        stacks_win[hero_index] += risk_amount
        stacks_win[caller_idx] = max(stacks_win[caller_idx] - risk_amount, 1.0)
        win_eq = malmuth_harville_icm(stacks_win, payouts)
        ev_hero_win = float(win_eq.equities[hero_index])

        # Lose scenario
        stacks_loss = stacks.copy()
        stacks_loss[hero_index] = max(stacks_loss[hero_index] - risk_amount, 1.0)
        stacks_loss[caller_idx] += risk_amount
        loss_eq = malmuth_harville_icm(stacks_loss, payouts)
        ev_hero_loss = float(loss_eq.equities[hero_index])

        ev_called = equity_vs_caller * ev_hero_win + (1.0 - equity_vs_caller) * ev_hero_loss
        ev_push += p_call * p_others_fold * ev_called

    return {
        "ev_push": ev_push,
        "ev_fold": ev_fold,
        "ev_diff": ev_push - ev_fold,
    }
