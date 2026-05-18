"""Bankroll simulation, ruin probability, and drawdown modeling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .kelly import binary_kelly, KellyResult


@dataclass(frozen=True)
class RuinResult:
    """Result of a ruin probability estimation.

    Attributes:
        ruin_probability: Estimated probability of hitting the ruin threshold.
        n_simulations: Number of Monte Carlo paths simulated.
        mean_survival_steps: Average number of steps before ruin (among ruined paths).
    """
    ruin_probability: float
    n_simulations: int
    mean_survival_steps: float


@dataclass(frozen=True)
class DrawdownResult:
    """Result of a drawdown analysis.

    Attributes:
        max_drawdown: Maximum drawdown fraction observed (0 to 1).
        mean_drawdown: Average drawdown fraction.
        median_drawdown: Median drawdown fraction.
        max_drawdown_duration: Longest drawdown in steps.
    """
    max_drawdown: float
    mean_drawdown: float
    median_drawdown: float
    max_drawdown_duration: int


def simulate_bankroll(
    p_win: float,
    odds: float,
    fraction: float,
    n_steps: int = 10_000,
    n_simulations: int = 10_000,
    ruin_threshold: float = 0.0,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Monte Carlo bankroll simulation.

    Args:
        p_win: Probability of winning each bet.
        odds: Net decimal odds on win.
        fraction: Bankroll fraction wagered each step.
        n_steps: Number of bets per simulation path.
        n_simulations: Number of independent paths.
        ruin_threshold: Bankroll level at which path is considered ruined (default 0).
        seed: Random seed for reproducibility.

    Returns:
        2-D array of shape (n_simulations, n_steps) with bankroll levels.
        Ruined paths continue at the ruin_threshold level.
    """
    if not (0.0 < p_win < 1.0):
        raise ValueError(f"p_win must be in (0,1), got {p_win}")
    if odds <= 0:
        raise ValueError(f"odds must be positive, got {odds}")
    if not (0.0 <= fraction <= 1.0):
        raise ValueError(f"fraction must be in [0,1], got {fraction}")

    rng = np.random.default_rng(seed)
    # Pre-generate all random outcomes
    wins = rng.random((n_simulations, n_steps)) < p_win

    bankrolls = np.ones((n_simulations, n_steps), dtype=np.float64)
    current = np.ones(n_simulations, dtype=np.float64)
    ruined = np.zeros(n_simulations, dtype=bool)

    for t in range(n_steps):
        bankrolls[:, t] = current
        bet_amount = fraction * current
        w = wins[:, t]
        # Win: multiply by (1 + fraction * odds), Loss: multiply by (1 - fraction)
        current = np.where(
            w,
            current + bet_amount * odds,
            current - bet_amount,
        )
        current = np.maximum(current, ruin_threshold)
        ruined |= current <= ruin_threshold

    return bankrolls


def estimate_ruin_probability(
    p_win: float,
    odds: float,
    fraction: float,
    n_steps: int = 10_000,
    n_simulations: int = 10_000,
    ruin_threshold: float = 0.0,
    seed: int | None = None,
) -> RuinResult:
    """Estimate ruin probability via Monte Carlo.

    Args:
        p_win: Probability of winning each bet.
        odds: Net decimal odds on win.
        fraction: Bankroll fraction wagered each step.
        n_steps: Max steps per path.
        n_simulations: Number of independent paths.
        ruin_threshold: Bankroll level considered ruin.
        seed: Random seed.

    Returns:
        RuinResult with probability and mean survival steps.
    """
    rng = np.random.default_rng(seed)
    wins = rng.random((n_simulations, n_steps)) < p_win

    current = np.ones(n_simulations, dtype=np.float64)
    ruined_at = np.full(n_simulations, -1, dtype=np.int64)

    for t in range(n_steps):
        active = ruined_at == -1
        if not np.any(active):
            break
        bet_amount = fraction * current
        w = wins[:, t]
        # Only update non-ruined players
        next_current = np.where(
            w,
            current + bet_amount * odds,
            current - bet_amount,
        )
        # Ruined players stay at 0
        current = np.where(active, np.maximum(next_current, 0.0), 0.0)
        newly_ruined = (current <= ruin_threshold) & active
        ruined_at[newly_ruined] = t

    ruined_mask = ruined_at >= 0
    ruin_prob = float(np.mean(ruined_mask))
    mean_survival = float(np.mean(ruined_at[ruined_mask])) if np.any(ruined_mask) else float(n_steps)

    return RuinResult(
        ruin_probability=ruin_prob,
        n_simulations=n_simulations,
        mean_survival_steps=mean_survival,
    )


def analyze_drawdowns(
    bankrolls: NDArray[np.float64],
) -> DrawdownResult:
    """Analyze drawdowns from a bankroll simulation matrix.

    Args:
        bankrolls: 2-D array (n_paths, n_steps) of bankroll levels.

    Returns:
        DrawdownResult with max, mean, median drawdowns and duration.
    """
    if bankrolls.ndim != 2:
        raise ValueError(f"Expected 2-D array, got {bankrolls.ndim}-D")

    n_paths, _ = bankrolls.shape
    running_max = np.maximum.accumulate(bankrolls, axis=1)
    drawdowns = (running_max - bankrolls) / np.where(running_max > 0, running_max, 1.0)

    max_dd = float(np.max(drawdowns))
    mean_dd = float(np.mean(drawdowns))
    median_dd = float(np.median(drawdowns))

    # Max drawdown duration: longest consecutive period in drawdown > 0
    in_drawdown = drawdowns > 1e-10
    max_duration = 0
    for i in range(n_paths):
        durations = np.diff(
            np.where(np.concatenate(([False], in_drawdown[i, :], [False])))[0]
        )
        if len(durations) > 0:
            max_duration = max(max_duration, int(np.max(durations)))

    return DrawdownResult(
        max_drawdown=max_dd,
        mean_drawdown=mean_dd,
        median_drawdown=median_dd,
        max_drawdown_duration=max_duration,
    )


def kelly_fraction_sweep(
    p_win: float,
    odds: float,
    fractions: NDArray[np.float64] | None = None,
    n_steps: int = 5_000,
    n_simulations: int = 5_000,
    seed: int | None = None,
) -> list[tuple[float, float, float]]:
    """Sweep Kelly fractions to find the empirically optimal one.

    Args:
        p_win: Probability of winning.
        odds: Net decimal odds on win.
        fractions: Array of fractions to test (default 0.01 to 1.0 in 0.01 steps).
        n_steps: Steps per simulation.
        n_simulations: Paths per fraction.
        seed: Random seed.

    Returns:
        List of (fraction, mean_log_growth, ruin_probability) tuples.
    """
    if fractions is None:
        fractions = np.arange(0.01, 1.01, 0.01)

    results: list[tuple[float, float, float]] = []
    for i, f in enumerate(fractions):
        bankrolls = simulate_bankroll(
            p_win, odds, float(f), n_steps, n_simulations,
            seed=(seed + i if seed is not None else None),
        )
        # Mean log growth: average of log(final / initial) / n_steps
        final = bankrolls[:, -1]
        final = np.maximum(final, 1e-15)
        mean_log_growth = float(np.mean(np.log(final) / n_steps))

        # Ruin probability
        ruin_prob = float(np.mean(final <= 0.001))

        results.append((float(f), mean_log_growth, ruin_prob))

    return results
