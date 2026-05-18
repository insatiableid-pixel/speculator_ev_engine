"""Betting strategy backtesting with proper walk-forward validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .odds import american_to_implied_prob, remove_vig


# TODO: Implement walk-forward validation with expanding/rolling window
# TODO: Implement unit profit/loss tracking
# TODO: Implement CLV-weighted backtesting
# TODO: Implement multi-strategy comparison framework


@dataclass(frozen=True)
class BacktestResult:
    """Result of a betting strategy backtest.

    Attributes:
        n_bets: Total number of bets placed.
        roi: Return on investment (net profit / total staked).
        n_wins: Number of winning bets.
        n_losses: Number of losing bets.
        total_staked: Total amount wagered.
        total_return: Total amount returned (including stake).
        net_profit: total_return - total_staked.
        clv_mean: Mean closing line value across all bets.
        kelly_growth: Simulated Kelly-scaled bankroll growth.
        max_drawdown: Maximum drawdown fraction.
        sharpe_ratio: Sharpe ratio of returns.
    """
    n_bets: int
    roi: float
    n_wins: int
    n_losses: int
    total_staked: float
    total_return: float
    net_profit: float
    clv_mean: float
    kelly_growth: float
    max_drawdown: float
    sharpe_ratio: float


def backtest_strategy(
    model_probs: NDArray[np.float64],
    odds_american: NDArray[np.int64],
    outcomes: NDArray[np.float64],
    clv_values: NDArray[np.float64] | None = None,
    edge_threshold: float = 0.0,
    kelly_fraction: float = 1.0,
    initial_bankroll: float = 1000.0,
    stake_flat: float | None = None,
) -> BacktestResult:
    """Backtest a betting strategy with model probabilities vs. market odds.

    A bet is placed when model_prob * decimal_odds - 1 > edge_threshold.

    Args:
        model_probs: Model-estimated probabilities (n_bets,).
        odds_american: American odds for each bet.
        outcomes: Actual outcomes (1.0 = win, 0.0 = loss).
        clv_values: Optional CLV values per bet.
        edge_threshold: Minimum edge required to place a bet.
        kelly_fraction: Fraction of Kelly to size bets (default full Kelly).
        initial_bankroll: Starting bankroll for growth simulation.
        stake_flat: If set, use flat stakes instead of Kelly sizing.

    Returns:
        BacktestResult with all performance metrics.
    """
    model_probs = np.asarray(model_probs, dtype=np.float64)
    odds_american = np.asarray(odds_american, dtype=np.int64)
    outcomes = np.asarray(outcomes, dtype=np.float64)

    # Calculate edges
    decimal_odds = np.array([
        o / 100.0 + 1.0 if o >= 100 else 100.0 / abs(o) + 1.0
        for o in odds_american
    ])

    edges = model_probs * decimal_odds - 1.0
    bet_mask = edges > edge_threshold

    if not np.any(bet_mask):
        return BacktestResult(
            n_bets=0, roi=0.0, n_wins=0, n_losses=0,
            total_staked=0.0, total_return=0.0, net_profit=0.0,
            clv_mean=0.0, kelly_growth=1.0, max_drawdown=0.0, sharpe_ratio=0.0,
        )

    bet_probs = model_probs[bet_mask]
    bet_odds = decimal_odds[bet_mask]
    bet_american = odds_american[bet_mask]
    bet_outcomes = outcomes[bet_mask]

    # Size bets
    if stake_flat is not None:
        stakes = np.full(np.sum(bet_mask), stake_flat)
    else:
        # Kelly sizing: f = (p * b - q) / b, where b = decimal_odds - 1
        b = bet_odds - 1.0
        kelly_fracs = (bet_probs * b - (1 - bet_probs)) / b
        kelly_fracs = np.clip(kelly_fracs, 0.0, 0.25) * kelly_fraction
        stakes = kelly_fracs * initial_bankroll

    # Calculate returns
    wins = bet_outcomes == 1.0
    returns = np.where(wins, stakes * (bet_odds - 1.0), -stakes)

    n_bets = int(np.sum(bet_mask))
    total_staked = float(np.sum(stakes))
    total_return = float(np.sum(returns) + total_staked)
    net_profit = float(np.sum(returns))
    roi = net_profit / total_staked if total_staked > 0 else 0.0

    # Bankroll growth simulation
    bankroll = initial_bankroll
    peak = initial_bankroll
    max_dd = 0.0
    bankroll_path = [initial_bankroll]

    for r in returns:
        bankroll += r
        bankroll_path.append(bankroll)
        if bankroll > peak:
            peak = bankroll
        dd = (peak - bankroll) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    kelly_growth = bankroll / initial_bankroll

    # Sharpe ratio
    if np.std(returns) > 0:
        sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252))
    else:
        sharpe = 0.0

    clv_mean = float(np.mean(clv_values[bet_mask])) if clv_values is not None else 0.0

    return BacktestResult(
        n_bets=n_bets,
        roi=roi,
        n_wins=int(np.sum(wins)),
        n_losses=int(np.sum(~wins)),
        total_staked=total_staked,
        total_return=total_return,
        net_profit=net_profit,
        clv_mean=clv_mean,
        kelly_growth=kelly_growth,
        max_drawdown=max_dd,
        sharpe_ratio=sharpe,
    )


def walk_forward_backtest(
    model_probs: NDArray[np.float64],
    odds_american: NDArray[np.int64],
    outcomes: NDArray[np.float64],
    train_window: int = 500,
    retrain_interval: int = 100,
) -> list[BacktestResult]:
    """Walk-forward backtest: only use data available at bet time.

    Args:
        model_probs: Model probabilities (pre-computed for each period).
        odds_american: American odds per bet.
        outcomes: Actual outcomes.
        train_window: Minimum observations before betting begins.
        retrain_interval: Retrain model every N new observations.

    Returns:
        List of BacktestResult objects, one per evaluation window.

    TODO: This is a framework stub — actual walk-forward requires model retraining.
    """
    # Stub: return single-period backtest
    result = backtest_strategy(model_probs, odds_american, outcomes)
    return [result]
