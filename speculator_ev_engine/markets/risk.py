"""Risk metrics: VaR, CVaR, max drawdown, correlation under stress."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


# TODO: Implement historical VaR with configurable confidence
# TODO: Implement parametric VaR (normal, t-distribution)
# TODO: Implement Monte Carlo VaR
# TODO: Implement stress-test correlation breakdown models


@dataclass(frozen=True)
class RiskMetrics:
    """Computed risk metrics for a return series or portfolio.

    Attributes:
        var_95: Value at Risk at 95% confidence (as fraction of portfolio).
        var_99: Value at Risk at 99% confidence.
        cvar_95: Conditional VaR (expected shortfall) at 95%.
        cvar_99: Conditional VaR at 99%.
        max_drawdown: Maximum drawdown fraction.
        annualized_volatility: Annualized volatility.
        downside_deviation: Downside deviation (returns below 0).
    """
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown: float
    annualized_volatility: float
    downside_deviation: float


def compute_risk_metrics(
    returns: NDArray[np.float64],
    periods_per_year: int = 252,
) -> RiskMetrics:
    """Compute comprehensive risk metrics from a return series.

    Args:
        returns: Array of periodic returns (e.g. daily).
        periods_per_year: Number of periods per year for annualization.

    Returns:
        RiskMetrics with all standard risk measures.
    """
    returns = np.asarray(returns, dtype=np.float64)

    var_95 = float(-np.percentile(returns, 5))
    var_99 = float(-np.percentile(returns, 1))

    # CVaR: mean of returns below the VaR threshold
    cvar_95 = float(-np.mean(returns[returns <= np.percentile(returns, 5)]))
    cvar_99 = float(-np.mean(returns[returns <= np.percentile(returns, 1)]))

    # Max drawdown
    cum_returns = np.cumsum(returns)
    running_max = np.maximum.accumulate(np.concatenate([[0], cum_returns]))[1:]
    drawdowns = cum_returns - running_max
    max_dd = float(-np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Annualized volatility
    ann_vol = float(np.std(returns) * np.sqrt(periods_per_year))

    # Downside deviation
    negative_returns = returns[returns < 0]
    downside_dev = float(np.std(negative_returns)) if len(negative_returns) > 0 else 0.0

    return RiskMetrics(
        var_95=var_95,
        var_99=var_99,
        cvar_95=cvar_95,
        cvar_99=cvar_99,
        max_drawdown=max_dd,
        annualized_volatility=ann_vol,
        downside_deviation=downside_dev,
    )


def correlation_under_stress(
    returns_matrix: NDArray[np.float64],
    threshold_percentile: float = 10.0,
) -> NDArray[np.float64]:
    """Compute correlation matrix conditional on stressed market conditions.

    Filters to periods where the market return (first column) is below
    the given percentile, then computes correlation on those periods.

    Args:
        returns_matrix: 2-D array (n_periods, n_assets). First column is market.
        threshold_percentile: Percentile threshold for stress definition.

    Returns:
        Stress-conditioned correlation matrix.
    """
    returns_matrix = np.asarray(returns_matrix, dtype=np.float64)
    market_returns = returns_matrix[:, 0]
    threshold = np.percentile(market_returns, threshold_percentile)
    stressed = returns_matrix[market_returns <= threshold]
    if stressed.shape[0] < 5:
        return np.corrcoef(returns_matrix.T)
    return np.corrcoef(stressed.T)
