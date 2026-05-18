"""Kelly-optimal portfolio construction, mean-variance vs. EV maximization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize


# TODO: Implement full Kelly portfolio with correlated assets
# TODO: Implement constrained Kelly (max position size, sector limits)
# TODO: Implement mean-variance frontier for comparison with Kelly
# TODO: Implement risk-parity baseline


@dataclass(frozen=True)
class PortfolioAllocation:
    """Result of a portfolio optimization.

    Attributes:
        weights: Array of portfolio weights (sum to 1).
        expected_growth: Expected log-growth rate.
        expected_return: Expected arithmetic return.
        portfolio_volatility: Portfolio standard deviation.
        sharpe_ratio: Return / volatility.
    """
    weights: NDArray[np.float64]
    expected_growth: float
    expected_return: float
    portfolio_volatility: float
    sharpe_ratio: float


def kelly_portfolio(
    expected_returns: NDArray[np.float64],
    covariance: NDArray[np.float64],
    risk_free_rate: float = 0.0,
    max_weight: float = 1.0,
) -> PortfolioAllocation:
    """Kelly-optimal portfolio allocation for continuous-time setting.

    In continuous time, the Kelly portfolio weights are:
    w = Σ^(-1) (μ - r)

    This is the growth-optimal portfolio that maximizes E[log(wealth)].

    Args:
        expected_returns: Array of expected returns per asset.
        covariance: Covariance matrix of asset returns.
        risk_free_rate: Risk-free rate.
        max_weight: Maximum weight per asset (for position limits).

    Returns:
        PortfolioAllocation with optimal weights and metrics.
    """
    mu = np.asarray(expected_returns, dtype=np.float64)
    cov = np.asarray(covariance, dtype=np.float64)

    excess = mu - risk_free_rate
    inv_cov = np.linalg.inv(cov)
    raw_weights = inv_cov @ excess

    # Apply position limits
    weights = np.clip(raw_weights, -max_weight, max_weight)
    # Re-normalize long-only or allow shorts?
    # For now, normalize to sum to 1 if all positive, else keep raw
    if np.all(weights > 0):
        weights = weights / np.sum(weights)

    port_return = float(weights @ mu)
    port_var = float(weights @ cov @ weights)
    port_vol = np.sqrt(port_var)
    growth = float(port_return - 0.5 * port_var)  # continuous-time log growth

    sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 1e-12 else 0.0

    return PortfolioAllocation(
        weights=weights,
        expected_growth=growth,
        expected_return=port_return,
        portfolio_volatility=port_vol,
        sharpe_ratio=sharpe,
    )


def mean_variance_portfolio(
    expected_returns: NDArray[np.float64],
    covariance: NDArray[np.float64],
    target_return: float | None = None,
    risk_free_rate: float = 0.0,
) -> PortfolioAllocation:
    """Mean-variance optimized portfolio (Markowitz).

    Args:
        expected_returns: Array of expected returns.
        covariance: Covariance matrix.
        target_return: Target portfolio return (None = max Sharpe).
        risk_free_rate: Risk-free rate.

    Returns:
        PortfolioAllocation.
    """
    mu = np.asarray(expected_returns, dtype=np.float64)
    cov = np.asarray(covariance, dtype=np.float64)
    n = len(mu)

    if target_return is not None:
        # Minimize variance subject to target return
        constraints = [
            {"type": "eq", "fun": lambda w: float(w @ mu) - target_return},
            {"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0},
        ]
    else:
        # Max Sharpe ratio
        constraints = [
            {"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0},
        ]

    bounds = [(0.0, 1.0)] * n

    def objective(w: NDArray[np.float64]) -> float:
        if target_return is not None:
            return float(w @ cov @ w)
        else:
            ret = float(w @ mu) - risk_free_rate
            vol = np.sqrt(float(w @ cov @ w))
            return -ret / vol if vol > 1e-12 else 0.0

    result = minimize(
        objective,
        x0=np.ones(n) / n,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
    )

    weights = np.asarray(result.x)
    port_return = float(weights @ mu)
    port_var = float(weights @ cov @ weights)
    port_vol = np.sqrt(port_var)
    growth = float(port_return - 0.5 * port_var)
    sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 1e-12 else 0.0

    return PortfolioAllocation(
        weights=weights,
        expected_growth=growth,
        expected_return=port_return,
        portfolio_volatility=port_vol,
        sharpe_ratio=sharpe,
    )
