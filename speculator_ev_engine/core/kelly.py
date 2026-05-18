"""Kelly criterion: full, fractional, multi-outcome, correlated, and uncertain-edge variants."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import norm


@dataclass(frozen=True)
class KellyResult:
    """Output of a Kelly calculation.

    Attributes:
        fraction: Optimal bankroll fraction to wager (0 to 1).
        expected_log_growth: Expected logarithmic growth rate at optimal fraction.
        ruin_probability: Approximate ruin probability at the recommended fraction.
    """
    fraction: float
    expected_log_growth: float
    ruin_probability: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.fraction <= 1.0):
            raise ValueError(f"Kelly fraction must be in [0,1], got {self.fraction}")
        if not np.isfinite(self.expected_log_growth):
            raise ValueError(
                f"Expected log growth must be finite, got {self.expected_log_growth}"
            )
        if not (0.0 <= self.ruin_probability <= 1.0):
            raise ValueError(
                f"Ruin probability must be in [0,1], got {self.ruin_probability}"
            )


def binary_kelly(p: float, b: float) -> KellyResult:
    """Full Kelly for a binary outcome.

    Args:
        p: Probability of winning.
        b: Net decimal odds received on win (e.g. 1.0 for even money, 2.0 for 2:1).

    Returns:
        KellyResult with fraction, expected log growth, and ruin probability.
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must be in (0,1), got {p}")
    if b <= 0.0:
        raise ValueError(f"b must be positive, got {b}")

    q = 1.0 - p
    f = p - q / b
    f = max(f, 0.0)

    log_growth = _binary_log_growth(f, p, b)
    ruin_prob = _approx_ruin_probability(f, p, b)

    return KellyResult(
        fraction=f,
        expected_log_growth=log_growth,
        ruin_probability=ruin_prob,
    )


def fractional_kelly(p: float, b: float, fraction: float = 0.5) -> KellyResult:
    """Fractional Kelly — scales the full Kelly fraction by a configurable multiplier.

    Args:
        p: Probability of winning.
        b: Net decimal odds on win.
        fraction: Fraction of full Kelly to use (default 0.5 = half Kelly).

    Returns:
        KellyResult with reduced fraction, log growth, and ruin probability.
    """
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0,1], got {fraction}")

    full = binary_kelly(p, b)
    f = full.fraction * fraction
    log_growth = _binary_log_growth(f, p, b)
    ruin_prob = _approx_ruin_probability(f, p, b)

    return KellyResult(
        fraction=f,
        expected_log_growth=log_growth,
        ruin_probability=ruin_prob,
    )


def multi_outcome_kelly(probabilities: NDArray[np.float64],
                         payouts: NDArray[np.float64]) -> KellyResult:
    """Kelly for a wager with more than two discrete outcomes.

    Solves: maximize E[log(1 + f * payout_i)] over f in [0, 1].

    Args:
        probabilities: Array of outcome probabilities (must sum to 1).
        payouts: Array of net payouts per unit wagered for each outcome.

    Returns:
        KellyResult with optimal fraction, log growth, and ruin probability.
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    pays = np.asarray(payouts, dtype=np.float64)

    if probs.shape != pays.shape:
        raise ValueError("probabilities and payouts must have same shape")
    if abs(probs.sum() - 1.0) > 1e-6:
        raise ValueError(f"probabilities must sum to 1.0, got {probs.sum():.6f}")
    if np.any(probs < 0):
        raise ValueError("probabilities must be non-negative")

    def neg_log_growth(f: float) -> float:
        returns = 1.0 + f * pays
        if np.any(returns <= 0):
            return 1e12  # penalty for ruin
        return -float(np.sum(probs * np.log(returns)))

    result = minimize(neg_log_growth, x0=0.01, bounds=[(0.0, 1.0)], method="L-BFGS-B")
    f_opt = float(result.x[0])

    # Compute log growth and ruin probability at optimal f
    log_growth = -result.fun if result.success else _multi_log_growth(f_opt, probs, pays)
    ruin_prob = _approx_multi_ruin(f_opt, probs, pays)

    return KellyResult(
        fraction=max(f_opt, 0.0),
        expected_log_growth=log_growth,
        ruin_probability=ruin_prob,
    )


def correlated_kelly(
    edges: NDArray[np.float64],
    odds: NDArray[np.float64],
    covariance: NDArray[np.float64],
) -> KellyResult:
    """Kelly for simultaneous correlated bets via covariance adjustment.

    Solves: maximize E[log(1 + f^T r)] subject to sum(f) <= 1, f >= 0,
    where r is the random return vector with given edge/odds/covariance structure.

    Args:
        edges: Array of edges (p_i * b_i - q_i) for each bet.
        odds: Array of net decimal odds for each bet.
        covariance: Covariance matrix of the return indicators (n_bets x n_bets).

    Returns:
        KellyResult: combined optimal fraction (sum of all bet fractions),
        log growth, and ruin probability.
    """
    edges = np.asarray(edges, dtype=np.float64)
    odds = np.asarray(odds, dtype=np.float64)
    cov = np.asarray(covariance, dtype=np.float64)

    n = len(edges)
    if edges.shape != odds.shape:
        raise ValueError("edges and odds must have same shape")
    if cov.shape != (n, n):
        raise ValueError(f"covariance must be {n}x{n}, got {cov.shape}")

    def neg_log_growth(f_vec: NDArray[np.float64]) -> float:
        f_vec = np.asarray(f_vec)
        # Approximate: use multivariate normal log-growth expansion
        # E[log(1 + f^T r)] ≈ f^T μ - 0.5 * f^T Σ f (small-f approximation)
        mu = edges
        growth = float(f_vec @ mu - 0.5 * f_vec @ cov @ f_vec)
        return -growth

    from scipy.optimize import minimize as scipy_minimize

    constraints = [{"type": "ineq", "fun": lambda f: 1.0 - np.sum(f)}]
    bounds = [(0.0, 1.0)] * n
    result = scipy_minimize(
        neg_log_growth,
        x0=np.full(n, 0.01),
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
    )

    f_opt = np.asarray(result.x)
    total_fraction = float(np.sum(f_opt))
    log_growth = -result.fun if result.success else 0.0
    ruin_prob = _approx_correlated_ruin(f_opt, edges, cov)

    return KellyResult(
        fraction=min(total_fraction, 1.0),
        expected_log_growth=log_growth,
        ruin_probability=ruin_prob,
    )


def uncertain_edge_kelly(
    edge_mean: float,
    edge_std: float,
    odds: float,
    n_samples: int = 10_000,
    seed: int | None = None,
) -> KellyResult:
    """Kelly when the edge estimate itself has uncertainty.

    Draws edge estimates from N(edge_mean, edge_std^2), computes Kelly for each,
    and returns the median fraction. This is the principled response to edge uncertainty:
    it naturally shrinks toward zero as uncertainty increases.

    Args:
        edge_mean: Mean of the edge distribution.
        edge_std: Standard deviation of the edge distribution.
        odds: Net decimal odds on win.
        n_samples: Number of Monte Carlo draws.

    Returns:
        KellyResult using the median fraction across samples.
    """
    if edge_std < 0.0:
        raise ValueError(f"edge_std must be non-negative, got {edge_std}")

    # Convert edge to win probability: edge = p*b - (1-p) → p = (1 + edge) / (1 + b)
    # But it's cleaner to sample p directly
    # edge = p*b - (1-p) = p*(1+b) - 1 → p = (1 + edge) / (1 + b)
    b = odds
    sampled_edges = np.random.default_rng(seed).normal(edge_mean, edge_std, n_samples)
    fractions = []
    for e in sampled_edges:
        p = (1.0 + e) / (1.0 + b)
        p = np.clip(p, 0.001, 0.999)
        k = binary_kelly(p, b)
        fractions.append(k.fraction)

    median_f = float(np.median(fractions))
    # Use the mean p for the growth/ruin calculations
    p_mean = (1.0 + edge_mean) / (1.0 + b)
    p_mean = np.clip(p_mean, 0.001, 0.999)
    log_growth = _binary_log_growth(median_f, p_mean, b)
    ruin_prob = _approx_ruin_probability(median_f, p_mean, b)

    return KellyResult(
        fraction=median_f,
        expected_log_growth=log_growth,
        ruin_probability=ruin_prob,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _binary_log_growth(f: float, p: float, b: float) -> float:
    """E[log(1 + f*b)] for win, E[log(1 - f)] for loss."""
    if f <= 0.0:
        return 0.0
    q = 1.0 - p
    win_term = 1.0 + f * b
    loss_term = 1.0 - f
    if win_term <= 0 or loss_term <= 0:
        return -np.inf
    return p * np.log(win_term) + q * np.log(loss_term)


def _approx_ruin_probability(f: float, p: float, b: float) -> float:
    """Approximate ruin probability using the classic gambler's ruin formula.

    For repeated bets at fraction f, ruin prob ≈ ((q*p_loss) / (p*p_win))^N as N → ∞,
    simplified to the ratio of the loss term to the win term raised to bankroll depth.
    """
    if f <= 0.0:
        return 0.0
    q = 1.0 - p
    win_growth = np.log(1.0 + f * b)
    loss_growth = np.log(1.0 - f)
    if loss_growth >= 0:
        return 0.0  # Can't lose
    if win_growth <= 0:
        return 1.0  # Always declining
    # Ruin probability approximation: exp(-2 * E[log_growth] / Var[log_return])
    # per the diffusion approximation
    expected_log = p * win_growth + q * loss_growth
    var_log = p * (win_growth - expected_log) ** 2 + q * (loss_growth - expected_log) ** 2
    if var_log < 1e-15:
        return 0.0
    return float(np.exp(-2.0 * expected_log / var_log))


def _multi_log_growth(f: float, probs: NDArray[np.float64],
                      pays: NDArray[np.float64]) -> float:
    """Log growth for multi-outcome at fraction f."""
    returns = 1.0 + f * pays
    if np.any(returns <= 0):
        return -np.inf
    return float(np.sum(probs * np.log(returns)))


def _approx_multi_ruin(f: float, probs: NDArray[np.float64],
                       pays: NDArray[np.float64]) -> float:
    """Approximate ruin probability for multi-outcome Kelly using diffusion approximation."""
    if f <= 0.0:
        return 0.0
    returns = 1.0 + f * pays
    if np.any(returns <= 0):
        return 1.0
    log_returns = np.log(returns)
    expected_log = float(np.sum(probs * log_returns))
    var_log = float(np.sum(probs * (log_returns - expected_log) ** 2))
    if var_log < 1e-15:
        return 0.0
    return float(np.exp(-2.0 * expected_log / var_log))


def _approx_correlated_ruin(
    f_vec: NDArray[np.float64],
    edges: NDArray[np.float64],
    cov: NDArray[np.float64],
) -> float:
    """Approximate ruin probability for correlated bets using diffusion approximation."""
    total_f = float(np.sum(f_vec))
    if total_f <= 0.0:
        return 0.0
    expected_log = float(f_vec @ edges - 0.5 * f_vec @ cov @ f_vec)
    var_log = float(f_vec @ cov @ f_vec)
    if var_log < 1e-15:
        return 0.0
    return float(np.exp(-2.0 * expected_log / var_log))
