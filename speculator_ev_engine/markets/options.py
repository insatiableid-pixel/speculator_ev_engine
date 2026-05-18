"""Options pricing: Black-Scholes, binomial tree, Greeks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm


# TODO: Implement American option pricing via binomial tree
# TODO: Implement volatility surface fitting
# TODO: Implement dividend-adjusted models


@dataclass(frozen=True)
class OptionPrice:
    """Result of an option pricing calculation.

    Attributes:
        price: Option price.
        delta: First-order price sensitivity to underlying.
        gamma: Second-order price sensitivity to underlying.
        theta: Time decay (per calendar day).
        vega: Sensitivity to volatility (per 1% change).
        rho: Sensitivity to interest rate.
    """
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


def black_scholes(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"] = "call",
    q: float = 0.0,
) -> OptionPrice:
    """Black-Scholes option pricing with Greeks.

    Args:
        S: Underlying price.
        K: Strike price.
        T: Time to expiration in years.
        r: Risk-free rate (annualized).
        sigma: Implied volatility (annualized).
        option_type: "call" or "put".
        q: Continuous dividend yield.

    Returns:
        OptionPrice with price and all Greeks.
    """
    if S <= 0:
        raise ValueError(f"Underlying price must be positive, got {S}")
    if K <= 0:
        raise ValueError(f"Strike price must be positive, got {K}")
    if T <= 0:
        raise ValueError(f"Time to expiration must be positive, got {T}")
    if sigma <= 0:
        raise ValueError(f"Volatility must be positive, got {sigma}")

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = np.exp(-q * T) * norm.cdf(d1)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        delta = -np.exp(-q * T) * norm.cdf(-d1)

    gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (
        -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        - r * K * np.exp(-r * T) * norm.cdf(d2 if option_type == "call" else -d2)
        + q * S * np.exp(-q * T) * norm.cdf(d1 if option_type == "call" else -d1)
    ) / 365.0
    vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100.0
    rho = K * T * np.exp(-r * T) * norm.cdf(d2 if option_type == "call" else -d2) / 100.0

    return OptionPrice(
        price=float(price),
        delta=float(delta),
        gamma=float(gamma),
        theta=float(theta),
        vega=float(vega),
        rho=float(rho),
    )


def binomial_tree(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_steps: int = 100,
    option_type: Literal["call", "put"] = "call",
    american: bool = False,
    q: float = 0.0,
) -> float:
    """Binomial tree option pricing (Cox-Ross-Rubinstein).

    Args:
        S: Underlying price.
        K: Strike price.
        T: Time to expiration in years.
        r: Risk-free rate.
        sigma: Volatility.
        n_steps: Number of time steps.
        option_type: "call" or "put".
        american: If True, price American-style option.
        q: Continuous dividend yield.

    Returns:
        Option price.
    """
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)
    p_up = (np.exp((r - q) * dt) - d) / (u - d)

    # Terminal values
    ST = S * u ** np.arange(n_steps, -1, -1) * d ** np.arange(0, n_steps + 1)

    if option_type == "call":
        values = np.maximum(ST - K, 0.0)
    else:
        values = np.maximum(K - ST, 0.0)

    for t in range(n_steps - 1, -1, -1):
        values = disc * (p_up * values[:-1] + (1 - p_up) * values[1:])

        if american:
            S_t = S * u ** np.arange(t, -1, -1) * d ** np.arange(0, t + 1)
            if option_type == "call":
                values = np.maximum(values, S_t - K)
            else:
                values = np.maximum(values, K - S_t)

    return float(values[0])


def implied_volatility(
    S: float,
    K: float,
    T: float,
    r: float,
    market_price: float,
    option_type: Literal["call", "put"] = "call",
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """Compute implied volatility via Newton-Raphson.

    Args:
        S: Underlying price.
        K: Strike price.
        T: Time to expiration.
        r: Risk-free rate.
        market_price: Observed option price.
        option_type: "call" or "put".
        tol: Convergence tolerance.
        max_iter: Maximum iterations.

    Returns:
        Implied volatility.
    """
    sigma = 0.3  # initial guess

    for _ in range(max_iter):
        result = black_scholes(S, K, T, r, sigma, option_type)
        diff = result.price - market_price
        if abs(diff) < tol:
            return sigma
        if abs(result.vega) < 1e-12:
            break
        sigma -= diff / (result.vega * 100.0)  # vega is per 1%
        sigma = max(sigma, 0.001)

    raise RuntimeError(f"Implied vol did not converge after {max_iter} iterations")
