"""Probability distribution utilities, calibration, and Brier scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm


@dataclass(frozen=True)
class BrierDecomposition:
    """Decomposition of the Brier score into reliability, resolution, and uncertainty.

    Attributes:
        brier_score: Raw Brier score.
        reliability: Calibration component (lower is better calibrated).
        resolution: Discrimination component (higher is better).
        uncertainty: Base-rate uncertainty (inherent to the problem).
    """
    brier_score: float
    reliability: float
    resolution: float
    uncertainty: float


def brier_score(forecasts: NDArray[np.float64],
                outcomes: NDArray[np.float64]) -> float:
    """Compute the mean Brier score for probabilistic forecasts of binary events.

    Args:
        forecasts: Array of predicted probabilities in [0, 1].
        outcomes: Array of actual outcomes (0 or 1).

    Returns:
        Mean Brier score (lower is better; 0 is perfect).
    """
    forecasts = np.asarray(forecasts, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)

    if forecasts.shape != outcomes.shape:
        raise ValueError(
            f"Shape mismatch: forecasts {forecasts.shape} vs outcomes {outcomes.shape}"
        )
    if np.any(forecasts < 0) or np.any(forecasts > 1):
        raise ValueError("Forecasts must be in [0, 1]")
    if np.any((outcomes != 0) & (outcomes != 1)):
        raise ValueError("Outcomes must be 0 or 1")

    return float(np.mean((forecasts - outcomes) ** 2))


def brier_decomposition(forecasts: NDArray[np.float64],
                        outcomes: NDArray[np.float64],
                        n_bins: int = 10) -> BrierDecomposition:
    """Decompose Brier score into reliability, resolution, and uncertainty.

    Uses binning by forecast probability.

    Args:
        forecasts: Array of predicted probabilities.
        outcomes: Array of actual outcomes (0 or 1).
        n_bins: Number of bins for the calibration curve.

    Returns:
        BrierDecomposition with all components.
    """
    forecasts = np.asarray(forecasts, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)

    bs = brier_score(forecasts, outcomes)
    mean_outcome = float(np.mean(outcomes))

    # Bin forecasts
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    reliability = 0.0
    resolution = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (forecasts >= lo) & (forecasts <= hi)
        else:
            mask = (forecasts >= lo) & (forecasts < hi)

        n_k = int(np.sum(mask))
        if n_k == 0:
            continue

        mean_forecast_k = float(np.mean(forecasts[mask]))
        mean_outcome_k = float(np.mean(outcomes[mask]))

        reliability += n_k * (mean_forecast_k - mean_outcome_k) ** 2
        resolution += n_k * (mean_outcome_k - mean_outcome) ** 2

    n = len(forecasts)
    reliability /= n
    resolution /= n
    uncertainty = mean_outcome * (1.0 - mean_outcome)

    return BrierDecomposition(
        brier_score=bs,
        reliability=reliability,
        resolution=resolution,
        uncertainty=uncertainty,
    )


def calibration_curve(
    forecasts: NDArray[np.float64],
    outcomes: NDArray[np.float64],
    n_bins: int = 10,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Compute calibration curve data.

    Args:
        forecasts: Array of predicted probabilities.
        outcomes: Array of actual outcomes (0 or 1).
        n_bins: Number of equal-width bins.

    Returns:
        Tuple of (mean_forecasts, mean_outcomes, counts) per bin.
        Bins with zero count are excluded.
    """
    forecasts = np.asarray(forecasts, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    mean_forecasts_list: list[float] = []
    mean_outcomes_list: list[float] = []
    counts_list: list[int] = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (forecasts >= lo) & (forecasts <= hi)
        else:
            mask = (forecasts >= lo) & (forecasts < hi)

        n_k = int(np.sum(mask))
        if n_k == 0:
            continue

        mean_forecasts_list.append(float(np.mean(forecasts[mask])))
        mean_outcomes_list.append(float(np.mean(outcomes[mask])))
        counts_list.append(n_k)

    return (
        np.array(mean_forecasts_list),
        np.array(mean_outcomes_list),
        np.array(counts_list, dtype=np.int64),
    )


def reliability_diagram_data(
    forecasts: NDArray[np.float64],
    outcomes: NDArray[np.float64],
    n_bins: int = 10,
) -> dict[str, NDArray[np.float64]]:
    """Return data for a reliability diagram plot.

    Args:
        forecasts: Array of predicted probabilities.
        outcomes: Array of actual outcomes.
        n_bins: Number of bins.

    Returns:
        Dict with keys: 'bin_centers', 'observed_frequencies', 'perfect_line',
        'counts'.
    """
    mean_f, mean_o, counts = calibration_curve(forecasts, outcomes, n_bins)
    return {
        "bin_centers": mean_f,
        "observed_frequencies": mean_o,
        "perfect_line": np.array([0.0, 1.0]),
        "counts": counts.astype(np.float64),
    }


def normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Standard normal CDF with configurable parameters.

    Args:
        x: Value at which to evaluate CDF.
        mu: Mean of the distribution.
        sigma: Standard deviation (must be positive).

    Returns:
        CDF value at x.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    return float(norm.cdf(x, loc=mu, scale=sigma))


def normal_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Standard normal PDF.

    Args:
        x: Value at which to evaluate PDF.
        mu: Mean.
        sigma: Standard deviation.

    Returns:
        PDF value at x.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    return float(norm.pdf(x, loc=mu, scale=sigma))


def confidence_interval(
    mean: float,
    std: float,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute a symmetric confidence interval for a normal distribution.

    Args:
        mean: Distribution mean.
        std: Distribution standard deviation.
        confidence: Confidence level (0, 1).

    Returns:
        Tuple of (lower, upper) bounds.
    """
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0,1), got {confidence}")
    if std <= 0:
        raise ValueError(f"std must be positive, got {std}")

    z = float(norm.ppf((1.0 + confidence) / 2.0))
    return (mean - z * std, mean + z * std)


def entropy(probabilities: NDArray[np.float64]) -> float:
    """Shannon entropy of a discrete probability distribution.

    Args:
        probabilities: Array of probabilities (must sum to 1).

    Returns:
        Entropy in nats.
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    probs = probs[probs > 0]  # 0 * log(0) = 0 by convention
    return float(-np.sum(probs * np.log(probs)))


def kl_divergence(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
    """KL divergence D_KL(P || Q) between two discrete distributions.

    Args:
        p: True distribution probabilities.
        q: Approximating distribution probabilities.

    Returns:
        KL divergence in nats.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    if p.shape != q.shape:
        raise ValueError(f"Shape mismatch: p {p.shape} vs q {q.shape}")

    mask = p > 0
    if np.any(q[mask] <= 0):
        raise ValueError("KL divergence is infinite when p > 0 and q = 0")

    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))
