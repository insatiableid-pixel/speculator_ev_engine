"""Probability calibration: reliability diagrams, Brier decomposition, calibration curves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.distributions import (
    brier_score,
    brier_decomposition,
    calibration_curve,
    BrierDecomposition,
)


@dataclass(frozen=True)
class CalibrationReport:
    """Full calibration analysis for a set of probabilistic forecasts.

    Attributes:
        brier_score: Raw Brier score.
        brier_decomp: Decomposed Brier score.
        bin_centers: Mean forecast probability per bin.
        bin_frequencies: Observed frequency per bin.
        bin_counts: Number of forecasts per bin.
        ece: Expected Calibration Error.
        mce: Maximum Calibration Error.
    """
    brier_score: float
    brier_decomp: BrierDecomposition
    bin_centers: NDArray[np.float64]
    bin_frequencies: NDArray[np.float64]
    bin_counts: NDArray[np.int64]
    ece: float
    mce: float


def full_calibration(
    forecasts: NDArray[np.float64],
    outcomes: NDArray[np.float64],
    n_bins: int = 10,
) -> CalibrationReport:
    """Run full calibration analysis on a set of forecasts.

    Args:
        forecasts: Array of predicted probabilities in [0, 1].
        outcomes: Array of actual outcomes (0 or 1).
        n_bins: Number of bins for the calibration curve.

    Returns:
        CalibrationReport with all calibration diagnostics.
    """
    forecasts = np.asarray(forecasts, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)

    bs = brier_score(forecasts, outcomes)
    decomp = brier_decomposition(forecasts, outcomes, n_bins)
    centers, frequencies, counts = calibration_curve(forecasts, outcomes, n_bins)

    # Expected Calibration Error
    ece = float(np.sum(counts * np.abs(centers - frequencies)) / max(np.sum(counts), 1))

    # Maximum Calibration Error
    mce = float(np.max(np.abs(centers - frequencies))) if len(centers) > 0 else 0.0

    return CalibrationReport(
        brier_score=bs,
        brier_decomp=decomp,
        bin_centers=centers,
        bin_frequencies=frequencies,
        bin_counts=counts,
        ece=ece,
        mce=mce,
    )


def calibration_correction(
    forecasts: NDArray[np.float64],
    outcomes: NDArray[np.float64],
    n_bins: int = 10,
) -> NDArray[np.float64]:
    """Apply Platt-scaling-style calibration correction using binned frequencies.

    Maps forecast probabilities through the empirical calibration curve,
    producing better-calibrated probabilities.

    Args:
        forecasts: Array of predicted probabilities.
        outcomes: Array of actual outcomes.
        n_bins: Number of bins.

    Returns:
        Calibrated probability estimates.
    """
    forecasts = np.asarray(forecasts, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)

    centers, frequencies, counts = calibration_curve(forecasts, outcomes, n_bins)

    if len(centers) == 0:
        return forecasts.copy()

    # Map each forecast to the nearest bin's observed frequency
    calibrated = np.empty_like(forecasts)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    for i, f in enumerate(forecasts):
        bin_idx = np.searchsorted(bin_edges[1:], f, side="right")
        bin_idx = min(bin_idx, n_bins - 1)

        # Find the matching bin in our calibration data
        best_bin = np.argmin(np.abs(centers - f))
        calibrated[i] = frequencies[best_bin]

    return np.clip(calibrated, 0.001, 0.999)


def cross_validated_calibration(
    forecasts: NDArray[np.float64],
    outcomes: NDArray[np.float64],
    n_folds: int = 5,
    n_bins: int = 10,
) -> list[CalibrationReport]:
    """Cross-validated calibration to avoid overfitting calibration curves.

    Args:
        forecasts: Array of predicted probabilities.
        outcomes: Array of actual outcomes.
        n_folds: Number of CV folds.
        n_bins: Number of bins per fold.

    Returns:
        List of CalibrationReport objects, one per fold.
    """
    forecasts = np.asarray(forecasts, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)

    n = len(forecasts)
    indices = np.arange(n)
    np.random.shuffle(indices)

    fold_size = n // n_folds
    reports: list[CalibrationReport] = []

    for fold in range(n_folds):
        start = fold * fold_size
        end = start + fold_size if fold < n_folds - 1 else n
        val_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])

        report = full_calibration(
            forecasts[val_idx], outcomes[val_idx], n_bins
        )
        reports.append(report)

    return reports


def calibration_score(ece: float, brier: float) -> float:
    """Composite calibration quality score (0 = perfect, higher = worse).

    Args:
        ece: Expected Calibration Error.
        brier: Brier score.

    Returns:
        Composite score.
    """
    return 0.5 * ece + 0.5 * brier
