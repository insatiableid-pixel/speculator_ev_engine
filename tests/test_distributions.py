"""Tests for core.distributions module."""

import numpy as np
import pytest

from speculator_ev_engine.core.distributions import (
    brier_score,
    brier_decomposition,
    calibration_curve,
    reliability_diagram_data,
    normal_cdf,
    normal_pdf,
    confidence_interval,
    entropy,
    kl_divergence,
)


class TestBrierScore:
    def test_perfect_forecasts(self) -> None:
        forecasts = np.array([1.0, 0.0, 1.0])
        outcomes = np.array([1, 0, 1])
        assert brier_score(forecasts, outcomes) == pytest.approx(0.0, abs=1e-10)

    def test_worst_forecasts(self) -> None:
        forecasts = np.array([0.0, 1.0])
        outcomes = np.array([1, 0])
        assert brier_score(forecasts, outcomes) == pytest.approx(1.0, abs=1e-10)

    def test_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="Shape mismatch"):
            brier_score(np.array([0.5]), np.array([0.5, 1.0]))

    def test_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="Forecasts must be in"):
            brier_score(np.array([1.5]), np.array([1.0]))


class TestBrierDecomposition:
    def test_perfect(self) -> None:
        forecasts = np.array([1.0, 0.0, 1.0, 0.0])
        outcomes = np.array([1, 0, 1, 0])
        decomp = brier_decomposition(forecasts, outcomes)
        assert decomp.reliability == pytest.approx(0.0, abs=0.01)
        assert decomp.brier_score == pytest.approx(0.0, abs=1e-10)

    def test_components_non_negative(self) -> None:
        rng = np.random.default_rng(42)
        forecasts = rng.uniform(0.1, 0.9, 100)
        outcomes = (rng.random(100) < 0.5).astype(float)
        decomp = brier_decomposition(forecasts, outcomes)
        assert decomp.reliability >= 0.0
        assert decomp.resolution >= 0.0
        assert 0.0 <= decomp.uncertainty <= 0.25


class TestCalibrationCurve:
    def test_perfect_calibration(self) -> None:
        forecasts = np.array([0.2, 0.4, 0.6, 0.8])
        outcomes = np.array([0, 0, 1, 1])
        centers, freqs, counts = calibration_curve(forecasts, outcomes, n_bins=4)
        # With only 1 sample per bin, the observed frequency should be 0 or 1
        assert len(centers) > 0

    def test_empty_bins_excluded(self) -> None:
        forecasts = np.array([0.1, 0.1, 0.1])
        outcomes = np.array([0, 1, 1])
        centers, freqs, counts = calibration_curve(forecasts, outcomes, n_bins=10)
        # Many bins should be empty, excluded from output
        assert len(counts) < 10


class TestNormalCDF:
    def test_at_zero(self) -> None:
        assert normal_cdf(0.0) == pytest.approx(0.5, abs=1e-10)

    def test_at_infinity(self) -> None:
        assert normal_cdf(10.0) == pytest.approx(1.0, abs=1e-6)

    def test_parameters(self) -> None:
        assert normal_cdf(0.0, mu=1.0, sigma=2.0) == pytest.approx(0.3085, abs=1e-3)

    def test_negative_sigma(self) -> None:
        with pytest.raises(ValueError, match="sigma must be positive"):
            normal_cdf(0.0, sigma=-1.0)


class TestConfidenceInterval:
    def test_95_percent(self) -> None:
        lo, hi = confidence_interval(0.0, 1.0, 0.95)
        assert lo == pytest.approx(-1.96, abs=0.01)
        assert hi == pytest.approx(1.96, abs=0.01)

    def test_invalid_confidence(self) -> None:
        with pytest.raises(ValueError, match="confidence must be in"):
            confidence_interval(0.0, 1.0, 1.5)


class TestEntropy:
    def test_uniform(self) -> None:
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        assert entropy(probs) == pytest.approx(np.log(4), abs=1e-10)

    def test_deterministic(self) -> None:
        probs = np.array([1.0, 0.0, 0.0])
        assert entropy(probs) == pytest.approx(0.0, abs=1e-10)


class TestKLDivergence:
    def test_identical(self) -> None:
        p = np.array([0.5, 0.5])
        assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-10)

    def test_different(self) -> None:
        p = np.array([1.0, 0.0])
        q = np.array([0.5, 0.5])
        assert kl_divergence(p, q) > 0

    def test_zero_in_q(self) -> None:
        with pytest.raises(ValueError, match="infinite"):
            kl_divergence(np.array([0.5, 0.5]), np.array([1.0, 0.0]))

    def test_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="Shape mismatch"):
            kl_divergence(np.array([0.5]), np.array([0.5, 0.5]))
