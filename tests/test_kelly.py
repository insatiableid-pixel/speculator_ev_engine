"""Tests for core.kelly module."""

import numpy as np
import pytest

from speculator_ev_engine.core.kelly import (
    KellyResult,
    binary_kelly,
    fractional_kelly,
    multi_outcome_kelly,
    correlated_kelly,
    uncertain_edge_kelly,
)


class TestBinaryKelly:
    def test_even_money_positive_edge(self) -> None:
        result = binary_kelly(p=0.55, b=1.0)
        assert result.fraction == pytest.approx(0.10, abs=1e-10)
        assert result.expected_log_growth > 0
        assert result.ruin_probability < 0.5

    def test_no_edge_zero_fraction(self) -> None:
        result = binary_kelly(p=0.5, b=1.0)
        assert result.fraction == pytest.approx(0.0, abs=1e-10)

    def test_negative_edge_zero_fraction(self) -> None:
        result = binary_kelly(p=0.45, b=1.0)
        assert result.fraction == pytest.approx(0.0, abs=1e-10)

    def test_asymmetric_odds(self) -> None:
        result = binary_kelly(p=0.40, b=2.0)
        # f = p - q/b = 0.4 - 0.6/2.0 = 0.1
        assert result.fraction == pytest.approx(0.10, abs=1e-10)

    def test_invalid_p(self) -> None:
        with pytest.raises(ValueError, match="p must be in"):
            binary_kelly(p=0.0, b=1.0)

    def test_invalid_b(self) -> None:
        with pytest.raises(ValueError, match="b must be positive"):
            binary_kelly(p=0.5, b=-1.0)

    def test_result_bounds(self) -> None:
        result = binary_kelly(p=0.6, b=2.0)
        assert 0.0 <= result.fraction <= 1.0
        assert 0.0 <= result.ruin_probability <= 1.0
        assert np.isfinite(result.expected_log_growth)


class TestFractionalKelly:
    def test_half_kelly(self) -> None:
        full = binary_kelly(p=0.55, b=1.0)
        half = fractional_kelly(p=0.55, b=1.0, fraction=0.5)
        assert half.fraction == pytest.approx(full.fraction * 0.5, abs=1e-10)

    def test_quarter_kelly(self) -> None:
        full = binary_kelly(p=0.6, b=1.0)
        quarter = fractional_kelly(p=0.6, b=1.0, fraction=0.25)
        assert quarter.fraction == pytest.approx(full.fraction * 0.25, abs=1e-10)

    def test_invalid_fraction(self) -> None:
        with pytest.raises(ValueError, match="fraction must be in"):
            fractional_kelly(p=0.55, b=1.0, fraction=0.0)

    def test_fractional_reduces_ruin(self) -> None:
        full = binary_kelly(p=0.55, b=1.0)
        half = fractional_kelly(p=0.55, b=1.0, fraction=0.5)
        assert half.ruin_probability < full.ruin_probability


class TestMultiOutcomeKelly:
    def test_three_outcome(self) -> None:
        probs = np.array([0.5, 0.3, 0.2])
        payouts = np.array([2.0, -1.0, -1.0])
        result = multi_outcome_kelly(probs, payouts)
        assert 0.0 <= result.fraction <= 1.0
        assert result.expected_log_growth > 0

    def test_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            multi_outcome_kelly(np.array([0.5, 0.5]), np.array([1.0]))

    def test_probabilities_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            multi_outcome_kelly(np.array([0.5, 0.3]), np.array([1.0, -1.0]))

    def test_negative_probability(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            multi_outcome_kelly(np.array([0.5, -0.1, 0.6]), np.array([1.0, -1.0, -1.0]))


class TestCorrelatedKelly:
    def test_two_uncorrelated_bets(self) -> None:
        edges = np.array([0.03, 0.04])
        odds = np.array([1.0, 1.0])
        cov = np.eye(2)  # uncorrelated
        result = correlated_kelly(edges, odds, cov)
        assert 0.0 <= result.fraction <= 1.0

    def test_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            correlated_kelly(np.array([0.03]), np.array([0.03, 0.04]), np.eye(2))

    def test_covariance_shape(self) -> None:
        with pytest.raises(ValueError, match="covariance must be"):
            correlated_kelly(np.array([0.03, 0.04]), np.array([1.0, 1.0]), np.eye(3))


class TestUncertainEdgeKelly:
    def test_certain_edge(self) -> None:
        # When edge_std=0, should match binary Kelly
        result = uncertain_edge_kelly(edge_mean=0.05, edge_std=0.0, odds=1.0, n_samples=5000, seed=42)
        # Should be close to binary_kelly(0.525, 1.0).fraction ≈ 0.05
        assert 0.0 <= result.fraction <= 1.0

    def test_uncertain_edge_shrinks(self) -> None:
        certain = uncertain_edge_kelly(edge_mean=0.05, edge_std=0.0, odds=1.0, n_samples=5000, seed=42)
        uncertain = uncertain_edge_kelly(edge_mean=0.05, edge_std=0.05, odds=1.0, n_samples=5000, seed=42)
        assert uncertain.fraction < certain.fraction

    def test_negative_std_raises(self) -> None:
        with pytest.raises(ValueError, match="edge_std must be non-negative"):
            uncertain_edge_kelly(edge_mean=0.05, edge_std=-0.01, odds=1.0)


class TestKellyResult:
    def test_fraction_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="Kelly fraction must be in"):
            KellyResult(fraction=1.5, expected_log_growth=0.0, ruin_probability=0.0)

    def test_infinite_log_growth(self) -> None:
        with pytest.raises(ValueError, match="Expected log growth must be finite"):
            KellyResult(fraction=0.1, expected_log_growth=float("inf"), ruin_probability=0.0)

    def test_ruin_probability_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="Ruin probability must be in"):
            KellyResult(fraction=0.1, expected_log_growth=0.0, ruin_probability=-0.1)
