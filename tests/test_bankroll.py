"""Tests for core.bankroll module."""

import numpy as np
import pytest

from speculator_ev_engine.core.bankroll import (
    simulate_bankroll,
    estimate_ruin_probability,
    analyze_drawdowns,
    kelly_fraction_sweep,
)


class TestSimulateBankroll:
    def test_shape(self) -> None:
        paths = simulate_bankroll(0.55, 1.0, 0.1, n_steps=100, n_simulations=50, seed=42)
        assert paths.shape == (50, 100)

    def test_all_start_at_one(self) -> None:
        paths = simulate_bankroll(0.55, 1.0, 0.1, n_steps=10, n_simulations=10, seed=42)
        assert np.all(paths[:, 0] == pytest.approx(1.0))

    def test_zero_fraction_flat(self) -> None:
        paths = simulate_bankroll(0.55, 1.0, 0.0, n_steps=100, n_simulations=10, seed=42)
        assert np.all(paths == pytest.approx(1.0))

    def test_invalid_p_win(self) -> None:
        with pytest.raises(ValueError, match="p_win must be in"):
            simulate_bankroll(0.0, 1.0, 0.1)

    def test_invalid_odds(self) -> None:
        with pytest.raises(ValueError, match="odds must be positive"):
            simulate_bankroll(0.55, 0.0, 0.1)

    def test_invalid_fraction(self) -> None:
        with pytest.raises(ValueError, match="fraction must be in"):
            simulate_bankroll(0.55, 1.0, 1.5)


class TestEstimateRuinProbability:
    def test_low_risk_low_ruin(self) -> None:
        result = estimate_ruin_probability(0.55, 1.0, 0.05, n_steps=1000, n_simulations=1000, seed=42)
        assert result.ruin_probability < 0.5

    def test_high_risk_high_ruin(self) -> None:
        # p=0.40, high fraction=0.25, negative EV — likely to hit 10% of bankroll
        result = estimate_ruin_probability(0.40, 1.0, 0.25, n_steps=5000, n_simulations=1000, ruin_threshold=0.1, seed=42)
        assert result.ruin_probability > 0.3

    def test_n_simulations_recorded(self) -> None:
        result = estimate_ruin_probability(0.55, 1.0, 0.05, n_simulations=500, seed=42)
        assert result.n_simulations == 500


class TestAnalyzeDrawdowns:
    def test_basic(self) -> None:
        paths = simulate_bankroll(0.55, 1.0, 0.1, n_steps=500, n_simulations=100, seed=42)
        result = analyze_drawdowns(paths)
        assert 0.0 <= result.max_drawdown <= 1.0
        assert result.mean_drawdown >= 0.0
        assert result.max_drawdown_duration > 0

    def test_flat_paths(self) -> None:
        paths = np.ones((10, 100))
        result = analyze_drawdowns(paths)
        assert result.max_drawdown == pytest.approx(0.0, abs=1e-10)

    def test_invalid_dimensions(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            analyze_drawdowns(np.ones(100))


class TestKellyFractionSweep:
    def test_sweep(self) -> None:
        results = kelly_fraction_sweep(0.55, 1.0, n_steps=500, n_simulations=200, seed=42)
        assert len(results) > 0
        # Peak growth should be near Kelly fraction
        best = max(results, key=lambda r: r[1])
        assert best[1] > 0  # positive growth at optimal
