"""Tests for sports.edge module."""

import numpy as np
import pytest

from speculator_ev_engine.sports.edge import (
    EdgeResult,
    CLVResult,
    CLVTracker,
    compute_edge,
    compute_clv,
    roi_vs_ev_reconciliation,
)


class TestComputeEdge:
    def test_positive_edge(self) -> None:
        result = compute_edge(model_prob=0.55, odds_american=-110, other_outcomes=[-110])
        assert result.edge > 0

    def test_no_edge(self) -> None:
        result = compute_edge(model_prob=0.5, odds_american=100)
        assert result.edge == pytest.approx(0.0, abs=0.05)

    def test_invalid_model_prob(self) -> None:
        with pytest.raises(ValueError, match="model_prob must be in"):
            compute_edge(model_prob=1.5, odds_american=-110)


class TestComputeCLV:
    def test_positive_clv(self) -> None:
        # You bet at +150, line closes at +120 — you got the better number
        # Closing implied prob is higher, meaning the market moved in your direction
        clv = compute_clv(open_odds=150, close_odds=120, other_open_outcomes=[-170], other_close_outcomes=[-140])
        # When open odds are longer (+150) and close is shorter (+120),
        # close_implied > open_implied, so CLV is positive
        assert clv.clv > 0

    def test_negative_clv(self) -> None:
        # Line moved against us: from +120 to +150
        # Close implied is lower, so CLV is negative
        clv = compute_clv(open_odds=120, close_odds=150, other_open_outcomes=[-140], other_close_outcomes=[-170])
        assert clv.clv < 0

    def test_no_movement(self) -> None:
        clv = compute_clv(open_odds=-110, close_odds=-110)
        assert clv.clv == pytest.approx(0.0, abs=1e-10)


class TestCLVTracker:
    def test_add_and_summary(self) -> None:
        tracker = CLVTracker()
        tracker.add_record("nfl", "pinnacle", "spread", "medium", -110, -105, [-110, -105], [-105, -115])
        tracker.add_record("nba", "draftkings", "total", "large", +120, +100, [-140, +120], [-120, +100])

        summary = tracker.summary_by("sport")
        assert "nfl" in summary
        assert "nba" in summary

    def test_flag_patterns_insufficient_samples(self) -> None:
        tracker = CLVTracker()
        tracker.add_record("nfl", "pinnacle", "spread", "medium", -110, -105)
        flags = tracker.flag_patterns(min_samples=10)
        assert len(flags) == 0

    def test_invalid_group_key(self) -> None:
        tracker = CLVTracker()
        with pytest.raises(ValueError, match="group_key"):
            tracker.summary_by("invalid_key")


class TestROIvsEVReconciliation:
    def test_reconciliation_keys(self) -> None:
        np.random.seed(42)
        n = 100
        probs = np.random.uniform(0.3, 0.7, n).astype(np.float64)
        odds = np.full(n, -110, dtype=np.int64)
        outcomes = (np.random.random(n) < probs).astype(np.float64)

        result = roi_vs_ev_reconciliation(probs, odds, outcomes)
        assert "actual_roi" in result
        assert "expected_ev_per_unit" in result
        assert "divergence" in result
        assert "n_bets" in result

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            roi_vs_ev_reconciliation(
                np.array([0.5]),
                np.array([-110, -110], dtype=np.int64),
                np.array([1.0]),
            )
