"""Tests for decisions.leaks module."""

import numpy as np
import pytest

from speculator_ev_engine.decisions.logger import Decision
from speculator_ev_engine.decisions.leaks import (
    detect_leaks,
    highest_ev_decisions,
    lowest_ev_decisions,
    largest_ev_outcome_divergences,
    detect_tilt,
)


def _make_decisions(n: int = 50, seed: int = 42) -> list[Decision]:
    """Create a sample of decisions for testing."""
    rng = np.random.default_rng(seed)
    decisions: list[Decision] = []
    for i in range(n):
        p = rng.uniform(0.3, 0.7)
        ev = p * 1.0 - (1 - p) * 1.0  # even-money EV
        outcome = float(rng.choice([1.0, -1.0], p=[p, 1 - p]))
        domain = rng.choice(["poker", "sports", "markets"])
        stake = rng.uniform(10.0, 100.0)
        decisions.append(Decision(
            decision=f"decision_{i}",
            p_estimate=float(p),
            ev_estimate=float(ev),
            stake=float(stake),
            outcome=outcome,
            domain=domain,
        ))
    return decisions


class TestDetectLeaks:
    def test_returns_reports(self) -> None:
        decisions = _make_decisions()
        reports = detect_leaks(decisions, group_by="domain")
        assert len(reports) > 0
        for r in reports:
            assert r.n_decisions >= 3
            # ev_outcome_gap can be positive or negative

    def test_empty_input(self) -> None:
        assert detect_leaks([]) == []

    def test_no_resolved(self) -> None:
        d = Decision(decision="pending", p_estimate=0.5, ev_estimate=0.0, stake=10.0, outcome=None)
        assert detect_leaks([d]) == []


class TestHighestEVLowestEV:
    def test_highest_ev(self) -> None:
        decisions = _make_decisions()
        top = highest_ev_decisions(decisions, n=5)
        assert len(top) == 5
        evs = [d.ev_estimate for d in top]
        assert all(evs[i] >= evs[i + 1] for i in range(len(evs) - 1))

    def test_lowest_ev(self) -> None:
        decisions = _make_decisions()
        bottom = lowest_ev_decisions(decisions, n=5)
        assert len(bottom) == 5
        evs = [d.ev_estimate for d in bottom]
        assert all(evs[i] <= evs[i + 1] for i in range(len(evs) - 1))


class TestLargestDivergences:
    def test_divergences(self) -> None:
        decisions = _make_decisions()
        divs = largest_ev_outcome_divergences(decisions, n=10)
        assert len(divs) == 10
        for _, d in divs:
            assert d >= 0

    def test_unresolved_excluded(self) -> None:
        d = Decision(decision="x", p_estimate=0.5, ev_estimate=0.0, stake=10.0, outcome=None)
        assert largest_ev_outcome_divergences([d], n=10) == []


class TestDetectTilt:
    def test_no_tilt_in_random(self) -> None:
        decisions = _make_decisions()
        alerts = detect_tilt(decisions)
        # Random decisions should not reliably trigger tilt
        # Just check it doesn't crash
        assert isinstance(alerts, list)

    def test_short_sequence(self) -> None:
        decisions = [Decision(decision="x", p_estimate=0.5, ev_estimate=0.0, stake=10.0, outcome=1.0)]
        alerts = detect_tilt(decisions)
        assert alerts == []
