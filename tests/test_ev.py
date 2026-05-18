"""Tests for core.ev module."""

import numpy as np
import pytest

from speculator_ev_engine.core.ev import (
    EVResult,
    MultiOutcomeEV,
    binary_ev,
    multi_outcome_ev,
    ev_per_unit_risk,
    DecisionNode,
    DecisionTree,
    ev_grid,
)


class TestBinaryEV:
    def test_positive_ev(self) -> None:
        result = binary_ev(p_win=0.6, payout_win=1.0, payout_loss=-1.0)
        assert result.ev == pytest.approx(0.2, abs=1e-10)
        assert result.p_win == 0.6
        assert result.p_loss == 0.4

    def test_negative_ev(self) -> None:
        result = binary_ev(p_win=0.4, payout_win=1.0, payout_loss=-1.0)
        assert result.ev == pytest.approx(-0.2, abs=1e-10)

    def test_fifty_fifty(self) -> None:
        result = binary_ev(p_win=0.5, payout_win=1.0, payout_loss=-1.0)
        assert result.ev == pytest.approx(0.0, abs=1e-10)

    def test_variance_positive(self) -> None:
        result = binary_ev(p_win=0.6, payout_win=1.0, payout_loss=-1.0)
        assert result.variance > 0

    def test_invalid_p_win(self) -> None:
        with pytest.raises(ValueError, match="p_win must be in"):
            binary_ev(p_win=1.5, payout_win=1.0, payout_loss=-1.0)

    def test_asymmetric_payout(self) -> None:
        result = binary_ev(p_win=0.4, payout_win=3.0, payout_loss=-1.0)
        assert result.ev == pytest.approx(0.4 * 3.0 + 0.6 * (-1.0), abs=1e-10)


class TestMultiOutcomeEV:
    def test_three_outcomes(self) -> None:
        outcomes = {"win": (0.5, 2.0), "draw": (0.3, 0.0), "loss": (0.2, -1.0)}
        result = multi_outcome_ev(outcomes)
        assert result.ev == pytest.approx(0.5 * 2.0 + 0.3 * 0.0 + 0.2 * (-1.0), abs=1e-10)

    def test_probabilities_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="Probabilities must sum to 1"):
            multi_outcome_ev({"a": (0.5, 1.0), "b": (0.3, -1.0)})

    def test_variance(self) -> None:
        outcomes = {"a": (0.5, 1.0), "b": (0.5, -1.0)}
        result = multi_outcome_ev(outcomes)
        assert result.variance == pytest.approx(1.0, abs=1e-10)


class TestEVResult:
    def test_finite_ev_required(self) -> None:
        with pytest.raises(ValueError, match="EV must be finite"):
            EVResult(ev=float("inf"), p_win=0.5, p_loss=0.5, payout_win=1.0, payout_loss=-1.0)

    def test_p_win_range(self) -> None:
        with pytest.raises(ValueError, match="p_win must be in"):
            EVResult(ev=0.0, p_win=-0.1, p_loss=0.5, payout_win=1.0, payout_loss=-1.0)


class TestEVPerUnitRisk:
    def test_positive(self) -> None:
        result = binary_ev(0.6, 1.0, -1.0)
        ratio = ev_per_unit_risk(result)
        assert ratio == pytest.approx(0.2, abs=1e-10)

    def test_zero_risk_raises(self) -> None:
        result = EVResult(ev=0.0, p_win=1.0, p_loss=0.0, payout_win=0.0, payout_loss=0.0)
        with pytest.raises(ValueError, match="Cannot compute EV per unit risk"):
            ev_per_unit_risk(result)


class TestDecisionTree:
    def test_optimal_path(self) -> None:
        node_a = DecisionNode(name="fold", ev_func=lambda: binary_ev(0.5, 1.0, -1.0))
        node_b = DecisionNode(name="call", ev_func=lambda: binary_ev(0.6, 1.0, -1.0))
        root = DecisionNode(
            name="decision",
            ev_func=lambda: binary_ev(0.5, 1.0, -1.0),
            children=[node_a, node_b],
        )
        tree = DecisionTree(root=root)
        path = tree.optimal_path()
        assert len(path) == 2
        assert path[1][0] == "call"
        assert path[1][1] == pytest.approx(0.2, abs=1e-10)


class TestEVGrid:
    def test_grid(self) -> None:
        p_range = np.linspace(0.0, 1.0, 11)
        grid = ev_grid(p_range, payout_win=1.0, payout_loss=-1.0)
        assert grid[5] == pytest.approx(0.0, abs=1e-10)  # p=0.5
        assert grid[0] == pytest.approx(-1.0, abs=1e-10)  # p=0
        assert grid[-1] == pytest.approx(1.0, abs=1e-10)  # p=1
