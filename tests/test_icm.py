"""Tests for poker.icm module."""

import numpy as np
import pytest

from speculator_ev_engine.poker.icm import (
    ICMResult,
    BubbleFactorResult,
    malmuth_harville_icm,
    chip_chop_icm,
    bubble_factor,
    icm_push_fold_ev,
)


class TestMalmuthHarvilleICM:
    def test_two_player(self) -> None:
        stacks = np.array([6000.0, 4000.0])
        payouts = np.array([600.0, 400.0])
        result = malmuth_harville_icm(stacks, payouts)
        assert np.sum(result.equities) == pytest.approx(1000.0, abs=1e-2)
        assert result.equities[0] > result.equities[1]

    def test_three_player(self) -> None:
        stacks = np.array([5000.0, 3000.0, 2000.0])
        payouts = np.array([500.0, 300.0, 200.0])
        result = malmuth_harville_icm(stacks, payouts)
        assert np.sum(result.equities) == pytest.approx(1000.0, abs=1e-2)

    def test_equal_stacks_equal_equity(self) -> None:
        stacks = np.array([3000.0, 3000.0, 3000.0])
        payouts = np.array([500.0, 300.0, 200.0])
        result = malmuth_harville_icm(stacks, payouts)
        assert result.equities[0] == pytest.approx(result.equities[1], abs=1e-6)
        assert result.equities[1] == pytest.approx(result.equities[2], abs=1e-6)

    def test_negative_stack_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            malmuth_harville_icm(np.array([-100.0, 5000.0]), np.array([500.0, 300.0]))

    def test_negative_payout_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            malmuth_harville_icm(np.array([5000.0, 5000.0]), np.array([-100.0, 300.0]))


class TestChipChopICM:
    def test_pure_chip_chop(self) -> None:
        stacks = np.array([6000.0, 4000.0])
        payouts = np.array([600.0, 400.0])
        result = chip_chop_icm(stacks, payouts, blend_weight=0.0)
        # Pure chip chop: equity proportional to stack
        assert result.equities[0] == pytest.approx(600.0, abs=1e-6)
        assert result.equities[1] == pytest.approx(400.0, abs=1e-6)

    def test_blend_weight_range(self) -> None:
        with pytest.raises(ValueError, match="blend_weight"):
            chip_chop_icm(np.array([5000.0, 5000.0]), np.array([600.0, 400.0]), blend_weight=-0.1)

    def test_sum_preserved(self) -> None:
        stacks = np.array([5000.0, 3000.0, 2000.0])
        payouts = np.array([500.0, 300.0, 200.0])
        result = chip_chop_icm(stacks, payouts, blend_weight=0.5)
        assert np.sum(result.equities) == pytest.approx(1000.0, abs=1e-2)


class TestBubbleFactor:
    def test_bubble_factor_greater_than_one_on_bubble(self) -> None:
        # 4 players, 3 paid: classic bubble
        stacks = np.array([4000.0, 4000.0, 4000.0, 3000.0])
        payouts = np.array([500.0, 300.0, 200.0])
        bf = bubble_factor(stacks, payouts, player_index=3)  # short stack
        assert bf.bubble_factor > 1.0

    def test_equity_lost_positive(self) -> None:
        stacks = np.array([5000.0, 3000.0, 2000.0])
        payouts = np.array([500.0, 300.0, 200.0])
        bf = bubble_factor(stacks, payouts, player_index=2)
        assert bf.equity_lost > 0
        assert bf.equity_gained > 0


class TestICMPushFoldEV:
    def test_push_fold_returns_dict(self) -> None:
        stacks = np.array([2000.0, 5000.0, 3000.0])
        payouts = np.array([500.0, 300.0, 200.0])
        result = icm_push_fold_ev(
            stacks=stacks,
            payouts=payouts,
            hero_index=0,
            hero_hand_strength=0.55,
            caller_indices=[1, 2],
            caller_call_frequencies=np.array([0.15, 0.25]),
            caller_hand_ranges=np.array([0.45, 0.40]),
        )
        assert "ev_push" in result
        assert "ev_fold" in result
        assert "ev_diff" in result
