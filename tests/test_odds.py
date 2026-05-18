"""Tests for sports.odds module."""

import numpy as np
import pytest

from speculator_ev_engine.sports.odds import (
    american_to_decimal,
    decimal_to_american,
    american_to_implied_prob,
    decimal_to_implied_prob,
    implied_prob_to_decimal,
    extract_vig,
    remove_vig,
    convert_odds,
)


class TestAmericanToDecimal:
    def test_positive(self) -> None:
        assert american_to_decimal(150) == pytest.approx(2.5, abs=1e-10)

    def test_negative(self) -> None:
        assert american_to_decimal(-110) == pytest.approx(1.0 + 100.0 / 110.0, abs=1e-10)

    def test_even(self) -> None:
        assert american_to_decimal(100) == pytest.approx(2.0, abs=1e-10)

    def test_invalid(self) -> None:
        with pytest.raises(ValueError, match="American odds must be"):
            american_to_decimal(50)


class TestDecimalToAmerican:
    def test_from_positive(self) -> None:
        assert decimal_to_american(2.5) == 150

    def test_from_negative(self) -> None:
        result = decimal_to_american(1.909)
        assert result <= -100  # should be around -110

    def test_invalid(self) -> None:
        with pytest.raises(ValueError, match="Decimal odds must be >= 1"):
            decimal_to_american(0.5)


class TestImpliedProbability:
    def test_positive_odds(self) -> None:
        assert american_to_implied_prob(100) == pytest.approx(0.5, abs=1e-10)

    def test_negative_odds(self) -> None:
        assert american_to_implied_prob(-110) == pytest.approx(110.0 / 210.0, abs=1e-10)

    def test_roundtrip(self) -> None:
        dec = american_to_decimal(-110)
        prob = decimal_to_implied_prob(dec)
        prob_from_american = american_to_implied_prob(-110)
        assert prob == pytest.approx(prob_from_american, abs=1e-6)


class TestVig:
    def test_extract_vig(self) -> None:
        probs = [american_to_implied_prob(-110), american_to_implied_prob(-110)]
        vig = extract_vig(probs)
        assert vig == pytest.approx(0.0454, abs=0.01)

    def test_remove_vig(self) -> None:
        probs = [american_to_implied_prob(-110), american_to_implied_prob(-110)]
        vig_free = remove_vig(probs)
        assert np.sum(vig_free) == pytest.approx(1.0, abs=1e-10)

    def test_no_vig_returns_zero(self) -> None:
        probs = [0.5, 0.5]
        vig = extract_vig(probs)
        assert vig == pytest.approx(0.0, abs=1e-10)


class TestConvertOdds:
    def test_with_other_outcomes(self) -> None:
        result = convert_odds(150, other_outcomes=[-170])
        assert result.decimal == pytest.approx(2.5, abs=1e-10)
        assert result.true_prob is not None
        assert result.true_prob is not None
        # +150 with -170 other side: true prob ~0.39
        assert 0.35 < result.true_prob < 0.45

    def test_without_other_outcomes(self) -> None:
        result = convert_odds(-110)
        assert result.true_prob is None
