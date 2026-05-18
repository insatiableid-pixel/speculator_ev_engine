"""Tests for decisions.logger module."""

import tempfile
from pathlib import Path

import pytest

from speculator_ev_engine.decisions.logger import Decision, DecisionLogger


class TestDecision:
    def test_valid_decision(self) -> None:
        d = Decision(
            decision="bet NFL spread",
            p_estimate=0.55,
            ev_estimate=0.1,
            stake=100.0,
            domain="sports",
        )
        assert d.p_estimate == 0.55

    def test_invalid_p_estimate(self) -> None:
        with pytest.raises(ValueError, match="p_estimate must be in"):
            Decision(decision="bad", p_estimate=1.5, ev_estimate=0.0, stake=100.0)

    def test_invalid_domain(self) -> None:
        with pytest.raises(ValueError, match="domain must be"):
            Decision(decision="bad", p_estimate=0.5, ev_estimate=0.0, stake=100.0, domain="crypto")

    def test_negative_stake(self) -> None:
        with pytest.raises(ValueError, match="stake must be non-negative"):
            Decision(decision="bad", p_estimate=0.5, ev_estimate=0.0, stake=-10.0)


class TestDecisionLogger:
    def test_log_and_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = DecisionLogger(db_path=Path(tmp) / "test.db")
            d = Decision(decision="test", p_estimate=0.6, ev_estimate=0.2, stake=50.0, domain="poker")
            row_id = logger.log(d)
            assert row_id > 0

            results = logger.query(domain="poker")
            assert len(results) == 1
            assert results[0].decision == "test"
            logger.close()

    def test_resolve(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = DecisionLogger(db_path=Path(tmp) / "test.db")
            d = Decision(decision="test", p_estimate=0.6, ev_estimate=0.2, stake=50.0)
            row_id = logger.log(d)
            logger.resolve(row_id, 1.0)

            results = logger.query(resolved_only=True)
            assert len(results) == 1
            assert results[0].outcome == 1.0
            logger.close()

    def test_tag_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = DecisionLogger(db_path=Path(tmp) / "test.db")
            d1 = Decision(decision="a", p_estimate=0.5, ev_estimate=0.0, stake=10.0, tags={"sport": "nfl"})
            d2 = Decision(decision="b", p_estimate=0.5, ev_estimate=0.0, stake=10.0, tags={"sport": "nba"})
            logger.log(d1)
            logger.log(d2)

            results = logger.query(tag_filter={"sport": "nfl"})
            assert len(results) == 1
            assert results[0].decision == "a"
            logger.close()

    def test_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = DecisionLogger(db_path=Path(tmp) / "test.db")
            logger.log(Decision(decision="a", p_estimate=0.5, ev_estimate=0.0, stake=10.0, domain="poker"))
            logger.log(Decision(decision="b", p_estimate=0.5, ev_estimate=0.0, stake=10.0, domain="sports"))

            assert logger.count() == 2
            assert logger.count(domain="poker") == 1
            logger.close()

    def test_context_manager(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with DecisionLogger(db_path=Path(tmp) / "test.db") as logger:
                logger.log(Decision(decision="x", p_estimate=0.5, ev_estimate=0.0, stake=10.0))
            # Connection should be closed
