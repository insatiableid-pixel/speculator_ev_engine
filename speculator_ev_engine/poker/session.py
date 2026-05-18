"""Session logging, leak detection, EV reconstruction from hand histories."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

from ..decisions.logger import Decision, DecisionLogger
from ..decisions.leaks import detect_leaks, detect_tilt, LeakReport, TiltAlert


# TODO: Implement hand history parser (PokerStars, GG, WPN formats)
# TODO: Implement EV reconstruction from hand histories
# TODO: Implement session-level aggregation and reporting


@dataclass(frozen=True)
class SessionSummary:
    """Summary of a poker session.

    Attributes:
        session_id: Unique session identifier.
        start_time: Session start timestamp.
        end_time: Session end timestamp.
        n_hands: Number of hands played.
        total_ev: Sum of estimated EV across all decisions.
        total_outcome: Sum of actual outcomes.
        ev_realized: total_outcome / total_ev (if total_ev > 0).
        leaks_detected: Leak reports for this session.
        tilt_alerts: Tilt alerts triggered during session.
    """
    session_id: str
    start_time: str
    end_time: str
    n_hands: int
    total_ev: float
    total_outcome: float
    ev_realized: float
    leaks_detected: list[LeakReport]
    tilt_alerts: list[TiltAlert]


class SessionLogger:
    """Session-scoped poker decision logger.

    Extends DecisionLogger with poker-specific tracking: hand ID,
    position, street, pot size, and stack depth.

    TODO: Implement full hand history parsing and EV reconstruction.
    """

    def __init__(self, decision_logger: DecisionLogger | None = None) -> None:
        self._logger = decision_logger or DecisionLogger()
        self._session_id = datetime.utcnow().isoformat()
        self._decisions: list[Decision] = []

    def log_hand(
        self,
        hand_id: str,
        p_estimate: float,
        ev_estimate: float,
        stake: float,
        position: str = "",
        street: str = "",
        pot_size: float = 0.0,
        stack_depth: float = 0.0,
        outcome: float | None = None,
        tags: dict[str, str] | None = None,
    ) -> int:
        """Log a poker hand decision.

        Args:
            hand_id: Hand history identifier.
            p_estimate: Estimated equity/probability.
            ev_estimate: Estimated EV of the decision.
            stake: Amount wagered.
            position: Table position (e.g. "BTN", "BB").
            street: Decision street ("preflop", "flop", "turn", "river").
            pot_size: Pot size at decision point.
            stack_depth: Effective stack depth.
            outcome: Actual outcome (None if unresolved).
            tags: Additional tags.

        Returns:
            Row ID in decision database.
        """
        all_tags = {"hand_id": hand_id, "position": position, "street": street}
        if tags:
            all_tags.update(tags)

        decision = Decision(
            decision=f"hand:{hand_id}:{street}",
            p_estimate=p_estimate,
            ev_estimate=ev_estimate,
            stake=stake,
            outcome=outcome,
            domain="poker",
            tags=all_tags,
            notes=f"pot={pot_size}, stack={stack_depth}, pos={position}",
        )
        self._decisions.append(decision)
        return self._logger.log(decision)

    def session_summary(self) -> SessionSummary:
        """Generate a session summary with leak and tilt analysis.

        Returns:
            SessionSummary with all session-level metrics.
        """
        leaks = detect_leaks(self._decisions, group_by="tag:street")
        tilts = detect_tilt(self._decisions)
        total_ev = sum(d.ev_estimate for d in self._decisions)
        total_outcome = sum(d.outcome or 0.0 for d in self._decisions)
        ev_realized = total_outcome / total_ev if abs(total_ev) > 1e-9 else 0.0

        return SessionSummary(
            session_id=self._session_id,
            start_time=self._decisions[0].timestamp if self._decisions else "",
            end_time=self._decisions[-1].timestamp if self._decisions else "",
            n_hands=len(self._decisions),
            total_ev=total_ev,
            total_outcome=total_outcome,
            ev_realized=ev_realized,
            leaks_detected=leaks,
            tilt_alerts=tilts,
        )
