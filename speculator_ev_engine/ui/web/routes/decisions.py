"""FastAPI routes for decision log, leak analysis, and calibration."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

from speculator_ev_engine.decisions.logger import Decision, DecisionLogger
from speculator_ev_engine.decisions.leaks import detect_leaks, detect_tilt
from speculator_ev_engine.decisions.calibration import full_calibration

router = APIRouter(prefix="/decisions", tags=["decisions"])

# Module-level logger using a temp DB per process lifetime
_db_path = Path(tempfile.mkdtemp()) / "decisions.db"
_logger = DecisionLogger(db_path=_db_path)


class DecisionInput(BaseModel):
    decision: str
    p_estimate: float = Field(..., ge=0, le=1)
    ev_estimate: float
    stake: float = Field(..., ge=0)
    domain: str = "general"
    tags: dict[str, str] = Field(default_factory=dict)
    notes: str = ""


class ResolveInput(BaseModel):
    row_id: int
    outcome: float


class DecisionResponse(BaseModel):
    row_id: int


class LeakReportResponse(BaseModel):
    dimension: str
    group: str
    n_decisions: int
    mean_ev: float
    mean_outcome: float
    ev_outcome_gap: float
    stake_inflation: bool
    edge_threshold_drop: bool


class CalibrationResponse(BaseModel):
    brier_score: float
    ece: float
    mce: float


@router.post("/log", response_model=DecisionResponse)
def log_decision(inp: DecisionInput) -> DecisionResponse:
    d = Decision(
        decision=inp.decision,
        p_estimate=inp.p_estimate,
        ev_estimate=inp.ev_estimate,
        stake=inp.stake,
        domain=inp.domain,
        tags=inp.tags,
        notes=inp.notes,
    )
    row_id = _logger.log(d)
    return DecisionResponse(row_id=row_id)


@router.post("/resolve")
def resolve_decision(inp: ResolveInput) -> dict[str, str]:
    _logger.resolve(inp.row_id, inp.outcome)
    return {"status": "resolved"}


@router.get("/leaks", response_model=list[LeakReportResponse])
def get_leaks() -> list[LeakReportResponse]:
    decisions = _logger.query(resolved_only=True, limit=2000)
    if not decisions:
        return []
    reports = detect_leaks(decisions)
    return [
        LeakReportResponse(
            dimension=r.dimension, group=r.group, n_decisions=r.n_decisions,
            mean_ev=r.mean_ev, mean_outcome=r.mean_outcome,
            ev_outcome_gap=r.ev_outcome_gap,
            stake_inflation=r.stake_inflation,
            edge_threshold_drop=r.edge_threshold_drop,
        )
        for r in reports
    ]


@router.get("/calibration", response_model=CalibrationResponse)
def get_calibration() -> CalibrationResponse:
    decisions = _logger.query(resolved_only=True, limit=2000)
    if not decisions:
        return CalibrationResponse(brier_score=0, ece=0, mce=0)
    forecasts = np.array([d.p_estimate for d in decisions])
    outcomes = np.array([(d.outcome or 0) > 0 for d in decisions], dtype=float)
    cal = full_calibration(forecasts, outcomes)
    return CalibrationResponse(brier_score=cal.brier_score, ece=cal.ece, mce=cal.mce)
