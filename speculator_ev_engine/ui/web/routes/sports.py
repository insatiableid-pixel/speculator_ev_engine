"""FastAPI routes for sports betting computations."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from speculator_ev_engine.sports.edge import compute_edge, compute_clv
from speculator_ev_engine.ui.core.formatters import fmt_ev, fmt_pct

router = APIRouter(prefix="/sports", tags=["sports"])


class EdgeRequest(BaseModel):
    model_prob: float = Field(..., gt=0, lt=1)
    odds_american: int = Field(...)
    other_outcomes: list[int] | None = None


class CLVRequest(BaseModel):
    open_odds: int = Field(...)
    close_odds: int = Field(...)
    other_open: list[int] | None = None
    other_close: list[int] | None = None


class EdgeResponse(BaseModel):
    model_prob: float
    market_prob: float
    edge: float
    ev_per_unit: float


class CLVResponse(BaseModel):
    clv: float
    open_implied: float
    close_implied: float


@router.post("/edge", response_model=EdgeResponse)
def sports_edge(req: EdgeRequest) -> EdgeResponse:
    r = compute_edge(req.model_prob, req.odds_american, req.other_outcomes)
    return EdgeResponse(
        model_prob=r.model_prob,
        market_prob=r.market_prob,
        edge=r.edge,
        ev_per_unit=r.ev_per_unit,
    )


@router.post("/clv", response_model=CLVResponse)
def sports_clv(req: CLVRequest) -> CLVResponse:
    r = compute_clv(req.open_odds, req.close_odds, req.other_open, req.other_close)
    return CLVResponse(
        clv=r.clv,
        open_implied=r.open_implied,
        close_implied=r.close_implied,
    )
