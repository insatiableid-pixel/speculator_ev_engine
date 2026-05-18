"""FastAPI routes for Kelly criterion computations."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

from speculator_ev_engine.core.kelly import (
    binary_kelly, fractional_kelly, multi_outcome_kelly, uncertain_edge_kelly,
)

router = APIRouter(prefix="/kelly", tags=["kelly"])


class BinaryKellyRequest(BaseModel):
    p: float = Field(..., gt=0, lt=1, description="Win probability")
    b: float = Field(..., gt=0, description="Net decimal odds")


class FractionalKellyRequest(BinaryKellyRequest):
    fraction: float = Field(..., gt=0, le=1, description="Fraction of full Kelly")


class MultiOutcomeKellyRequest(BaseModel):
    probabilities: list[float] = Field(..., description="Outcome probabilities (sum to 1)")
    payouts: list[float] = Field(..., description="Net payouts per outcome")


class UncertainEdgeKellyRequest(BaseModel):
    edge_mean: float = Field(..., description="Mean edge")
    edge_std: float = Field(..., ge=0, description="Edge standard deviation")
    odds: float = Field(..., gt=0, description="Net decimal odds")


class KellyResponse(BaseModel):
    fraction: float
    expected_log_growth: float
    ruin_probability: float


@router.post("/binary", response_model=KellyResponse)
def kelly_binary(req: BinaryKellyRequest) -> KellyResponse:
    r = binary_kelly(req.p, req.b)
    return KellyResponse(
        fraction=r.fraction,
        expected_log_growth=r.expected_log_growth,
        ruin_probability=r.ruin_probability,
    )


@router.post("/fractional", response_model=KellyResponse)
def kelly_fractional(req: FractionalKellyRequest) -> KellyResponse:
    r = fractional_kelly(req.p, req.b, fraction=req.fraction)
    return KellyResponse(
        fraction=r.fraction,
        expected_log_growth=r.expected_log_growth,
        ruin_probability=r.ruin_probability,
    )


@router.post("/multi-outcome", response_model=KellyResponse)
def kelly_multi_outcome(req: MultiOutcomeKellyRequest) -> KellyResponse:
    r = multi_outcome_kelly(np.array(req.probabilities), np.array(req.payouts))
    return KellyResponse(
        fraction=r.fraction,
        expected_log_growth=r.expected_log_growth,
        ruin_probability=r.ruin_probability,
    )


@router.post("/uncertain-edge", response_model=KellyResponse)
def kelly_uncertain_edge(req: UncertainEdgeKellyRequest) -> KellyResponse:
    r = uncertain_edge_kelly(req.edge_mean, req.edge_std, req.odds, seed=42)
    return KellyResponse(
        fraction=r.fraction,
        expected_log_growth=r.expected_log_growth,
        ruin_probability=r.ruin_probability,
    )
