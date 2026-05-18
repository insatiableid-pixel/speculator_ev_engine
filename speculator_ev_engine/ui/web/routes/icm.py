"""FastAPI routes for ICM computations."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

from speculator_ev_engine.poker.icm import malmuth_harville_icm, chip_chop_icm, bubble_factor

router = APIRouter(prefix="/icm", tags=["icm"])


class ICMRequest(BaseModel):
    stacks: list[float] = Field(..., min_length=2, description="Chip stacks")
    payouts: list[float] = Field(..., description="Payout by position")
    blend: float = Field(1.0, ge=0, le=1, description="ICM/chip chop blend")


class BubbleFactorRequest(BaseModel):
    stacks: list[float] = Field(..., min_length=2)
    payouts: list[float] = Field(...)
    player_index: int = Field(..., ge=0)


class ICMResponse(BaseModel):
    equities: list[float]
    total_prize_pool: float


class BubbleFactorResponse(BaseModel):
    bubble_factor: float
    equity_lost: float
    equity_gained: float


@router.post("/equity", response_model=ICMResponse)
def icm_equity(req: ICMRequest) -> ICMResponse:
    stacks = np.array(req.stacks)
    payouts = np.array(req.payouts)
    if req.blend >= 1.0:
        result = malmuth_harville_icm(stacks, payouts)
    else:
        result = chip_chop_icm(stacks, payouts, blend_weight=req.blend)
    return ICMResponse(
        equities=result.equities.tolist(),
        total_prize_pool=result.total_prize_pool,
    )


@router.post("/bubble-factor", response_model=BubbleFactorResponse)
def icm_bubble_factor(req: BubbleFactorRequest) -> BubbleFactorResponse:
    stacks = np.array(req.stacks)
    payouts = np.array(req.payouts)
    result = bubble_factor(stacks, payouts, req.player_index)
    return BubbleFactorResponse(
        bubble_factor=result.bubble_factor,
        equity_lost=result.equity_lost,
        equity_gained=result.equity_gained,
    )
