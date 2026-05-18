"""Market movement analysis: sharp vs. public money signals."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.typing import NDArray


# TODO: Implement line movement classification (sharp vs. public)
# TODO: Implement reverse line movement detection
# TODO: Implement steam move detection
# TODO: Implement book-specific movement profiles


@dataclass(frozen=True)
class LineMovement:
    """A single line movement event.

    Attributes:
        book: Sportsbook identifier.
        market: Market identifier.
        open_odds: Opening American odds.
        close_odds: Closing American odds.
        timestamp: When the movement occurred.
        volume: Betting volume (if available).
        is_sharp: Whether the movement is classified as sharp-driven.
    """
    book: str
    market: str
    open_odds: int
    close_odds: int
    timestamp: str = ""
    volume: float | None = None
    is_sharp: bool | None = None


def classify_movement(
    open_odds: int,
    close_odds: int,
    volume_ratio: float | None = None,
    time_to_close_hours: float | None = None,
) -> str:
    """Classify a line movement as sharp-driven, public-driven, or ambiguous.

    Heuristics:
    - Movements early (high time_to_close) are more likely sharp
    - Movements with high volume relative to normal are more likely sharp
    - Reverse line movements (against the public side) are sharp

    Args:
        open_odds: Opening American odds.
        close_odds: Closing American odds.
        volume_ratio: Ratio of current volume to average volume.
        time_to_close_hours: Hours until market close when movement occurred.

    Returns:
        "sharp", "public", or "ambiguous".
    """
    if open_odds == close_odds:
        return "ambiguous"

    # Reverse line movement heuristic: favorite gets more public action
    # but line moves the other way
    moved_toward_underdog = (
        (open_odds < 0 and close_odds > open_odds) or  # favorite got longer
        (open_odds > 0 and close_odds < open_odds)     # underdog got shorter
    )

    if moved_toward_underdog and volume_ratio is not None and volume_ratio > 1.5:
        return "sharp"

    if time_to_close_hours is not None:
        if time_to_close_hours > 24 and moved_toward_underdog:
            return "sharp"
        if time_to_close_hours < 2:
            return "public"

    return "ambiguous"


def detect_steam(
    movements: list[LineMovement],
    movement_threshold: int = 15,
    window_minutes: int = 30,
) -> list[LineMovement]:
    """Detect steam moves — rapid, coordinated line movements across multiple books.

    Args:
        movements: List of line movements across books.
        movement_threshold: Minimum odds movement (in cents) to qualify.
        window_minutes: Time window for coordinated movements.

    Returns:
        List of LineMovement events classified as steam.

    TODO: Implement full steam detection with multi-book coordination.
    """
    # Stub — requires timestamped multi-book data
    return []
