"""Pluggable data source abstraction for odds APIs, hand history parsers, market data feeds."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generator

import pandas as pd


# TODO: Implement odds API adapters (Pinnacle, TheOdds, etc.)
# TODO: Implement hand history parsers (PokerStars, GG, WPN)
# TODO: Implement market data feed adapters (yfinance, polygon, etc.)
# TODO: Implement async streaming for live data


@dataclass(frozen=True)
class DataFeedConfig:
    """Configuration for a data feed connection.

    Attributes:
        source: Feed identifier (e.g. "pinnacle", "pokerstars", "polygon").
        api_key: API key if required.
        base_url: API base URL.
        rate_limit: Maximum requests per second.
        params: Additional connection parameters.
    """
    source: str
    api_key: str | None = None
    base_url: str | None = None
    rate_limit: float = 1.0
    params: dict[str, Any] | None = None


class BaseDataFeed(ABC):
    """Abstract base class for all data feeds.

    Every feed must implement fetch (batch) and optionally stream (live).
    """

    def __init__(self, config: DataFeedConfig) -> None:
        self.config = config

    @abstractmethod
    def fetch(self, query: dict[str, Any]) -> pd.DataFrame:
        """Fetch data for a given query.

        Args:
            query: Domain-specific query parameters.

        Returns:
            DataFrame with standardized columns.
        """
        ...

    def stream(self, query: dict[str, Any]) -> Generator[dict[str, Any], None, None]:
        """Stream live data updates. Override for live feeds.

        Args:
            query: Stream filter parameters.

        Yields:
            Dict records as they arrive.
        """
        raise NotImplementedError(f"{self.config.source} does not support streaming")

    @abstractmethod
    def validate(self, data: pd.DataFrame) -> bool:
        """Validate fetched data against expected schema.

        Args:
            data: DataFrame to validate.

        Returns:
            True if data passes validation.
        """
        ...
