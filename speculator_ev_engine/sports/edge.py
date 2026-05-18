"""Edge calculation, CLV tracking, closing line value as ground truth."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .odds import american_to_implied_prob, remove_vig


@dataclass(frozen=True)
class EdgeResult:
    """Edge calculation for a single bet.

    Attributes:
        model_prob: Model-estimated true probability.
        market_prob: Market implied probability (vig-free).
        edge: model_prob - market_prob.
        ev_per_unit: Expected value per unit wagered.
        odds_american: Original American odds.
    """
    model_prob: float
    market_prob: float
    edge: float
    ev_per_unit: float
    odds_american: int


@dataclass(frozen=True)
class CLVResult:
    """Closing line value result.

    Attributes:
        clv: CLV percentage (positive = you beat the close).
        open_implied: Implied probability at open (vig-free).
        close_implied: Implied probability at close (vig-free).
        open_odds: American odds at open.
        close_odds: American odds at close.
    """
    clv: float
    open_implied: float
    close_implied: float
    open_odds: int
    close_odds: int


@dataclass
class CLVTracker:
    """Tracks CLV patterns across bets for diagnosing edge quality.

    Aggregates CLV data by sport, book, market type, and bet size tier
    to surface consistent patterns of CLV success or failure.

    Attributes:
        records: List of CLV tracking records.
    """
    records: list[dict[str, object]] = field(default_factory=list)

    def add_record(
        self,
        sport: str,
        book: str,
        market_type: str,
        bet_size_tier: str,
        open_odds: int,
        close_odds: int,
        other_open_outcomes: list[int] | None = None,
        other_close_outcomes: list[int] | None = None,
        timestamp: datetime | None = None,
    ) -> CLVResult:
        """Add a CLV record and return the CLV result.

        Args:
            sport: Sport identifier (e.g. "nfl", "nba").
            book: Sportsbook identifier.
            market_type: Market type (e.g. "spread", "moneyline", "total").
            bet_size_tier: Tier label (e.g. "small", "medium", "large").
            open_odds: American odds at open.
            close_odds: American odds at close.
            other_open_outcomes: Odds for other outcomes at open (for vig removal).
            other_close_outcomes: Odds for other outcomes at close.
            timestamp: When the bet was placed.

        Returns:
            CLVResult for this record.
        """
        clv_result = compute_clv(open_odds, close_odds,
                                 other_open_outcomes, other_close_outcomes)

        record: dict[str, object] = {
            "sport": sport,
            "book": book,
            "market_type": market_type,
            "bet_size_tier": bet_size_tier,
            "clv": clv_result.clv,
            "open_implied": clv_result.open_implied,
            "close_implied": clv_result.close_implied,
            "open_odds": open_odds,
            "close_odds": close_odds,
            "timestamp": timestamp or datetime.now(),
        }
        self.records.append(record)
        return clv_result

    def summary_by(self, group_key: str = "sport") -> dict[str, dict[str, float]]:
        """Summarize CLV statistics grouped by a field.

        Args:
            group_key: One of "sport", "book", "market_type", "bet_size_tier".

        Returns:
            Dict mapping group value to {"mean_clv", "n_bets", "positive_clv_pct"}.
        """
        if group_key not in {"sport", "book", "market_type", "bet_size_tier"}:
            raise ValueError(
                f"group_key must be sport/book/market_type/bet_size_tier, got {group_key}"
            )

        groups: dict[str, list[float]] = {}
        for rec in self.records:
            key = str(rec.get(group_key, "unknown"))
            groups.setdefault(key, []).append(float(rec["clv"]))

        result: dict[str, dict[str, float]] = {}
        for key, clvs in groups.items():
            arr = np.array(clvs)
            result[key] = {
                "mean_clv": float(np.mean(arr)),
                "n_bets": float(len(arr)),
                "positive_clv_pct": float(np.mean(arr > 0)),
            }
        return result

    def flag_patterns(self, min_samples: int = 20) -> list[dict[str, object]]:
        """Flag groups with consistent positive or negative CLV patterns.

        Args:
            min_samples: Minimum number of bets in a group to flag.

        Returns:
            List of dicts with group info and flag direction.
        """
        flags: list[dict[str, object]] = []
        for group_key in ("sport", "book", "market_type", "bet_size_tier"):
            summary = self.summary_by(group_key)
            for name, stats in summary.items():
                if stats["n_bets"] < min_samples:
                    continue
                mean_clv = stats["mean_clv"]
                if mean_clv > 0.02:
                    flags.append({
                        "group": group_key,
                        "value": name,
                        "direction": "positive",
                        "mean_clv": mean_clv,
                        "n_bets": int(stats["n_bets"]),
                    })
                elif mean_clv < -0.02:
                    flags.append({
                        "group": group_key,
                        "value": name,
                        "direction": "negative",
                        "mean_clv": mean_clv,
                        "n_bets": int(stats["n_bets"]),
                    })
        return flags


def compute_edge(model_prob: float, odds_american: int,
                 other_outcomes: list[int] | None = None) -> EdgeResult:
    """Compute edge from model probability vs. market odds.

    Args:
        model_prob: Model-estimated probability of the outcome.
        odds_american: American odds for the bet.
        other_outcomes: American odds for other outcomes (for vig removal).

    Returns:
        EdgeResult with edge, EV per unit, and probabilities.
    """
    if not (0.0 < model_prob < 1.0):
        raise ValueError(f"model_prob must be in (0,1), got {model_prob}")

    implied_raw = american_to_implied_prob(odds_american)

    if other_outcomes is not None:
        all_probs = [implied_raw] + [american_to_implied_prob(o) for o in other_outcomes]
        market_prob = float(remove_vig(all_probs)[0])
    else:
        market_prob = implied_raw

    edge = model_prob - market_prob
    ev_per_unit = model_prob * (american_to_decimal_payout(odds_american) - 1.0) - (1.0 - model_prob)

    return EdgeResult(
        model_prob=model_prob,
        market_prob=market_prob,
        edge=edge,
        ev_per_unit=ev_per_unit,
        odds_american=odds_american,
    )


def compute_clv(
    open_odds: int,
    close_odds: int,
    other_open_outcomes: list[int] | None = None,
    other_close_outcomes: list[int] | None = None,
) -> CLVResult:
    """Compute closing line value.

    CLV = open_implied_prob - close_implied_prob.
    Positive CLV means you got a better number than the close —
    the market moved in your direction.

    Args:
        open_odds: American odds at the time of bet placement.
        close_odds: American odds at market close.
        other_open_outcomes: Other outcomes' odds at open.
        other_close_outcomes: Other outcomes' odds at close.

    Returns:
        CLVResult.
    """
    open_implied_raw = american_to_implied_prob(open_odds)
    close_implied_raw = american_to_implied_prob(close_odds)

    if other_open_outcomes is not None:
        all_open = [open_implied_raw] + [american_to_implied_prob(o) for o in other_open_outcomes]
        open_implied = float(remove_vig(all_open)[0])
    else:
        open_implied = open_implied_raw

    if other_close_outcomes is not None:
        all_close = [close_implied_raw] + [american_to_implied_prob(o) for o in other_close_outcomes]
        close_implied = float(remove_vig(all_close)[0])
    else:
        close_implied = close_implied_raw

    clv = close_implied - open_implied

    return CLVResult(
        clv=clv,
        open_implied=open_implied,
        close_implied=close_implied,
        open_odds=open_odds,
        close_odds=close_odds,
    )


def expected_clv(
    open_odds: int,
    line_movement_std: float,
    other_open_outcomes: list[int] | None = None,
    n_simulations: int = 10_000,
) -> float:
    """Expected CLV given a model of line movement.

    Simulates closing lines drawn from a normal distribution centered on
    the open line and computes average CLV.

    Args:
        open_odds: American odds at open.
        line_movement_std: Standard deviation of line movement in cents.
        other_open_outcomes: Other outcomes' odds at open.
        n_simulations: Number of Monte Carlo draws.

    Returns:
        Expected CLV.
    """
    open_implied_raw = american_to_implied_prob(open_odds)

    if other_open_outcomes is not None:
        all_open = [open_implied_raw] + [american_to_implied_prob(o) for o in other_open_outcomes]
        open_implied = float(remove_vig(all_open)[0])
    else:
        open_implied = open_implied_raw

    # Convert line_movement_std from cents to probability movement
    # Approximate: 1 cent ≈ 0.0025 probability change for near-even odds
    prob_std = line_movement_std * 0.0025

    close_implied_samples = np.random.normal(open_implied, prob_std, n_simulations)
    close_implied_samples = np.clip(close_implied_samples, 0.01, 0.99)

    clv_samples = open_implied - close_implied_samples
    return float(np.mean(clv_samples))


def roi_vs_ev_reconciliation(
    model_probs: NDArray[np.float64],
    odds_american: NDArray[np.int64],
    outcomes: NDArray[np.float64],
) -> dict[str, float]:
    """Reconcile actual ROI vs. expected EV over a sample of bets.

    This is the critical sanity check: over large samples, ROI should
    converge to average EV if your model is well-calibrated.

    Args:
        model_probs: Array of model-estimated probabilities.
        odds_american: Array of American odds (same length).
        outcomes: Array of outcomes (1.0 = win, 0.0 = loss).

    Returns:
        Dict with 'actual_roi', 'expected_ev_per_unit', 'divergence', 'n_bets'.
    """
    model_probs = np.asarray(model_probs, dtype=np.float64)
    odds_american = np.asarray(odds_american, dtype=np.int64)
    outcomes = np.asarray(outcomes, dtype=np.float64)

    if not (model_probs.shape == odds_american.shape == outcomes.shape):
        raise ValueError("All arrays must have the same shape")

    # Compute per-bet EV
    evs = np.array([
        p * (american_to_decimal_payout(int(o)) - 1.0) - (1.0 - p)
        for p, o in zip(model_probs, odds_american)
    ])

    # Compute actual returns
    returns = np.array([
        (american_to_decimal_payout(int(o)) - 1.0) if outcome == 1.0 else -1.0
        for o, outcome in zip(odds_american, outcomes)
    ])

    return {
        "actual_roi": float(np.mean(returns)),
        "expected_ev_per_unit": float(np.mean(evs)),
        "divergence": float(np.mean(returns) - np.mean(evs)),
        "n_bets": float(len(outcomes)),
    }


def american_to_decimal_payout(odds: int) -> float:
    """Convert American odds to decimal payout per unit wagered.

    Args:
        odds: American odds.

    Returns:
        Decimal payout (e.g. +100 → 2.0, -110 → 1.909).
    """
    if odds >= 100:
        return odds / 100.0 + 1.0
    elif odds <= -100:
        return 100.0 / abs(odds) + 1.0
    else:
        raise ValueError(f"American odds must be >= 100 or <= -100, got {odds}")
