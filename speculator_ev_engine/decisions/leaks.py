"""Leak analysis engine — pattern detection across logged decisions."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .logger import Decision, DecisionLogger


@dataclass(frozen=True)
class LeakReport:
    """Summary of detected leaks in a decision sample.

    Attributes:
        dimension: The grouping dimension (e.g. "domain", "tag:sport", "stake_tier").
        group: The specific group value.
        n_decisions: Number of decisions in this group.
        mean_ev: Mean estimated EV of decisions.
        mean_outcome: Mean actual outcome.
        ev_outcome_gap: Mean EV minus mean outcome (positive = EV not realized).
        stake_inflation: Flagged if recent losses correlate with higher stakes.
        edge_threshold_drop: Flagged if recent losses correlate with lower edge thresholds.
    """
    dimension: str
    group: str
    n_decisions: int
    mean_ev: float
    mean_outcome: float
    ev_outcome_gap: float
    stake_inflation: bool
    edge_threshold_drop: bool


@dataclass(frozen=True)
class TiltAlert:
    """A tilt detection alert.

    Attributes:
        start_index: Index in the decision sequence where tilt begins.
        severity: 0-1 scale (1 = severe tilt).
        pattern: Description of the tilt pattern detected.
    """
    start_index: int
    severity: float
    pattern: str


def detect_leaks(
    decisions: Sequence[Decision],
    group_by: str = "domain",
    user_tag: str | None = None,
    stake_tier_boundaries: tuple[float, ...] = (0.01, 0.05, 0.10),
) -> list[LeakReport]:
    """Detect EV leaks across a decision sample.

    Groups decisions by a dimension and surfaces where EV is systematically
    not being realized (ev_outcome_gap), or where tilt patterns emerge.

    Args:
        decisions: Sequence of resolved decisions.
        group_by: "domain", "tag:<key>", or "stake_tier".
        user_tag: Tag key to group by if group_by starts with "tag:".
        stake_tier_boundaries: Boundaries for stake tiers (as fractions of bankroll).

    Returns:
        List of LeakReport objects sorted by ev_outcome_gap descending.
    """
    resolved = [d for d in decisions if d.outcome is not None]
    if not resolved:
        return []

    groups: dict[str, list[Decision]] = defaultdict(list)

    for d in resolved:
        if group_by == "domain":
            key = d.domain
        elif group_by.startswith("tag:"):
            tag_key = group_by[4:] if user_tag is None else user_tag
            key = d.tags.get(tag_key, "untagged")
        elif group_by == "stake_tier":
            key = _stake_tier(d.stake, stake_tier_boundaries)
        else:
            key = "all"
        groups[key].append(d)

    reports: list[LeakReport] = []
    for group, group_decisions in groups.items():
        n = len(group_decisions)
        if n < 3:
            continue
        evs = np.array([d.ev_estimate for d in group_decisions])
        outcomes = np.array([d.outcome for d in group_decisions])
        stakes = np.array([d.stake for d in group_decisions])
        edges = np.array([d.p_estimate for d in group_decisions])

        mean_ev = float(np.mean(evs))
        mean_outcome = float(np.mean(outcomes))
        ev_outcome_gap = mean_ev - mean_outcome

        # Tilt detection: stake inflation after losses
        stake_inflation = _detect_stake_inflation(stakes, outcomes)
        edge_drop = _detect_edge_threshold_drop(edges, outcomes)

        reports.append(LeakReport(
            dimension=group_by,
            group=group,
            n_decisions=n,
            mean_ev=mean_ev,
            mean_outcome=mean_outcome,
            ev_outcome_gap=ev_outcome_gap,
            stake_inflation=stake_inflation,
            edge_threshold_drop=edge_drop,
        ))

    return sorted(reports, key=lambda r: r.ev_outcome_gap, reverse=True)


def highest_ev_decisions(
    decisions: Sequence[Decision],
    n: int = 10,
) -> list[Decision]:
    """Return the N highest estimated-EV decisions.

    Args:
        decisions: Sequence of decisions.
        n: Number to return.

    Returns:
        List of Decision objects sorted by ev_estimate descending.
    """
    return sorted(decisions, key=lambda d: d.ev_estimate, reverse=True)[:n]


def lowest_ev_decisions(
    decisions: Sequence[Decision],
    n: int = 10,
) -> list[Decision]:
    """Return the N lowest estimated-EV decisions.

    Args:
        decisions: Sequence of decisions.
        n: Number to return.

    Returns:
        List of Decision objects sorted by ev_estimate ascending.
    """
    return sorted(decisions, key=lambda d: d.ev_estimate)[:n]


def largest_ev_outcome_divergences(
    decisions: Sequence[Decision],
    n: int = 10,
) -> list[tuple[Decision, float]]:
    """Surface decisions where outcome variance is masking EV leaks.

    Finds decisions with the largest gap between estimated EV and actual
    outcome — these are the spots where variance is loudest and the user
    is most likely to question good decisions or celebrate bad ones.

    Args:
        decisions: Sequence of resolved decisions.
        n: Number to return.

    Returns:
        List of (Decision, divergence) tuples sorted by |divergence| descending.
    """
    resolved = [d for d in decisions if d.outcome is not None]
    divergences = [(d, abs(d.ev_estimate - d.outcome)) for d in resolved]
    return sorted(divergences, key=lambda x: x[1], reverse=True)[:n]


def detect_tilt(
    decisions: Sequence[Decision],
    window_size: int = 10,
    stake_increase_threshold: float = 1.5,
    edge_drop_threshold: float = 0.05,
) -> list[TiltAlert]:
    """Detect tilt patterns in a decision sequence.

    Flags sequences where recent losses correlate with:
    - Stake size increases (chasing losses)
    - Edge threshold drops (taking worse spots)

    Args:
        decisions: Sequence of resolved decisions, chronologically ordered.
        window_size: Sliding window size for tilt detection.
        stake_increase_threshold: Ratio of post-loss stakes to baseline that triggers.
        edge_drop_threshold: Absolute drop in mean edge that triggers.

    Returns:
        List of TiltAlert objects.
    """
    resolved = [d for d in decisions if d.outcome is not None]
    if len(resolved) < window_size * 2:
        return []

    alerts: list[TiltAlert] = []
    stakes = np.array([d.stake for d in resolved])
    edges = np.array([d.p_estimate for d in resolved])
    outcomes = np.array([d.outcome for d in resolved])

    for i in range(window_size, len(resolved)):
        # Check if recent window has more losses than expected
        recent = outcomes[i - window_size : i]
        recent_stakes = stakes[i - window_size : i]
        recent_edges = edges[i - window_size : i]

        # Baseline from earlier decisions
        baseline_stakes = stakes[max(0, i - 2 * window_size) : i - window_size]
        baseline_edges = edges[max(0, i - 2 * window_size) : i - window_size]

        if len(baseline_stakes) == 0:
            continue

        mean_baseline_stake = float(np.mean(baseline_stakes))
        mean_baseline_edge = float(np.mean(baseline_edges))
        mean_recent_stake = float(np.mean(recent_stakes))
        mean_recent_edge = float(np.mean(recent_edges))

        loss_fraction = float(np.mean(recent < 0))

        stake_inflated = (
            mean_baseline_stake > 0
            and mean_recent_stake > mean_baseline_stake * stake_increase_threshold
        )
        edge_dropped = mean_recent_edge < mean_baseline_edge - edge_drop_threshold

        if loss_fraction > 0.5 and (stake_inflated or edge_dropped):
            severity = 0.0
            patterns: list[str] = []
            if stake_inflated:
                severity += 0.5
                patterns.append(
                    f"stake inflation: {mean_recent_stake:.2f} vs baseline {mean_baseline_stake:.2f}"
                )
            if edge_dropped:
                severity += 0.5
                patterns.append(
                    f"edge threshold drop: {mean_recent_edge:.3f} vs baseline {mean_baseline_edge:.3f}"
                )

            alerts.append(TiltAlert(
                start_index=i - window_size,
                severity=min(severity, 1.0),
                pattern="; ".join(patterns),
            ))

    return alerts


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _stake_tier(
    stake: float,
    boundaries: tuple[float, ...],
) -> str:
    """Classify stake into a tier."""
    for i, b in enumerate(boundaries):
        if stake < b:
            return f"tier_{i}"
    return f"tier_{len(boundaries)}"


def _detect_stake_inflation(
    stakes: NDArray[np.float64],
    outcomes: NDArray[np.float64],
) -> bool:
    """Flag if losses correlate with subsequent stake increases."""
    if len(stakes) < 4:
        return False
    # Correlation between loss indicator and next stake
    losses = (outcomes[:-1] < 0).astype(float)
    next_stakes = stakes[1:]
    if np.std(losses) == 0 or np.std(next_stakes) == 0:
        return False
    corr = float(np.corrcoef(losses, next_stakes)[0, 1])
    return corr > 0.3


def _detect_edge_threshold_drop(
    edges: NDArray[np.float64],
    outcomes: NDArray[np.float64],
) -> bool:
    """Flag if losses correlate with taking worse edges."""
    if len(edges) < 4:
        return False
    losses = (outcomes[:-1] < 0).astype(float)
    next_edges = edges[1:]
    if np.std(losses) == 0 or np.std(next_edges) == 0:
        return False
    corr = float(np.corrcoef(losses, next_edges)[0, 1])
    return corr < -0.3
