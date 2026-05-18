"""Decision review CLI — worst EV decisions, best EV decisions, biggest divergences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .logger import Decision, DecisionLogger
from .leaks import (
    highest_ev_decisions,
    lowest_ev_decisions,
    largest_ev_outcome_divergences,
)


# TODO: Implement rich-formatted CLI output
# TODO: Implement interactive session review mode
# TODO: Implement export to CSV/JSON


@dataclass(frozen=True)
class ReviewReport:
    """Summary of a decision review session.

    Attributes:
        best_decisions: Top N highest-EV decisions.
        worst_decisions: Top N lowest-EV decisions.
        biggest_divergences: Top N decisions where outcome most diverged from EV.
        total_decisions_reviewed: Total number of decisions in sample.
        mean_ev: Mean estimated EV.
        mean_outcome: Mean actual outcome.
    """
    best_decisions: list[Decision]
    worst_decisions: list[Decision]
    biggest_divergences: list[tuple[Decision, float]]
    total_decisions_reviewed: int
    mean_ev: float
    mean_outcome: float


def review_decisions(
    decisions: Sequence[Decision],
    n: int = 10,
) -> ReviewReport:
    """Generate a review report of recent decisions.

    Args:
        decisions: Sequence of resolved decisions.
        n: Number of top/bottom decisions to surface.

    Returns:
        ReviewReport with decision analysis.
    """
    best = highest_ev_decisions(decisions, n)
    worst = lowest_ev_decisions(decisions, n)
    divergences = largest_ev_outcome_divergences(decisions, n)

    resolved = [d for d in decisions if d.outcome is not None]
    mean_ev = sum(d.ev_estimate for d in resolved) / max(len(resolved), 1)
    mean_outcome = sum(d.outcome or 0.0 for d in resolved) / max(len(resolved), 1)

    return ReviewReport(
        best_decisions=best,
        worst_decisions=worst,
        biggest_divergences=divergences,
        total_decisions_reviewed=len(resolved),
        mean_ev=mean_ev,
        mean_outcome=mean_outcome,
    )


def format_review(report: ReviewReport) -> str:
    """Format a ReviewReport as a readable string.

    Args:
        report: ReviewReport to format.

    Returns:
        Formatted string for terminal display.
    """
    lines: list[str] = []
    lines.append(f"=== Decision Review ({report.total_decisions_reviewed} decisions) ===")
    lines.append(f"Mean EV: {report.mean_ev:+.4f}  Mean Outcome: {report.mean_outcome:+.4f}")
    lines.append("")

    lines.append("--- Best EV Decisions ---")
    for d in report.best_decisions:
        lines.append(f"  {d.decision:40s}  EV={d.ev_estimate:+.4f}  outcome={d.outcome}")

    lines.append("")
    lines.append("--- Worst EV Decisions ---")
    for d in report.worst_decisions:
        lines.append(f"  {d.decision:40s}  EV={d.ev_estimate:+.4f}  outcome={d.outcome}")

    lines.append("")
    lines.append("--- Biggest EV/Outcome Divergences ---")
    for d, div in report.biggest_divergences:
        lines.append(f"  {d.decision:40s}  div={div:+.4f}  EV={d.ev_estimate:+.4f}")

    return "\n".join(lines)
