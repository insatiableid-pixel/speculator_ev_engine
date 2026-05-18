"""EV primitives and decision tree engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

T = TypeVar("T")


class DecisionOutcome(Enum):
    """Possible outcomes for a binary decision node."""
    WIN = "win"
    LOSS = "loss"
    PUSH = "push"


@dataclass(frozen=True)
class EVResult:
    """Result of an EV calculation.

    Attributes:
        ev: Expected value in absolute units.
        p_win: Probability of positive outcome.
        p_loss: Probability of negative outcome.
        payout_win: Payout on win (positive).
        payout_loss: Payout on loss (negative).
        variance: Variance of the outcome distribution.
    """
    ev: float
    p_win: float
    p_loss: float
    payout_win: float
    payout_loss: float
    variance: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.ev):
            raise ValueError(f"EV must be finite, got {self.ev}")
        if not (0.0 <= self.p_win <= 1.0):
            raise ValueError(f"p_win must be in [0,1], got {self.p_win}")
        if not (0.0 <= self.p_loss <= 1.0):
            raise ValueError(f"p_loss must be in [0,1], got {self.p_loss}")
        if abs(self.p_win + self.p_loss - 1.0) > 1e-9 and self.p_win + self.p_loss > 1.0:
            raise ValueError(
                f"p_win + p_loss must be <= 1.0, got {self.p_win + self.p_loss}"
            )


@dataclass(frozen=True)
class MultiOutcomeEV:
    """EV result for a decision with arbitrarily many discrete outcomes.

    Attributes:
        outcomes: Mapping of outcome name to (probability, payout).
        ev: Computed expected value.
        variance: Computed variance.
    """
    outcomes: dict[str, tuple[float, float]] = field(repr=False)
    ev: float = 0.0
    variance: float = 0.0

    def __post_init__(self) -> None:
        total_p = sum(p for p, _ in self.outcomes.values())
        if abs(total_p - 1.0) > 1e-6:
            raise ValueError(f"Probabilities must sum to 1.0, got {total_p:.6f}")


def binary_ev(p_win: float, payout_win: float, payout_loss: float) -> EVResult:
    """Compute EV for a binary outcome decision.

    Args:
        p_win: Probability of winning.
        payout_win: Net payout on win (positive).
        payout_loss: Net payout on loss (negative).

    Returns:
        EVResult with ev, variance, and probabilities.
    """
    if not (0.0 <= p_win <= 1.0):
        raise ValueError(f"p_win must be in [0,1], got {p_win}")
    p_loss = 1.0 - p_win
    ev = p_win * payout_win + p_loss * payout_loss
    variance = p_win * (payout_win - ev) ** 2 + p_loss * (payout_loss - ev) ** 2
    return EVResult(
        ev=ev,
        p_win=p_win,
        p_loss=p_loss,
        payout_win=payout_win,
        payout_loss=payout_loss,
        variance=variance,
    )


def multi_outcome_ev(outcomes: dict[str, tuple[float, float]]) -> MultiOutcomeEV:
    """Compute EV for a decision with arbitrarily many discrete outcomes.

    Args:
        outcomes: Mapping of outcome name to (probability, payout).

    Returns:
        MultiOutcomeEV with ev and variance.
    """
    total_p = sum(p for p, _ in outcomes.values())
    if abs(total_p - 1.0) > 1e-6:
        raise ValueError(f"Probabilities must sum to 1.0, got {total_p:.6f}")
    ev = sum(p * v for p, v in outcomes.values())
    variance = sum(p * (v - ev) ** 2 for p, v in outcomes.values())
    return MultiOutcomeEV(outcomes=outcomes, ev=ev, variance=variance)


def ev_per_unit_risk(ev_result: EVResult) -> float:
    """EV divided by amount risked. Positive means +EV decision."""
    risked = abs(ev_result.payout_loss)
    if risked < 1e-12:
        raise ValueError("Cannot compute EV per unit risk when no risk is taken")
    return ev_result.ev / risked


@dataclass
class DecisionNode(Generic[T]):
    """A node in a decision tree.

    Attributes:
        name: Descriptive label.
        ev_func: Callable that returns EVResult for this node.
        children: Optional child nodes (for sequential decisions).
        action: Arbitrary action payload.
    """
    name: str
    ev_func: Callable[[], EVResult]
    children: list[DecisionNode[T]] = field(default_factory=list)
    action: T | None = None

    def evaluate(self) -> EVResult:
        """Evaluate this node's EV."""
        return self.ev_func()

    def best_child(self) -> DecisionNode[T] | None:
        """Return the child with the highest EV, or None if no children."""
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.evaluate().ev)


@dataclass
class DecisionTree:
    """Rooted decision tree for sequential EV analysis.

    Attributes:
        root: Root decision node.
    """
    root: DecisionNode

    def evaluate(self) -> EVResult:
        """Evaluate the root node."""
        return self.root.evaluate()

    def optimal_path(self) -> list[tuple[str, float]]:
        """Walk the tree choosing the highest-EV child at each level.

        Returns:
            List of (node_name, ev) tuples along the optimal path.
        """
        path: list[tuple[str, float]] = []
        node = self.root
        while node is not None:
            result = node.evaluate()
            path.append((node.name, result.ev))
            node = node.best_child()  # type: ignore[assignment]
        return path


def ev_grid(
    p_range: NDArray[np.float64],
    payout_win: float,
    payout_loss: float,
) -> NDArray[np.float64]:
    """Compute EV across a grid of win probabilities.

    Args:
        p_range: 1-D array of win probabilities.
        payout_win: Payout on win.
        payout_loss: Payout on loss.

    Returns:
        1-D array of EV values, same shape as p_range.
    """
    return p_range * payout_win + (1.0 - p_range) * payout_loss
