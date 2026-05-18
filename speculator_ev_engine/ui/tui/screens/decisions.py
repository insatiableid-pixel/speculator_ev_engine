"""Decision review screen."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Header, Footer, Static, Button, DataTable

import numpy as np

from speculator_ev_engine.decisions.logger import Decision, DecisionLogger
from speculator_ev_engine.decisions.leaks import detect_leaks, detect_tilt
from speculator_ev_engine.decisions.calibration import full_calibration
from speculator_ev_engine.decisions.review import review_decisions, format_review
from speculator_ev_engine.ui.core.formatters import fmt_ev, fmt_prob, fmt_dollar
from speculator_ev_engine.ui.tui.widgets import ContextStrip


class DecisionsScreen(Screen):
    """Decision log, leak analysis, calibration, and review."""

    BINDINGS = [("q", "pop_screen", "Back"), ("r", "refresh", "Refresh")]

    def __init__(self) -> None:
        super().__init__()
        self._logger = DecisionLogger()

    def compose(self) -> ComposeResult:
        yield Header()
        yield ContextStrip(id="context")
        with Vertical(id="output"):
            yield Static(id="review_text")
            yield DataTable(id="leak_table")
            yield Static(id="calibration_text")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh()

    def action_refresh(self) -> None:
        self._refresh()

    def _refresh(self) -> None:
        decisions = self._logger.query(resolved_only=True, limit=500)
        if not decisions:
            self.query_one("#review_text", Static).update("No resolved decisions yet. Log some via the API.")
            return

        # Review
        report = review_decisions(decisions)
        self.query_one("#review_text", Static).update(format_review(report))

        # Leak table
        table = self.query_one("#leak_table", DataTable)
        table.clear(columns=True)
        table.add_column("Dimension")
        table.add_column("Group")
        table.add_column("N")
        table.add_column("Mean EV")
        table.add_column("Mean Outcome")
        table.add_column("Gap")
        table.add_column("Stake ↑")
        table.add_column("Edge ↓")
        leaks = detect_leaks(decisions)
        for r in leaks:
            table.add_row(
                r.dimension, r.group, str(r.n_decisions),
                fmt_ev(r.mean_ev), fmt_ev(r.mean_outcome), fmt_ev(r.ev_outcome_gap),
                "⚠" if r.stake_inflation else "",
                "⚠" if r.edge_threshold_drop else "",
            )

        # Calibration
        forecasts = np.array([d.p_estimate for d in decisions])
        outcomes = np.array([d.outcome or 0.0 for d in decisions])
        cal = full_calibration(forecasts, outcomes)
        self.query_one("#calibration_text", Static).update(
            f"Brier: {cal.brier_score:.4f}  ECE: {cal.ece:.3f}  MCE: {cal.mce:.3f}"
        )

        # Context strip
        ctx = self.query_one("#context", ContextStrip)
        ctx.session_ev = report.mean_ev
