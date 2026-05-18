"""Sports betting screen."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Header, Footer, Static, Input, Button

import numpy as np

from speculator_ev_engine.sports.odds import convert_odds
from speculator_ev_engine.sports.edge import compute_edge, compute_clv, CLVTracker
from speculator_ev_engine.ui.core.formatters import fmt_ev, fmt_prob, fmt_pct, fmt_dollar
from speculator_ev_engine.ui.tui.widgets import ContextStrip


class SportsScreen(Screen):
    """Edge and CLV calculator."""

    BINDINGS = [("q", "pop_screen", "Back")]

    def __init__(self) -> None:
        super().__init__()
        self._tracker = CLVTracker()

    def compose(self) -> ComposeResult:
        yield Header()
        yield ContextStrip(id="context")
        with Horizontal():
            with Vertical(id="inputs"):
                yield Static("Model Probability")
                yield Input(placeholder="0.55", id="model_p_input")
                yield Static("American Odds")
                yield Input(placeholder="-110", id="odds_input")
                yield Static("Other Side Odds (optional)")
                yield Input(placeholder="-110", id="other_odds_input")
                yield Button("Compute Edge", id="edge_btn")
                yield Static("─── CLV Tracker ───")
                yield Static("Open Odds")
                yield Input(placeholder="-110", id="open_odds")
                yield Static("Close Odds")
                yield Input(placeholder="-105", id="close_odds")
                yield Static("Sport")
                yield Input(placeholder="nfl", id="sport_input")
                yield Static("Book")
                yield Input(placeholder="pinnacle", id="book_input")
                yield Button("Add CLV Record", id="clv_btn")
            with Vertical(id="output"):
                yield Static(id="result_text")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "edge_btn":
            self._compute_edge()
        elif event.button.id == "clv_btn":
            self._add_clv()

    def _compute_edge(self) -> None:
        try:
            model_p = float(self.query_one("#model_p_input", Input).value or "0.55")
            odds = int(self.query_one("#odds_input", Input).value or "-110")
            other_str = self.query_one("#other_odds_input", Input).value.strip()
            other = [int(other_str)] if other_str else None
        except ValueError:
            self.query_one("#result_text", Static).update("[red]Invalid input[/red]")
            return

        if not (0.0 < model_p < 1.0):
            self.query_one("#result_text", Static).update("[red]Probability must be in (0,1)[/red]")
            return

        result = compute_edge(model_p, odds, other)
        lines = [
            f"Edge: {fmt_pct(result.edge)}",
            f"Model prob: {fmt_prob(result.model_prob)}  Market prob: {fmt_prob(result.market_prob)}",
            f"EV per unit: {fmt_ev(result.ev_per_unit)}",
        ]
        self.query_one("#result_text", Static).update("\n".join(lines))

        ctx = self.query_one("#context", ContextStrip)
        ctx.session_ev = result.ev_per_unit

    def _add_clv(self) -> None:
        try:
            open_o = int(self.query_one("#open_odds", Input).value or "-110")
            close_o = int(self.query_one("#close_odds", Input).value or "-105")
            sport = self.query_one("#sport_input", Input).value or "general"
            book = self.query_one("#book_input", Input).value or "general"
        except ValueError:
            self.query_one("#result_text", Static).update("[red]Invalid CLV input[/red]")
            return

        clv_result = self._tracker.add_record(sport, book, "moneyline", "medium", open_o, close_o)
        flags = self._tracker.flag_patterns(min_samples=5)
        lines = [
            f"CLV: {fmt_pct(clv_result.clv)}",
            f"Open implied: {fmt_prob(clv_result.open_implied)}  Close implied: {fmt_prob(clv_result.close_implied)}",
            f"Records in tracker: {len(self._tracker.records)}",
        ]
        if flags:
            lines.append(f"Pattern flags: {len(flags)}")
            for f in flags[:3]:
                lines.append(f"  {f['group']}={f['value']}: {f['direction']} ({fmt_pct(float(f['mean_clv']))})")
        self.query_one("#result_text", Static).update("\n".join(lines))
