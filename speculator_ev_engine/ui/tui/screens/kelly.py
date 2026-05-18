"""Kelly criterion screen."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Header, Footer, Static, Input, Button, DataTable

import numpy as np

from speculator_ev_engine.core.kelly import (
    binary_kelly, fractional_kelly, multi_outcome_kelly, uncertain_edge_kelly, KellyResult,
)
from speculator_ev_engine.ui.core.charts import (
    kelly_fraction_vs_growth, uncertain_edge_kelly as uncertain_edge_chart,
)
from speculator_ev_engine.ui.core.formatters import fmt_ev, fmt_frac, fmt_prob, fmt_ruin
from speculator_ev_engine.ui.tui.widgets import ContextStrip, EVTable


class KellyScreen(Screen):
    """Interactive Kelly criterion explorer."""

    BINDINGS = [("q", "pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield ContextStrip(id="context")
        with Horizontal():
            with Vertical(id="inputs"):
                yield Static("Win Probability")
                yield Input(placeholder="0.55", id="p_input")
                yield Static("Decimal Odds")
                yield Input(placeholder="1.0", id="b_input")
                yield Static("Fraction (half=0.5)")
                yield Input(placeholder="0.5", id="frac_input")
                yield Static("Edge Uncertainty σ")
                yield Input(placeholder="0.0", id="edge_std_input")
                yield Button("Compute", id="compute_btn")
            with Vertical(id="output"):
                yield Static(id="result_text")
                yield Static(id="chart_area")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "compute_btn":
            return
        try:
            p = float(self.query_one("#p_input", Input).value or "0.55")
            b = float(self.query_one("#b_input", Input).value or "1.0")
            frac = float(self.query_one("#frac_input", Input).value or "0.5")
            edge_std = float(self.query_one("#edge_std_input", Input).value or "0.0")
        except ValueError:
            self.query_one("#result_text", Static).update("[red]Invalid input[/red]")
            return

        if not (0.0 < p < 1.0):
            self.query_one("#result_text", Static).update("[red]p must be in (0,1)[/red]")
            return
        if b <= 0:
            self.query_one("#result_text", Static).update("[red]Odds must be positive[/red]")
            return

        full = binary_kelly(p, b)
        half = fractional_kelly(p, b, fraction=frac)

        lines = [
            f"Full Kelly:  f={fmt_frac(full.fraction)}  g={fmt_ev(full.expected_log_growth)}  ruin={fmt_ruin(full.ruin_probability)}",
            f"Frac Kelly:  f={fmt_frac(half.fraction)}  g={fmt_ev(half.expected_log_growth)}  ruin={fmt_ruin(half.ruin_probability)}",
        ]

        if edge_std > 0:
            uncertain = uncertain_edge_kelly(edge_mean=p - (1-p)/b, edge_std=edge_std, odds=b, seed=42)
            lines.append(
                f"Uncertain:   f={fmt_frac(uncertain.fraction)}  g={fmt_ev(uncertain.expected_log_growth)}  ruin={fmt_ruin(uncertain.ruin_probability)}"
            )

        self.query_one("#result_text", Static).update("\n".join(lines))

        # Update context strip
        ctx = self.query_one("#context", ContextStrip)
        ctx.kelly_fraction = half.fraction
        ctx.session_ev = half.expected_log_growth
