"""ICM screen."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Header, Footer, Static, Input, Button

import numpy as np

from speculator_ev_engine.poker.icm import malmuth_harville_icm, chip_chop_icm, bubble_factor
from speculator_ev_engine.ui.core.formatters import fmt_dollar, fmt_prob
from speculator_ev_engine.ui.tui.widgets import ContextStrip


class ICMScreen(Screen):
    """ICM equity calculator."""

    BINDINGS = [("q", "pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield ContextStrip(id="context")
        with Horizontal():
            with Vertical(id="inputs"):
                yield Static("Stacks (comma-separated)")
                yield Input(placeholder="5000,3000,2000", id="stacks_input")
                yield Static("Payouts (comma-separated)")
                yield Input(placeholder="500,300,200", id="payouts_input")
                yield Static("Blend weight (0=chip chop, 1=ICM)")
                yield Input(placeholder="1.0", id="blend_input")
                yield Button("Compute", id="compute_btn")
            with Vertical(id="output"):
                yield Static(id="result_text")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "compute_btn":
            return
        try:
            stacks = np.array([float(x.strip()) for x in self.query_one("#stacks_input", Input).value.split(",")])
            payouts = np.array([float(x.strip()) for x in self.query_one("#payouts_input", Input).value.split(",")])
            blend = float(self.query_one("#blend_input", Input).value or "1.0")
        except ValueError:
            self.query_one("#result_text", Static).update("[red]Invalid input[/red]")
            return

        if np.any(stacks <= 0):
            self.query_one("#result_text", Static).update("[red]Stacks must be positive[/red]")
            return
        if not (0.0 <= blend <= 1.0):
            self.query_one("#result_text", Static).update("[red]Blend must be in [0,1][/red]")
            return

        if blend == 1.0:
            result = malmuth_harville_icm(stacks, payouts)
        else:
            result = chip_chop_icm(stacks, payouts, blend_weight=blend)

        lines = [f"Total prize pool: {fmt_dollar(result.total_prize_pool)}"]
        for i, (s, eq) in enumerate(zip(stacks, result.equities)):
            lines.append(f"  Stack {i+1} ({int(s):,} chips): {fmt_dollar(float(eq))}")

        self.query_one("#result_text", Static).update("\n".join(lines))
