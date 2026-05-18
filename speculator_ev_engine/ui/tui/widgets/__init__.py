"""Reusable TUI widgets: context strip, sparkline, EV table."""

from __future__ import annotations

from textual.widgets import Static, DataTable
from textual.reactive import reactive
from rich.text import Text
from rich.text import Text
# Sparkline removed — rich.sparkline was removed in Rich 15+
from rich.table import Table

from speculator_ev_engine.ui.core.themes import ev_color, EV_WARNING
from speculator_ev_engine.ui.core.formatters import fmt_ev, fmt_dollar, fmt_frac, fmt_prob


class ContextStrip(Static):
    """Persistent context bar: bankroll, Kelly fraction, session EV. Always visible."""

    bankroll: reactive[float] = reactive(1000.0)
    kelly_fraction: reactive[float] = reactive(0.0)
    session_ev: reactive[float] = reactive(0.0)
    tilt_flag: reactive[bool] = reactive(False)

    def watch_bankroll(self, val: float) -> None:
        self._refresh()

    def watch_kelly_fraction(self, val: float) -> None:
        self._refresh()

    def watch_session_ev(self, val: float) -> None:
        self._refresh()

    def watch_tilt_flag(self, val: bool) -> None:
        self._refresh()

    def _refresh(self) -> None:
        ev_color_val = ev_color(self.session_ev)
        parts = [
            f"Bankroll: {fmt_dollar(self.bankroll)}",
            f"Kelly: {fmt_frac(self.kelly_fraction)}",
            f"Session EV: {fmt_ev(self.session_ev)}",
        ]
        if self.tilt_flag:
            parts.append("[TILT DETECTED]")
        text = Text("  │  ".join(parts))
        # Color the session EV
        text.stylize(f"color: {ev_color_val}", text.plain.find("Session EV:"), text.plain.find("Session EV:") + len(f"Session EV: {fmt_ev(self.session_ev)}"))
        if self.tilt_flag:
            text.stylize(f"color: {EV_WARNING}", text.plain.find("[TILT"), text.plain.length)
        self.update(text)

    def compose(self):
        self._refresh()


class EVTable(DataTable):
    """Data table pre-configured for EV display with semantic coloring."""

    def add_ev_row(self, label: str, ev: float, *values: str) -> None:
        """Add a row with the EV column colored semantically."""
        ev_str = fmt_ev(ev)
        color = ev_color(ev)
        all_vals = [label, f"[{color}]{ev_str}[/{color}]"] + list(values)
        self.add_row(*all_vals)
