"""Textual TUI application entry point."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Header, Footer, TabbedContent
from textual.containers import Container

from speculator_ev_engine.ui.tui.widgets import ContextStrip
from speculator_ev_engine.ui.tui.screens.kelly import KellyScreen
from speculator_ev_engine.ui.tui.screens.icm import ICMScreen
from speculator_ev_engine.ui.tui.screens.sports import SportsScreen
from speculator_ev_engine.ui.tui.screens.decisions import DecisionsScreen


class SpeculatorApp(App):
    """speculator_ev_engine terminal interface. Keyboard-first, mouse-optional."""

    TITLE = "speculator_ev_engine"
    CSS = """
    Screen {
        layout: vertical;
    }
    #context {
        dock: top;
        height: 1;
        padding: 0 1;
    }
    TabbedContent {
        height: 1fr;
    }
    """

    BINDINGS = [
        Binding("k", "switch_tab('kelly')", "Kelly"),
        Binding("i", "switch_tab('icm')", "ICM"),
        Binding("s", "switch_tab('sports')", "Sports"),
        Binding("d", "switch_tab('decisions')", "Decisions"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield ContextStrip(id="context")
        with TabbedContent():
            yield KellyScreen(title="Kelly")
            yield ICMScreen(title="ICM")
            yield SportsScreen(title="Sports")
            yield DecisionsScreen(title="Decisions")
        yield Footer()

    def action_switch_tab(self, tab_id: str) -> None:
        try:
            self.query_one(TabbedContent).active = tab_id
        except Exception:
            pass


def main() -> None:
    app = SpeculatorApp()
    app.run()


if __name__ == "__main__":
    main()
