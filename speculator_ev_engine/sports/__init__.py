"""Sports betting modules: odds, edge, markets, models, backtesting."""

from .odds import (
    OddsConversion, american_to_decimal, decimal_to_american,
    american_to_implied_prob, decimal_to_implied_prob, implied_prob_to_decimal,
    extract_vig, remove_vig, convert_odds, spread_to_moneyline,
)
from .edge import (
    EdgeResult, CLVResult, CLVTracker, compute_edge, compute_clv,
    expected_clv, roi_vs_ev_reconciliation,
)
from .market import LineMovement, classify_movement, detect_steam
from .backtest import BacktestResult, backtest_strategy, walk_forward_backtest
