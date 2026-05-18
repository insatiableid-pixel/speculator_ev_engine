"""Financial markets modules: options, arbitrage, portfolio, risk."""

from .options import OptionPrice, black_scholes, binomial_tree, implied_volatility
from .arbitrage import ArbitrageOpportunity, detect_arbitrage
from .portfolio import PortfolioAllocation, kelly_portfolio, mean_variance_portfolio
from .risk import RiskMetrics, compute_risk_metrics, correlation_under_stress
