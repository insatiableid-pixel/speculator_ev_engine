# ROADMAP

Where things stand and where they could go. Not a commitment — just breadcrumbs.

## Fully Implemented

| Module | What's there |
|---|---|
| `core/ev.py` | Binary/multi-outcome EV, decision trees, EV grids |
| `core/kelly.py` | Binary, fractional, multi-outcome, correlated, uncertain-edge Kelly + ruin probability |
| `core/bankroll.py` | Monte Carlo bankroll simulation, ruin estimation, drawdown analysis, fraction sweep |
| `core/distributions.py` | Brier score/decomposition, calibration curves, normal CDF/PDF, confidence intervals, entropy, KL divergence |
| `poker/icm.py` | Malmuth-Harville ICM, chip chop hybrid, bubble factor, push/fold EV interface |
| `sports/odds.py` | American/decimal/fractional conversion, vig extraction, vig-free probabilities |
| `sports/edge.py` | Edge computation, CLV (sign-correct), CLV tracker with pattern flagging by sport/book/market/tier, ROI-vs-EV reconciliation |
| `decisions/logger.py` | SQLite decision logger, schema-validated writes, domain/tag/date filtering |
| `decisions/leaks.py` | Leak detection by dimension, tilt detection (stake inflation + edge threshold drop after losses), EV-outcome divergence surfacing |
| `decisions/calibration.py` | Full calibration report (ECE/MCE/Brier), calibration correction, cross-validated calibration |

## Stub Modules — Interfaces Exist, Logic TODO

### poker/gto.py
- Range construction and range-vs-range equity
- MDF solver for arbitrary bet sizes
- Multi-street indifference frequency solver
- Alpha (bluff-to-value ratio) for balanced ranges

### poker/hand_equity.py
- Full Monte Carlo equity engine (currently returns uniform stub)
- Hold'em, Omaha, Short Deck variants
- Weighted opponent ranges and multi-way equities
- Needs a hand evaluator backend (deuces or equivalent)

### poker/exploitative.py
- Population tendency profiles
- Villain-type-specific deviation strategies
- Deviation cost/benefit: EV gained vs. exploitability exposed
- `compute_deviation` stub — needs hand-level EV engine to fill in

### poker/session.py
- Hand history parsers: PokerStars, GG, WPN formats
- EV reconstruction from parsed hand histories
- Session-level aggregation (already has `SessionLogger` + `SessionSummary` plumbing)

### sports/market.py
- Line movement classification (sharp vs. public)
- Reverse line movement detection
- Steam move detection across multiple books
- Book-specific movement profiles
- `detect_steam` returns empty — needs timestamped multi-book data

### sports/models/base_model.py
- Model serialization/deserialization
- Feature store integration
- Versioned model registry

### sports/models/regression.py
- L1/L2 regularization tuning
- Feature importance extraction
- Calibration-aware training (temperature scaling post-fit)

### sports/models/ensemble.py
- Stacking meta-learner
- Learned weight optimization
- Diversity-aware weighting
- Ensemble pruning
- `evaluate` has a `calibration_error=0.0` placeholder

### sports/backtest.py
- Walk-forward validation with expanding/rolling window
- CLV-weighted backtesting
- Multi-strategy comparison framework
- `walk_forward_backtest` is a framework stub — needs model retraining loop

### markets/options.py
- American option pricing via binomial tree (European exists)
- Volatility surface fitting
- Dividend-adjusted models

### markets/arbitrage.py
- Cross-book arbitrage detection (binary exists, multi-outcome TODO)
- Correlated market arbitrage (same-game parlays, derivatives)
- Latency-aware arb detection for live markets

### markets/portfolio.py
- Constrained Kelly (max position size, sector limits)
- Full mean-variance frontier computation
- Risk-parity baseline

### markets/risk.py
- Parametric VaR (normal, t-distribution)
- Monte Carlo VaR
- Stress-test correlation breakdown models (correlation_under_stress exists but basic)

### decisions/review.py
- Rich-formatted CLI output
- Interactive session review mode
- CSV/JSON export

### pipelines/data_feeds.py
- Odds API adapters (Pinnacle, TheOdds, etc.)
- Hand history parsers (PokerStars, GG, WPN)
- Market data feed adapters (yfinance, polygon)
- Async streaming for live data

### pipelines/scrapers/
- Odds scraper for public bookmaker pages
- Injury report scraper
- Weather data scraper
- Line movement history scraper

### pipelines/storage.py
- Model output storage (schema exists, query helpers TODO)
- Session metadata storage
- Common analytical pattern queries

## Natural Next Steps (when motivation hits)

1. **Hand evaluator** — plug in deuces or similar, unlock `hand_equity.py` and everything downstream
2. **Hand history parser** — pick one site format (PokerStars is most documented), unlock `session.py` and real leak analysis
3. **Walk-forward backtest** — the actual model-retraining loop; makes the sports layer properly testable
4. **American options** — binomial tree already works for European; early exercise is one parameter flip + exercise check
5. **Live data feeds** — async streaming would make the sports/markets layers real-time instead of batch-only
