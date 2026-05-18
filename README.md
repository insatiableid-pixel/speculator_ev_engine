# speculator_ev_engine

EV-maximizing framework for poker, sports betting, and financial markets. Every module speaks in expected value. Outcome variance is tracked separately from decision quality at all times.

## Install

```bash
pip install -e .
```

Requires Python 3.12+. Core dependencies: numpy, scipy, pandas, matplotlib. Optional: jupyter for notebooks.

## Architecture

```
core/           EV primitives, Kelly criterion, bankroll simulation, probability distributions
poker/          ICM, GTO concepts, hand equity, exploitative deviations, session tracking
sports/         Odds conversion, edge/CLV calculation, market analysis, predictive models, backtesting
markets/        Options pricing, arbitrage detection, portfolio construction, risk metrics
decisions/      Universal decision logger, leak analysis, calibration, review
pipelines/      Data feeds, scrapers, persistent storage
```

## Core Modules

### EV (`core/ev.py`)

```python
from speculator_ev_engine.core.ev import binary_ev, multi_outcome_ev, DecisionTree

result = binary_ev(p_win=0.6, payout_win=1.0, payout_loss=-1.0)
# EVResult(ev=0.2, p_win=0.6, p_loss=0.4, variance=0.96)
```

### Kelly (`core/kelly.py`)

```python
from speculator_ev_engine.core.kelly import binary_kelly, fractional_kelly, multi_outcome_kelly, uncertain_edge_kelly

# Binary
k = binary_kelly(p=0.55, b=1.0)        # KellyResult(fraction=0.10, ...)

# Fractional (half-Kelly is the practical default)
k = fractional_kelly(p=0.55, b=1.0, fraction=0.5)

# Multi-outcome (3-way soccer market)
probs = np.array([0.50, 0.30, 0.20])
payouts = np.array([1.2, -1.0, -1.0])
k = multi_outcome_kelly(probs, payouts)

# Uncertain edge (shrinks Kelly toward zero as uncertainty grows)
k = uncertain_edge_kelly(edge_mean=0.05, edge_std=0.03, odds=1.0)
```

### Bankroll (`core/bankroll.py`)

```python
from speculator_ev_engine.core.bankroll import simulate_bankroll, estimate_ruin_probability, analyze_drawdowns

paths = simulate_bankroll(0.55, 1.0, 0.1, n_steps=10000, n_simulations=10000, seed=42)
ruin = estimate_ruin_probability(0.55, 1.0, 0.1, ruin_threshold=0.05, seed=42)
dd = analyze_drawdowns(paths)
```

## Poker

### ICM (`poker/icm.py`)

```python
from speculator_ev_engine.poker.icm import malmuth_harville_icm, chip_chop_icm, bubble_factor

stacks = np.array([5000.0, 3000.0, 2000.0])
payouts = np.array([500.0, 300.0, 200.0])

# Standard ICM
eq = malmuth_harville_icm(stacks, payouts)
# eq.equities sums to total prize pool

# Chip chop hybrid (0=pure chip chop, 1=pure ICM)
eq = chip_chop_icm(stacks, payouts, blend_weight=0.5)

# Bubble factor — how much more equity you lose than gain per chip
bf = bubble_factor(stacks, payouts, player_index=2)
# bf.bubble_factor > 1 means bubble pressure
```

## Sports Betting

### Odds (`sports/odds.py`)

```python
from speculator_ev_engine.sports.odds import convert_odds, remove_vig, extract_vig

result = convert_odds(-110, other_outcomes=[-110])
# OddsConversion with true_prob (vig-free)
```

### Edge & CLV (`sports/edge.py`)

```python
from speculator_ev_engine.sports.edge import compute_edge, compute_clv, CLVTracker

edge = compute_edge(model_prob=0.55, odds_american=-110, other_outcomes=[-110])
clv = compute_clv(open_odds=150, close_odds=120, other_open_outcomes=[-170], other_close_outcomes=[-140])

# Track CLV patterns across books/sports/tiers
tracker = CLVTracker()
tracker.add_record("nfl", "pinnacle", "spread", "medium", -110, -105)
summary = tracker.summary_by("sport")
flags = tracker.flag_patterns(min_samples=20)
```

## Decision Logger (`decisions/`)

```python
from speculator_ev_engine.decisions.logger import Decision, DecisionLogger

logger = DecisionLogger()
d = Decision(decision="NFL spread: KC -3", p_estimate=0.55, ev_estimate=0.1, stake=100, domain="sports")
row_id = logger.log(d)
logger.resolve(row_id, outcome=1.0)  # win

# Query and analyze
from speculator_ev_engine.decisions.leaks import detect_leaks, detect_tilt
from speculator_ev_engine.decisions.calibration import full_calibration
from speculator_ev_engine.decisions.review import review_decisions, format_review

decisions = logger.query(domain="sports", resolved_only=True)
leaks = detect_leaks(decisions, group_by="domain")
tilts = detect_tilt(decisions)
cal = full_calibration(np.array([d.p_estimate for d in decisions]), np.array([d.outcome for d in decisions]))
review = review_decisions(decisions)
print(format_review(review))
```

## UI

Three interfaces over the same shared chart engine. All charts from `ui/core/charts/` — no logic duplication.

```bash
pip install -e ".[ui-all]"     # everything
pip install -e ".[ui-tui]"     # terminal only
pip install -e ".[ui-web]"     # FastAPI + dark theme frontend
pip install -e ".[ui-jupyter]" # ipywidgets notebooks
```

**Terminal** — `seve-tui`. Textual app, keyboard-first, mouse-optional. Left nav, right output, bottom context strip. Degrades to 16 colors gracefully.

**Jupyter** — four notebooks (`kelly.ipynb`, `icm.ipynb`, `sports.ipynb`, `decisions.ipynb`) with ipywidgets sliders driving live chart updates.

**Web** — `seve-web`. FastAPI backend, single-page dark frontend with Plotly.js, persistent context strip pinned to viewport, no page reloads.

## Design Principles

1. **EV is the universal unit.** Every module speaks expected value. Results are tracked separately from decision quality.
2. **Kelly is the bankroll spine.** All sizing routes through the Kelly framework. Full multi-outcome and correlated-bet Kelly implemented — not just the binary formula.
3. **CLV is sports betting ground truth.** The CLV module validates whether edges were real.
4. **The decision logger is domain-agnostic.** A poker hand, a sports bet, and an options trade are the same schema.
5. **Calibration is first-class.** Probability estimation without calibration feedback is superstition.
6. **No hand-holding.** Function signatures assume you know what ICM, MDF, CLV, Greeks, and Kelly mean.

## Notebooks

| Notebook | Content |
|---|---|
| `kelly_deep_dive.ipynb` | Binary, fractional, multi-outcome, correlated, and uncertain-edge Kelly with parameter sweeps |
| `icm_vs_chipev.ipynb` | ICM vs chip chop comparison, bubble factor heatmaps, push/fold EV |
| `clv_analysis.ipynb` | CLV distribution, tracking patterns, ROI-vs-EV reconciliation |
| `bankroll_simulation.ipynb` | Monte Carlo paths under different Kelly fractions, drawdown analysis, ruin probability |

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```
