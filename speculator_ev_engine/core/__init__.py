"""Core EV primitives: expected value, Kelly criterion, bankroll, distributions."""

from .ev import EVResult, MultiOutcomeEV, binary_ev, multi_outcome_ev, ev_per_unit_risk, DecisionNode, DecisionTree, ev_grid
from .kelly import KellyResult, binary_kelly, fractional_kelly, multi_outcome_kelly, correlated_kelly, uncertain_edge_kelly
from .bankroll import RuinResult, DrawdownResult, simulate_bankroll, estimate_ruin_probability, analyze_drawdowns
from .distributions import (
    BrierDecomposition, brier_score, brier_decomposition, calibration_curve,
    reliability_diagram_data, normal_cdf, normal_pdf, confidence_interval,
    entropy, kl_divergence,
)
