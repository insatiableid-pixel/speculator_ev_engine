"""Chart package — pure functions producing matplotlib figures from engine data types."""

from .kelly_charts import kelly_fraction_vs_growth, correlated_kelly_heatmap, uncertain_edge_kelly
from .bankroll_charts import bankroll_monte_carlo, drawdown_distribution, fraction_sweep_small_multiples
from .icm_charts import icm_equity_by_stack, bubble_factor_heatmap, push_fold_range_grid
from .sports_charts import clv_distribution, clv_over_time, pattern_flags_small_multiples
from .decision_charts import ev_vs_outcome_scatter, tilt_detection_time_series, calibration_curve
