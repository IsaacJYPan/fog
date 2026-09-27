"""Evaluation protocol. Fixed before any experiments; results in every report.

  * gauntlet     -- vs Random / MaxDamage / SimpleHeuristics / BC-only, Wilson 95% CIs, ≥2000 games
  * elo          -- round-robin among {BC, N, L×β, set oracle, action oracle}
  * oracle_gap   -- closure = (N − blind_reactive)/(set_oracle − blind_reactive)
  * belief_metrics -- per-attribute top-1/top-3 vs turn, calibration (ECE), pre-reveal accuracy
  * leakage_metrics -- readability curves: how well a FRESH predictor reads L vs N at matched turns
"""

from fog.eval.stats import wilson_interval

__all__ = ["wilson_interval"]
