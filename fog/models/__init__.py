"""Model heads (spec §3). Shared backbone `s in R^256`, then:

  * policy + value + team-preview heads          (§3.1)
  * battle-conditioned autoregressive belief net (§3.2)  -> b_vec + sample_team(k)
  * self-belief predictor (leakage measurer)     (§3.3)  -> per-attribute CE = leakage signal
  * oracle variants (evaluation ceilings)        (§3.4)

Parameter budget: backbone ~1.5M + belief ~1M + self-belief ~1M + heads ~0.3M ≈ 3.8M (≤5M cap).
"""

from fog.models.agent import FogAgent
from fog.models.belief import BeliefModel
from fog.models.encoder import Backbone
from fog.models.heads import PolicyValueHead, TeamPreviewHead
from fog.models.self_belief import SelfBeliefPredictor

__all__ = [
    "FogAgent",
    "Backbone",
    "BeliefModel",
    "PolicyValueHead",
    "TeamPreviewHead",
    "SelfBeliefPredictor",
]
