"""Model heads (BELIEF_POLICY_PLAN §3). Shared backbone -> (s_board, c_dyn), then:

  * policy + value + team-preview heads          (§3.2)  <- (s_board ‖ c_dyn), no belief summary
  * battle-conditioned autoregressive belief net (§3.3)  -> chain CE + sample_team(k); reads c_dyn
  * self-belief predictor (leakage measurer)     (§3.5)  -> per-attribute CE = leakage signal
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
