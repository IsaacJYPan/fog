"""Model heads. Three encoders -> (private context, fused public context), then:

  * policy + value + team-preview heads          <- (private ‖ public), no belief summary
  * battle-conditioned autoregressive belief net -> chain CE + sample_team(k); public evidence plus
    the same observer's private context
  * self-belief predictor (leakage measurer)     (§3.5)  -> per-attribute CE = leakage signal
  * oracle variants (evaluation ceilings)        (§3.4)

The default generated-table model is approximately 4.24M parameters (≤5M cap).
"""

from fog.models.agent import FogAgent
from fog.models.belief import BeliefModel
from fog.models.encoder import Backbone
from fog.models.heads import PolicyValueHead, TeamPreviewHead
from fog.models.self_belief import SelfBeliefPredictor
from fog.models.shared_latent import SharedBattleLatent
from fog.models.structured_memory import StructuredMemoryEncoder, StructuredMemoryReader

__all__ = [
    "Backbone",
    "BeliefModel",
    "FogAgent",
    "PolicyValueHead",
    "SelfBeliefPredictor",
    "SharedBattleLatent",
    "StructuredMemoryEncoder",
    "StructuredMemoryReader",
    "TeamPreviewHead",
]
