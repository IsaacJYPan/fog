"""Policy / value / team-preview heads (spec §3.1)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fog.constants.schema import N_ACTIONS

# Posterior features (SEARCH_PLAN §4.2): the opponent's belief over MY possible private states,
# summarized per decision as [normalized entropy, mass on the state being evaluated, top-1 mass,
# effective-state fraction exp(H)/K]. The value net cannot price signaling without them — this is
# the consumer that makes joint search's belief coupling matter (see fog/search/joint.py).
N_POSTERIOR_FEATURES = 4


class PolicyValueHead(nn.Module):
    """Policy: s -> 14 logits (legality-masked). Value: s -> tanh scalar = E[win] in [-1, 1].

    `belief_vec` (b_vec, §3.2) is concatenated into the input when `use_belief_vec` is set — the cheap
    default path for belief conditioning. `posterior_dim > 0` (SEARCH_PLAN §4.2, config
    `model.use_posterior_features`) additionally concatenates posterior features describing the
    opponent's belief over my own hidden state; absent features fall back to zeros so training
    paths that don't compute them keep working unchanged.
    """

    def __init__(self, state_dim: int = 256, belief_dim: int = 128, use_belief_vec: bool = True,
                 posterior_dim: int = 0) -> None:
        super().__init__()
        self.use_belief_vec = use_belief_vec
        self.posterior_dim = int(posterior_dim)
        in_dim = state_dim + (belief_dim if use_belief_vec else 0) + self.posterior_dim
        self.policy = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, N_ACTIONS))
        self.value = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(
        self,
        s: torch.Tensor,                       # [B, state_dim]
        legality_mask: torch.Tensor,           # [B, 14] bool
        belief_vec: torch.Tensor | None = None,
        posterior_features: torch.Tensor | None = None,  # [B, posterior_dim] or None -> zeros
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = s if not self.use_belief_vec else torch.cat([s, belief_vec], dim=-1)
        if self.posterior_dim:
            if posterior_features is None:
                posterior_features = x.new_zeros(x.shape[0], self.posterior_dim)
            x = torch.cat([x, posterior_features], dim=-1)
        logits = self.policy(x)
        logits = logits.masked_fill(~legality_mask, float("-inf"))
        log_probs = F.log_softmax(logits, dim=-1)
        value = torch.tanh(self.value(x)).squeeze(-1)
        return log_probs, value


class TeamPreviewHead(nn.Module):
    """Team-preview lead selection (spec §3.1).

    Consumes the encoded preview tokens (my 6 mons in slots 0-5) and scores each as a lead. Played
    live by `FogPlayer.choose_teampreview` and trained as a bandit on game outcome in ppo_update.
    """

    def __init__(self, d: int = 128) -> None:
        super().__init__()
        self.lead = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, preview_tokens: torch.Tensor) -> torch.Tensor:
        """preview_tokens [B, 12, d] (0-5 = mine) -> lead logits [B, 6] over my own mons."""
        return self.lead(preview_tokens[:, :6]).squeeze(-1)
