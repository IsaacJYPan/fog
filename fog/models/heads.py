"""Policy / value / team-preview heads (spec §3.1)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fog.constants.schema import N_ACTIONS


class PolicyValueHead(nn.Module):
    """Policy: s -> 14 logits (legality-masked). Value: s -> tanh scalar = E[win] in [-1, 1].

    `belief_vec` (b_vec, §3.2) is concatenated into the input when `use_belief_vec` is set — the cheap
    default path for belief conditioning.
    """

    def __init__(self, state_dim: int = 256, belief_dim: int = 128, use_belief_vec: bool = True) -> None:
        super().__init__()
        self.use_belief_vec = use_belief_vec
        in_dim = state_dim + (belief_dim if use_belief_vec else 0)
        self.policy = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, N_ACTIONS))
        self.value = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(
        self,
        s: torch.Tensor,                       # [B, state_dim]
        legality_mask: torch.Tensor,           # [B, 14] bool
        belief_vec: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = s if not self.use_belief_vec else torch.cat([s, belief_vec], dim=-1)
        logits = self.policy(x)
        logits = logits.masked_fill(~legality_mask, float("-inf"))
        log_probs = F.log_softmax(logits, dim=-1)
        value = torch.tanh(self.value(x)).squeeze(-1)
        return log_probs, value


class TeamPreviewHead(nn.Module):
    """Team-preview lead selection (spec §3.1).

    Consumes the encoded preview tokens (my 6 mons in slots 0-5) and scores each as a lead. For
    standard OU that's a single lead over 6; for the POC order-of-3, `sample_order` decodes 3 picks
    sequentially without replacement from the same per-mon logits.
    """

    def __init__(self, d: int = 128) -> None:
        super().__init__()
        self.lead = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, preview_tokens: torch.Tensor) -> torch.Tensor:
        """preview_tokens [B, 12, d] (0-5 = mine) -> lead logits [B, 6] over my own mons."""
        return self.lead(preview_tokens[:, :6]).squeeze(-1)

    @torch.no_grad()
    def sample_order(self, preview_tokens: torch.Tensor, k: int = 3, argmax: bool = True) -> list[int]:
        """Pick `k` distinct leads in order (POC order-of-`k`). Returns team-slot indices 0-5."""
        logits = self.forward(preview_tokens)[0].clone()  # [6], single battle
        picks: list[int] = []
        for _ in range(k):
            choice = int(logits.argmax()) if argmax else int(torch.multinomial(logits.softmax(-1), 1))
            picks.append(choice)
            logits[choice] = float("-inf")
        return picks
