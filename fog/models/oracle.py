"""Oracle variants — evaluation ceilings, never deployed (spec §3.4).

  * Set oracle:    opponent entity records + belief tokens filled with true values, all epistemic
                   flags "known". Bounds the value of knowing the opponent's sets.
  * Action oracle: set oracle + the opponent's committed action index this turn appended to `s` before
                   the policy head. Bounds the additional value of predicting intent.

Both are config switches (`model.oracle: {none, set, action}`) consumed inside FogAgent.forward, so
turning them on is a one-line config change (deliberate — §3.4). The heavy lifting is:
  - `apply_set_oracle`: rewrite the belief-net input tokens (opp_revealed) to the opponent's full
    truth so the belief head/b_vec are computed from perfect information, and mark every opponent
    entity's epistemic flags known. Ground truth is supplied by the caller (simulator in self-play,
    the parsed row's uncensored labels where available).
  - `ActionOracleMixin`: a small embedding of the opponent's committed action concatenated to `s`.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from fog.constants.schema import N_ACTIONS


def apply_set_oracle(opp_revealed: dict, opp_truth_tokens) -> dict:
    """Return an opp_revealed dict with tokens replaced by full truth (epistemic = all known).

    `opp_truth_tokens`: [B,6,9] (or [6,9]) ground-truth opponent tokens. Masks are dropped — with full
    truth every attribute is clamped, so the belief net just re-emits the truth (the b_vec is then a
    perfect-information summary). Works for both the batched model path and single observations.
    """
    tokens = opp_truth_tokens
    if isinstance(tokens, np.ndarray):
        tokens = torch.as_tensor(tokens, dtype=torch.long)
    if tokens.ndim == 2:
        tokens = tokens.unsqueeze(0)
    return {"tokens": tokens.to(_ref_device(opp_revealed)), "masks": {}}


def _ref_device(opp_revealed: dict):
    tok = opp_revealed.get("tokens") if opp_revealed else None
    return tok.device if isinstance(tok, torch.Tensor) else torch.device("cpu")


class ActionOracleHead(nn.Module):
    """Embeds the opponent's committed action and concatenates it to `s` (spec §3.4 action oracle).

    Kept separate from the base policy so weights stay comparable across oracle modes: the base head
    still sees `s`; only this augmented head sees `s ‖ action_embed`.
    """

    def __init__(self, state_dim: int, belief_dim: int, action_embed: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(N_ACTIONS + 1, action_embed)  # +1 = <no committed action known>
        self.extra_dim = action_embed

    def augment(self, s: torch.Tensor, committed_action: torch.Tensor | None) -> torch.Tensor:
        if committed_action is None:
            idx = torch.full((s.shape[0],), N_ACTIONS, dtype=torch.long, device=s.device)
        else:
            idx = committed_action.to(s.device)
        return torch.cat([s, self.embed(idx)], dim=-1)
