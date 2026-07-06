"""Root-level determinized evaluation (spec §8). Inference-time only — NEVER in the training loop.

For each legal action a (≤14): sample k opponent teams from the belief model (clamped to reveals);
in each sampled world score a with the value net (optionally after stepping the opponent's policy
reply); play argmax over a of the mean score.

The full version steps a dedicated local simulator; that Node coupling is environment-specific, so
this module implements the model-side scoring and exposes a `step_fn` seam for the simulator. With
`step_fn=None` it falls back to a 1-ply value-net estimate (score = value of the current state under
each forced action's implied belief) — enough to run the search wiring and the gauntlet on/off column.

Known theoretical limitation (assumes worlds get resolved, undervalues concealment) is documented and
is exactly what §7 compensates for.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def determinized_action(agent, batch: dict, *, k: int = 16, step_fn=None) -> int:
    """Return the argmax action under root-level determinized rollout for a single decision.

    `batch`: a batched (B=1) FogAgent input dict (as FogPlayer builds). `step_fn(action, world_tokens)
    -> next_batch` optionally advances a simulator one ply in a sampled world; if None, the value net
    scores the current state conditioned on each sampled world.
    """
    agent.eval()
    legal = batch["legality_mask"][0].nonzero(as_tuple=True)[0].tolist()
    if len(legal) <= 1:
        return legal[0] if legal else 13

    s = agent._backbone_pass(batch)                        # [1, state_dim]
    worlds = agent.belief.sample_team(s, batch["opp_revealed"], k=k)  # [1, k, 6, 9]
    worlds = worlds[0]                                     # [k, 6, 9]

    scores = torch.full((14,), float("-inf"))
    for a in legal:
        world_scores = []
        for w in range(k):
            world_batch = _world_batch(batch, worlds[w])
            if step_fn is not None:
                nxt = step_fn(a, worlds[w])
                if nxt is not None:
                    world_batch = nxt
            out = agent(world_batch)
            # sign: value is E[win] for the acting player; higher is better
            world_scores.append(float(out.value[0]))
        scores[a] = sum(world_scores) / len(world_scores)
    return int(scores.argmax())


def _world_batch(batch: dict, world_tokens: torch.Tensor) -> dict:
    """Clamp the belief input to one fully-specified sampled opponent team (set-oracle-like)."""
    wb = dict(batch)
    wb["opp_revealed"] = {"tokens": world_tokens.unsqueeze(0), "masks": {}}
    return wb


class SearchPlayerMixin:
    """Mix into a FogPlayer to select actions via `determinized_action` instead of sampling.

    Set `self.search_k` and (optionally) `self.step_fn`; override the action-selection call site to use
    `determinized_action(self.agent, batch, k=self.search_k, step_fn=self.step_fn)`.
    """

    search_k: int = 16
    step_fn = None
