"""Leakage probes (spec §3.3/§7): the measurement path = (backbone -> self-belief predictor).

`FrozenProbe` is Φ̂ — a frozen deep copy of BOTH the backbone and the self-belief head. Freezing only
the head (the original v1 wiring) left the reward a function of the live, training backbone: as the
trunk drifted between refreezes the frozen head received out-of-distribution states, the CE stopped
meaning "readability", and the agent could look opaque by moving its representations instead of its
behavior. The probe closes that hole: every ΔI measurement runs end-to-end inside frozen weights.

`TrainableProbe` is the independent "fresh predictor" (§7 watchdog + §9 readability eval): same
architecture, separate weights, trained only to read the agent — never used for reward. Its CE vs the
frozen probe's CE is the staleness-exploitation detector; trained from scratch on eval rollouts it is
the readability curve measurement.
"""
from __future__ import annotations

import copy

import torch
from torch import nn

_BACKBONE_KEYS = (
    "entities", "entity_ids", "field", "history", "active_idx", "history_len",
    "history_ids", "history_state",
)


def _backbone_kwargs(batch: dict) -> dict:
    return {k: batch[k] for k in _BACKBONE_KEYS if k in batch and batch[k] is not None}


class FrozenProbe(nn.Module):
    """Φ̂: frozen (backbone + self-belief) pair. Built via `from_agent`; refrozen on the §7 schedule."""

    def __init__(
        self, backbone: nn.Module, private_to_belief: nn.Module, predictor: nn.Module
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.private_to_belief = private_to_belief
        self.predictor = predictor
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    @classmethod
    def from_agent(cls, agent) -> FrozenProbe:
        return cls(
            copy.deepcopy(agent.backbone),
            copy.deepcopy(agent.private_to_belief),
            copy.deepcopy(agent.self_belief),
        )

    def _belief_context(self, batch: dict) -> torch.Tensor:
        contexts = self.backbone(**_backbone_kwargs(batch))
        public_context = self.backbone.fuse_public(
            contexts.public_state, contexts.history
        )
        return public_context + self.private_to_belief(contexts.private_state)

    @torch.no_grad()
    def leakage_ce(
        self,
        mirror_batch: dict,
        my_view: dict,
        still_hidden_mask: dict | None = None,
        family_weights: dict[str, float] | None = None,
        per_family: bool = False,
    ):
        """Readability CE of the mirror observation under frozen weights. See
        `SelfBeliefPredictor.leakage_ce` for mask/label/weight semantics. Returns [B] (or a
        per-family dict of [B] when `per_family`)."""
        belief_context = self._belief_context(mirror_batch)
        return self.predictor.leakage_ce(
            belief_context, my_view, still_hidden_mask=still_hidden_mask,
            family_weights=family_weights, per_family=per_family,
        )


class TrainableProbe(nn.Module):
    """An independent reader of the agent: fresh backbone + self-belief predictor, same sizes.

    Used two ways:
      * inside the §6/§7 training loop, fit on each rollout batch's opp_view so `fresh_ce` tracks how
        readable the CURRENT agent is to an up-to-date adversary (staleness watchdog input);
      * in §9 eval, trained from scratch on collected trajectories of a finished checkpoint
        (the readability curve). Never used for reward.
    """

    def __init__(self, cfg: dict | None = None, sizes: dict | None = None) -> None:
        super().__init__()
        from fog.models.encoder import Backbone
        from fog.models.self_belief import SelfBeliefPredictor

        cfg = cfg or {}
        model_cfg = cfg.get("model", {})
        belief_kwargs = (
            {
                k: sizes[k]
                for k in (
                    "n_species", "n_items", "n_abilities", "n_moves",
                    "n_natures", "n_ev_units",
                )
            }
            if sizes
            else {}
        )
        self.backbone = Backbone(cfg.get("encoding", {}) | model_cfg, sizes=sizes)
        self.private_to_belief = nn.Linear(
            self.backbone.state_dim, self.backbone.dyn_dim, bias=False
        )
        nn.init.zeros_(self.private_to_belief.weight)
        self.predictor = SelfBeliefPredictor(
            state_dim=self.backbone.dyn_dim, d=model_cfg.get("belief_dim", 128), **belief_kwargs
        )

    @classmethod
    def from_agent(cls, agent, cfg: dict | None = None) -> TrainableProbe:
        """Same sizes as `agent` (built via FogAgent.from_tables or bare), freshly initialized."""
        probe = cls(cfg, sizes=agent.sizes)
        return probe

    def loss(self, opp_view_batch: dict, labels: dict | None = None) -> torch.Tensor:
        """Full-truth training CE over ALL attributes of the mirror view (labels default to
        opp_view_batch["my_view"]["labels"])."""
        from fog.training.losses import belief_loss

        my_view = opp_view_batch["my_view"]
        labels = labels if labels is not None else my_view["labels"]
        contexts = self.backbone(**_backbone_kwargs(opp_view_batch))
        public_context = self.backbone.fuse_public(
            contexts.public_state, contexts.history
        )
        belief_context = public_context + self.private_to_belief(contexts.private_state)
        out = self.predictor(belief_context, my_view)
        valid = {k: (v >= 0) for k, v in labels.items()}
        return belief_loss(out.logits, labels, valid)

    @torch.no_grad()
    def leakage_ce(
        self,
        mirror_batch: dict,
        my_view: dict,
        still_hidden_mask: dict | None = None,
        family_weights: dict[str, float] | None = None,
        per_family: bool = False,
    ):
        contexts = self.backbone(**_backbone_kwargs(mirror_batch))
        public_context = self.backbone.fuse_public(
            contexts.public_state, contexts.history
        )
        belief_context = public_context + self.private_to_belief(contexts.private_state)
        return self.predictor.leakage_ce(
            belief_context, my_view, still_hidden_mask=still_hidden_mask,
            family_weights=family_weights, per_family=per_family,
        )
