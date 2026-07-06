"""Assembled agent: backbone + belief + self-belief + heads (spec §3), with a param-budget check.

This is the single nn.Module that training and inference instantiate. It owns the wiring between the
belief net's `b_vec` and the policy/value head, and the `aux_into_trunk` switch controlling whether the
belief/self-belief losses backprop into the shared backbone (default true; an ablation switch).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from fog.models.belief import BeliefModel, BeliefOutput
from fog.models.encoder import Backbone
from fog.models.heads import PolicyValueHead
from fog.models.self_belief import SelfBeliefPredictor

PARAM_BUDGET = 5_000_000  # hard cap (≤5M); spec target ≈ 3.8M


@dataclass
class AgentOutput:
    log_probs: torch.Tensor
    value: torch.Tensor
    belief: BeliefOutput | None = None       # opponent-team belief (§3.2)
    self_belief: BeliefOutput | None = None  # leakage measurer output (§3.3; training only)


class FogAgent(nn.Module):
    def __init__(self, cfg: dict | None = None, sizes: dict | None = None) -> None:
        """`sizes` = RAW generated-table cardinalities (see `from_tables`). None keeps the scaffolding
        placeholder sizes — fine for unit tests, WRONG for real battles (real table indices exceed
        the placeholder embedding ranges and will silently alias or crash)."""
        super().__init__()
        cfg = cfg or {}
        model_cfg = cfg.get("model", {})
        self.aux_into_trunk = model_cfg.get("aux_into_trunk", True)
        self.use_belief_vec = model_cfg.get("use_belief_vec", True)
        self.oracle = model_cfg.get("oracle", "none")  # {none, set, action} — §3.4 eval ceilings
        # blind_reactive (§9 closure baseline): force history to 0 and zero the belief conditioning,
        # so the agent reacts only to the current board with no memory/belief. Same weights.
        self.blind_reactive = model_cfg.get("blind_reactive", False)
        belief_dim = model_cfg.get("belief_dim", 128)
        self.sizes = sizes

        # Belief nets take RAW sizes: their output heads emit n classes (no UNK class — you never
        # predict "unknown"), while their internal input embeddings allocate the +1 UNK row
        # themselves. The backbone embeddings get the +1 inside Backbone.
        belief_kwargs = (
            {k: sizes[k] for k in ("n_species", "n_items", "n_abilities", "n_moves", "n_spreads")}
            if sizes
            else {}
        )

        self.backbone = Backbone(cfg.get("encoding", {}) | model_cfg, sizes=sizes)
        self.belief = BeliefModel(state_dim=self.backbone.state_dim, d=belief_dim, **belief_kwargs)
        self.self_belief = SelfBeliefPredictor(
            state_dim=self.backbone.state_dim, d=belief_dim, **belief_kwargs
        )
        # Action oracle appends a committed-action embedding to `s`; a dedicated head keeps the base
        # policy's weights comparable across oracle modes (§3.4).
        self.action_oracle = None
        extra = 0
        if self.oracle == "action":
            from fog.models.oracle import ActionOracleHead

            self.action_oracle = ActionOracleHead(self.backbone.state_dim, belief_dim)
            extra = self.action_oracle.extra_dim
        self.policy_value = PolicyValueHead(
            state_dim=self.backbone.state_dim + extra,
            belief_dim=belief_dim,
            use_belief_vec=self.use_belief_vec,
        )
        # Team-preview lead selection (§3.1) — used LIVE by FogPlayer.teampreview and trained by the
        # REINFORCE-style preview term in ppo_update (game outcome as the bandit reward).
        from fog.models.heads import TeamPreviewHead

        self.preview = TeamPreviewHead(d=(cfg.get("encoding", {}) | model_cfg).get("entity_hidden", 128))

    def preview_logits(self, batch: dict) -> torch.Tensor:
        """Lead logits [B, 6] over my six preview mons. `batch` needs entities/entity_ids/field."""
        tokens = self.backbone.entity_tokens(batch["entities"], batch["entity_ids"], batch["field"])
        return self.preview(tokens)

    @classmethod
    def from_tables(cls, cfg: dict | None = None, tables=None) -> "FogAgent":
        """Build with real cardinalities from the generated constants (§1). Use this — not the bare
        constructor — for anything touching real battles or parsed replays.

        n_spreads is MAX_SPREADS_PER_SPECIES + 1 (<OTHER_SPREAD>): spread cluster ids live in a
        per-species index space, so the global embedding just needs the max per-species count.
        """
        from fog.constants.schema import MAX_SPREADS_PER_SPECIES

        if tables is None:
            from fog.constants.tables import TABLES as tables
        sizes = {
            "n_species": tables.n_species,
            "n_items": tables.n_items,
            "n_abilities": tables.n_abilities,
            "n_moves": tables.n_moves,
            "n_spreads": MAX_SPREADS_PER_SPECIES + 1,
        }
        agent = cls(cfg, sizes=sizes)
        agent.assert_within_budget()
        return agent

    def _backbone_pass(self, group: dict) -> torch.Tensor:
        return self.backbone(
            entities=group["entities"],
            entity_ids=group["entity_ids"],
            field=group["field"],
            history=group["history"],
            active_idx=group["active_idx"],
        )

    def forward(self, batch: dict) -> AgentOutput:
        """`batch` holds the stacked Observation tensors + auxiliary inputs.

        Required keys: entities [B,12,D_ent], entity_ids [B,12,N_id], field [B,D_field],
        history [B,T,D_turn], active_idx [B,2], legality_mask [B,14] bool.
        Optional:
          opp_revealed  -- belief-net input dict (see belief.py contract). Required when
                           `use_belief_vec` is set; if absent otherwise, belief output is skipped.
          opp_view      -- the OPPONENT's-view observation group (same tensor keys as above) plus a
                           "my_view" sub-dict for the self-belief predictor (§3.3). Training only.

        `aux_into_trunk=False` detaches `s` before the belief/self-belief nets so their losses (and the
        b_vec path) cannot shape the shared backbone — the ablation switch from §5.
        """
        if self.blind_reactive:
            # §9 closure baseline: no memory (history=0) and no epistemic info (opp reveal tokens
            # blanked, so b_vec is computed from nothing). Architecture is UNCHANGED so the same
            # checkpoint loads — only the inputs are ablated.
            batch = dict(batch)
            batch["history"] = torch.zeros_like(batch["history"][:, :1])
            opp = batch.get("opp_revealed")
            if opp is not None:
                batch["opp_revealed"] = {"tokens": torch.full_like(opp["tokens"], -1), "masks": {}}
        s = self._backbone_pass(batch)

        # Set oracle (§3.4): condition the belief net on the opponent's TRUE tokens instead of the
        # running reveal state, so b_vec is a perfect-information summary.
        opp_revealed = batch.get("opp_revealed")
        if self.oracle in ("set", "action") and batch.get("opp_truth_tokens") is not None:
            from fog.models.oracle import apply_set_oracle

            opp_revealed = apply_set_oracle(opp_revealed or {}, batch["opp_truth_tokens"])

        belief_out: BeliefOutput | None = None
        if opp_revealed is None and self.use_belief_vec:
            raise ValueError("use_belief_vec=True requires batch['opp_revealed'].")
        if opp_revealed is not None:
            s_aux = s if self.aux_into_trunk else s.detach()
            belief_out = self.belief(s_aux, opp_revealed)

        # Action oracle (§3.4): append the opponent's committed action to s before the policy head.
        s_head = s
        if self.action_oracle is not None:
            s_head = self.action_oracle.augment(s, batch.get("opp_committed_action"))

        b_vec = belief_out.b_vec if (belief_out is not None and self.use_belief_vec) else None
        log_probs, value = self.policy_value(s_head, batch["legality_mask"], b_vec)

        self_belief_out: BeliefOutput | None = None
        opp_view = batch.get("opp_view")
        if opp_view is not None:
            s_opp = self._backbone_pass(opp_view)
            if not self.aux_into_trunk:
                s_opp = s_opp.detach()
            self_belief_out = self.self_belief(s_opp, opp_view["my_view"])

        return AgentOutput(log_probs=log_probs, value=value, belief=belief_out, self_belief=self_belief_out)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def assert_within_budget(self) -> None:
        n = self.n_params()
        if n > PARAM_BUDGET:
            raise ValueError(f"Model has {n:,} params > budget {PARAM_BUDGET:,} (spec §3/§11).")
