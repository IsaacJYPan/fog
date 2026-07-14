"""Assembled agent: backbone + belief + self-belief + heads (BELIEF_POLICY_PLAN §3).

This is the single nn.Module that training and inference instantiate. The backbone emits two vectors
(s_board, c_dyn); the policy/value head consumes both (inputs (a)+(b) via s_board, (c) via c_dyn) and
NO explicit belief summary (the old b_vec is gone, §3.6). The belief and self-belief chains cross-
attend to the SAME c_dyn — one vector, two consumers — so their auxiliary CE (with `aux_into_trunk`
on, the default) shapes c_dyn to be belief-informative for the policy. `aux_into_trunk` is the
ablation switch controlling whether the belief/self-belief losses backprop into the shared backbone.
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
    belief: BeliefOutput | None = None       # opponent-team belief, pass A / marginals (§3.2)
    self_belief: BeliefOutput | None = None  # leakage measurer output, pass A (§3.3; training only)
    # Pass B (BELIEF_POLICY_PLAN §3.3): random-order, fully-teacher-forced chain logits for the chain-rule
    # CE. Populated only during a training-mode forward that carries belief/self-belief targets; the
    # deployed b_vec and marginals always come from pass A above.
    belief_chain: BeliefOutput | None = None
    self_belief_chain: BeliefOutput | None = None


class FogAgent(nn.Module):
    def __init__(self, cfg: dict | None = None, sizes: dict | None = None) -> None:
        """`sizes` = RAW generated-table cardinalities (see `from_tables`). None keeps the scaffolding
        placeholder sizes — fine for unit tests, WRONG for real battles (real table indices exceed
        the placeholder embedding ranges and will silently alias or crash)."""
        super().__init__()
        cfg = cfg or {}
        model_cfg = cfg.get("model", {})
        self.aux_into_trunk = model_cfg.get("aux_into_trunk", True)
        self.oracle = model_cfg.get("oracle", "none")  # {none, set, action} — §3.4 eval ceilings
        # blind_reactive (§9 closure baseline): force history to 0 and zero the belief conditioning,
        # so the agent reacts only to the current board with no memory/belief. Same weights.
        self.blind_reactive = model_cfg.get("blind_reactive", False)
        # SEARCH_PLAN §4.2: give the policy/value head a posterior-features input (the opponent's
        # belief over MY hidden state). Default OFF — existing checkpoints load unchanged.
        self.use_posterior_features = model_cfg.get("use_posterior_features", False)
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
        # Belief nets cross-attend to c_dyn (§3.1), so their conditioning-memory width is dyn_dim.
        self.belief = BeliefModel(state_dim=self.backbone.dyn_dim, d=belief_dim, **belief_kwargs)
        self.self_belief = SelfBeliefPredictor(
            state_dim=self.backbone.dyn_dim, d=belief_dim, **belief_kwargs
        )
        # Action oracle appends a committed-action embedding to `s`; a dedicated head keeps the base
        # policy's weights comparable across oracle modes (§3.4).
        self.action_oracle = None
        extra = 0
        if self.oracle == "action":
            from fog.models.oracle import ActionOracleHead

            self.action_oracle = ActionOracleHead(self.backbone.state_dim, belief_dim)
            extra = self.action_oracle.extra_dim
        from fog.models.heads import N_POSTERIOR_FEATURES

        self.policy_value = PolicyValueHead(
            state_dim=self.backbone.state_dim + extra,
            dyn_dim=self.backbone.dyn_dim,
            posterior_dim=N_POSTERIOR_FEATURES if self.use_posterior_features else 0,
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

    def _backbone_pass(self, group: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """(s_board, c_dyn) for one observation group (§3.1)."""
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
          opp_revealed  -- belief-net input dict (see belief.py contract). Drives the belief aux loss
                           and diagnostics; the POLICY no longer needs it, so it may be absent.
          opp_view      -- the OPPONENT's-view observation group (same tensor keys as above) plus a
                           "my_view" sub-dict for the self-belief predictor (§3.3). Training only.

        `aux_into_trunk=False` detaches c_dyn before the belief/self-belief nets so their losses cannot
        shape the shared backbone — the ablation switch from §5. The policy always sees the live c_dyn.
        """
        if self.blind_reactive:
            # §9 closure baseline: no memory (history=0) and no epistemic info (opp reveal tokens
            # blanked, so the belief chain is fed nothing). Architecture is UNCHANGED so the same
            # checkpoint loads — only the inputs are ablated.
            batch = dict(batch)
            batch["history"] = torch.zeros_like(batch["history"][:, :1])
            opp = batch.get("opp_revealed")
            if opp is not None:
                batch["opp_revealed"] = {"tokens": torch.full_like(opp["tokens"], -1), "masks": {}}
        s_board, c_dyn = self._backbone_pass(batch)

        # Set oracle (§3.4): condition the belief net on the opponent's TRUE tokens instead of the
        # running reveal state, so the belief marginals are a perfect-information summary.
        opp_revealed = batch.get("opp_revealed")
        if self.oracle in ("set", "action") and batch.get("opp_truth_tokens") is not None:
            from fog.models.oracle import apply_set_oracle

            opp_revealed = apply_set_oracle(opp_revealed or {}, batch["opp_truth_tokens"])

        # Belief cross-attends to c_dyn (§3.1). The marginal forward is a diagnostic (dashboards /
        # pool_id fast path / leakage) — no longer a policy input; the CHAIN forward (pass B) is the
        # auxiliary loss + search proposal. `c_dyn_aux` carries the aux_into_trunk detach.
        belief_out: BeliefOutput | None = None
        belief_chain: BeliefOutput | None = None
        c_dyn_aux = c_dyn if self.aux_into_trunk else c_dyn.detach()
        if opp_revealed is not None:
            belief_out = self.belief(c_dyn_aux, opp_revealed)             # marginals (diagnostic)
            if self.training and batch.get("belief_targets") is not None:
                from fog.models.self_belief import tokens_from_labels

                truth = tokens_from_labels(batch["belief_targets"])       # aligned [B, 6, 9]
                belief_chain = self.belief(c_dyn_aux, opp_revealed, teacher_values=truth)

        # Action oracle (§3.4): append the opponent's committed action to s_board before the head.
        s_head = s_board
        if self.action_oracle is not None:
            s_head = self.action_oracle.augment(s_board, batch.get("opp_committed_action"))

        # Policy/value take (a)+(b) via s_head and (c) via the LIVE c_dyn — no belief summary (§3.6).
        log_probs, value = self.policy_value(
            s_head, batch["legality_mask"], c_dyn,
            posterior_features=batch.get("posterior_features"),
        )

        self_belief_out: BeliefOutput | None = None
        self_belief_chain: BeliefOutput | None = None
        opp_view = batch.get("opp_view")
        if opp_view is not None:
            _, c_dyn_opp = self._backbone_pass(opp_view)
            if not self.aux_into_trunk:
                c_dyn_opp = c_dyn_opp.detach()
            self_belief_out = self.self_belief(c_dyn_opp, opp_view["my_view"])  # marginals
            if self.training and batch.get("self_belief_targets") is not None:
                from fog.models.self_belief import tokens_from_labels

                my_truth = tokens_from_labels(batch["self_belief_targets"])
                self_belief_chain = self.self_belief(
                    c_dyn_opp, opp_view["my_view"], teacher_values=my_truth
                )

        return AgentOutput(
            log_probs=log_probs, value=value, belief=belief_out, self_belief=self_belief_out,
            belief_chain=belief_chain, self_belief_chain=self_belief_chain,
        )

    def head_inputs(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """(s_board, c_dyn) for direct `policy_value` calls — the cached-trunk path for joint search
        (SEARCH_PLAN §4.3): the backbone pass over a leaf batch runs ONCE; each regret-matching
        iteration then re-scores leaves through the cheap head with fresh posterior features. No
        belief forward here (§3.6: the policy no longer conditions on a belief summary); no
        oracle/blind handling — search never runs in those modes.

        Gradient contract: NOT decorated no_grad — full-scope distillation (training/distill.py)
        backprops through this path. Inference callers (search, precompute loops) wrap their
        calls in torch.no_grad(); a decorator here silently froze the trunk under distillation
        (caught by tests/test_distill.py::test_distill_scope_freezes_correctly)."""
        return self._backbone_pass(batch)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def assert_within_budget(self) -> None:
        n = self.n_params()
        if n > PARAM_BUDGET:
            raise ValueError(f"Model has {n:,} params > budget {PARAM_BUDGET:,} (spec §3/§11).")
