"""Assembled agent: three-part backbone + belief/self-belief + policy/value heads.

The backbone explicitly returns the acting player's ``private_state`` context, volatile ``public_state``
context, and causal ``history`` context. Policy/value consume private plus a derived
``public_context`` fusion. Each perspective's belief model also receives its own private context:
damage, speed order, and action constraints are evidence about the opponent only when interpreted
relative to the observer's known item, nature, and EVs.  Private information never crosses
perspectives. ``aux_into_trunk`` controls whether auxiliary losses train the encoders and fusion.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from fog.encoding.dims import DIMS
from fog.encoding.entities import (
    ENTITY_RAW_DIM,
    N_ID_FIELDS,
    ORIGINAL_ABILITY_ID_FIELD,
    ORIGINAL_ITEM_ID_FIELD,
)
from fog.models.belief import BeliefModel, BeliefOutput
from fog.models.encoder import Backbone, EncoderContexts
from fog.models.heads import PolicyValueHead
from fog.models.self_belief import SelfBeliefPredictor
from fog.models.shared_latent import (
    ContextualLatentReader,
    OpponentContextualLatentReader,
    SharedBattleLatent,
)
from fog.models.structured_memory import StructuredMemoryEncoder, StructuredMemoryReader

# Parameter ceiling checked by assert_within_budget; token-memory models may exceed the older 5M limit.
PARAM_BUDGET = 50_000_000


@dataclass
class AgentOutput:
    log_probs: torch.Tensor
    value: torch.Tensor
    belief: BeliefOutput | None = None       # opponent-team field marginals (diagnostic)
    self_belief: BeliefOutput | None = None  # opponent-view field marginals (diagnostic)
    # Set-AR matched next-step outputs. Populated only when a training/evaluation batch carries full
    # truth; each row uses a random legal route and partial cut matching ancestral sampling.
    belief_chain: BeliefOutput | None = None
    self_belief_chain: BeliefOutput | None = None
    # Returned batch diagnostic used by reconstruction retention losses.  Keeping it in the
    # output avoids recomputing the shared trunk and does not retain mutable state on the module.
    shared_state: torch.Tensor | None = None


class TypedMatchupPrior(nn.Module):
    """Permutation-symmetric initial matchup prior over learned typed entity tokens."""

    def __init__(self, token_dim: int) -> None:
        super().__init__()
        self.entity = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
        )
        self.output = nn.Sequential(
            nn.Linear(64 * 2 + token_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        # Adding the optional branch is exactly behavior-preserving until explicitly trained.
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, entity_tokens: torch.Tensor, field_token: torch.Tensor) -> torch.Tensor:
        if entity_tokens.shape[1] != 12:
            raise ValueError("matchup prior requires twelve typed creature tokens")
        encoded = self.entity(entity_tokens)
        pooled = torch.cat(
            [encoded[:, :6].mean(1), encoded[:, 6:].mean(1), field_token], dim=-1
        )
        return self.output(pooled).squeeze(-1)


class FogAgent(nn.Module):
    def __init__(self, cfg: dict | None = None, sizes: dict | None = None) -> None:
        """Build from table cardinalities, excluding the unknown-token row.

        Use ``from_tables`` for battles. The default sizes support small unit-test fixtures
        only; real species and move IDs can exceed those embedding ranges.
        """
        super().__init__()
        cfg = cfg or {}
        model_cfg = cfg.get("model", {})
        self.aux_into_trunk = model_cfg.get("aux_into_trunk", True)
        self.oracle = model_cfg.get("oracle", "none")  # {none, set, action} — §3.4 eval ceilings
        # blind_reactive (§9 closure baseline): force history to 0 and zero the belief conditioning,
        # so the agent reacts only to the current board with no memory/belief. Same weights.
        self.blind_reactive = model_cfg.get("blind_reactive", False)
        # ARCHITECTURE.md invariant S2: give the policy/value head a posterior-features input (the opponent's
        # belief over MY hidden state). Default OFF — existing checkpoints load unchanged.
        self.use_posterior_features = model_cfg.get("use_posterior_features", False)
        belief_dim = model_cfg.get("belief_dim", 128)
        self.sizes = sizes

        # Belief nets take RAW sizes: their output heads emit n classes (no UNK class — you never
        # predict "unknown"), while their internal input embeddings allocate the +1 UNK row
        # themselves. The backbone embeddings get the +1 inside Backbone.
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
        self.shared_latent_enabled = bool(model_cfg.get("shared_latent_trunk", False))
        self.shared_latent = None
        self.shared_to_belief = None
        self.shared_to_policy = None
        self.shared_to_value = None
        self.matchup_value_prior = None
        self.matchup_value_prior_initial_only = bool(
            model_cfg.get("matchup_value_prior_initial_only", False)
        )
        shared_latent_dim = 0
        if self.shared_latent_enabled:
            shared_latent_dim = int(model_cfg.get("shared_latent_dim", 4096))
            contextual_shared = bool(model_cfg.get("shared_latent_contextual", False))
            self.shared_latent = SharedBattleLatent(
                self.backbone.embeds,
                history_dim=self.backbone.history_dim,
                latent_dim=shared_latent_dim,
                entity_dim=int(model_cfg.get("shared_entity_dim", 256)),
                field_dim=int(model_cfg.get("shared_field_dim", 128)),
                residual_dim=int(model_cfg.get("shared_residual_dim", 1024)),
                learned_only=bool(model_cfg.get("shared_latent_learned_only", False)),
                preserve_scale=bool(model_cfg.get("shared_latent_preserve_scale", False)),
                contextual=contextual_shared,
                contextual_layers=int(model_cfg.get("shared_contextual_layers", 3)),
                contextual_heads=int(model_cfg.get("shared_contextual_heads", 8)),
                contextual_ff_dim=int(model_cfg.get("shared_contextual_ff_dim", 512)),
                contextual_dynamic_residual=bool(
                    model_cfg.get("shared_contextual_dynamic_residual", False)
                ),
                contextual_data_global=bool(
                    model_cfg.get("shared_contextual_data_global", False)
                ),
                typed_world_tokens=bool(
                    model_cfg.get("shared_typed_world_tokens", False)
                ),
                recent_history_tokens=int(
                    model_cfg.get("shared_recent_history_tokens", 8)
                ),
                history_token_dim=self.backbone.entity_dim,
                contextual_dropout=float(
                    model_cfg.get("shared_contextual_dropout", 0.05)
                ),
                flat_autoencoder=bool(
                    model_cfg.get("shared_latent_flat_autoencoder", False)
                ),
                flat_hidden_dim=int(model_cfg.get("shared_flat_hidden_dim", 2048)),
                flat_layers=int(model_cfg.get("shared_flat_layers", 2)),
            )
            if contextual_shared:
                belief_reader = (
                    OpponentContextualLatentReader
                    if self.shared_latent.typed_world_tokens
                    else ContextualLatentReader
                )
                self.shared_to_belief = belief_reader(
                    shared_latent_dim,
                    self.shared_latent.token_dim,
                    self.backbone.dyn_dim,
                )
                if bool(model_cfg.get("shared_contextual_head_readers", False)):
                    # Belief, value, and policy all consume the same contextual token
                    # sequence.  Their readers are deliberately small task-specific queries;
                    # the expensive cross-Pokemon/field/history computation remains shared in
                    # ``SharedBattleLatent``.  Do not force policy/value through only the final
                    # 128-wide global token when the audited representation is all 16 tokens.
                    self.shared_to_policy = ContextualLatentReader(
                        shared_latent_dim,
                        self.shared_latent.token_dim,
                        self.shared_latent.token_dim,
                    )
                    self.shared_to_value = ContextualLatentReader(
                        shared_latent_dim,
                        self.shared_latent.token_dim,
                        self.shared_latent.token_dim,
                    )
                if bool(model_cfg.get("matchup_value_prior", False)):
                    if not self.shared_latent.typed_world_tokens:
                        raise ValueError("matchup_value_prior requires typed world tokens")
                    self.matchup_value_prior = TypedMatchupPrior(self.shared_latent.token_dim)
            else:
                self.shared_to_belief = nn.Linear(
                    shared_latent_dim, self.backbone.dyn_dim
                )
        if self.matchup_value_prior_initial_only and self.matchup_value_prior is None:
            raise ValueError(
                "matchup_value_prior_initial_only requires matchup_value_prior=true"
            )
        # Optional lossless head input.  The backbone still learns compact summaries, but it also
        # carries the exact legitimate observation records in EncoderContexts.  This reader lets
        # belief/value/policy attend to those records instead of forcing all information through a
        # mean-pooled 256-vector.  It does not add or infer any unavailable opponent information.
        self.structured_memory_enabled = bool(model_cfg.get("structured_memory", False))
        self.structured_encoder = None
        self.belief_structured_memory = None
        self.value_structured_memory = None
        self.policy_structured_memory = None
        self.structured_to_belief = None
        structured_dim = 0
        if self.structured_memory_enabled:
            public_dim = ENTITY_RAW_DIM + self.backbone.embeds.out_dim + N_ID_FIELDS
            private_dim = public_dim + 2 * self.backbone.embeds.item.embedding_dim
            history_dim = DIMS.turn_dim + DIMS.hist_id_fields + DIMS.hist_state
            token_dim = int(model_cfg.get("structured_token_dim", 512))
            self.structured_encoder = StructuredMemoryEncoder(
                public_dim=public_dim,
                private_dim=private_dim,
                field_dim=DIMS.field_dim,
                history_dim=history_dim,
                token_dim=token_dim,
            )
            # Separate task readers prevent an average that is useful for one objective from
            # becoming a new bottleneck for the other two objectives.
            self.belief_structured_memory = StructuredMemoryReader(token_dim)
            self.value_structured_memory = StructuredMemoryReader(token_dim)
            self.policy_structured_memory = StructuredMemoryReader(token_dim)
            structured_dim = self.belief_structured_memory.output_dim
            self.structured_to_belief = nn.Linear(
                structured_dim, self.backbone.dyn_dim, bias=False
            )
            # Architectural seeding must preserve the source checkpoint exactly.  Training the
            # belief reader later makes this route live without perturbing a seeded policy.
            nn.init.zeros_(self.structured_to_belief.weight)
        # Add the observer's private evidence to the public belief context without changing the
        # BeliefModel input shape. Zero initialization makes old checkpoints behavior-preserving:
        # compatible loading retains every existing tensor, while continuation learns this channel.
        self.private_to_belief = nn.Linear(
            self.backbone.state_dim, self.backbone.dyn_dim, bias=False
        )
        nn.init.zeros_(self.private_to_belief.weight)

        last_context_probability = float(
            model_cfg.get("belief_last_context_probability", 0.0)
        )
        self.belief = BeliefModel(
            state_dim=self.backbone.dyn_dim,
            d=belief_dim,
            last_context_probability=last_context_probability,
            **belief_kwargs,
        )
        self.self_belief = SelfBeliefPredictor(
            state_dim=self.backbone.dyn_dim,
            d=belief_dim,
            last_context_probability=last_context_probability,
            **belief_kwargs,
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

        action_feature_dim = 0
        if model_cfg.get("use_action_features", False):
            from fog.encoding.action_features import ACTION_FEATURE_DIM

            action_feature_dim = ACTION_FEATURE_DIM

        # Independent experiments may use the existing per-choice scorer with only public battle
        # facts.  The raw optional table also supports a rule-guided player and contains the exact
        # SimpleHeuristics choice, so erase every rule-decision column inside the model before the
        # scorer can consume it.  This is a non-persistent input mask, not a model parameter; old
        # configurations and checkpoints are unchanged.
        public_action_features_only = bool(
            model_cfg.get("public_action_features_only", False)
        )
        if public_action_features_only and not action_feature_dim:
            raise ValueError(
                "public_action_features_only requires model.use_action_features=true"
            )
        if public_action_features_only and float(
            model_cfg.get("simple_rule_prior_strength", 0.0)
        ) != 0.0:
            raise ValueError(
                "public-action-feature models require simple_rule_prior_strength=0"
            )
        public_action_feature_mask = None
        if public_action_features_only:
            from fog.encoding.action_features import ACTION_FEATURE_NAMES
            from fog.encoding.public_action_features import PUBLIC_ACTION_FEATURE_NAMES

            allowed = set(PUBLIC_ACTION_FEATURE_NAMES)
            public_action_feature_mask = torch.tensor(
                [float(name in allowed) for name in ACTION_FEATURE_NAMES],
                dtype=torch.float32,
            )
        self.register_buffer(
            "_public_action_feature_mask",
            public_action_feature_mask,
            persistent=False,
        )

        self.token_action_policy = bool(model_cfg.get("token_action_policy", False))
        self.action_query_policy = bool(model_cfg.get("action_query_policy", False))
        if self.action_query_policy and not self.token_action_policy:
            raise ValueError("action_query_policy requires token_action_policy=true")
        self.exact_action_context = bool(model_cfg.get("exact_action_context", False))
        if self.exact_action_context and not self.token_action_policy:
            raise ValueError("exact_action_context requires token_action_policy=true")
        action_context_dim = 0
        if self.token_action_policy:
            if self.exact_action_context:
                public_dim = ENTITY_RAW_DIM + self.backbone.embeds.out_dim
                private_dim = public_dim + 2 * self.backbone.embeds.item.embedding_dim
                action_context_dim = (
                    public_dim + private_dim + self.backbone.embeds.move.embedding_dim
                )
            else:
                action_entity_dim = (
                    self.shared_latent.token_dim
                    if self.shared_latent is not None and self.shared_latent.contextual
                    else self.backbone.entity_dim
                )
                action_context_dim = (
                    2 * action_entity_dim + self.backbone.embeds.move.embedding_dim
                )
            # Candidate semantics travel with the candidate, so permuting candidates and these
            # descriptors permutes logits.  No learned absolute action-index embedding is used.
            action_context_dim += 4
        self.policy_value = PolicyValueHead(
            state_dim=self.backbone.state_dim + extra,
            dyn_dim=self.backbone.dyn_dim,
            posterior_dim=N_POSTERIOR_FEATURES if self.use_posterior_features else 0,
            action_feature_dim=action_feature_dim,
            action_context_dim=action_context_dim,
            structured_dim=structured_dim,
            shared_state_dim=shared_latent_dim,
            shared_token_dim=(
                self.shared_latent.token_dim
                if self.shared_latent is not None and self.shared_latent.contextual
                else 0
            ),
            action_query_policy=self.action_query_policy,
            action_query_heads=int(model_cfg.get("action_query_heads", 4)),
            simple_rule_prior_strength=float(
                model_cfg.get("simple_rule_prior_strength", 0.0)
            ),
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
    def from_tables(cls, cfg: dict | None = None, tables=None) -> FogAgent:
        """Build with real cardinalities from the generated constants (§1). Use this — not the bare
        constructor — for anything touching real battles or parsed replays.

        Nature and EV cardinalities are fixed schema values: 25 natures and 64 effective EV units.
        """
        from fog.constants.schema import N_EV_UNITS, N_NATURES

        if tables is None:
            from fog.constants.tables import TABLES as tables
        sizes = {
            "n_species": tables.n_species,
            "n_items": tables.n_items,
            "n_abilities": tables.n_abilities,
            "n_moves": tables.n_moves,
            "n_natures": N_NATURES,
            "n_ev_units": N_EV_UNITS,
        }
        agent = cls(cfg, sizes=sizes)
        agent.assert_within_budget()
        return agent

    def encode_contexts(self, group: dict) -> EncoderContexts:
        """Return the explicit private, public-state, and history encoder outputs."""
        return self.backbone(
            entities=group["entities"],
            entity_ids=group["entity_ids"],
            field=group["field"],
            history=group["history"],
            active_idx=group["active_idx"],
            history_len=group.get("history_len"),
            history_ids=group.get("history_ids"),
            history_state=group.get("history_state"),
        )

    def public_context(self, contexts: EncoderContexts) -> torch.Tensor:
        """Fuse only the public-state and history outputs."""
        return self.backbone.fuse_public(contexts.public_state, contexts.history)

    def structured_context(
        self, contexts: EncoderContexts, consumer: str = "belief", memory=None
    ) -> torch.Tensor | None:
        """Read the exact structured observation without destroying the lossless sidecar."""
        readers = {
            "belief": self.belief_structured_memory,
            "value": self.value_structured_memory,
            "policy": self.policy_structured_memory,
        }
        if consumer not in readers:
            raise ValueError("structured-memory consumer must be belief, value, or policy")
        reader = readers[consumer]
        if reader is None:
            return None
        if memory is None:
            memory = self.structured_encoder(contexts, self.backbone.embeds)
        return reader(memory)

    def belief_context(
        self,
        contexts: EncoderContexts,
        structured_context: torch.Tensor | None = None,
        shared_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse evidence available to one observer for its opponent-belief prediction."""
        if self.shared_to_belief is not None:
            if shared_state is None:
                raise ValueError("shared belief context requires shared_state")
            return self.shared_to_belief(shared_state)
        result = self.public_context(contexts) + self.private_to_belief(contexts.private_state)
        if self.structured_to_belief is not None:
            if structured_context is None:
                structured_context = self.structured_context(contexts, "belief")
            result = result + self.structured_to_belief(structured_context)
        return result

    def action_context(
        self,
        contexts: EncoderContexts,
        batch: dict,
        shared_state: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Align exact private move/switch candidates and the active opponent to 14 actions."""
        if not self.token_action_policy:
            return None
        entity_ids = batch["entity_ids"]
        active_idx = batch["active_idx"]
        batch_size = entity_ids.shape[0]
        rows = torch.arange(batch_size, device=entity_ids.device)
        mine = active_idx[:, 0].clamp(min=0, max=5)
        opponent = active_idx[:, 1].clamp(min=6, max=11)
        if self.exact_action_context:
            public_memory = torch.cat(
                [
                    contexts.public_exact_raw,
                    self.backbone.embeds(contexts.public_exact_ids),
                ],
                dim=-1,
            )
            own_ids = contexts.private_exact_ids
            item_ids = own_ids[..., ORIGINAL_ITEM_ID_FIELD].clamp(
                0, self.backbone.embeds.item.num_embeddings - 1
            )
            ability_ids = own_ids[..., ORIGINAL_ABILITY_ID_FIELD].clamp(
                0, self.backbone.embeds.ability.num_embeddings - 1
            )
            private_memory = torch.cat(
                [
                    contexts.private_exact_raw,
                    self.backbone.embeds(own_ids),
                    self.backbone.embeds.item(item_ids),
                    self.backbone.embeds.ability(ability_ids),
                ],
                dim=-1,
            )
        elif self.shared_latent is not None and self.shared_latent.contextual:
            if shared_state is None:
                raise ValueError("contextual action policy requires shared_state")
            entity_memory = self.shared_latent.tokens(shared_state)[:, :12]
            public_memory = entity_memory
            private_memory = entity_memory[:, :6]
        else:
            public_memory = contexts.public_state_tokens
            private_memory = contexts.private_tokens
        active_token = private_memory[rows, mine]
        opponent_token = public_memory[rows, opponent]
        candidate = active_token.unsqueeze(1).expand(-1, 14, -1).clone()
        team_slots = torch.arange(6, device=entity_ids.device).unsqueeze(0).expand(
            batch_size, -1
        )
        bench = team_slots[team_slots != mine.unsqueeze(1)].reshape(batch_size, 5)
        candidate[:, 8:13] = private_memory.gather(
            1, bench.unsqueeze(-1).expand(-1, -1, private_memory.shape[-1])
        )
        move_ids = entity_ids[rows, mine, 3 : 3 + 4]
        move_tokens = self.backbone.embeds.move(move_ids)
        action_moves = move_tokens.new_zeros(batch_size, 14, move_tokens.shape[-1])
        action_moves[:, :4] = move_tokens
        action_moves[:, 4:8] = move_tokens
        action_type = action_moves.new_zeros(batch_size, 14, 4)
        action_type[:, :4, 0] = 1.0
        action_type[:, 4:8, 1] = 1.0
        action_type[:, 8:13, 2] = 1.0
        action_type[:, 13, 3] = 1.0
        return torch.cat(
            [
                candidate,
                opponent_token.unsqueeze(1).expand(-1, 14, -1),
                action_moves,
                action_type,
            ],
            dim=-1,
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

        `aux_into_trunk=False` detaches the fused belief context before belief/self-belief so their
        losses cannot shape either encoder path. The policy always sees the live paths.
        """
        if self.blind_reactive:
            # §9 closure baseline: no memory (history=0) and no epistemic info (opp reveal tokens
            # blanked, so the belief chain is fed nothing). Architecture is UNCHANGED so the same
            # checkpoint loads — only the inputs are ablated.
            batch = dict(batch)
            batch["history"] = torch.zeros_like(batch["history"][:, :1])
            if batch.get("history_ids") is not None:
                batch["history_ids"] = torch.zeros_like(batch["history_ids"][:, :1])
            if batch.get("history_state") is not None:
                batch["history_state"] = torch.zeros_like(batch["history_state"][:, :1])
            batch["history_len"] = torch.zeros(
                batch["history"].shape[0], dtype=torch.long, device=batch["history"].device
            )
            opp = batch.get("opp_revealed")
            if opp is not None:
                batch["opp_revealed"] = {"tokens": torch.full_like(opp["tokens"], -1), "masks": {}}
        contexts = self.encode_contexts(batch)
        public_context = self.public_context(contexts)
        matchup_entity_tokens = None
        matchup_field_token = None
        if self.shared_latent is not None:
            shared_result = self.shared_latent(
                batch,
                contexts,
                self.backbone.embeds,
                return_precontext=self.matchup_value_prior is not None,
            )
            if isinstance(shared_result, tuple):
                shared_state, matchup_entity_tokens, matchup_field_token = shared_result
            else:
                shared_state = shared_result
        else:
            shared_state = None
        structured_memory = (
            self.structured_encoder(contexts, self.backbone.embeds)
            if self.structured_encoder is not None
            else None
        )
        belief_structured_context = self.structured_context(
            contexts, "belief", structured_memory
        )
        value_structured_context = self.structured_context(
            contexts, "value", structured_memory
        )
        policy_structured_context = self.structured_context(
            contexts, "policy", structured_memory
        )
        policy_shared_context = (
            self.shared_to_policy(shared_state)
            if self.shared_to_policy is not None and shared_state is not None
            else None
        )
        value_shared_context = (
            self.shared_to_value(shared_state)
            if self.shared_to_value is not None and shared_state is not None
            else None
        )
        matchup_value_logit = (
            self.matchup_value_prior(matchup_entity_tokens, matchup_field_token)
            if self.matchup_value_prior is not None
            else None
        )
        if matchup_value_logit is not None and self.matchup_value_prior_initial_only:
            # The empirical matchup target is defined before play.  Until an immutable static-team
            # representation is separately validated, never leak that target into later-state
            # supervision or erase the ordinary value head's within-battle update.
            initial = batch["history_len"].eq(0).to(matchup_value_logit.dtype)
            matchup_value_logit = matchup_value_logit * initial

        # Set oracle (§3.4): condition the belief net on the opponent's TRUE tokens instead of the
        # running reveal state, so the belief marginals are a perfect-information summary.
        opp_revealed = batch.get("opp_revealed")
        if self.oracle in ("set", "action") and batch.get("opp_truth_tokens") is not None:
            from fog.models.oracle import apply_set_oracle

            opp_revealed = apply_set_oracle(opp_revealed or {}, batch["opp_truth_tokens"])

        # Belief combines public current/history with this observer's private context. This is a
        # diagnostic/auxiliary path, not a policy input. The second forward samples a matched set-AR
        # route/cut and supplies the auxiliary loss.
        belief_out: BeliefOutput | None = None
        belief_chain: BeliefOutput | None = None
        belief_context = self.belief_context(
            contexts, belief_structured_context, shared_state
        )
        belief_context_aux = (
            belief_context if self.aux_into_trunk else belief_context.detach()
        )
        if opp_revealed is not None:
            belief_out = self.belief(
                belief_context_aux, opp_revealed
            )                                                             # marginals (diagnostic)
            if ((self.training or batch.get("compute_belief_chain", False))
                    and batch.get("belief_targets") is not None):
                from fog.models.self_belief import tokens_from_labels

                truth = tokens_from_labels(batch["belief_targets"])       # aligned [B,6,15]
                belief_chain = self.belief(
                    belief_context_aux, opp_revealed, teacher_values=truth
                )

        # Action oracle (§3.4): append the opponent's committed action to the private context.
        policy_private = contexts.private_state
        if self.action_oracle is not None:
            policy_private = self.action_oracle.augment(
                contexts.private_state, batch.get("opp_committed_action")
            )

        # Policy/value take private context plus the live fused public context; no belief summary.
        action_features = batch.get("action_features")
        if self._public_action_feature_mask is not None and action_features is not None:
            action_features = action_features * self._public_action_feature_mask.to(
                dtype=action_features.dtype, device=action_features.device
            )
        log_probs, value = self.policy_value(
            policy_private, batch["legality_mask"], public_context,
            posterior_features=batch.get("posterior_features"),
            action_features=action_features,
            action_context=self.action_context(contexts, batch, shared_state),
            policy_structured_context=policy_structured_context,
            value_structured_context=value_structured_context,
            shared_state=shared_state,
            policy_shared_context=policy_shared_context,
            value_shared_context=value_shared_context,
            value_logit_bias=matchup_value_logit,
        )

        self_belief_out: BeliefOutput | None = None
        self_belief_chain: BeliefOutput | None = None
        opp_view = batch.get("opp_view")
        if opp_view is not None:
            opponent_contexts = self.encode_contexts(opp_view)
            opponent_shared_state = (
                self.shared_latent(opp_view, opponent_contexts, self.backbone.embeds)
                if self.shared_latent is not None
                else None
            )
            opponent_structured_memory = (
                self.structured_encoder(opponent_contexts, self.backbone.embeds)
                if self.structured_encoder is not None
                else None
            )
            opponent_structured_context = self.structured_context(
                opponent_contexts, "belief", opponent_structured_memory
            )
            self_belief_context = self.belief_context(
                opponent_contexts, opponent_structured_context, opponent_shared_state
            )
            if not self.aux_into_trunk:
                self_belief_context = self_belief_context.detach()
            self_belief_out = self.self_belief(
                self_belief_context, opp_view["my_view"]
            )                                                               # marginals
            if ((self.training or batch.get("compute_belief_chain", False))
                    and batch.get("self_belief_targets") is not None):
                from fog.models.self_belief import tokens_from_labels

                my_truth = tokens_from_labels(batch["self_belief_targets"])
                self_belief_chain = self.self_belief(
                    self_belief_context, opp_view["my_view"], teacher_values=my_truth
                )

        return AgentOutput(
            log_probs=log_probs, value=value, belief=belief_out, self_belief=self_belief_out,
            belief_chain=belief_chain, self_belief_chain=self_belief_chain,
            shared_state=shared_state,
        )

    def head_inputs(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """(private_state, public_context) for direct policy/value calls and joint-search caching.
        Per ARCHITECTURE.md invariant S2, the backbone pass over a leaf batch runs ONCE; each regret-matching
        iteration then re-scores leaves through the cheap head with fresh posterior features. No
        belief forward here (§3.6: the policy no longer conditions on a belief summary); no
        oracle/blind handling — search never runs in those modes.

        Gradient contract: NOT decorated no_grad — full-scope distillation (training/distill.py)
        backprops through this path. Inference callers (search, precompute loops) wrap their
        calls in torch.no_grad(); a decorator here silently froze the trunk under distillation
        (caught by tests/test_distill.py::test_distill_scope_freezes_correctly)."""
        contexts = self.encode_contexts(batch)
        return contexts.private_state, self.public_context(contexts)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def load_compatible_state_dict(self, state_dict: dict) -> dict:
        """Warm-start every parameter whose name and shape still match.

        Architectural research changes sometimes replace only the belief generator while leaving
        the trained backbone, policy, value, preview head, and even the belief value embeddings and
        output heads usable.  PyTorch's ``strict=False`` does not ignore shape mismatches, so a
        deliberate compatibility loader is required.  The returned manifest makes every retained,
        newly initialized, obsolete, and shape-mismatched tensor explicit; callers should record it
        beside the run configuration.
        """
        current = self.state_dict()
        compatible = {
            name: value
            for name, value in state_dict.items()
            if name in current and tuple(value.shape) == tuple(current[name].shape)
        }
        migrated: dict[str, dict] = {}

        # V2 checkpoints predate the explicit three encoders. Reuse semantically identical input
        # projections in both the public-current and private paths. The sequence projection and
        # categorical sidecars also retain their meanings; a transformer history checkpoint can
        # reuse transformer layers, while the usual GRU checkpoint intentionally initializes the
        # new causal transformer.
        legacy_names: dict[str, tuple[str, ...]] = {
            "backbone.entity_in.weight": (
                "backbone.current_state_encoder.entity_in.weight",
                "backbone.private_state_encoder.entity_in.weight",
            ),
            "backbone.entity_in.bias": (
                "backbone.current_state_encoder.entity_in.bias",
                "backbone.private_state_encoder.entity_in.bias",
            ),
            "backbone.private_sheet_in.weight": (
                "backbone.private_state_encoder.original_in.weight",
            ),
            "backbone.field_in.weight": (
                "backbone.current_state_encoder.field_in.weight",
            ),
            "backbone.field_in.bias": (
                "backbone.current_state_encoder.field_in.bias",
            ),
            "backbone.hist_in.weight": (
                "backbone.turn_history_encoder.hist_in.weight",
            ),
            "backbone.hist_in.bias": (
                "backbone.turn_history_encoder.hist_in.bias",
            ),
            "backbone.history_cat_in.weight": (
                "backbone.turn_history_encoder.history_cat_in.weight",
            ),
            "backbone.history_state_in.weight": (
                "backbone.turn_history_encoder.history_state_in.weight",
            ),
        }
        for old_name, new_names in legacy_names.items():
            old_value = state_dict.get(old_name)
            if old_value is None:
                continue
            for new_name in new_names:
                if new_name in current and tuple(old_value.shape) == tuple(current[new_name].shape):
                    compatible[new_name] = old_value
                    migrated[new_name] = {
                        "from": old_name,
                        "checkpoint": list(old_value.shape),
                        "model": list(current[new_name].shape),
                    }

        for old_name, old_value in state_dict.items():
            prefixes = (
                ("backbone.history_categorical.", "backbone.turn_history_encoder.categorical."),
                ("backbone.hist.layers.", "backbone.turn_history_encoder.sequence.layers."),
            )
            for old_prefix, new_prefix in prefixes:
                if not old_name.startswith(old_prefix):
                    continue
                new_name = new_prefix + old_name[len(old_prefix) :]
                if new_name in current and tuple(old_value.shape) == tuple(current[new_name].shape):
                    compatible[new_name] = old_value
                    migrated[new_name] = {
                        "from": old_name,
                        "checkpoint": list(old_value.shape),
                        "model": list(current[new_name].shape),
                    }

        # Native nature/EV flags inserted six entity-raw columns after older spread checkpoints.
        # Apply that one-way column migration before seeding both new entity encoders.
        old_entity_name = "backbone.entity_in.weight"
        if old_entity_name in state_dict:
            old = state_dict[old_entity_name]
            for new_name in (
                "backbone.current_state_encoder.entity_in.weight",
                "backbone.private_state_encoder.entity_in.weight",
            ):
                if new_name not in current:
                    continue
                new = current[new_name]
                if (
                    old.ndim == new.ndim == 2
                    and old.shape[0] == new.shape[0]
                    and old.shape[1] + 6 == new.shape[1]
                ):
                    from fog.encoding.entities import ENTITY_RAW_DIM, L

                    old_raw_dim = ENTITY_RAW_DIM - 6
                    ep = L.epistemic
                    if old.shape[1] - old_raw_dim == new.shape[1] - ENTITY_RAW_DIM:
                        merged = new.clone()
                        merged[:, :ep] = old[:, :ep]
                        merged[:, ep : ep + 6] = old[:, ep : ep + 6]
                        merged[:, ep + 6 : ep + 13] = 0
                        merged[:, ep + 13] = old[:, ep + 7]
                        merged[:, ep + 14 : ENTITY_RAW_DIM] = old[
                            :, ep + 8 : old_raw_dim
                        ]
                        merged[:, ENTITY_RAW_DIM:] = old[:, old_raw_dim:]
                        compatible[new_name] = merged
                        migrated[new_name] = {
                            "from": old_entity_name,
                            "checkpoint": list(old.shape),
                            "model": list(new.shape),
                            "inserted_zero_columns": list(range(ep + 6, ep + 13)),
                        }
                        if new_name.endswith("current_state_encoder.entity_in.weight"):
                            # Stable manifest alias used by native-schema audit tooling.
                            migrated[old_entity_name] = {
                                **migrated[new_name],
                                "to": new_name,
                            }
        mismatched = {
            name: {"checkpoint": list(value.shape), "model": list(current[name].shape)}
            for name, value in state_dict.items()
            if (
                name in current
                and tuple(value.shape) != tuple(current[name].shape)
                and name not in migrated
            )
        }
        obsolete = sorted(name for name in state_dict if name not in current)
        initialized = sorted(name for name in current if name not in compatible)
        self.load_state_dict(compatible, strict=False)
        return {
            "loaded": sorted(compatible),
            "initialized": initialized,
            "obsolete": obsolete,
            "mismatched": mismatched,
            "migrated": migrated,
        }

    def assert_within_budget(self) -> None:
        n = self.n_params()
        if n > PARAM_BUDGET:
            raise ValueError(f"Model has {n:,} params > budget {PARAM_BUDGET:,} (spec §3/§11).")
