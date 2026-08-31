"""One learned battle-state trunk shared by belief, value, and policy."""
from __future__ import annotations

import torch
from torch import nn

from fog.encoding.dims import DIMS
from fog.encoding.entities import (
    ENTITY_RAW_DIM,
    EV_ID_FIELDS,
    N_ID_FIELDS,
    ORIGINAL_ABILITY_ID_FIELD,
    ORIGINAL_ITEM_ID_FIELD,
)


class ContextualLatentReader(nn.Module):
    """Read one task-specific summary from a shared contextual token sequence.

    The expensive cross-record computation happens once in :class:`SharedBattleLatent`.
    A reader contributes only a learned query for the information its consumer needs.
    """

    def __init__(
        self,
        latent_dim: int,
        token_dim: int,
        output_dim: int,
        *,
        heads: int = 4,
    ) -> None:
        super().__init__()
        if latent_dim % token_dim:
            raise ValueError("contextual latent width must be divisible by token width")
        if token_dim % heads:
            raise ValueError("contextual token width must be divisible by reader heads")
        self.latent_dim = int(latent_dim)
        self.token_dim = int(token_dim)
        self.token_count = self.latent_dim // self.token_dim
        self.query = nn.Parameter(torch.empty(1, 1, self.token_dim))
        nn.init.normal_(self.query, std=0.02)
        self.attention = nn.MultiheadAttention(
            self.token_dim,
            heads,
            batch_first=True,
        )
        self.output = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, output_dim),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if tuple(latent.shape[1:]) != (self.latent_dim,):
            raise ValueError(
                f"contextual latent must have shape [batch,{self.latent_dim}]"
            )
        tokens = latent.reshape(latent.shape[0], self.token_count, self.token_dim)
        query = self.query.expand(tokens.shape[0], -1, -1)
        read, _ = self.attention(query, tokens, tokens, need_weights=False)
        return self.output(read[:, 0])


class OpponentContextualLatentReader(nn.Module):
    """Return one complete-state belief context per public opponent entity."""

    def __init__(
        self,
        latent_dim: int,
        token_dim: int,
        output_dim: int,
        *,
        heads: int = 4,
    ) -> None:
        super().__init__()
        if latent_dim % token_dim:
            raise ValueError("contextual latent width must be divisible by token width")
        if token_dim % heads:
            raise ValueError("contextual token width must be divisible by reader heads")
        self.latent_dim = int(latent_dim)
        self.token_dim = int(token_dim)
        self.token_count = self.latent_dim // self.token_dim
        self.query_offset = nn.Parameter(torch.empty(1, 6, self.token_dim))
        nn.init.normal_(self.query_offset, std=0.02)
        self.attention = nn.MultiheadAttention(
            self.token_dim, heads, batch_first=True
        )
        self.output = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, output_dim),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if tuple(latent.shape[1:]) != (self.latent_dim,):
            raise ValueError(
                f"contextual latent must have shape [batch,{self.latent_dim}]"
            )
        tokens = latent.reshape(latent.shape[0], self.token_count, self.token_dim)
        if self.token_count < 12:
            raise ValueError("opponent belief reader requires twelve creature tokens")
        # Start each query from its corresponding opponent-public Pokemon token, then let it attend
        # to every own/opponent/field/control/history/global token.
        query = tokens[:, 6:12] + self.query_offset
        read, _ = self.attention(query, tokens, tokens, need_weights=False)
        return self.output(query + read)


class FlatResidualBlock(nn.Module):
    """Global nonlinear computation inside a flat shared latent space."""

    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Linear(hidden, width),
        )
        # Start as an exact residual identity after the learned input projection.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.net(value)


class SharedBattleLatent(nn.Module):
    """Encode the complete legitimate observation into one fixed-width learned vector.

    There is no exact-value skip from inputs to outputs.  Information retention is established by
    a separately trained decoder that receives only this vector on held-out observations.
    """

    def __init__(
        self,
        embeds,
        *,
        history_dim: int,
        latent_dim: int = 4096,
        entity_dim: int = 256,
        field_dim: int = 128,
        residual_dim: int = 1024,
        learned_only: bool = False,
        preserve_scale: bool = False,
        contextual: bool = False,
        contextual_layers: int = 3,
        contextual_heads: int = 8,
        contextual_ff_dim: int = 512,
        contextual_dynamic_residual: bool = False,
        contextual_data_global: bool = False,
        typed_world_tokens: bool = False,
        recent_history_tokens: int = 8,
        history_token_dim: int | None = None,
        contextual_dropout: float = 0.05,
        flat_autoencoder: bool = False,
        flat_hidden_dim: int = 2048,
        flat_layers: int = 2,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.entity_dim = int(entity_dim)
        self.field_dim = int(field_dim)
        self.learned_only = bool(learned_only)
        self.preserve_scale = bool(preserve_scale)
        self.contextual = bool(contextual)
        self.contextual_dynamic_residual = bool(contextual_dynamic_residual)
        self.contextual_data_global = bool(contextual_data_global)
        self.typed_world_tokens = bool(typed_world_tokens)
        self.flat_autoencoder = bool(flat_autoencoder)
        semantic_dim = embeds.out_dim + embeds.item.embedding_dim + embeds.ability.embedding_dim
        exact_input_dim = ENTITY_RAW_DIM + N_ID_FIELDS
        if self.flat_autoencoder:
            if not self.learned_only:
                raise ValueError("flat shared autoencoder requires learned_only=true")
            if self.contextual:
                raise ValueError("flat shared autoencoder cannot also be contextual")
            if flat_hidden_dim < 1 or flat_layers < 0:
                raise ValueError("flat hidden width must be positive and layers nonnegative")
            self.flat_input_dim = (
                12 * (exact_input_dim + semantic_dim)
                + DIMS.field_dim
                + 12
                + 14
                + history_dim
            )
            self.flat_input_encoder = nn.Sequential(
                nn.Linear(self.flat_input_dim, int(flat_hidden_dim)),
                nn.GELU(),
                nn.Linear(int(flat_hidden_dim), self.latent_dim),
            )
            self.flat_blocks = nn.ModuleList(
                FlatResidualBlock(self.latent_dim, int(flat_hidden_dim))
                for _ in range(int(flat_layers))
            )
            self.exact_entity = None
            self.semantic_entity = None
            self.field = None
            self.global_context = None
            self.entity_encoder = None
            self.field_encoder = None
            self.history_encoder = None
            return
        if self.contextual:
            if not self.learned_only:
                raise ValueError("contextual shared trunk requires learned_only=true")
            self.token_dim = self.entity_dim
            self.recent_history_tokens = (
                int(recent_history_tokens) if self.typed_world_tokens else 0
            )
            if self.recent_history_tokens < 1 and self.typed_world_tokens:
                raise ValueError("typed world trunk requires at least one recent-history token")
            # Revised architecture: twelve distinct creatures, field, request/control,
            # several recent information-set events, a long-history summary, and global.
            self.token_count = (
                12 + 1 + 1 + self.recent_history_tokens + 1 + 1
                if self.typed_world_tokens
                else 16
            )
            expected = self.token_count * self.token_dim
            if self.latent_dim != expected:
                raise ValueError(
                    "contextual shared latent must equal token_count * shared_entity_dim "
                    f"({expected}), got {self.latent_dim}"
                )
            if self.token_dim % contextual_heads:
                raise ValueError(
                    "shared_entity_dim must be divisible by shared_contextual_heads"
                )
            hidden = max(2 * self.token_dim, 256)
            self.entity_encoder = nn.Sequential(
                nn.Linear(exact_input_dim + semantic_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.token_dim),
            )
            # Keep dynamic numeric facts from competing entirely with categorical semantics in
            # one projection. This is a learned nonlinear residual, never a raw-value bypass.
            self.dynamic_entity_encoder = (
                nn.Sequential(
                    nn.Linear(ENTITY_RAW_DIM, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, self.token_dim),
                )
                if self.contextual_dynamic_residual
                else None
            )
            if self.dynamic_entity_encoder is not None:
                nn.init.zeros_(self.dynamic_entity_encoder[-1].weight)
                nn.init.zeros_(self.dynamic_entity_encoder[-1].bias)
            self.field_encoder = nn.Sequential(
                nn.Linear(DIMS.field_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.token_dim),
            )
            self.history_encoder = nn.Sequential(
                nn.Linear(history_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.token_dim),
            )
            self.history_event_encoder = (
                nn.Sequential(
                    nn.Linear(int(history_token_dim or history_dim), hidden),
                    nn.GELU(),
                    nn.Linear(hidden, self.token_dim),
                )
                if self.typed_world_tokens
                else None
            )
            self.control_encoder = nn.Sequential(
                nn.Linear(12 + 14 + (6 if self.typed_world_tokens else 0), hidden),
                nn.GELU(),
                nn.Linear(hidden, self.token_dim),
            )
            # Compatibility name for old experimental checkpoints.
            self.metadata_encoder = self.control_encoder
            self.global_token = nn.Parameter(torch.empty(1, 1, self.token_dim))
            nn.init.normal_(self.global_token, std=0.02)
            exact_current_dim = 12 * exact_input_dim + DIMS.field_dim + 12 + 14
            self.global_state_encoder = (
                nn.Sequential(
                    nn.Linear(exact_current_dim, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, self.token_dim),
                )
                if self.contextual_data_global
                else None
            )
            if self.global_state_encoder is not None:
                nn.init.zeros_(self.global_state_encoder[-1].weight)
                nn.init.zeros_(self.global_state_encoder[-1].bias)
            self.token_position = nn.Embedding(self.token_count, self.token_dim)
            # Own-private creature, opponent-public creature, field, control,
            # recent history, long history, global.
            kind_count = 7 if self.typed_world_tokens else 6
            self.token_kind = nn.Embedding(kind_count, self.token_dim)
            kind_ids = (
                torch.tensor(
                    [0] * 6
                    + [1] * 6
                    + [2, 3]
                    + [4] * self.recent_history_tokens
                    + [5, 6]
                )
                if self.typed_world_tokens
                else torch.tensor([0] * 6 + [1] * 6 + [2, 3, 4, 5])
            )
            self.register_buffer("contextual_kind_ids", kind_ids, persistent=False)
            self.token_side = nn.Embedding(3, self.token_dim)
            side_ids = torch.tensor(
                [0] * 6 + [1] * 6 + [2] * (self.token_count - 12)
            )
            self.register_buffer("contextual_side_ids", side_ids, persistent=False)
            self.party_position = nn.Embedding(7, self.token_dim)
            party_ids = torch.tensor(
                list(range(6)) + list(range(6)) + [6] * (self.token_count - 12)
            )
            self.register_buffer("contextual_party_ids", party_ids, persistent=False)
            layer = nn.TransformerEncoderLayer(
                d_model=self.token_dim,
                nhead=contextual_heads,
                dim_feedforward=contextual_ff_dim,
                dropout=float(contextual_dropout),
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.contextual_encoder = nn.TransformerEncoder(
                layer,
                num_layers=contextual_layers,
                enable_nested_tensor=False,
            )
            self.exact_entity = None
            self.semantic_entity = None
            self.field = None
            self.global_context = None
            return
        if self.learned_only:
            # Every current-state fact must pass through a nonlinear learned encoder.  There is no
            # exact-value side channel and no analytically invertible projection in this mode.
            entity_hidden = max(self.entity_dim, 384)
            self.entity_encoder = nn.Sequential(
                nn.Linear(exact_input_dim + semantic_dim, entity_hidden),
                nn.GELU(),
                nn.Linear(entity_hidden, self.entity_dim),
                nn.LayerNorm(self.entity_dim),
            )
            self.field_encoder = nn.Sequential(
                nn.Linear(DIMS.field_dim + 12 + 14, self.field_dim),
                nn.GELU(),
                nn.Linear(self.field_dim, self.field_dim),
                nn.LayerNorm(self.field_dim),
            )
            self.history_encoder = nn.Sequential(
                nn.Linear(history_dim, history_dim),
                nn.GELU(),
                nn.LayerNorm(history_dim),
            )
            self.exact_entity = None
            self.semantic_entity = None
            self.field = None
            metadata_dim = 0
        else:
            self.exact_entity_dim = min(192, self.entity_dim - 1)
            self.semantic_entity_dim = self.entity_dim - self.exact_entity_dim
            # Compatibility mode retained for old experimental checkpoints.  New shared-trunk
            # experiments use learned_only=true and never instantiate this reversible path.
            self.exact_entity = nn.Linear(exact_input_dim, self.exact_entity_dim, bias=False)
            nn.init.orthogonal_(self.exact_entity.weight)
            self.semantic_entity = nn.Sequential(
                nn.Linear(semantic_dim, self.semantic_entity_dim),
                nn.LeakyReLU(0.1),
                nn.Linear(self.semantic_entity_dim, self.semantic_entity_dim),
                nn.LeakyReLU(0.1),
            )
            if self.field_dim < DIMS.field_dim:
                raise ValueError("shared field token must be wide enough for an injective projection")
            self.field = nn.Linear(DIMS.field_dim, self.field_dim, bias=False)
            nn.init.orthogonal_(self.field.weight)
            self.entity_encoder = None
            self.field_encoder = None
            self.history_encoder = None
            # Six active-slot bits per side plus fourteen exact legality bits.
            metadata_dim = 12 + 14
        self.entity_offset = 0
        self.entity_width = 12 * self.entity_dim
        self.field_offset = self.entity_width
        self.history_offset = self.field_offset + self.field_dim
        self.metadata_offset = self.history_offset + history_dim
        structured_dim = self.metadata_offset + metadata_dim
        self.global_dim = self.latent_dim - structured_dim
        if self.global_dim < 1:
            raise ValueError(
                f"shared latent {self.latent_dim} is too small for {structured_dim} structured dimensions"
            )
        # The first part of the single vector retains learned slot structure.  A learned global
        # suffix supplies cross-slot interactions without forcing exact slot facts through it.
        self.global_context = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Linear(structured_dim, residual_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(residual_dim, self.global_dim),
        )

    def tokens(self, latent: torch.Tensor) -> torch.Tensor:
        """Return the contextual token view without copying storage."""
        if not self.contextual:
            raise ValueError("token view is available only for the contextual shared trunk")
        if tuple(latent.shape[1:]) != (self.latent_dim,):
            raise ValueError(f"latent must have shape [batch,{self.latent_dim}]")
        return latent.reshape(latent.shape[0], self.token_count, self.token_dim)

    @staticmethod
    def normalized_ids(entity_ids: torch.Tensor, embeds) -> torch.Tensor:
        scales = entity_ids.new_tensor(
            [
                embeds.species.num_embeddings - 1,
                embeds.item.num_embeddings - 1,
                embeds.ability.num_embeddings - 1,
                *([embeds.move.num_embeddings - 1] * 5),
                embeds.nature.num_embeddings - 1,
                *([embeds.ev.num_embeddings - 1] * len(EV_ID_FIELDS)),
                embeds.item.num_embeddings - 1,
                embeds.ability.num_embeddings - 1,
            ],
            dtype=torch.float32,
        ).clamp_min(1.0)
        return entity_ids.float() / scales

    def forward(
        self, batch: dict, contexts, embeds, *, return_precontext: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ids = batch["entity_ids"]
        original_item = embeds.item(
            ids[..., ORIGINAL_ITEM_ID_FIELD].clamp(0, embeds.item.num_embeddings - 1)
        )
        original_ability = embeds.ability(
            ids[..., ORIGINAL_ABILITY_ID_FIELD].clamp(0, embeds.ability.num_embeddings - 1)
        )
        exact_input = torch.cat(
            [batch["entities"], self.normalized_ids(ids, embeds)], dim=-1
        )
        semantic_input = torch.cat(
            [embeds(ids), original_item, original_ability], dim=-1
        )
        active = torch.zeros(
            ids.shape[0], 12, dtype=batch["field"].dtype, device=batch["field"].device
        )
        active.scatter_(1, batch["active_idx"].clamp(0, 11), 1.0)
        legality = batch["legality_mask"].to(batch["field"].dtype)
        if self.flat_autoencoder:
            flat_input = torch.cat(
                [
                    exact_input.flatten(1),
                    semantic_input.flatten(1),
                    batch["field"],
                    active,
                    legality,
                    contexts.history,
                ],
                dim=-1,
            )
            if flat_input.shape[-1] != self.flat_input_dim:
                raise RuntimeError(
                    f"flat shared input has {flat_input.shape[-1]} values, "
                    f"expected {self.flat_input_dim}"
                )
            latent = self.flat_input_encoder(flat_input)
            for block in self.flat_blocks:
                latent = block(latent)
            return latent
        if self.contextual:
            entity_input = torch.cat([exact_input, semantic_input], dim=-1)
            entity = self.entity_encoder(entity_input)
            if self.dynamic_entity_encoder is not None:
                entity = entity + self.dynamic_entity_encoder(batch["entities"])
            field = self.field_encoder(batch["field"]).unsqueeze(1)
            history = self.history_encoder(contexts.history).unsqueeze(1)
            if self.typed_world_tokens:
                phase = self._decision_phase(batch, legality)
                control = self.control_encoder(
                    torch.cat([active, legality, phase], dim=-1)
                ).unsqueeze(1)
                recent = self._recent_history(contexts)
                recent = self.history_event_encoder(recent)
            else:
                control = self.control_encoder(
                    torch.cat([active, legality], dim=-1)
                ).unsqueeze(1)
            global_token = self.global_token.expand(ids.shape[0], -1, -1)
            if self.global_state_encoder is not None:
                exact_current = torch.cat(
                    [exact_input.flatten(1), batch["field"], active, legality], dim=-1
                )
                global_token = global_token + self.global_state_encoder(
                    exact_current
                ).unsqueeze(1)
            tokens = (
                torch.cat(
                    [entity, field, control, recent, history, global_token], dim=1
                )
                if self.typed_world_tokens
                else torch.cat([entity, field, history, control, global_token], dim=1)
            )
            positions = torch.arange(self.token_count, device=tokens.device)
            tokens = (
                tokens
                + self.token_position(positions).unsqueeze(0)
                + self.token_kind(self.contextual_kind_ids).unsqueeze(0)
                + self.token_side(self.contextual_side_ids).unsqueeze(0)
                + self.party_position(self.contextual_party_ids).unsqueeze(0)
            )
            latent = self.contextual_encoder(tokens).flatten(1)
            if return_precontext:
                # Learned typed creature/field tokens before cross-token contextualization.  This
                # is not an exact-input bypass: every value has passed through the same nonlinear
                # shared encoders used by the trunk.  It supports a permutation-symmetric matchup
                # prior without redundantly recomputing those encoders in a task head.
                return latent, entity, field.squeeze(1)
            return latent
        if self.learned_only:
            entity_input = torch.cat([exact_input, semantic_input], dim=-1)
            field_input = torch.cat([batch["field"], active, legality], dim=-1)
            if self.preserve_scale:
                # LayerNorm removes two sample-specific degrees of freedom (mean and scale).
                # Reconstruction-focused trunks retain the same learned nonlinear transforms but
                # skip that final destructive normalization; there is still no input bypass.
                entity = self.entity_encoder[:-1](entity_input).flatten(1)
                field = self.field_encoder[:-1](field_input)
                history = self.history_encoder[:-1](contexts.history)
            else:
                entity = self.entity_encoder(entity_input).flatten(1)
                field = self.field_encoder(field_input)
                history = self.history_encoder(contexts.history)
            structured = torch.cat([entity, field, history], dim=-1)
        else:
            entity = torch.cat(
                [self.exact_entity(exact_input), self.semantic_entity(semantic_input)], dim=-1
            ).flatten(1)
            field = self.field(batch["field"])
            structured = torch.cat([entity, field, contexts.history, active, legality], dim=-1)
        return torch.cat([structured, self.global_context(structured)], dim=-1)

    def _recent_history(self, contexts) -> torch.Tensor:
        """Select the most recent real event tokens in chronological order.

        ``TurnHistoryEncoder`` right-packs real events at the start of its sequence.  This
        selection therefore never reads padded or future positions.
        """
        sequence = contexts.history_tokens
        lengths = contexts.history_lengths.to(device=sequence.device, dtype=torch.long)
        keep = torch.minimum(
            lengths,
            torch.full_like(lengths, self.recent_history_tokens),
        )
        offsets = torch.arange(self.recent_history_tokens, device=sequence.device)
        source = lengths.unsqueeze(1) - keep.unsqueeze(1) + offsets.unsqueeze(0)
        valid = offsets.unsqueeze(0) < keep.unsqueeze(1)
        source = source.clamp(min=0, max=max(0, sequence.shape[1] - 1))
        selected = sequence.gather(
            1, source.unsqueeze(-1).expand(-1, -1, sequence.shape[-1])
        )
        return selected.masked_fill(~valid.unsqueeze(-1), 0.0)

    @staticmethod
    def _decision_phase(batch: dict, legality: torch.Tensor) -> torch.Tensor:
        """Return a six-way player-visible request phase.

        New collectors may provide ``decision_phase`` directly.  Legacy rows can only
        distinguish an ordinary move request from a forced switch using their request-derived
        legality mask; they are never assigned a more specific phase by guesswork.
        """
        supplied = batch.get("decision_phase")
        if supplied is None:
            move_available = legality[:, :8].bool().any(dim=-1)
            switch_available = legality[:, 8:13].bool().any(dim=-1)
            phase_ids = torch.where(
                ~move_available & switch_available,
                torch.ones_like(move_available, dtype=torch.long),
                torch.zeros_like(move_available, dtype=torch.long),
            )
        else:
            phase_ids = torch.as_tensor(
                supplied, device=legality.device, dtype=torch.long
            ).reshape(legality.shape[0])
            if bool(((phase_ids < 0) | (phase_ids >= 6)).any()):
                raise ValueError("decision_phase values must be in [0, 5]")
        return torch.nn.functional.one_hot(phase_ids, num_classes=6).to(legality.dtype)
