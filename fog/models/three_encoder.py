"""Three-part observation backbone with explicit information boundaries.

The current-state encoder sees only the public board, the turn-history encoder sees only the
chronological public event sequence, and the private-state encoder sees only the acting player's
six submitted sets. Policy/value use the private vector plus a fused public vector. Each
perspective's belief consumer combines that fused public vector with a learned projection of the
same perspective's private vector, so private mechanics can contextualize public battle evidence
without crossing the player-information boundary.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from fog.constants.schema import N_EV_UNITS, N_NATURES, N_TERA_TYPES
from fog.encoding.dims import DIMS
from fog.encoding.entities import (
    BASE_ID_FIELDS,
    ENTITY_RAW_DIM,
    EV_ID_FIELDS,
    N_ID_FIELDS,
    NATURE_ID_FIELD,
    ORIGINAL_ABILITY_ID_FIELD,
    ORIGINAL_ITEM_ID_FIELD,
    L,
)
from fog.encoding.events import (
    H_FIRST_ACTOR,
    H_MY_ABILITY,
    H_MY_ITEM,
    H_MY_MOVE,
    H_MY_SWITCH_PHASE,
    H_OPP_ABILITY,
    H_OPP_ITEM,
    H_OPP_MOVE,
    H_OPP_SWITCH_PHASE,
    N_HISTORY_ID_FIELDS,
)


class EntityEmbeddings(nn.Module):
    """Shared categorical tables used independently by all three encoders."""

    def __init__(
        self,
        n_species: int = 320,
        n_items: int = 64,
        n_abilities: int = 160,
        n_moves: int = 720,
        n_natures: int = N_NATURES + 1,
        n_ev_units: int = N_EV_UNITS + 1,
        species_dim: int = 64,
        small_dim: int = 24,
    ) -> None:
        super().__init__()
        self.species = nn.Embedding(n_species, species_dim)
        self.item = nn.Embedding(n_items, small_dim)
        self.ability = nn.Embedding(n_abilities, small_dim)
        self.move = nn.Embedding(n_moves, small_dim)
        self.nature = nn.Embedding(n_natures, small_dim)
        self.ev = nn.Embedding(n_ev_units, small_dim)
        self.native_stats = nn.Sequential(
            nn.Linear(7 * small_dim, 2 * small_dim),
            nn.GELU(),
            nn.Linear(2 * small_dim, small_dim),
        )
        nn.init.zeros_(self.native_stats[-1].weight)
        nn.init.zeros_(self.native_stats[-1].bias)
        # species, item, ability, move0..3, last move, compressed native nature/EV block
        self.out_dim = species_dim + small_dim * (2 + DIMS.n_move_slots + 1 + 1)

    def forward(self, entity_ids: torch.Tensor) -> torch.Tensor:
        assert entity_ids.shape[-1] >= BASE_ID_FIELDS
        species = self.species(entity_ids[..., 0])
        item = self.item(entity_ids[..., 1])
        ability = self.ability(entity_ids[..., 2])
        moves = self.move(entity_ids[..., 3 : 3 + DIMS.n_move_slots]).flatten(-2)
        last_move = self.move(entity_ids[..., 3 + DIMS.n_move_slots])
        nature = self.nature(entity_ids[..., NATURE_ID_FIELD])
        evs = self.ev(entity_ids[..., list(EV_ID_FIELDS)]).flatten(-2)
        native_stats = self.native_stats(torch.cat([nature, evs], dim=-1))
        return torch.cat(
            [species, item, ability, moves, last_move, native_stats], dim=-1
        )


class HistoryCategoricalEmbeddings(nn.Module):
    """Categorical identity sidecar for one completed public turn."""

    def __init__(self, embeds: EntityEmbeddings) -> None:
        super().__init__()
        self.tera = nn.Embedding(N_TERA_TYPES + 1, 16)
        self.actor = nn.Embedding(3, 8, padding_idx=0)
        self.phase = nn.Embedding(3, 8, padding_idx=0)
        species_dim = embeds.species.embedding_dim
        small_dim = embeds.move.embedding_dim
        self.species_fields = (0, 1, 3, 5, 6, 7)
        self.out_dim = 6 * species_dim + 6 * small_dim + 2 * 16 + 3 * 8

    @staticmethod
    def _known(table: nn.Embedding, ids: torch.Tensor) -> torch.Tensor:
        unknown = table.num_embeddings - 1
        safe = ids.clamp(min=0, max=unknown)
        return table(safe) * (ids != unknown).unsqueeze(-1)

    def forward(
        self, history_ids: torch.Tensor, embeds: EntityEmbeddings
    ) -> torch.Tensor:
        assert history_ids.shape[-1] == N_HISTORY_ID_FIELDS
        pieces = [
            self._known(embeds.species, history_ids[..., index])
            for index in self.species_fields
        ]
        pieces.extend(
            [
                self._known(embeds.move, history_ids[..., H_MY_MOVE]),
                self._known(embeds.move, history_ids[..., H_OPP_MOVE]),
                self._known(embeds.item, history_ids[..., H_MY_ITEM]),
                self._known(embeds.item, history_ids[..., H_OPP_ITEM]),
                self._known(embeds.ability, history_ids[..., H_MY_ABILITY]),
                self._known(embeds.ability, history_ids[..., H_OPP_ABILITY]),
                self._known(self.tera, history_ids[..., 12]),
                self._known(self.tera, history_ids[..., 13]),
                self.actor(history_ids[..., H_FIRST_ACTOR].clamp(min=0, max=2)),
                self.phase(
                    history_ids[..., H_MY_SWITCH_PHASE].clamp(min=0, max=2)
                ),
                self.phase(
                    history_ids[..., H_OPP_SWITCH_PHASE].clamp(min=0, max=2)
                ),
            ]
        )
        return torch.cat(pieces, dim=-1)


@dataclass
class EncoderContexts:
    """The three explicit encoder outputs, plus their inspectable entity tokens."""

    private_state: torch.Tensor
    public_state: torch.Tensor
    history: torch.Tensor
    public_state_tokens: torch.Tensor
    private_tokens: torch.Tensor
    # Lossless structured memory. These are legitimate current inputs carried alongside learned
    # features so downstream heads never depend exclusively on a pooled compression.
    public_exact_raw: torch.Tensor
    public_exact_ids: torch.Tensor
    private_exact_raw: torch.Tensor
    private_exact_ids: torch.Tensor
    field_exact: torch.Tensor
    history_tokens: torch.Tensor
    history_exact_raw: torch.Tensor
    history_exact_ids: torch.Tensor | None
    history_exact_state: torch.Tensor | None
    history_lengths: torch.Tensor


class CurrentStateEncoder(nn.Module):
    """Embed the volatile public board with per-entity MLPs and pooling."""

    def __init__(self, embedded_dim: int, d: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.entity_in = nn.Linear(ENTITY_RAW_DIM + embedded_dim, d)
        self.field_in = nn.Linear(DIMS.field_dim, d)
        # Preserve a short, directly learnable route for volatile field facts instead of forcing
        # them through the same pooled bottleneck as twelve entity records.
        self.field_skip = nn.Linear(DIMS.field_dim, out_dim, bias=False)
        self.entity_mlp = nn.Sequential(
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d, d),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(4 * d, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    @staticmethod
    def public_inputs(
        entities: torch.Tensor,
        entity_ids: torch.Tensor,
        embeds: EntityEmbeddings,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Remove acting-player sheet facts from the otherwise perspective-private input."""
        public_entities = entities.clone()
        public_ids = entity_ids.clone()
        own = slice(0, 6)

        public_ids[:, own, 1] = embeds.item.num_embeddings - 1
        public_ids[:, own, 2] = embeds.ability.num_embeddings - 1
        public_ids[:, own, 3 : 3 + DIMS.n_move_slots] = embeds.move.num_embeddings - 1
        public_ids[:, own, NATURE_ID_FIELD] = embeds.nature.num_embeddings - 1
        public_ids[:, own, list(EV_ID_FIELDS)] = embeds.ev.num_embeddings - 1
        if public_ids.shape[-1] >= N_ID_FIELDS:
            public_ids[:, own, ORIGINAL_ITEM_ID_FIELD] = embeds.item.num_embeddings - 1
            public_ids[:, own, ORIGINAL_ABILITY_ID_FIELD] = (
                embeds.ability.num_embeddings - 1
            )

        # PP is request-private. A sheet Tera type is public only after tera_used is set.
        public_entities[:, own, L.move_pp : L.move_pp + DIMS.move_pp] = 0.0
        tera_used = public_entities[:, own, L.tera_used : L.tera_used + 1] > 0.5
        tera = public_entities[:, own, L.tera_type : L.tera_type + DIMS.tera_type]
        public_entities[:, own, L.tera_type : L.tera_type + DIMS.tera_type] = (
            tera * tera_used.to(tera.dtype)
        )
        public_entities[
            :, own, L.epistemic : L.epistemic + DIMS.epistemic_flags
        ] = 0.0
        return public_entities, public_ids

    def forward(
        self,
        entities: torch.Tensor,
        entity_ids: torch.Tensor,
        field: torch.Tensor,
        active_idx: torch.Tensor,
        embeds: EntityEmbeddings,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        public_entities, public_ids = self.public_inputs(
            entities, entity_ids, embeds
        )
        tokens = self.entity_mlp(
            self.entity_in(
                torch.cat([public_entities, embeds(public_ids)], dim=-1)
            )
        )
        rows = torch.arange(entities.shape[0], device=entities.device)
        active_mine = tokens[rows, active_idx[:, 0]]
        active_opp = tokens[rows, active_idx[:, 1]]
        field_token = self.field_in(field)
        current = self.head(
            torch.cat(
                [tokens.mean(dim=1), active_mine, active_opp, field_token], dim=-1
            )
        ) + self.field_skip(field)
        return current, tokens


class PrivateStateEncoder(nn.Module):
    """Encode only the acting player's six full private team records."""

    def __init__(
        self,
        embedded_dim: int,
        original_dim: int,
        d: int,
        out_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.entity_in = nn.Linear(ENTITY_RAW_DIM + embedded_dim, d)
        self.original_in = nn.Linear(original_dim, d, bias=False)
        self.entity_mlp = nn.Sequential(
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d, d),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * d, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    @staticmethod
    def _private_raw(entities: torch.Tensor) -> torch.Tensor:
        """Keep private mechanics without duplicating the public current-state blocks."""
        raw = torch.zeros_like(entities[:, :6])
        raw[..., L.move_pp : L.move_pp + DIMS.move_pp] = entities[
            :, :6, L.move_pp : L.move_pp + DIMS.move_pp
        ]
        raw[..., L.tera_type : L.tera_type + DIMS.tera_type] = entities[
            :, :6, L.tera_type : L.tera_type + DIMS.tera_type
        ]
        raw[..., L.is_active] = entities[:, :6, L.is_active]
        raw[..., L.party_slot : L.party_slot + DIMS.party_slot] = entities[
            :, :6, L.party_slot : L.party_slot + DIMS.party_slot
        ]
        return raw

    def forward(
        self,
        entities: torch.Tensor,
        entity_ids: torch.Tensor,
        active_idx: torch.Tensor,
        embeds: EntityEmbeddings,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        own_ids = entity_ids[:, :6]
        private_raw = self._private_raw(entities)
        item_ids = own_ids[..., ORIGINAL_ITEM_ID_FIELD]
        ability_ids = own_ids[..., ORIGINAL_ABILITY_ID_FIELD]
        item_known = item_ids != embeds.item.num_embeddings - 1
        ability_known = ability_ids != embeds.ability.num_embeddings - 1
        original = torch.cat(
            [
                embeds.item(item_ids.clamp(0, embeds.item.num_embeddings - 1))
                * item_known.unsqueeze(-1),
                embeds.ability(
                    ability_ids.clamp(0, embeds.ability.num_embeddings - 1)
                )
                * ability_known.unsqueeze(-1),
            ],
            dim=-1,
        )
        tokens = self.entity_mlp(
            self.entity_in(torch.cat([private_raw, embeds(own_ids)], dim=-1))
            + self.original_in(original)
        )
        rows = torch.arange(entities.shape[0], device=entities.device)
        active = tokens[rows, active_idx[:, 0].clamp(min=0, max=5)]
        private = self.head(torch.cat([tokens.mean(dim=1), active], dim=-1))
        return private, tokens


class TurnHistoryEncoder(nn.Module):
    """Public sequence encoder with learned positions and a true causal mask."""

    def __init__(
        self,
        embeds: EntityEmbeddings,
        d: int,
        out_dim: int,
        cfg: dict,
        dropout: float,
    ) -> None:
        super().__init__()
        self.kind = str(cfg.get("history_encoder", "causal_transformer"))
        if self.kind == "transformer":
            self.kind = "causal_transformer"
        if self.kind not in {"causal_transformer", "gru"}:
            raise ValueError(
                "history_encoder must be 'causal_transformer' or 'gru'"
            )
        self.history_cap = int(cfg.get("history_cap", DIMS.history_cap))
        self.hist_in = nn.Linear(DIMS.turn_dim, d)
        self.categorical = HistoryCategoricalEmbeddings(embeds)
        self.history_cat_in = nn.Linear(self.categorical.out_dim, d, bias=False)
        self.history_state_in = nn.Linear(DIMS.hist_state, d, bias=False)
        self.position = nn.Embedding(self.history_cap, d)

        if self.kind == "gru":
            hidden = int(cfg.get("history_gru_hidden", out_dim))
            self.sequence = nn.GRU(d, hidden, batch_first=True)
            sequence_dim = hidden
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=d,
                nhead=int(cfg.get("history_heads", 4)),
                dim_feedforward=int(cfg.get("history_ff_dim", 4 * d)),
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.sequence = nn.TransformerEncoder(
                layer,
                num_layers=int(cfg.get("history_transformer_layers", 2)),
                # ``norm_first=True`` intentionally uses the standard padded
                # path; opting out explicitly avoids PyTorch emitting the same
                # unsupported nested-tensor warning for every model instance.
                enable_nested_tensor=False,
            )
            sequence_dim = d
        self.head = nn.Sequential(
            nn.Linear(sequence_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    @staticmethod
    def _right_pack(
        values: torch.Tensor,
        source: torch.Tensor,
        real: torch.Tensor,
    ) -> torch.Tensor:
        packed = values.gather(
            1, source.unsqueeze(-1).expand(-1, -1, values.shape[-1])
        )
        return packed.masked_fill(~real.unsqueeze(-1), 0)

    def encode_sequence(
        self,
        history: torch.Tensor,
        embeds: EntityEmbeddings,
        history_len: torch.Tensor | None = None,
        history_ids: torch.Tensor | None = None,
        history_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return all chronological token states for causality and padding audits."""
        batch, turns = history.shape[:2]
        if turns > self.history_cap:
            raise ValueError(
                f"history length {turns} exceeds configured cap {self.history_cap}"
            )
        if history_len is None:
            lengths = history.abs().sum(dim=-1).ne(0).sum(dim=-1)
        else:
            lengths = torch.as_tensor(
                history_len, device=history.device, dtype=torch.long
            )
        lengths = lengths.reshape(batch).clamp(min=0, max=turns)
        positions = torch.arange(turns, device=history.device).unsqueeze(0).expand(
            batch, -1
        )
        source = ((turns - lengths).unsqueeze(1) + positions).clamp(
            min=0, max=max(0, turns - 1)
        )
        real = positions < lengths.unsqueeze(1)
        history_right = self._right_pack(history, source, real)
        encoded_input = self.hist_in(history_right)

        if history_ids is not None:
            if history_ids.shape[:2] != history.shape[:2]:
                raise ValueError("history_ids must align with history [B,T]")
            ids_right = self._right_pack(history_ids, source, real)
            encoded_input = encoded_input + self.history_cat_in(
                self.categorical(ids_right, embeds)
            )
        if history_state is not None:
            if history_state.shape[:2] != history.shape[:2]:
                raise ValueError("history_state must align with history [B,T]")
            state_right = self._right_pack(history_state, source, real)
            encoded_input = encoded_input + self.history_state_in(state_right)
        encoded_input = (
            encoded_input + self.position(positions)
        ).masked_fill(~real.unsqueeze(-1), 0.0)

        safe_lengths = lengths.clamp_min(1)
        if self.kind == "gru":
            packed = nn.utils.rnn.pack_padded_sequence(
                encoded_input,
                safe_lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            encoded_packed, _ = self.sequence(packed)
            encoded, _ = nn.utils.rnn.pad_packed_sequence(
                encoded_packed, batch_first=True, total_length=turns
            )
        else:
            padding_mask = positions >= safe_lengths.unsqueeze(1)
            causal_mask = torch.triu(
                torch.ones(
                    turns,
                    turns,
                    dtype=torch.bool,
                    device=history.device,
                ),
                diagonal=1,
            )
            encoded = self.sequence(
                encoded_input,
                mask=causal_mask,
                src_key_padding_mask=padding_mask,
            )
        return encoded.masked_fill(~real.unsqueeze(-1), 0.0), lengths

    def forward(
        self,
        history: torch.Tensor,
        embeds: EntityEmbeddings,
        history_len: torch.Tensor | None = None,
        history_ids: torch.Tensor | None = None,
        history_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoded, lengths = self.encode_sequence(
            history, embeds, history_len, history_ids, history_state
        )
        return self.pool_encoded(encoded, lengths)

    def pool_encoded(
        self, encoded: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Pool the final real turn while keeping ``encode_sequence`` available to consumers."""
        rows = torch.arange(encoded.shape[0], device=encoded.device)
        final = encoded[rows, lengths.clamp_min(1) - 1]
        final = torch.where(
            lengths.unsqueeze(1) > 0, final, torch.zeros_like(final)
        )
        return self.head(final)


class Backbone(nn.Module):
    """Three explicit encoders: private, public state, and public history."""

    def __init__(self, cfg: dict | None = None, sizes: dict | None = None) -> None:
        super().__init__()
        cfg = cfg or {}
        hidden = int(cfg.get("entity_hidden", 128))
        self.entity_dim = hidden
        dropout = float(cfg.get("dropout", 0.1))
        self.state_dim = int(cfg.get("state_dim", 256))
        self.dyn_dim = int(cfg.get("dyn_dim", self.state_dim))
        self.current_dim = int(cfg.get("current_dim", self.dyn_dim))
        self.history_dim = int(cfg.get("history_dim", self.dyn_dim))

        if sizes:
            self.embeds = EntityEmbeddings(
                n_species=sizes["n_species"] + 1,
                n_items=sizes["n_items"] + 1,
                n_abilities=sizes["n_abilities"] + 1,
                n_moves=sizes["n_moves"] + 1,
                n_natures=sizes["n_natures"] + 1,
                n_ev_units=sizes["n_ev_units"] + 1,
            )
        else:
            self.embeds = EntityEmbeddings()

        self.current_state_encoder = CurrentStateEncoder(
            self.embeds.out_dim, hidden, self.current_dim, dropout
        )
        self.turn_history_encoder = TurnHistoryEncoder(
            self.embeds, hidden, self.history_dim, cfg, dropout
        )
        self.private_state_encoder = PrivateStateEncoder(
            self.embeds.out_dim,
            self.embeds.item.embedding_dim + self.embeds.ability.embedding_dim,
            hidden,
            self.state_dim,
            dropout,
        )
        self.public_fusion = nn.Sequential(
            nn.Linear(self.current_dim + self.history_dim, self.dyn_dim),
            nn.GELU(),
            nn.Linear(self.dyn_dim, self.dyn_dim),
        )
        self.preview_context = nn.Linear(self.current_dim, hidden, bias=False)

    @property
    def history_encoder(self) -> str:
        return self.turn_history_encoder.kind

    @property
    def entity_in(self) -> nn.Linear:
        """Legacy inspection alias for the public-current entity projection."""
        return self.current_state_encoder.entity_in

    def encode_contexts(
        self,
        entities: torch.Tensor,
        entity_ids: torch.Tensor,
        field: torch.Tensor,
        history: torch.Tensor,
        active_idx: torch.Tensor,
        history_len: torch.Tensor | None = None,
        history_ids: torch.Tensor | None = None,
        history_state: torch.Tensor | None = None,
    ) -> EncoderContexts:
        public_state, public_state_tokens = self.current_state_encoder(
            entities, entity_ids, field, active_idx, self.embeds
        )
        history_tokens, history_lengths = self.turn_history_encoder.encode_sequence(
            history, self.embeds, history_len, history_ids, history_state
        )
        history_context = self.turn_history_encoder.pool_encoded(
            history_tokens, history_lengths
        )
        private_state, private_tokens = self.private_state_encoder(
            entities, entity_ids, active_idx, self.embeds
        )
        public_exact_raw, public_exact_ids = CurrentStateEncoder.public_inputs(
            entities, entity_ids, self.embeds
        )
        return EncoderContexts(
            private_state=private_state,
            public_state=public_state,
            history=history_context,
            public_state_tokens=public_state_tokens,
            private_tokens=private_tokens,
            public_exact_raw=public_exact_raw,
            public_exact_ids=public_exact_ids,
            private_exact_raw=entities[:, :6],
            private_exact_ids=entity_ids[:, :6],
            field_exact=field,
            history_tokens=history_tokens,
            history_exact_raw=history,
            history_exact_ids=history_ids,
            history_exact_state=history_state,
            history_lengths=history_lengths,
        )

    def fuse_public(
        self, public_state: torch.Tensor, history: torch.Tensor
    ) -> torch.Tensor:
        """Create the derived public context used by policy/value and belief heads."""
        return self.public_fusion(torch.cat([public_state, history], dim=-1))

    def entity_tokens(
        self,
        entities: torch.Tensor,
        entity_ids: torch.Tensor,
        field: torch.Tensor,
    ) -> torch.Tensor:
        """Lead-selection tokens: public matchup context plus own private set context."""
        active_idx = torch.tensor(
            [[0, 6]], dtype=torch.long, device=entities.device
        ).expand(entities.shape[0], -1)
        current, current_tokens = self.current_state_encoder(
            entities, entity_ids, field, active_idx, self.embeds
        )
        _, private_tokens = self.private_state_encoder(
            entities, entity_ids, active_idx, self.embeds
        )
        tokens = current_tokens + self.preview_context(current).unsqueeze(1)
        tokens = tokens.clone()
        tokens[:, :6] = tokens[:, :6] + private_tokens
        return tokens

    def forward(
        self,
        entities: torch.Tensor,
        entity_ids: torch.Tensor,
        field: torch.Tensor,
        history: torch.Tensor,
        active_idx: torch.Tensor,
        history_len: torch.Tensor | None = None,
        history_ids: torch.Tensor | None = None,
        history_state: torch.Tensor | None = None,
    ) -> EncoderContexts:
        return self.encode_contexts(
            entities,
            entity_ids,
            field,
            history,
            active_idx,
            history_len,
            history_ids,
            history_state,
        )


__all__ = [
    "Backbone",
    "CurrentStateEncoder",
    "EncoderContexts",
    "EntityEmbeddings",
    "HistoryCategoricalEmbeddings",
    "PrivateStateEncoder",
    "TurnHistoryEncoder",
]
