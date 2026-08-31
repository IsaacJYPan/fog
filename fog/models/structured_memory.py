"""Wide structured observation memory and independent task readers."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class StructuredMemory:
    """Learned per-record latent tokens plus their padding mask.

    Exact inputs remain separately inspectable in ``EncoderContexts``.  These tokens are the
    learned representation whose information retention is tested with held-out decoders.
    """

    tokens: torch.Tensor
    padding: torch.Tensor
    public: torch.Tensor
    private: torch.Tensor
    field: torch.Tensor
    history: torch.Tensor


class StructuredMemoryEncoder(nn.Module):
    """Give every record an exact latent prefix plus a learned feature suffix."""

    def __init__(
        self,
        *,
        public_dim: int,
        private_dim: int,
        field_dim: int,
        history_dim: int,
        token_dim: int = 512,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        if self.token_dim < max(public_dim, private_dim, field_dim, history_dim):
            raise ValueError(
                "structured token width must be at least the widest exact input record "
                "to avoid a forced linear bottleneck"
            )
        self.input_dims = {
            "public": int(public_dim),
            "private": int(private_dim),
            "field": int(field_dim),
            "history": int(history_dim),
        }
        self.public = nn.Linear(public_dim, self.token_dim - public_dim)
        self.private = nn.Linear(private_dim, self.token_dim - private_dim)
        self.field = nn.Linear(field_dim, self.token_dim - field_dim)
        self.history = nn.Linear(history_dim, self.token_dim - history_dim)
        self.public_slots = nn.Embedding(12, self.token_dim)
        self.private_slots = nn.Embedding(6, self.token_dim)
        self.history_slots = nn.Embedding(60, self.token_dim)
        self.kind = nn.Embedding(4, self.token_dim)
        # There is deliberately no normalization or nonlinear operation over the exact prefix.
        # The learned suffix can change freely without ever corrupting an input value.

    @staticmethod
    def _token(
        values: torch.Tensor,
        learned: nn.Linear,
        metadata: torch.Tensor,
    ) -> torch.Tensor:
        width = values.shape[-1]
        return torch.cat([values, learned(values) + metadata[..., width:]], dim=-1)

    def forward(self, contexts, embeds) -> StructuredMemory:
        batch = contexts.public_exact_raw.shape[0]
        from fog.encoding.entities import (
            EV_ID_FIELDS,
            ORIGINAL_ABILITY_ID_FIELD,
            ORIGINAL_ITEM_ID_FIELD,
        )

        def normalized_ids(ids: torch.Tensor) -> torch.Tensor:
            scales = ids.new_tensor(
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
            return ids.float() / scales

        public_input = torch.cat(
            [
                contexts.public_exact_raw,
                embeds(contexts.public_exact_ids),
                normalized_ids(contexts.public_exact_ids),
            ],
            dim=-1,
        )
        own_ids = contexts.private_exact_ids

        original_item = embeds.item(
            own_ids[..., ORIGINAL_ITEM_ID_FIELD].clamp(0, embeds.item.num_embeddings - 1)
        )
        original_ability = embeds.ability(
            own_ids[..., ORIGINAL_ABILITY_ID_FIELD].clamp(0, embeds.ability.num_embeddings - 1)
        )
        private_input = torch.cat(
            [
                contexts.private_exact_raw,
                embeds(own_ids),
                original_item,
                original_ability,
                normalized_ids(own_ids),
            ],
            dim=-1,
        )
        public_metadata = self.public_slots.weight.unsqueeze(0) + self.kind.weight[0]
        public = self._token(public_input, self.public, public_metadata)
        private_metadata = self.private_slots.weight.unsqueeze(0) + self.kind.weight[1]
        private = self._token(private_input, self.private, private_metadata)
        field_input = contexts.field_exact
        field_metadata = self.kind.weight[2].expand(batch, -1)
        field = self._token(field_input, self.field, field_metadata).unsqueeze(1)
        history_pieces = [contexts.history_exact_raw]
        if contexts.history_exact_ids is not None:
            history_pieces.append(contexts.history_exact_ids.float())
        if contexts.history_exact_state is not None:
            history_pieces.append(contexts.history_exact_state)
        history_input = torch.cat(history_pieces, dim=-1)
        turns = history_input.shape[1]
        if turns > self.history_slots.num_embeddings:
            raise ValueError("structured history exceeds configured 60-turn memory")
        history_metadata = (
            self.history_slots.weight[:turns].unsqueeze(0) + self.kind.weight[3]
        )
        history = self._token(history_input, self.history, history_metadata)
        tokens = torch.cat([public, private, field, history], dim=1)

        prefix = torch.zeros(batch, 19, dtype=torch.bool, device=tokens.device)
        positions = torch.arange(turns, device=tokens.device).unsqueeze(0)
        history_padding = positions < (turns - contexts.history_lengths).unsqueeze(1)
        padding = torch.cat([prefix, history_padding], dim=1)
        return StructuredMemory(
            tokens=tokens,
            padding=padding,
            public=tokens[:, :12],
            private=tokens[:, 12:18],
            field=tokens[:, 18:19],
            history=tokens[:, 19:],
        )


class StructuredMemoryReader(nn.Module):
    """One task-specific query over the shared, information-preserving token memory."""

    def __init__(self, token_dim: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.empty(1, 1, token_dim))
        nn.init.normal_(self.query, std=0.02)
        heads = 8 if token_dim % 8 == 0 else 4
        self.attention = nn.MultiheadAttention(token_dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(token_dim)
        self.output_dim = token_dim

    def forward(self, memory: StructuredMemory) -> torch.Tensor:
        query = self.query.expand(memory.tokens.shape[0], -1, -1)
        attended, _ = self.attention(
            query,
            memory.tokens,
            memory.tokens,
            key_padding_mask=memory.padding,
            need_weights=False,
        )
        return self.norm(attended[:, 0])
