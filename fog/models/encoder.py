"""Shared backbone encoder (spec §2.4). Produces the state vector `s in R^256`.

  * entity records -> linear(->128) -> 2 transformer encoder layers over the 12 entity tokens + field
    token (13 tokens), 4 heads, d=128. Positional info comes from the positional bits in each record.
  * history records -> linear(->128) -> GRU(1 layer, hidden 256)  [or 2-layer causal transformer;
    config switch, default GRU]
  * concat[ mean-pool(entity tokens) || active-mine token || active-opp token || history final ]
    -> MLP -> s in R^256

Backbone budget ~1.5M params. Embedding tables for species/item/ability/move/spread/tera live here so
the belief heads and the backbone share them.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from fog.encoding.dims import DIMS
from fog.encoding.entities import ENTITY_RAW_DIM, N_ID_FIELDS


class EntityEmbeddings(nn.Module):
    """Embedding tables for the integer id fields inside an entity record.

    Cardinalities come from the generated tables at build time. During scaffolding, defaults are
    placeholders; pass real sizes from `constants.tables` when wiring M1.
    """

    def __init__(
        self,
        n_species: int = 320,
        n_items: int = 64,
        n_abilities: int = 160,
        n_moves: int = 720,
        n_spreads: int = 17,
        species_dim: int = 64,
        small_dim: int = 24,
    ) -> None:
        super().__init__()
        self.species = nn.Embedding(n_species, species_dim)
        self.item = nn.Embedding(n_items, small_dim)
        self.ability = nn.Embedding(n_abilities, small_dim)
        self.move = nn.Embedding(n_moves, small_dim)  # shared by the 4 move slots AND last_used_move
        self.spread = nn.Embedding(n_spreads, small_dim)
        # id order: species, item, ability, move0..3, last_used_move, spread (see encoding/entities.py)
        self.out_dim = species_dim + small_dim * (2 + DIMS.n_move_slots + 1 + 1)

    def forward(self, entity_ids: torch.Tensor) -> torch.Tensor:
        """entity_ids [B, 12, N_ID_FIELDS] -> [B, 12, out_dim]."""
        assert entity_ids.shape[-1] == N_ID_FIELDS
        sp = self.species(entity_ids[..., 0])
        it = self.item(entity_ids[..., 1])
        ab = self.ability(entity_ids[..., 2])
        mv = self.move(entity_ids[..., 3 : 3 + DIMS.n_move_slots]).flatten(-2)
        last_mv = self.move(entity_ids[..., 3 + DIMS.n_move_slots])
        spread = self.spread(entity_ids[..., -1])
        return torch.cat([sp, it, ab, mv, last_mv, spread], dim=-1)


class Backbone(nn.Module):
    def __init__(self, cfg: dict | None = None, sizes: dict | None = None) -> None:
        """`sizes` carries RAW generated-table cardinalities (n_species, n_items, n_abilities,
        n_moves, n_spreads). The embedding tables get one extra row each: index n is the <UNKNOWN>
        id that encoding/entities.py writes for hidden fields. None -> placeholder defaults."""
        super().__init__()
        cfg = cfg or {}
        d = cfg.get("entity_hidden", 128)
        self.state_dim = cfg.get("state_dim", 256)

        if sizes:
            self.embeds = EntityEmbeddings(
                n_species=sizes["n_species"] + 1,
                n_items=sizes["n_items"] + 1,
                n_abilities=sizes["n_abilities"] + 1,
                n_moves=sizes["n_moves"] + 1,
                n_spreads=sizes["n_spreads"] + 1,
            )
        else:
            self.embeds = EntityEmbeddings()
        self.entity_in = nn.Linear(ENTITY_RAW_DIM + self.embeds.out_dim, d)
        self.field_in = nn.Linear(DIMS.field_dim, d)  # field as the 13th token
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.get("entity_heads", 4), dim_feedforward=4 * d, batch_first=True
        )
        self.entity_tf = nn.TransformerEncoder(enc_layer, num_layers=cfg.get("entity_transformer_layers", 2))

        self.history_encoder = cfg.get("history_encoder", "gru")
        self.hist_in = nn.Linear(DIMS.turn_dim, d)
        if self.history_encoder == "gru":
            self.hist = nn.GRU(d, cfg.get("history_gru_hidden", 256), batch_first=True)
            hist_out = cfg.get("history_gru_hidden", 256)
        else:  # 2-layer causal transformer over turns
            layer = nn.TransformerEncoderLayer(d_model=d, nhead=4, dim_feedforward=4 * d, batch_first=True)
            self.hist = nn.TransformerEncoder(layer, num_layers=2)
            hist_out = d

        # mean-pool(entities) || active-mine || active-opp || history-final
        self.head = nn.Sequential(
            nn.Linear(d * 3 + hist_out, self.state_dim),
            nn.ReLU(),
            nn.Linear(self.state_dim, self.state_dim),
        )

    def entity_tokens(
        self,
        entities: torch.Tensor,      # [B, 12, ENTITY_RAW_DIM]
        entity_ids: torch.Tensor,    # [B, 12, N_ID_FIELDS]
        field: torch.Tensor,         # [B, D_field]
    ) -> torch.Tensor:
        """Contextualized entity tokens [B, 12, d] (post entity-transformer, field token dropped).
        Consumed by the team-preview head; `forward` builds on the same pass."""
        ent = torch.cat([entities, self.embeds(entity_ids)], dim=-1)
        ent_tok = self.entity_in(ent)                             # [B, 12, d]
        fld_tok = self.field_in(field).unsqueeze(1)               # [B, 1, d]
        tokens = torch.cat([ent_tok, fld_tok], dim=1)             # [B, 13, d]
        return self.entity_tf(tokens)[:, :12]

    def forward(
        self,
        entities: torch.Tensor,      # [B, 12, ENTITY_RAW_DIM]
        entity_ids: torch.Tensor,    # [B, 12, N_ID_FIELDS]
        field: torch.Tensor,         # [B, D_field]
        history: torch.Tensor,       # [B, T, D_turn]
        active_idx: torch.Tensor,    # [B, 2] (mine, opp) token indices into the 12 entity slots
    ) -> torch.Tensor:
        b = entities.shape[0]
        ent_tokens = self.entity_tokens(entities, entity_ids, field)
        pooled = ent_tokens.mean(dim=1)
        ar = torch.arange(b, device=entities.device)
        active_mine = ent_tokens[ar, active_idx[:, 0]]
        active_opp = ent_tokens[ar, active_idx[:, 1]]

        h = self.hist_in(history)
        if self.history_encoder == "gru":
            _, hn = self.hist(h)
            hist_final = hn[-1]
        else:
            hist_final = self.hist(h)[:, -1]

        s = self.head(torch.cat([pooled, active_mine, active_opp, hist_final], dim=-1))
        return s
