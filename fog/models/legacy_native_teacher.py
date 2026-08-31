"""Inference-only reconstruction of the pre-three-encoder native-EV policy."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from fog.encoding.dims import DIMS
from fog.encoding.entities import (
    ENTITY_RAW_DIM,
    ORIGINAL_ABILITY_ID_FIELD,
    ORIGINAL_ITEM_ID_FIELD,
)
from fog.models.heads import PolicyValueHead, TeamPreviewHead
from fog.models.three_encoder import EntityEmbeddings, HistoryCategoricalEmbeddings


class LegacyNativeBackbone(nn.Module):
    """Exact checkpoint-shaped board-transformer + GRU backbone used by the 200k teacher."""

    def __init__(self, cfg: dict, sizes: dict[str, int]) -> None:
        super().__init__()
        d = int(cfg.get("entity_hidden", 128))
        self.embeds = EntityEmbeddings(
            n_species=sizes["n_species"] + 1,
            n_items=sizes["n_items"] + 1,
            n_abilities=sizes["n_abilities"] + 1,
            n_moves=sizes["n_moves"] + 1,
            n_natures=sizes["n_natures"] + 1,
            n_ev_units=sizes["n_ev_units"] + 1,
        )
        self.entity_in = nn.Linear(ENTITY_RAW_DIM + self.embeds.out_dim, d)
        self.field_in = nn.Linear(DIMS.field_dim, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=int(cfg.get("entity_heads", 4)),
            dim_feedforward=4 * d,
            batch_first=True,
        )
        self.entity_tf = nn.TransformerEncoder(
            layer, num_layers=int(cfg.get("entity_transformer_layers", 2))
        )
        self.private_sheet_in = nn.Linear(
            self.embeds.item.embedding_dim + self.embeds.ability.embedding_dim,
            d,
            bias=False,
        )
        self.hist_in = nn.Linear(DIMS.turn_dim, d)
        self.history_categorical = HistoryCategoricalEmbeddings(self.embeds)
        self.history_cat_in = nn.Linear(
            self.history_categorical.out_dim, d, bias=False
        )
        self.history_state_in = nn.Linear(DIMS.hist_state, d, bias=False)
        hidden = int(cfg.get("history_gru_hidden", 256))
        self.hist = nn.GRU(d, hidden, batch_first=True)
        state_dim = int(cfg.get("state_dim", 256))
        dyn_dim = int(cfg.get("dyn_dim", state_dim))
        self.board_head = nn.Sequential(
            nn.Linear(3 * d, state_dim), nn.ReLU(), nn.Linear(state_dim, state_dim)
        )
        self.dyn_head = nn.Sequential(
            nn.Linear(hidden + DIMS.field_dim, dyn_dim),
            nn.ReLU(),
            nn.Linear(dyn_dim, dyn_dim),
        )

    def entity_tokens(
        self, entities: torch.Tensor, entity_ids: torch.Tensor, field: torch.Tensor
    ) -> torch.Tensor:
        tokens = self.entity_in(
            torch.cat([entities, self.embeds(entity_ids)], dim=-1)
        )
        original_item = entity_ids[:, :6, ORIGINAL_ITEM_ID_FIELD]
        original_ability = entity_ids[:, :6, ORIGINAL_ABILITY_ID_FIELD]
        original = torch.cat(
            [self.embeds.item(original_item), self.embeds.ability(original_ability)],
            dim=-1,
        )
        tokens = tokens.clone()
        tokens[:, :6] = tokens[:, :6] + self.private_sheet_in(original)
        field_token = self.field_in(field).unsqueeze(1)
        return self.entity_tf(torch.cat([tokens, field_token], dim=1))[:, :12]

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.entity_tokens(
            batch["entities"], batch["entity_ids"], batch["field"]
        )
        rows = torch.arange(tokens.shape[0], device=tokens.device)
        board = self.board_head(
            torch.cat(
                [
                    tokens.mean(dim=1),
                    tokens[rows, batch["active_idx"][:, 0]],
                    tokens[rows, batch["active_idx"][:, 1]],
                ],
                dim=-1,
            )
        )
        history = self.hist_in(batch["history"])
        if batch.get("history_ids") is not None:
            history = history + self.history_cat_in(
                self.history_categorical(batch["history_ids"], self.embeds)
            )
        if batch.get("history_state") is not None:
            history = history + self.history_state_in(batch["history_state"])
        _, hidden = self.hist(history)
        dynamics = self.dyn_head(torch.cat([hidden[-1], batch["field"]], dim=-1))
        return board, dynamics


@dataclass
class LegacyOutput:
    log_probs: torch.Tensor
    value: torch.Tensor


class LegacyNativeTeacher(nn.Module):
    def __init__(self, cfg: dict, sizes: dict[str, int]) -> None:
        super().__init__()
        merged = cfg.get("encoding", {}) | cfg.get("model", {})
        self.backbone = LegacyNativeBackbone(merged, sizes)
        posterior_dim = 4 if cfg.get("model", {}).get("use_posterior_features", False) else 0
        self.policy_value = PolicyValueHead(256, 256, posterior_dim=posterior_dim)
        self.preview = TeamPreviewHead(d=int(merged.get("entity_hidden", 128)))

    @classmethod
    def from_checkpoint(cls, cfg: dict, checkpoint: str, device="cpu"):
        from fog.constants.schema import N_EV_UNITS, N_NATURES
        from fog.constants.tables import TABLES

        sizes = {
            "n_species": TABLES.n_species,
            "n_items": TABLES.n_items,
            "n_abilities": TABLES.n_abilities,
            "n_moves": TABLES.n_moves,
            "n_natures": N_NATURES,
            "n_ev_units": N_EV_UNITS,
        }
        model = cls(cfg, sizes).to(device)
        state = torch.load(checkpoint, map_location=device)
        result = model.load_state_dict(state, strict=False)
        if result.missing_keys:
            raise RuntimeError(f"legacy teacher missing checkpoint tensors: {result.missing_keys}")
        return model.eval()

    def forward(self, batch: dict) -> LegacyOutput:
        board, dynamics = self.backbone(batch)
        log_probs, value = self.policy_value(
            board, batch["legality_mask"], dynamics
        )
        return LegacyOutput(log_probs, value)

    def preview_logits(self, batch: dict) -> torch.Tensor:
        return self.preview(
            self.backbone.entity_tokens(
                batch["entities"], batch["entity_ids"], batch["field"]
            )
        )
