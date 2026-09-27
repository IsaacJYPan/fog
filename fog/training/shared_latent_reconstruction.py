"""Train and verify one learned state vector without giving its decoder raw inputs."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from fog.config import load_config
from fog.encoding.dims import DIMS
from fog.encoding.entities import L
from fog.eval.encoder_information_probe_live import TensorObservationDataset


def _digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _class_sizes(agent) -> list[int]:
    embeds = agent.backbone.embeds
    return [
        embeds.species.num_embeddings,
        embeds.item.num_embeddings,
        embeds.ability.num_embeddings,
        *([embeds.move.num_embeddings] * 5),
        embeds.nature.num_embeddings,
        *([embeds.ev.num_embeddings] * 6),
        embeds.item.num_embeddings,
        embeds.ability.num_embeddings,
    ]


class SharedLatentDecoder(nn.Module):
    """Decode exact current inputs from *only* the shared latent vector."""

    def __init__(
        self,
        latent_dim: int,
        class_sizes: list[int],
        *,
        slot_dim: int = 256,
        latent_entity_dim: int | None = None,
        raw_dim: int = 140,
        field_dim: int = 51,
        identity_weights: list[torch.Tensor] | None = None,
        contextual: bool = False,
        flat: bool = False,
        simple_flat: bool = False,
        linear_flat: bool = False,
        categorical_identity_heads: bool = False,
        linear_exact_values: bool = False,
    ) -> None:
        super().__init__()
        self.class_sizes = tuple(int(size) for size in class_sizes)
        self.raw_dim = int(raw_dim)
        self.field_dim = int(field_dim)
        self.slot_dim = int(slot_dim)
        self.latent_entity_dim = int(latent_entity_dim or slot_dim)
        self.contextual = bool(contextual)
        self.flat = bool(flat)
        self.simple_flat = bool(simple_flat)
        self.linear_flat = bool(linear_flat)
        self.categorical_identity_heads = bool(categorical_identity_heads)
        if self.simple_flat and not self.flat:
            raise ValueError("simple flat decoder requires flat=true")
        if self.categorical_identity_heads and not self.simple_flat:
            raise ValueError("categorical identity heads require the simple flat decoder")
        if self.linear_flat and not self.simple_flat:
            raise ValueError("linear flat decoding requires the simple flat decoder")
        self.linear_exact_values = bool(linear_exact_values)
        self.slot_projection = (
            nn.Identity() if self.latent_entity_dim == self.slot_dim
            else nn.Linear(self.latent_entity_dim, self.slot_dim)
        )
        if self.simple_flat:
            self.simple_flat_output_dim = (
                12 * raw_dim + 12 * len(self.class_sizes) + field_dim + 14 + 12
            )
            self.simple_flat_projection = (
                nn.Linear(latent_dim, self.simple_flat_output_dim)
                if self.linear_flat
                else nn.Sequential(
                    nn.Linear(latent_dim, latent_dim),
                    nn.GELU(),
                    nn.Linear(latent_dim, self.simple_flat_output_dim),
                )
            )
            self.simple_flat_identity_slots = (
                nn.Linear(latent_dim, 12 * self.slot_dim)
                if self.categorical_identity_heads else None
            )
            self.simple_flat_ids = (
                nn.ModuleList(
                    nn.Linear(self.slot_dim, size) for size in self.class_sizes
                )
                if self.categorical_identity_heads else None
            )
            self.flat_slot_projection = None
        elif self.flat:
            self.flat_slot_projection = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, 12 * self.slot_dim),
            )
        else:
            self.flat_slot_projection = None
            self.simple_flat_projection = None
            self.simple_flat_identity_slots = None
            self.simple_flat_ids = None
        if self.contextual:
            self.context_projection = nn.Linear(latent_dim, self.slot_dim)
            # Training-only high-capacity probe: each reconstructed creature
            # gets an independent linear read of the complete latent.  Without
            # this, twelve slots compete inside one shared 256-wide context and
            # decoder failure can be mistaken for encoder information loss.
            self.full_slot_projection = nn.Linear(latent_dim, 12 * self.slot_dim)
            self.slot_position = nn.Embedding(12, self.slot_dim)
            self.slot_fusion = nn.Sequential(
                nn.LayerNorm(self.slot_dim),
                nn.Linear(self.slot_dim, self.slot_dim),
                nn.GELU(),
                nn.Linear(self.slot_dim, self.slot_dim),
            )
            if latent_dim % self.latent_entity_dim:
                raise ValueError(
                    "contextual decoder requires latent width divisible by token width"
                )
            self.memory_token_count = latent_dim // self.latent_entity_dim
            self.memory_projection = (
                nn.Identity()
                if self.latent_entity_dim == self.slot_dim
                else nn.Linear(self.latent_entity_dim, self.slot_dim)
            )
            self.memory_position = nn.Embedding(self.memory_token_count, self.slot_dim)
            heads = 8 if self.slot_dim % 8 == 0 else 4
            self.cross_attention = nn.MultiheadAttention(
                self.slot_dim,
                heads,
                batch_first=True,
            )
            self.cross_norm = nn.LayerNorm(self.slot_dim)
            self.cross_ff = nn.Sequential(
                nn.Linear(self.slot_dim, 2 * self.slot_dim),
                nn.GELU(),
                nn.Linear(2 * self.slot_dim, self.slot_dim),
            )
            self.cross_ff_norm = nn.LayerNorm(self.slot_dim)
        else:
            self.context_projection = None
            self.full_slot_projection = None
            self.slot_position = None
            self.slot_fusion = None
            self.memory_token_count = 0
            self.memory_projection = None
            self.memory_position = None
            self.cross_attention = None
            self.cross_norm = None
            self.cross_ff = None
            self.cross_ff_norm = None
        self.raw = nn.Sequential(
            nn.Linear(self.slot_dim, self.slot_dim), nn.LeakyReLU(0.1),
            nn.Linear(self.slot_dim, raw_dim),
        )
        self.tied_identity = identity_weights is not None
        if identity_weights is None:
            self.ids = nn.ModuleList(nn.Linear(self.slot_dim, size) for size in self.class_sizes)
            self.id_queries = None
        else:
            if len(identity_weights) != len(self.class_sizes):
                raise ValueError("one fixed identity table is required per entity id field")
            self.ids = None
            self.id_queries = nn.ModuleList(
                nn.Linear(self.slot_dim, int(weight.shape[1])) for weight in identity_weights
            )
            for index, weight in enumerate(identity_weights):
                if int(weight.shape[0]) != self.class_sizes[index]:
                    raise ValueError("identity table size does not match encoded class count")
                self.register_buffer(f"identity_weight_{index}", weight.detach().clone())
        # Stable table indices are part of the exact encoded input.  Regressing their normalized
        # values tests invertibility even when a held-out team contains a category the temporary
        # decoder never saw during training; categorical heads alone cannot make that distinction.
        self.id_values = nn.Linear(self.slot_dim, len(self.class_sizes))
        self.field = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.LeakyReLU(0.1), nn.Linear(256, field_dim)
        )
        self.legality = nn.Linear(latent_dim, 14)
        self.active_mine = nn.Linear(latent_dim, 6)
        self.active_opp = nn.Linear(latent_dim, 6)

    def forward(self, latent: torch.Tensor) -> dict[str, object]:
        if self.simple_flat:
            if self.linear_flat:
                hidden = latent
                packed = self.simple_flat_projection(latent)
            else:
                hidden = self.simple_flat_projection[1](
                    self.simple_flat_projection[0](latent)
                )
                packed = self.simple_flat_projection[2](hidden)
            offset = 0
            raw_width = 12 * self.raw_dim
            raw_logits = packed[:, offset : offset + raw_width].reshape(
                -1, 12, self.raw_dim
            )
            offset += raw_width
            id_width = 12 * len(self.class_sizes)
            id_values = packed[:, offset : offset + id_width].reshape(
                -1, 12, len(self.class_sizes)
            )
            offset += id_width
            field_logits = packed[:, offset : offset + self.field_dim]
            offset += self.field_dim
            legality = packed[:, offset : offset + 14]
            offset += 14
            active_mine = packed[:, offset : offset + 6]
            offset += 6
            active_opp = packed[:, offset : offset + 6]
            if self.categorical_identity_heads:
                identity_slots = self.simple_flat_identity_slots(hidden).reshape(
                    latent.shape[0], 12, self.slot_dim
                )
                identity_logits = [head(identity_slots) for head in self.simple_flat_ids]
            else:
                identity_logits = []
                for index, size in enumerate(self.class_sizes):
                    positions = torch.arange(
                        size, device=latent.device, dtype=latent.dtype
                    )
                    predicted_index = id_values[..., index].unsqueeze(-1) * float(size - 1)
                    identity_logits.append(-(predicted_index - positions).square())
            return {
                "raw": raw_logits if self.linear_exact_values else _raw_values(raw_logits),
                "raw_logits": raw_logits,
                "ids": identity_logits,
                "id_queries": None,
                "identity_weights": None,
                "id_values": id_values,
                "field": field_logits if self.linear_exact_values else torch.sigmoid(field_logits),
                "field_logits": field_logits,
                "legality": legality,
                "active_mine": active_mine,
                "active_opp": active_opp,
            }
        if self.flat:
            slots = self.flat_slot_projection(latent).reshape(
                latent.shape[0], 12, self.slot_dim
            )
        else:
            slots = latent[:, : 12 * self.latent_entity_dim].reshape(
                latent.shape[0], 12, self.latent_entity_dim
            )
            slots = self.slot_projection(slots)
        if self.contextual:
            positions = self.slot_position.weight.unsqueeze(0)
            context = self.context_projection(latent).unsqueeze(1)
            full_slots = self.full_slot_projection(latent).reshape(
                latent.shape[0], 12, self.slot_dim
            )
            slots = slots + full_slots + self.slot_fusion(
                slots + full_slots + context + positions
            )
            memory = latent.reshape(
                latent.shape[0], self.memory_token_count, self.latent_entity_dim
            )
            memory = self.memory_projection(memory) + self.memory_position.weight.unsqueeze(0)
            attended, _ = self.cross_attention(
                slots + positions,
                memory,
                memory,
                need_weights=False,
            )
            slots = self.cross_norm(slots + attended)
            slots = self.cross_ff_norm(slots + self.cross_ff(slots))
        if self.tied_identity:
            identity_logits = []
            identity_queries = []
            for index, query_head in enumerate(self.id_queries):
                query = F.normalize(query_head(slots), dim=-1)
                weights = F.normalize(getattr(self, f"identity_weight_{index}"), dim=-1)
                identity_queries.append(query)
                identity_logits.append(20.0 * query @ weights.transpose(0, 1))
        else:
            identity_logits = [head(slots) for head in self.ids]
            identity_queries = None
        raw_logits = self.raw(slots)
        field_logits = self.field(latent)
        return {
            "raw": raw_logits if self.linear_exact_values else _raw_values(raw_logits),
            "raw_logits": raw_logits,
            "ids": identity_logits,
            "id_queries": identity_queries,
            "identity_weights": (
                [getattr(self, f"identity_weight_{index}") for index in range(len(self.class_sizes))]
                if self.tied_identity else None
            ),
            "id_values": self.id_values(slots),
            "field": field_logits if self.linear_exact_values else torch.sigmoid(field_logits),
            "field_logits": field_logits,
            "legality": self.legality(latent),
            "active_mine": self.active_mine(latent),
            "active_opp": self.active_opp(latent),
        }

    def initialize_linear_flat_identity_control(self) -> None:
        """Make the evaluation-only linear control an exact layout-preserving copy.

        A positive control should validate the recovery metric, not ask an optimizer to discover a
        1,961-value permutation from a small correlated battle dataset. The deployable trunk never
        calls this method. It is valid only with the direct-input layout produced below.
        """
        if not (self.simple_flat and self.linear_flat):
            raise ValueError("identity control requires the linear simple-flat decoder")
        projection = self.simple_flat_projection
        with torch.no_grad():
            projection.weight.zero_()
            projection.bias.zero_()
            width = min(projection.out_features, projection.in_features)
            projection.weight[:width, :width] = torch.eye(
                width,
                dtype=projection.weight.dtype,
                device=projection.weight.device,
            )


_RAW_BLOCKS = (
    ("bounded", L.base_stats, DIMS.base_stats),
    ("binary", L.typing, DIMS.typing),
    ("categorical", L.weight, DIMS.weight_bucket),
    ("bounded", L.hp, DIMS.hp_fraction),
    ("binary", L.fainted, DIMS.fainted_flag),
    ("categorical", L.status, DIMS.status),
    ("bounded", L.counters, DIMS.sleep_tox_counters),
    ("signed", L.boosts, DIMS.boosts),
    ("binary", L.volatiles, DIMS.volatiles),
    ("bounded", L.vol_counters, DIMS.volatile_counters),
    ("binary", L.paradox, DIMS.paradox_boost),
    ("binary", L.booster, DIMS.booster_consumed),
    ("bounded", L.fallen, DIMS.fallen_count),
    ("bounded", L.move_pp, DIMS.move_pp),
    ("binary", L.tera_type, DIMS.tera_type),
    ("binary", L.tera_used, DIMS.tera_used_flag),
    ("binary", L.epistemic, DIMS.epistemic_flags),
    ("binary", L.is_mine, DIMS.is_mine),
    ("binary", L.is_active, DIMS.is_active),
    ("categorical", L.party_slot, DIMS.party_slot),
)
_RAW_BLOCK_NAMES = (
    "base_stats", "typing", "weight_bucket", "hp", "fainted", "status",
    "status_counters", "boosts", "volatiles", "volatile_counters", "paradox_boost",
    "booster_consumed", "fallen_count", "move_pp", "tera_type", "tera_used",
    "epistemic_flags", "is_mine", "is_active", "party_slot",
)


def _raw_values(logits: torch.Tensor) -> torch.Tensor:
    """Apply the correct range transform to every documented entity-input block."""
    pieces = []
    for kind, offset, width in _RAW_BLOCKS:
        block = logits[..., offset : offset + width]
        if kind == "categorical":
            block = block.softmax(-1)
        elif kind == "signed":
            block = block.tanh()
        else:
            block = block.sigmoid()
        pieces.append(block)
    result = torch.cat(pieces, dim=-1)
    if result.shape[-1] != logits.shape[-1]:
        raise RuntimeError("raw reconstruction layout does not cover every entity column")
    return result


def _weighted_mean(loss: torch.Tensor, target: torch.Tensor, rare_value_weight: float) -> torch.Tensor:
    if rare_value_weight <= 1:
        return loss.mean()
    nonzero = target.abs() > 1e-6
    rate = nonzero.float().mean()
    if float(rate) <= 0.0 or float(rate) >= 1.0:
        return loss.mean()
    multiplier = ((1.0 - rate) / rate).clamp(1.0, rare_value_weight)
    weights = torch.where(nonzero, multiplier.to(loss.dtype), torch.ones_like(loss))
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def _tail_mean(values: torch.Tensor, fraction: float) -> torch.Tensor:
    """Stable worst-field pressure: optimize a group of weak fields, not one oscillating maximum."""
    if fraction <= 0:
        return values.mean()
    count = max(1, min(values.numel(), round(values.numel() * fraction)))
    return values.topk(count).values.mean()


def _checkpoint_selection_score(
    *,
    epoch_index: int,
    mse_warmup_epochs: int,
    team: dict,
    battle: dict,
    recovery_score: float,
) -> tuple[str, float]:
    """Choose checkpoints with the objective that is actually being optimized.

    During the MSE warmup, strict exact-recovery gates commonly remain at zero even while the
    decoder is learning rapidly. Selecting and early-stopping on that zero discarded every later
    warmup update and made an exact-input control look incapable of copying its own input.  The
    warmup therefore selects the lowest held-out reconstruction loss; the strict recovery gate
    takes over only after warmup ends.
    """
    if epoch_index < mse_warmup_epochs:
        heldout_loss = 0.5 * (float(team["loss"]) + float(battle["loss"]))
        return "mse_warmup", -heldout_loss
    return "strict_recovery", float(recovery_score)


def _raw_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    worst_field_weight: float = 0.0,
    rare_value_weight: float = 0.0,
    tail_fraction: float = 0.25,
    linear_exact_values: bool = False,
    categorical_cross_entropy: bool = False,
    priority_blocks: tuple[str, ...] = (),
    priority_weight: float = 1.0,
) -> torch.Tensor:
    """Give each semantic input block equal weight instead of diluting rare facts over 140 columns."""
    losses = []
    block_weights = []
    per_field = []
    for block_name, (kind, offset, width) in zip(_RAW_BLOCK_NAMES, _RAW_BLOCKS):
        block = logits[..., offset : offset + width]
        truth = target[..., offset : offset + width]
        if kind == "categorical" and categorical_cross_entropy:
            value = F.cross_entropy(block.flatten(0, -2), truth.argmax(-1).flatten())
            losses.append(value)
            per_field.append(value)
        elif linear_exact_values:
            element = (block - truth).square()
            value = _weighted_mean(element, truth, rare_value_weight)
            losses.append(value)
            per_field.extend(
                _weighted_mean(element[..., i], truth[..., i], rare_value_weight)
                for i in range(width)
            )
        elif kind == "categorical":
            value = F.cross_entropy(block.flatten(0, -2), truth.argmax(-1).flatten())
            losses.append(value)
            per_field.append(value)
        elif kind == "binary":
            element = F.binary_cross_entropy_with_logits(block, truth, reduction="none")
            value = _weighted_mean(element, truth, rare_value_weight)
            losses.append(value)
            per_field.extend(_weighted_mean(element[..., i], truth[..., i], rare_value_weight) for i in range(width))
        elif kind == "signed":
            element = (block.tanh() - truth).square()
            value = _weighted_mean(element, truth, rare_value_weight)
            losses.append(value)
            per_field.extend(_weighted_mean(element[..., i], truth[..., i], rare_value_weight) for i in range(width))
        else:
            element = (block.sigmoid() - truth).square()
            value = _weighted_mean(element, truth, rare_value_weight)
            losses.append(value)
            per_field.extend(_weighted_mean(element[..., i], truth[..., i], rare_value_weight) for i in range(width))
        block_weights.append(
            priority_weight if block_name in priority_blocks else 1.0
        )
    field_values = torch.stack(per_field)
    weights = field_values.new_tensor(block_weights)
    block_loss = (torch.stack(losses) * weights).sum() / weights.sum().clamp_min(1.0)
    return block_loss + worst_field_weight * _tail_mean(field_values, tail_fraction)


_CONTINUOUS_FIELD_COLUMNS = frozenset((5, 11, 13, 14, 19, 20, 28, 33, 35, 36, 40, 42, 43, 45, 47, 48, 50))
_IDENTITY_FIELD_NAMES = (
    "species", "item", "ability", "move0", "move1", "move2", "move3", "move4",
    "nature", "ev_hp", "ev_atk", "ev_def", "ev_spa", "ev_spd", "ev_spe",
    "original_item", "original_ability",
)


def _weakest_components(report: dict, limit: int = 12) -> list[dict[str, float | str]]:
    """Return the lowest legitimate recovery components for concise live diagnostics."""
    components: list[tuple[str, float]] = []
    for block_name, values in report["raw_semantic_recovery"].items():
        if not isinstance(values, list):
            values = [values]
        components.extend(
            (f"raw.{block_name}[{index}]", float(value))
            for index, value in enumerate(values)
        )
    components.extend(
        (f"field[{index}]", float(value))
        for index, value in enumerate(report["field_semantic_recovery_by_field"])
    )
    components.extend(
        (f"identity.{name}", float(value))
        for name, value in zip(_IDENTITY_FIELD_NAMES, report["id_accuracy_by_field"])
    )
    components.extend(
        (
            ("legality.bit", float(report["legality_bit_accuracy"])),
            ("legality.row", float(report["legality_row_accuracy"])),
            ("active.mine", float(report["active_mine_accuracy"])),
            ("active.opponent", float(report["active_opp_accuracy"])),
        )
    )
    return [
        {"component": name, "recovery": value}
        for name, value in sorted(components, key=lambda item: item[1])[:limit]
    ]


def _field_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    worst_field_weight: float,
    rare_value_weight: float,
    tail_fraction: float,
    linear_exact_values: bool = False,
) -> torch.Tensor:
    per_field = []
    for index in range(target.shape[-1]):
        if linear_exact_values:
            element = (logits[..., index] - target[..., index]).square()
        elif index in _CONTINUOUS_FIELD_COLUMNS:
            element = (logits[..., index].sigmoid() - target[..., index]).square()
        else:
            element = F.binary_cross_entropy_with_logits(
                logits[..., index], target[..., index], reduction="none"
            )
        per_field.append(_weighted_mean(element, target[..., index], rare_value_weight))
    values = torch.stack(per_field)
    return values.mean() + worst_field_weight * _tail_mean(values, tail_fraction)


def _loss(
    predicted: dict[str, object], batch: dict, *, id_weight: float,
    id_regression_weight: float = 5.0, raw_weight: float = 1.0,
    field_weight: float = 1.0, worst_field_weight: float = 0.0,
    rare_value_weight: float = 0.0, identity_embedding_weight: float = 0.0,
    tail_fraction: float = 0.25,
    ordinal_ev_decoder: bool = False,
    ordinal_all_identities: bool = False,
    cumulative_all_identities: bool = False,
    linear_id_values: bool = False,
    linear_exact_values: bool = False,
    categorical_cross_entropy: bool = False,
    code_scaled_identities: bool = False,
    priority_raw_blocks: tuple[str, ...] = (),
    priority_raw_weight: float = 1.0,
    legality_weight: float = 1.0,
    active_weight: float = 1.0,
) -> torch.Tensor:
    id_losses = (
        [
            F.cross_entropy(
                logits.flatten(0, 1), batch["entity_ids"][..., index].flatten()
            )
            for index, logits in enumerate(predicted["ids"])
        ]
        if id_weight > 0
        else []
    )
    scales = batch["entity_ids"].new_tensor(
        [logits.shape[-1] - 1 for logits in predicted["ids"]], dtype=torch.float32
    ).clamp_min(1.0)
    # Numeric index regression remains available as a diagnostic/optional auxiliary. Exact
    # categorical recovery is judged by the classification heads for every identity field.
    # A sigmoid cannot exactly reproduce normalized identity codes at the 0/1
    # endpoints without infinite logits.  The linear option is therefore the
    # appropriate invertibility probe; evaluation clamps only after decoding.
    predicted_id_values = (
        predicted["id_values"] if linear_id_values else predicted["id_values"].sigmoid()
    )
    normalized_target = batch["entity_ids"].float() / scales
    if code_scaled_identities:
        # Optimize error in exact table-index units rather than normalized [0,1]
        # units.  One wrong species index must remain a meaningful error even
        # when the table contains hundreds of entries.
        element = F.smooth_l1_loss(
            predicted_id_values * scales,
            batch["entity_ids"].float(),
            reduction="none",
        )
        per_field = element.flatten(0, 1).mean(0)
        id_regression = per_field.mean() + worst_field_weight * _tail_mean(
            per_field, tail_fraction
        )
    elif cumulative_all_identities:
        # Each identity is an ordered integer table index.  A cumulative ordinal
        # target gives a well-scaled exact-bin objective: unlike normalized
        # regression it cannot hide a one-index error, and unlike the old
        # Gaussian softmax its loss does not grow quadratically with table size.
        cumulative_losses = []
        for field_index in range(len(predicted["ids"])):
            scale = int(scales[field_index].item())
            predicted_bin = predicted_id_values[..., field_index] * float(scale)
            thresholds = torch.arange(
                scale, device=predicted_bin.device, dtype=predicted_bin.dtype
            ) + 0.5
            logits = predicted_bin.unsqueeze(-1) - thresholds
            targets = (
                batch["entity_ids"][..., field_index].unsqueeze(-1) > thresholds
            ).to(logits.dtype)
            cumulative_losses.append(F.binary_cross_entropy_with_logits(logits, targets))
        cumulative_values = torch.stack(cumulative_losses)
        id_regression = cumulative_values.mean() + worst_field_weight * _tail_mean(
            cumulative_values, tail_fraction
        )
    elif ordinal_ev_decoder or ordinal_all_identities:
        ordinal_losses = []
        ordinal_fields = range(len(predicted["ids"])) if ordinal_all_identities else range(9, 15)
        for field_index in ordinal_fields:
            scale = scales[field_index]
            predicted_bin = predicted_id_values[..., field_index] * scale
            positions = torch.arange(
                int(scale.item()) + 1, device=predicted_bin.device, dtype=predicted_bin.dtype
            )
            ordinal_logits = -(predicted_bin.unsqueeze(-1) - positions).square()
            ordinal_losses.append(F.cross_entropy(
                ordinal_logits.flatten(0, -2),
                batch["entity_ids"][..., field_index].flatten(),
            ))
        ordinal_values = torch.stack(ordinal_losses)
        id_regression = ordinal_values.mean() + worst_field_weight * _tail_mean(
            ordinal_values, tail_fraction
        )
    else:
        id_regression = F.smooth_l1_loss(predicted_id_values, normalized_target)
    if id_losses:
        id_loss_values = torch.stack(id_losses)
        id_loss = id_loss_values.mean() + worst_field_weight * _tail_mean(
            id_loss_values, tail_fraction
        )
    else:
        id_loss = batch["field"].new_zeros(())
    identity_embedding = batch["field"].new_zeros(())
    if predicted.get("id_queries") is not None and identity_embedding_weight > 0:
        embedding_losses = []
        for index, query in enumerate(predicted["id_queries"]):
            weights = F.normalize(predicted["identity_weights"][index], dim=-1)
            target_query = weights[batch["entity_ids"][..., index]]
            embedding_losses.append((1.0 - (query * target_query).sum(-1)).mean())
        embedding_values = torch.stack(embedding_losses)
        identity_embedding = embedding_values.mean() + worst_field_weight * _tail_mean(
            embedding_values, tail_fraction
        )
    return (
        raw_weight * _raw_loss(
            predicted["raw_logits"], batch["entities"],
            worst_field_weight=worst_field_weight,
            rare_value_weight=rare_value_weight,
            tail_fraction=tail_fraction,
            linear_exact_values=linear_exact_values,
            categorical_cross_entropy=categorical_cross_entropy,
            priority_blocks=priority_raw_blocks,
            priority_weight=priority_raw_weight,
        )
        + field_weight * _field_loss(
            predicted["field_logits"], batch["field"],
            worst_field_weight=worst_field_weight,
            rare_value_weight=rare_value_weight,
            tail_fraction=tail_fraction,
            linear_exact_values=linear_exact_values,
        )
        + legality_weight * F.binary_cross_entropy_with_logits(
            predicted["legality"], batch["legality_mask"].float()
        )
        + active_weight * F.cross_entropy(
            predicted["active_mine"], batch["active_idx"][:, 0]
        )
        + active_weight * F.cross_entropy(
            predicted["active_opp"], batch["active_idx"][:, 1] - 6
        )
        + id_weight * id_loss
        + identity_embedding_weight * identity_embedding
        + id_regression_weight * id_regression
    ) / (
        legality_weight + 2.0 * active_weight + raw_weight + field_weight
        + id_weight + id_regression_weight + identity_embedding_weight
    )


def _simple_mse_warmup_loss(predicted: dict[str, object], batch: dict) -> torch.Tensor:
    """Learn the coarse flat inverse before exact categorical/bin-level refinement."""
    scales = batch["entity_ids"].new_tensor(
        [logits.shape[-1] - 1 for logits in predicted["ids"]], dtype=torch.float32
    ).clamp_min(1.0)
    normalized_ids = batch["entity_ids"].float() / scales
    return torch.stack(
        [
            F.mse_loss(predicted["raw"], batch["entities"]),
            F.mse_loss(predicted["id_values"], normalized_ids),
            F.mse_loss(predicted["field"], batch["field"]),
            F.binary_cross_entropy_with_logits(
                predicted["legality"], batch["legality_mask"].float()
            ),
            F.cross_entropy(predicted["active_mine"], batch["active_idx"][:, 0]),
            F.cross_entropy(predicted["active_opp"], batch["active_idx"][:, 1] - 6),
        ]
    ).mean()


def _denoised_batch(
    batch: dict,
    *,
    fraction: float,
    generator: torch.Generator,
) -> dict:
    """Hide a small training-only subset of current facts without changing targets.

    Independent feature masking makes exact copying insufficient: the contextual trunk must use
    relationships among creatures and field state.  Evaluation is always performed on untouched
    inputs, and the original batch remains the reconstruction target.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError("denoise fraction must be strictly between zero and one")
    result = dict(batch)
    for key in ("entities", "entity_ids", "field"):
        values = batch[key]
        mask = torch.rand(
            values.shape,
            device=values.device,
            generator=generator,
        ) < fraction
        corrupted = values.clone()
        corrupted.masked_fill_(mask, 0)
        result[key] = corrupted
    return result


def _flat_direct_input_latent(agent, batch: dict) -> torch.Tensor:
    """Pack exact current inputs into the flat latent for a positive-control run only."""
    trunk = agent.shared_latent
    if trunk is None or not trunk.flat_autoencoder:
        raise ValueError("flat direct control requires a flat shared trunk")
    normalized_ids = trunk.normalized_ids(batch["entity_ids"], agent.backbone.embeds)
    active = batch["field"].new_zeros(batch["field"].shape[0], 12)
    active.scatter_(1, batch["active_idx"].clamp(0, 11), 1.0)
    exact = torch.cat(
        [
            batch["entities"].flatten(1),
            normalized_ids.flatten(1),
            batch["field"],
            active,
            batch["legality_mask"].float(),
        ],
        dim=-1,
    )
    if exact.shape[-1] > trunk.latent_dim:
        raise ValueError(
            "flat direct control is too narrow for exact current inputs: "
            f"{exact.shape[-1]} > {trunk.latent_dim}"
        )
    return F.pad(exact, (0, trunk.latent_dim - exact.shape[-1]))


def run(args: argparse.Namespace) -> dict:
    from fog.models.agent import FogAgent

    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    priority_raw_blocks = tuple(
        name.strip() for name in args.priority_raw_blocks.split(",") if name.strip()
    )
    unknown_priority_blocks = sorted(set(priority_raw_blocks) - set(_RAW_BLOCK_NAMES))
    if unknown_priority_blocks:
        raise ValueError(f"unknown priority raw blocks: {unknown_priority_blocks}")
    if args.priority_raw_weight < 1.0:
        raise ValueError("priority raw weight must be at least 1")
    if args.legality_weight <= 0 or args.active_weight <= 0:
        raise ValueError("legality and active reconstruction weights must be positive")
    if args.local_token_aux_weight < 0:
        raise ValueError("local token auxiliary weight must be non-negative")
    if args.mse_warmup_epochs < 0 or args.mse_warmup_epochs > args.epochs:
        raise ValueError("--mse-warmup-epochs must be between zero and --epochs")
    if args.categorical_all_identity_gate and not args.categorical_identity_heads:
        raise ValueError(
            "--categorical-all-identity-gate requires --categorical-identity-heads"
        )
    cfg = load_config(args.config)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    initial = torch.load(args.checkpoint, map_location=device, weights_only=False)
    agent = FogAgent.from_tables(cfg).to(device)
    agent.load_state_dict(initial)
    if agent.shared_latent is None:
        raise ValueError("model.shared_latent_trunk must be enabled")
    if args.local_token_aux_weight and (
        args.direct_input_control or args.freeze_trunk or agent.shared_latent.flat_autoencoder
    ):
        raise ValueError(
            "local token auxiliary requires a trainable structured shared trunk"
        )
    agent.requires_grad_(False)
    if not args.direct_input_control and not args.freeze_trunk:
        agent.shared_latent.requires_grad_(True)
    embeds = agent.backbone.embeds
    identity_weights = None
    if args.tied_identity_decoder:
        identity_weights = [
            embeds.species.weight, embeds.item.weight, embeds.ability.weight,
            *([embeds.move.weight] * 5), embeds.nature.weight,
            *([embeds.ev.weight] * 6), embeds.item.weight, embeds.ability.weight,
        ]
    decoder = SharedLatentDecoder(
        agent.shared_latent.latent_dim,
        _class_sizes(agent),
        slot_dim=args.decoder_slot_dim,
        latent_entity_dim=agent.shared_latent.entity_dim,
        identity_weights=identity_weights,
        contextual=args.contextual_decoder,
        flat=args.flat_decoder,
        simple_flat=args.simple_flat_decoder,
        linear_flat=args.linear_flat_decoder,
        categorical_identity_heads=args.categorical_identity_heads,
        linear_exact_values=args.linear_exact_values,
    ).to(device)
    local_decoder = None
    if args.local_token_aux_weight:
        # Training-only factorization pressure.  Raw creature facts and exact identities must be
        # recoverable from the corresponding output token alone; the shared Transformer is still
        # free to place cross-creature strategy in the rest of each token and in global tokens.
        local_decoder = SharedLatentDecoder(
            agent.shared_latent.latent_dim,
            _class_sizes(agent),
            slot_dim=args.decoder_slot_dim,
            latent_entity_dim=agent.shared_latent.entity_dim,
            identity_weights=identity_weights,
            contextual=False,
            linear_exact_values=args.linear_exact_values,
        ).to(device)
    if args.direct_input_control and args.simple_flat_decoder and args.linear_flat_decoder:
        decoder.initialize_linear_flat_identity_control()
    if args.decoder_checkpoint is not None:
        decoder.load_state_dict(
            torch.load(args.decoder_checkpoint, map_location=device, weights_only=False)
        )
    datasets: dict[str, object] = {
        name: TensorObservationDataset(args.data / f"{name}.pt")
        for name in ("train", "team_holdout", "battle_holdout")
    }
    live_train_dataset = datasets["train"]
    training_source = "authoritative_live_training_split"
    if args.replay_train_manifest is not None:
        from fog.training.dataset import StreamingReplayDataset

        datasets["train"] = StreamingReplayDataset(
            args.replay_train_manifest, shard_buffer=args.replay_shard_buffer
        )
        training_source = "leak_checked_replay_training_split"
    fixed_overfit_batch = None
    fixed_overfit_dataset = None
    if args.overfit_first_batch:
        repeats = args.max_train_batches or 100
        fixed_overfit_batch = next(
            live_train_dataset.batches(
                args.batch_size,
                shuffle=True,
                seed=args.seed,
                device=device,
            )
        )
        training_source = f"fixed_authoritative_live_batch_repeated_{repeats}_times"

        class _FixedBatchDataset:
            def batches(self, *_args, **_kwargs):
                yield fixed_overfit_batch

        fixed_overfit_dataset = _FixedBatchDataset()
    decoder_parameters = list(decoder.parameters())
    if local_decoder is not None:
        decoder_parameters += list(local_decoder.parameters())
    parameter_groups = [{"params": decoder_parameters, "lr": args.decoder_lr}]
    if not args.direct_input_control and not args.freeze_trunk:
        parameter_groups.insert(0, {"params": agent.shared_latent.parameters(), "lr": args.trunk_lr})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    if not 0.0 < args.lr_final_factor <= 1.0:
        raise ValueError("--lr-final-factor must be in (0, 1]")
    decay_denominator = max(1, args.epochs - 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda completed_epochs: args.lr_final_factor
        ** (completed_epochs / decay_denominator),
    )

    def direct_input_latent(batch: dict) -> torch.Tensor:
        """Exact-input control in the same structured layout as the learned latent.

        This is deliberately not a deployable representation.  It tests whether the temporary
        decoder, objective, and train/holdout split can pass before they are used to judge an
        encoder.  The decoder still receives one vector and no side arguments.
        """
        trunk = agent.shared_latent
        if trunk.flat_autoencoder:
            return _flat_direct_input_latent(agent, batch)
        if trunk.contextual:
            # Pack every exact current-state value once into the same total-width vector.  Token
            # boundaries deliberately do not help this evaluation-only control: a contextual
            # decoder must demonstrate that the data, loss, and probe can recover the source
            # before a learned contextual encoder is judged by that probe.
            normalized_ids = trunk.normalized_ids(
                batch["entity_ids"], agent.backbone.embeds
            )
            active = batch["field"].new_zeros(batch["field"].shape[0], 12)
            active.scatter_(1, batch["active_idx"].clamp(0, 11), 1.0)
            if args.simple_flat_decoder:
                # Match SharedLatentDecoder's packed output order exactly. This makes the direct
                # control an actual positive control for the metric rather than a learned
                # high-dimensional permutation test.
                exact = torch.cat(
                    [
                        batch["entities"].flatten(1),
                        normalized_ids.flatten(1),
                        batch["field"],
                        batch["legality_mask"].float(),
                        active[:, :6],
                        active[:, 6:],
                    ],
                    dim=-1,
                )
            else:
                entity_exact = torch.cat(
                    [batch["entities"], normalized_ids], dim=-1
                ).flatten(1)
                exact = torch.cat(
                    [
                        entity_exact,
                        batch["field"],
                        active,
                        batch["legality_mask"].float(),
                    ],
                    dim=-1,
                )
            if exact.shape[-1] > trunk.latent_dim:
                raise ValueError(
                    "contextual direct control is too narrow for exact current inputs: "
                    f"{exact.shape[-1]} > {trunk.latent_dim}"
                )
            return F.pad(exact, (0, trunk.latent_dim - exact.shape[-1]))
        if trunk.entity_dim < batch["entities"].shape[-1] + batch["entity_ids"].shape[-1]:
            raise ValueError("direct control requires room for raw entity values and normalized ids")
        normalized_ids = trunk.normalized_ids(batch["entity_ids"], agent.backbone.embeds)
        exact_entity = torch.cat([batch["entities"], normalized_ids], dim=-1)
        entity_padding = exact_entity.new_zeros(
            *exact_entity.shape[:-1], trunk.entity_dim - exact_entity.shape[-1]
        )
        entity = torch.cat([exact_entity, entity_padding], dim=-1).flatten(1)
        active = batch["field"].new_zeros(batch["field"].shape[0], 12)
        active.scatter_(1, batch["active_idx"].clamp(0, 11), 1.0)
        metadata = torch.cat([active, batch["legality_mask"].float()], dim=-1)
        field_exact = (
            torch.cat([batch["field"], metadata], dim=-1)
            if trunk.learned_only else batch["field"]
        )
        field_padding = batch["field"].new_zeros(
            batch["field"].shape[0], trunk.field_dim - field_exact.shape[-1]
        )
        field = torch.cat([field_exact, field_padding], dim=-1)
        history = batch["field"].new_zeros(
            batch["field"].shape[0], trunk.metadata_offset - trunk.history_offset
        )
        global_padding = batch["field"].new_zeros(batch["field"].shape[0], trunk.global_dim)
        pieces = [entity, field, history]
        if not trunk.learned_only:
            pieces.append(metadata)
        pieces.append(global_padding)
        latent = torch.cat(pieces, dim=-1)
        if latent.shape[-1] != trunk.latent_dim:
            raise RuntimeError(f"direct control produced {latent.shape[-1]} rather than {trunk.latent_dim}")
        return latent

    def encode_latent(batch: dict) -> torch.Tensor:
        if args.direct_input_control:
            return direct_input_latent(batch)
        contexts = agent.encode_contexts(batch)
        return agent.shared_latent(batch, contexts, agent.backbone.embeds)

    def predict(batch: dict) -> dict[str, object]:
        return decoder(encode_latent(batch))

    def local_reconstruction_loss(latent: torch.Tensor, batch: dict) -> torch.Tensor:
        if local_decoder is None:
            return latent.new_zeros(())
        return _loss(
            local_decoder(latent), batch,
            id_weight=args.id_weight,
            id_regression_weight=args.id_regression_weight,
            raw_weight=args.raw_weight,
            field_weight=0.0,
            worst_field_weight=args.worst_field_weight,
            rare_value_weight=args.rare_value_weight,
            identity_embedding_weight=args.identity_embedding_weight,
            tail_fraction=args.tail_fraction,
            ordinal_ev_decoder=args.ordinal_ev_decoder,
            ordinal_all_identities=args.ordinal_all_identities,
            cumulative_all_identities=args.cumulative_all_identities,
            linear_id_values=args.linear_id_values,
            linear_exact_values=args.linear_exact_values,
            categorical_cross_entropy=args.categorical_cross_entropy,
            code_scaled_identities=args.code_scaled_identities,
            priority_raw_blocks=priority_raw_blocks,
            priority_raw_weight=args.priority_raw_weight,
            legality_weight=0.0,
            active_weight=0.0,
        )

    coverage_generator = torch.Generator(device=device)
    coverage_generator.manual_seed(args.seed + 50000)
    denoise_generator = torch.Generator(device=device)
    denoise_generator.manual_seed(args.seed + 60000)

    def synthetic_coverage_batch(batch: dict) -> dict:
        """Training-only schema coverage for rare exact values; holdouts are never modified.

        Only current inputs are changed, and only to values valid for their documented fields.
        This teaches invertibility across the schema rather than memorizing the compact live teams.
        """
        augmented = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        entities = augmented["entities"]
        entity_ids = augmented["entity_ids"]
        fraction = float(args.synthetic_coverage_fraction)
        entity_mask = torch.rand(
            entities.shape[:2], device=device, generator=coverage_generator
        ) < fraction
        present = entity_ids[..., 0] != 0
        entity_mask &= present
        own_mask = entity_mask.clone()
        own_mask[:, 6:] = False
        if args.direct_input_control:
            identity_masks = {
                field_index: entity_mask
                for field_index in range(len(decoder.class_sizes))
            }
        elif args.synthetic_all_learned_identities:
            # All identities on our own team are exact inputs.  On the opponent's
            # side, species/item/ability/moves/original revealed identities may
            # legitimately be known, while nature and EVs remain private.  Cover
            # the former across both sides and the latter on our side only.
            private_set_fields = set(range(8, 15))
            identity_masks = {
                field_index: (
                    own_mask if field_index in private_set_fields else entity_mask
                )
                for field_index in range(len(decoder.class_sizes))
            }
        else:
            identity_masks = {field_index: own_mask for field_index in range(9, 15)}
        for field_index, identity_mask in identity_masks.items():
            random_ids = torch.randint(
                0, decoder.class_sizes[field_index], entity_ids.shape[:2],
                device=device, generator=coverage_generator,
            )
            entity_ids[..., field_index] = torch.where(
                identity_mask, random_ids, entity_ids[..., field_index]
            )
        # Dynamic and categorical raw facts use their actual encoded domains.
        random_base = torch.rand(
            *entities.shape[:2], DIMS.base_stats,
            device=device, generator=coverage_generator,
        )
        entities[..., L.base_stats : L.base_stats + DIMS.base_stats] = torch.where(
            entity_mask.unsqueeze(-1),
            random_base,
            entities[..., L.base_stats : L.base_stats + DIMS.base_stats],
        )
        type_index = torch.randint(
            0, DIMS.typing, entities.shape[:2],
            device=device, generator=coverage_generator,
        )
        random_typing = F.one_hot(type_index, DIMS.typing).to(entities.dtype)
        entities[..., L.typing : L.typing + DIMS.typing] = torch.where(
            entity_mask.unsqueeze(-1),
            random_typing,
            entities[..., L.typing : L.typing + DIMS.typing],
        )
        weight_index = torch.randint(
            0, DIMS.weight_bucket, entities.shape[:2],
            device=device, generator=coverage_generator,
        )
        random_weight = F.one_hot(weight_index, DIMS.weight_bucket).to(entities.dtype)
        entities[..., L.weight : L.weight + DIMS.weight_bucket] = torch.where(
            entity_mask.unsqueeze(-1),
            random_weight,
            entities[..., L.weight : L.weight + DIMS.weight_bucket],
        )
        fainted = torch.rand(
            entities.shape[:2], device=device, generator=coverage_generator
        ) < 0.10
        hp = torch.rand(
            entities.shape[:2], device=device, generator=coverage_generator
        ).clamp_min(1e-3)
        hp = torch.where(fainted, torch.zeros_like(hp), hp)
        entities[..., L.hp] = torch.where(entity_mask, hp, entities[..., L.hp])
        entities[..., L.fainted] = torch.where(
            entity_mask,
            fainted.to(entities.dtype),
            entities[..., L.fainted],
        )
        status_index = torch.randint(
            0, DIMS.status, entities.shape[:2],
            device=device, generator=coverage_generator,
        )
        random_status = F.one_hot(status_index, DIMS.status).to(entities.dtype)
        entities[..., L.status : L.status + DIMS.status] = torch.where(
            entity_mask.unsqueeze(-1),
            random_status,
            entities[..., L.status : L.status + DIMS.status],
        )
        for offset, width in (
            (L.counters, DIMS.sleep_tox_counters),
            (L.vol_counters, DIMS.volatile_counters),
        ):
            values = torch.rand(
                *entities.shape[:2], width,
                device=device, generator=coverage_generator,
            )
            entities[..., offset : offset + width] = torch.where(
                entity_mask.unsqueeze(-1),
                values,
                entities[..., offset : offset + width],
            )
        for column in range(L.boosts, L.boosts + DIMS.boosts):
            values = torch.randint(
                -6, 7, entities.shape[:2], device=device, generator=coverage_generator
            ).float() / 6.0
            entities[..., column] = torch.where(entity_mask, values, entities[..., column])
        random_volatiles = (
            torch.rand(
                *entities.shape[:2], DIMS.volatiles,
                device=device, generator=coverage_generator,
            ) < 0.10
        ).to(entities.dtype)
        entities[..., L.volatiles : L.volatiles + DIMS.volatiles] = torch.where(
            entity_mask.unsqueeze(-1),
            random_volatiles,
            entities[..., L.volatiles : L.volatiles + DIMS.volatiles],
        )
        paradox_index = torch.randint(
            0, DIMS.paradox_boost + 1, entities.shape[:2],
            device=device, generator=coverage_generator,
        )
        random_paradox = F.one_hot(
            paradox_index.clamp_max(DIMS.paradox_boost - 1), DIMS.paradox_boost
        ).to(entities.dtype)
        random_paradox *= (paradox_index < DIMS.paradox_boost).unsqueeze(-1)
        entities[..., L.paradox : L.paradox + DIMS.paradox_boost] = torch.where(
            entity_mask.unsqueeze(-1),
            random_paradox,
            entities[..., L.paradox : L.paradox + DIMS.paradox_boost],
        )
        random_booster = (
            torch.rand(entities.shape[:2], device=device, generator=coverage_generator) > 0.5
        ).to(entities.dtype)
        entities[..., L.booster] = torch.where(
            entity_mask, random_booster, entities[..., L.booster]
        )
        for side_start in (0, 6):
            side = slice(side_start, side_start + 6)
            fallen = entities[:, side, L.fainted].sum(-1).clamp(0, 5) / 5.0
            side_mask = entity_mask[:, side].any(-1, keepdim=True)
            entities[:, side, L.fallen] = torch.where(
                side_mask,
                fallen.unsqueeze(-1).expand(-1, 6),
                entities[:, side, L.fallen],
            )
        for move_index, column in enumerate(range(L.move_pp, L.move_pp + DIMS.move_pp)):
            values = torch.rand(entities.shape[:2], device=device, generator=coverage_generator)
            has_move = entity_ids[..., 3 + move_index] != 0
            pp_mask = entity_mask & has_move
            entities[..., column] = torch.where(pp_mask, values, entities[..., column])
        tera_index = torch.randint(
            0, DIMS.tera_type, entities.shape[:2],
            device=device, generator=coverage_generator,
        )
        random_tera = F.one_hot(tera_index, DIMS.tera_type).to(entities.dtype)
        entities[..., L.tera_type : L.tera_type + DIMS.tera_type] = torch.where(
            entity_mask.unsqueeze(-1),
            random_tera,
            entities[..., L.tera_type : L.tera_type + DIMS.tera_type],
        )
        random_tera_used = (
            torch.rand(entities.shape[:2], device=device, generator=coverage_generator) > 0.5
        ).to(entities.dtype)
        entities[..., L.tera_used] = torch.where(
            entity_mask, random_tera_used, entities[..., L.tera_used]
        )
        random_epistemic = (
            torch.rand(
                *entities.shape[:2], DIMS.epistemic_flags,
                device=device, generator=coverage_generator,
            ) > 0.5
        ).to(entities.dtype)
        entities[..., L.epistemic : L.epistemic + DIMS.epistemic_flags] = torch.where(
            entity_mask.unsqueeze(-1),
            random_epistemic,
            entities[..., L.epistemic : L.epistemic + DIMS.epistemic_flags],
        )
        field = augmented["field"]
        row_mask = torch.rand(field.shape[0], device=device, generator=coverage_generator) < fraction
        covered_rows = int(row_mask.sum().item())
        if covered_rows:
            covered = field[row_mask].clone()
            weather = torch.randint(0, 5, (covered_rows,), device=device, generator=coverage_generator)
            covered[:, 0:5] = F.one_hot(weather, 5).to(field.dtype)
            covered[:, 5] = torch.where(
                weather == 4,
                torch.zeros(covered_rows, device=device, dtype=field.dtype),
                torch.rand(covered_rows, device=device, generator=coverage_generator),
            )
            terrain = torch.randint(0, 5, (covered_rows,), device=device, generator=coverage_generator)
            covered[:, 6:11] = F.one_hot(terrain, 5).to(field.dtype)
            covered[:, 11] = torch.where(
                terrain == 4,
                torch.zeros(covered_rows, device=device, dtype=field.dtype),
                torch.rand(covered_rows, device=device, generator=coverage_generator),
            )
            for side_offset in (12, 18):
                covered[:, side_offset] = torch.randint(
                    0, 2, (covered_rows,), device=device, generator=coverage_generator
                ).to(field.dtype)
                covered[:, side_offset + 1] = torch.randint(
                    0, 4, (covered_rows,), device=device, generator=coverage_generator
                ).to(field.dtype) / 3.0
                covered[:, side_offset + 2] = torch.randint(
                    0, 3, (covered_rows,), device=device, generator=coverage_generator
                ).to(field.dtype) / 2.0
                covered[:, side_offset + 3] = torch.randint(
                    0, 2, (covered_rows,), device=device, generator=coverage_generator
                ).to(field.dtype)
                covered[:, side_offset + 4 : side_offset + 6] = 0.0
            for side_offset in (24, 29):
                screen_bits = torch.randint(
                    0, 2, (covered_rows, 4), device=device, generator=coverage_generator
                ).to(field.dtype)
                covered[:, side_offset : side_offset + 4] = screen_bits
                covered[:, side_offset + 4] = torch.where(
                    screen_bits.any(-1),
                    torch.rand(covered_rows, device=device, generator=coverage_generator),
                    torch.zeros(covered_rows, device=device, dtype=field.dtype),
                )
            trick = torch.randint(
                0, 2, (covered_rows,), device=device, generator=coverage_generator
            ).to(field.dtype)
            covered[:, 34] = trick
            covered[:, 35] = trick * torch.rand(
                covered_rows, device=device, generator=coverage_generator
            )
            covered[:, 36] = torch.randint(
                0, 51, (covered_rows,), device=device, generator=coverage_generator
            ).to(field.dtype) / 50.0
            covered[:, 37] = torch.randint(
                0, 2, (covered_rows,), device=device, generator=coverage_generator
            ).to(field.dtype)
            covered[:, 38] = 0.0
            for side_offset in (39, 44):
                future = torch.randint(
                    0, 2, (covered_rows,), device=device, generator=coverage_generator
                ).to(field.dtype)
                covered[:, side_offset] = future
                covered[:, side_offset + 1] = future * torch.randint(
                    1, 3, (covered_rows,), device=device, generator=coverage_generator
                ).to(field.dtype) / 3.0
                wish = torch.randint(
                    0, 2, (covered_rows,), device=device, generator=coverage_generator
                ).to(field.dtype)
                covered[:, side_offset + 2] = wish
                covered[:, side_offset + 3] = wish * torch.randint(
                    1, 3, (covered_rows,), device=device, generator=coverage_generator
                ).to(field.dtype) / 2.0
                covered[:, side_offset + 4] = wish * torch.rand(
                    covered_rows, device=device, generator=coverage_generator
                )
            gravity = torch.randint(
                0, 2, (covered_rows,), device=device, generator=coverage_generator
            ).to(field.dtype)
            covered[:, 49] = gravity
            covered[:, 50] = gravity * torch.rand(
                covered_rows, device=device, generator=coverage_generator
            )
            field[row_mask] = covered
        if args.direct_input_control:
            # A positive copy-control should cover the complete valid schema, not
            # inherit behavioral correlations from the replay policy.  These are
            # still current input targets and never touch either holdout.
            augmented["legality_mask"] = torch.where(
                row_mask.unsqueeze(-1),
                torch.rand(
                    augmented["legality_mask"].shape,
                    device=device,
                    generator=coverage_generator,
                ) > 0.5,
                augmented["legality_mask"],
            )
            active_idx = augmented["active_idx"]
            random_active = torch.stack(
                [
                    torch.randint(0, 6, active_idx.shape[:1], device=device, generator=coverage_generator),
                    torch.randint(6, 12, active_idx.shape[:1], device=device, generator=coverage_generator),
                ],
                dim=-1,
            )
            augmented["active_idx"] = torch.where(
                row_mask.unsqueeze(-1), random_active, active_idx
            )
        return augmented

    def evaluate(dataset: TensorObservationDataset) -> dict:
        agent.eval()
        decoder.eval()
        totals = {
            "loss": 0.0, "batches": 0, "rows": 0,
            "raw_sq": 0.0, "raw_target_sum": 0.0, "raw_target_square": 0.0,
            "raw_n": 0, "raw_within": 0,
            "field_sq": 0.0, "field_target_sum": 0.0, "field_target_square": 0.0,
            "field_n": 0, "field_within": 0,
            "legal_correct": 0, "legal_n": 0, "legal_rows": 0,
            "active_mine": 0, "active_opp": 0,
            "raw_within_by_field": torch.zeros(140, dtype=torch.long),
            "raw_count_by_field": torch.zeros(140, dtype=torch.long),
            "field_within_by_field": torch.zeros(51, dtype=torch.long),
            "field_count_by_field": torch.zeros(51, dtype=torch.long),
            "field_binary_correct_by_field": torch.zeros(51, dtype=torch.long),
            "field_fractional_by_field": torch.zeros(51, dtype=torch.bool),
        }
        raw_semantic_correct: dict[str, torch.Tensor] = {}
        raw_semantic_count: dict[str, torch.Tensor] = {}
        id_correct = torch.zeros(len(decoder.class_sizes), dtype=torch.long)
        id_class_correct = torch.zeros(len(decoder.class_sizes), dtype=torch.long)
        id_count = torch.zeros(len(decoder.class_sizes), dtype=torch.long)
        with torch.no_grad():
            for batch in dataset.batches(args.batch_size, shuffle=False, seed=args.seed, device=device):
                predicted = predict(batch)
                totals["loss"] += float(_loss(
                    predicted, batch, id_weight=args.id_weight,
                    id_regression_weight=args.id_regression_weight,
                    raw_weight=args.raw_weight, field_weight=args.field_weight,
                    worst_field_weight=args.worst_field_weight,
                    rare_value_weight=args.rare_value_weight,
                    identity_embedding_weight=args.identity_embedding_weight,
                    tail_fraction=args.tail_fraction,
                    ordinal_ev_decoder=args.ordinal_ev_decoder,
                    ordinal_all_identities=args.ordinal_all_identities,
                    cumulative_all_identities=args.cumulative_all_identities,
                    linear_id_values=args.linear_id_values,
                    linear_exact_values=args.linear_exact_values,
                    categorical_cross_entropy=args.categorical_cross_entropy,
                    code_scaled_identities=args.code_scaled_identities,
                    priority_raw_blocks=priority_raw_blocks,
                    priority_raw_weight=args.priority_raw_weight,
                ))
                totals["batches"] += 1
                rows = int(batch["entities"].shape[0])
                totals["rows"] += rows
                for name, target_name in (("raw", "entities"), ("field", "field")):
                    target = batch[target_name]
                    error = predicted[name] - target
                    totals[f"{name}_sq"] += float(error.square().sum())
                    totals[f"{name}_target_sum"] += float(target.sum())
                    totals[f"{name}_target_square"] += float(target.square().sum())
                    totals[f"{name}_n"] += target.numel()
                    totals[f"{name}_within"] += int((error.abs() <= args.numeric_tolerance).sum())
                    within_by_field = (error.abs() <= args.numeric_tolerance).reshape(
                        -1, error.shape[-1]
                    ).sum(0).cpu()
                    totals[f"{name}_within_by_field"] += within_by_field
                    totals[f"{name}_count_by_field"] += torch.full_like(
                        within_by_field, error.numel() // error.shape[-1]
                    )
                field_target = batch["field"]
                totals["field_binary_correct_by_field"] += (
                    ((predicted["field"] >= 0.5) == (field_target >= 0.5))
                    .sum(0).cpu()
                )
                totals["field_fractional_by_field"] |= (
                    ((field_target > 1e-6) & (field_target < 1.0 - 1e-6)).any(0).cpu()
                )
                for block_name, (kind, offset, width) in zip(_RAW_BLOCK_NAMES, _RAW_BLOCKS):
                    values = predicted["raw"][..., offset : offset + width]
                    truth = batch["entities"][..., offset : offset + width]
                    if kind == "categorical":
                        correct = (values.argmax(-1) == truth.argmax(-1)).sum().cpu()
                        count = torch.tensor(truth.numel() // width, dtype=torch.long)
                    elif kind == "binary":
                        correct = ((values >= 0.5) == (truth >= 0.5)).reshape(
                            -1, width
                        ).sum(0).cpu()
                        count = torch.full(
                            (width,), truth.numel() // width, dtype=torch.long
                        )
                    else:
                        correct = (values.sub(truth).abs() <= args.numeric_tolerance).reshape(
                            -1, width
                        ).sum(0).cpu()
                        count = torch.full(
                            (width,), truth.numel() // width, dtype=torch.long
                        )
                    raw_semantic_correct[block_name] = (
                        raw_semantic_correct.get(block_name, torch.zeros_like(correct)) + correct
                    )
                    raw_semantic_count[block_name] = (
                        raw_semantic_count.get(block_name, torch.zeros_like(count)) + count
                    )
                for index, logits in enumerate(predicted["ids"]):
                    target = batch["entity_ids"][..., index]
                    scale = decoder.class_sizes[index] - 1
                    normalized_id = (
                        predicted["id_values"][..., index]
                        if args.linear_id_values
                        else predicted["id_values"][..., index].sigmoid()
                    ).clamp(0.0, 1.0)
                    recovered = torch.round(normalized_id * scale).long()
                    recovered.clamp_(0, scale)
                    id_correct[index] += int((recovered == target).sum())
                    id_class_correct[index] += int((logits.argmax(-1) == target).sum())
                    id_count[index] += target.numel()
                legal = predicted["legality"] > 0
                target_legal = batch["legality_mask"]
                totals["legal_correct"] += int((legal == target_legal).sum())
                totals["legal_n"] += target_legal.numel()
                totals["legal_rows"] += int((legal == target_legal).all(-1).sum())
                totals["active_mine"] += int((predicted["active_mine"].argmax(-1) == batch["active_idx"][:, 0]).sum())
                totals["active_opp"] += int((predicted["active_opp"].argmax(-1) == batch["active_idx"][:, 1] - 6).sum())
        report = {"rows": totals["rows"], "loss": totals["loss"] / max(1, totals["batches"])}
        recovery_values: list[float] = []
        for name in ("raw", "field"):
            count = totals[f"{name}_n"]
            mse = totals[f"{name}_sq"] / max(1, count)
            mean = totals[f"{name}_target_sum"] / max(1, count)
            variance = totals[f"{name}_target_square"] / max(1, count) - mean * mean
            within = totals[f"{name}_within"] / max(1, count)
            recovered = 1.0 - mse / max(variance, 1e-12)
            report[f"{name}_mse"] = mse
            report[f"{name}_variance_recovered"] = recovered
            report[f"{name}_within_tolerance"] = within
            within_by_field = (
                totals[f"{name}_within_by_field"].float()
                / totals[f"{name}_count_by_field"].clamp_min(1)
            ).tolist()
            report[f"{name}_within_tolerance_by_field"] = within_by_field
            report[f"{name}_minimum_field_recovery"] = min(within_by_field)
            # Per-field exact recovery is the gate. Variance explained is retained as a useful
            # continuous diagnostic but is ill-defined for constant binary/input columns.
            if name == "raw":
                semantic_recovery = {
                    block_name: (
                        raw_semantic_correct[block_name].float()
                        / raw_semantic_count[block_name].clamp_min(1)
                    ).tolist()
                    for block_name in _RAW_BLOCK_NAMES
                }
                semantic_values = [
                    float(value)
                    for values in semantic_recovery.values()
                    for value in (values if isinstance(values, list) else [values])
                ]
                report["raw_semantic_recovery"] = semantic_recovery
                report["raw_semantic_minimum_recovery"] = min(semantic_values)
                recovery_values.append(report["raw_semantic_minimum_recovery"])
            else:
                binary = (
                    totals["field_binary_correct_by_field"].float()
                    / totals["field_count_by_field"].clamp_min(1)
                )
                fractional = totals["field_fractional_by_field"]
                selected = torch.where(
                    fractional,
                    totals["field_within_by_field"].float()
                    / totals["field_count_by_field"].clamp_min(1),
                    binary,
                ).tolist()
                report["field_semantic_recovery_by_field"] = selected
                report["field_semantic_minimum_recovery"] = min(selected)
                recovery_values.append(report["field_semantic_minimum_recovery"])
        id_roundtrip = (id_correct.float() / id_count.clamp_min(1)).tolist()
        id_classification = (id_class_correct.float() / id_count.clamp_min(1)).tolist()
        selected_identity = list(id_classification if decoder.tied_identity else id_roundtrip)
        if args.categorical_all_identity_gate:
            selected_identity = list(id_classification)
        elif (
            args.ordinal_all_identities
            or args.numeric_all_identity_gate
            or args.cumulative_all_identities
        ):
            selected_identity = list(id_roundtrip)
        elif args.ordinal_ev_decoder:
            selected_identity[9:15] = id_roundtrip[9:15]
        report["id_accuracy_by_field"] = selected_identity
        report["id_accuracy"] = (
            float(id_class_correct.sum() / id_count.sum()) if decoder.tied_identity
            else float(id_correct.sum() / id_count.sum())
        )
        report["id_min_accuracy"] = min(selected_identity)
        report["id_numeric_roundtrip_accuracy_by_field"] = id_roundtrip
        report["id_classification_accuracy_by_field"] = id_classification
        report["identity_recovery_method"] = (
            "categorical_classification_for_all_exact_identity_codes"
            if args.categorical_all_identity_gate
            else "normalized_numeric_roundtrip_for_all_exact_identity_codes"
            if args.numeric_all_identity_gate
            else "cumulative_ordinal_roundtrip_for_all_exact_identity_codes"
            if args.cumulative_all_identities
            else "ordinal_roundtrip_for_exact_gate_with_auxiliary_semantic_classification"
            if args.ordinal_all_identities
            else "categorical_semantic_embedding_with_ordinal_ev_roundtrip"
            if decoder.tied_identity and args.ordinal_ev_decoder
            else "nearest_fixed_semantic_embedding_for_all_identity_fields"
            if decoder.tied_identity else "rounded_numeric_index"
        )
        report["legality_bit_accuracy"] = totals["legal_correct"] / max(1, totals["legal_n"])
        report["legality_row_accuracy"] = totals["legal_rows"] / max(1, totals["rows"])
        report["active_mine_accuracy"] = totals["active_mine"] / max(1, totals["rows"])
        report["active_opp_accuracy"] = totals["active_opp"] / max(1, totals["rows"])
        recovery_values.extend([
            report["id_min_accuracy"], report["legality_bit_accuracy"],
            report["legality_row_accuracy"],
            report["active_mine_accuracy"], report["active_opp_accuracy"],
        ])
        report["minimum_recovery"] = min(recovery_values)
        return report

    rows: list[dict] = []
    best_score = float("-inf")
    best_recovery_score = float("-inf")
    best_agent = None
    best_decoder = None
    best_local_decoder = None
    stale = 0
    selection_phase = None
    if args.epochs == 0:
        if args.decoder_checkpoint is None:
            raise ValueError("--epochs 0 requires --decoder-checkpoint for evaluation-only use")
        best_agent = copy.deepcopy(agent.state_dict())
        best_decoder = copy.deepcopy(decoder.state_dict())
    for epoch in range(args.epochs):
        agent.train()
        decoder.train()
        if local_decoder is not None:
            local_decoder.train()
        running = 0.0
        running_local = 0.0
        batches = 0
        source_batches = (
            (fixed_overfit_batch for _ in range(args.max_train_batches or 100))
            if fixed_overfit_batch is not None
            else datasets["train"].batches(
                args.batch_size, shuffle=True, seed=args.seed + epoch, device=device
            )
        )
        live_batches = None
        if (
            fixed_overfit_batch is None
            and args.replay_train_manifest is not None
            and args.live_mix_every
        ):
            live_batches = iter(live_train_dataset.batches(
                args.batch_size, shuffle=True, seed=args.seed + 10000 + epoch, device=device
            ))
        for mixed_batches, source_batch in enumerate(source_batches, start=1):
            pending = [source_batch]
            if live_batches is not None and mixed_batches % args.live_mix_every == 0:
                try:
                    pending.append(next(live_batches))
                except StopIteration:
                    live_batches = iter(live_train_dataset.batches(
                        args.batch_size, shuffle=True,
                        seed=args.seed + 20000 + epoch + mixed_batches, device=device,
                    ))
                    pending.append(next(live_batches))
            if args.synthetic_coverage_every and mixed_batches % args.synthetic_coverage_every == 0:
                pending.append(synthetic_coverage_batch(source_batch))
            for batch in pending:
                optimizer.zero_grad(set_to_none=True)
                latent = encode_latent(batch)
                predicted = decoder(latent)
                if epoch < args.mse_warmup_epochs:
                    loss = _simple_mse_warmup_loss(predicted, batch)
                else:
                    loss = _loss(
                        predicted, batch, id_weight=args.id_weight,
                        id_regression_weight=args.id_regression_weight,
                        raw_weight=args.raw_weight, field_weight=args.field_weight,
                        worst_field_weight=args.worst_field_weight,
                        rare_value_weight=args.rare_value_weight,
                        identity_embedding_weight=args.identity_embedding_weight,
                        tail_fraction=args.tail_fraction,
                        ordinal_ev_decoder=args.ordinal_ev_decoder,
                        ordinal_all_identities=args.ordinal_all_identities,
                        cumulative_all_identities=args.cumulative_all_identities,
                        linear_id_values=args.linear_id_values,
                        linear_exact_values=args.linear_exact_values,
                        categorical_cross_entropy=args.categorical_cross_entropy,
                        code_scaled_identities=args.code_scaled_identities,
                        priority_raw_blocks=priority_raw_blocks,
                        priority_raw_weight=args.priority_raw_weight,
                        legality_weight=args.legality_weight,
                        active_weight=args.active_weight,
                    )
                local_loss = local_reconstruction_loss(latent, batch)
                loss = loss + args.local_token_aux_weight * local_loss
                if args.denoise_fraction and args.denoise_weight:
                    denoised = _denoised_batch(
                        batch,
                        fraction=args.denoise_fraction,
                        generator=denoise_generator,
                    )
                    denoised_latent = encode_latent(denoised)
                    denoised_prediction = decoder(denoised_latent)
                    loss = loss + args.denoise_weight * _loss(
                        denoised_prediction,
                        batch,
                        id_weight=args.id_weight,
                        id_regression_weight=args.id_regression_weight,
                        raw_weight=args.raw_weight,
                        field_weight=args.field_weight,
                        worst_field_weight=args.worst_field_weight,
                        rare_value_weight=args.rare_value_weight,
                        identity_embedding_weight=args.identity_embedding_weight,
                        tail_fraction=args.tail_fraction,
                        ordinal_ev_decoder=args.ordinal_ev_decoder,
                        ordinal_all_identities=args.ordinal_all_identities,
                        cumulative_all_identities=args.cumulative_all_identities,
                        linear_id_values=args.linear_id_values,
                        linear_exact_values=args.linear_exact_values,
                        categorical_cross_entropy=args.categorical_cross_entropy,
                        code_scaled_identities=args.code_scaled_identities,
                        priority_raw_blocks=priority_raw_blocks,
                        priority_raw_weight=args.priority_raw_weight,
                        legality_weight=args.legality_weight,
                        active_weight=args.active_weight,
                    )
                    loss = loss + (
                        args.denoise_weight
                        * args.local_token_aux_weight
                        * local_reconstruction_loss(denoised_latent, batch)
                    )
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite reconstruction loss at epoch {epoch + 1}")
                loss.backward()
                trainable = list(decoder.parameters())
                if local_decoder is not None:
                    trainable += list(local_decoder.parameters())
                if not args.direct_input_control and not args.freeze_trunk:
                    trainable += list(agent.shared_latent.parameters())
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                running += float(loss.detach())
                running_local += float(local_loss.detach())
                batches += 1
                if args.max_train_batches and batches >= args.max_train_batches:
                    break
            if args.max_train_batches and batches >= args.max_train_batches:
                break
        team = evaluate(datasets["team_holdout"])
        battle = evaluate(datasets["battle_holdout"])
        overfit = evaluate(fixed_overfit_dataset) if fixed_overfit_dataset is not None else None
        recovery_score = (
            float(overfit["minimum_recovery"])
            if overfit is not None
            else min(float(team["minimum_recovery"]), float(battle["minimum_recovery"]))
        )
        phase, score = _checkpoint_selection_score(
            epoch_index=epoch,
            mse_warmup_epochs=args.mse_warmup_epochs,
            team=team,
            battle=battle,
            recovery_score=recovery_score,
        )
        if phase != selection_phase:
            selection_phase = phase
            best_score = float("-inf")
            stale = 0
        best_recovery_score = max(best_recovery_score, recovery_score)
        row = {
            "epoch": epoch + 1, "train_loss": running / max(1, batches),
            "local_token_aux_loss": running_local / max(1, batches),
            "trunk_lr": optimizer.param_groups[0]["lr"]
            if not args.direct_input_control and not args.freeze_trunk else 0.0,
            "decoder_lr": optimizer.param_groups[-1]["lr"],
            "team_loss": team["loss"], "battle_loss": battle["loss"],
            "team_minimum_recovery": team["minimum_recovery"],
            "battle_minimum_recovery": battle["minimum_recovery"],
            "team_id_min_accuracy": team["id_min_accuracy"],
            "battle_id_min_accuracy": battle["id_min_accuracy"],
            "selection_phase": phase,
        }
        if overfit is not None:
            row["overfit_minimum_recovery"] = overfit["minimum_recovery"]
            row["overfit_loss"] = overfit["loss"]
        rows.append(row)
        print(f"[shared-latent-reconstruction] {row}", flush=True)
        with (args.out / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (args.out / "progress.json").write_text(
            json.dumps(
                {
                    "status": "training",
                    "latest": row,
                    "best_minimum_recovery": best_recovery_score,
                    "team_weakest_components": _weakest_components(team),
                    "battle_weakest_components": _weakest_components(battle),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if score > best_score + args.min_delta:
            best_score = score
            stale = 0
            best_agent = copy.deepcopy(agent.state_dict())
            best_decoder = copy.deepcopy(decoder.state_dict())
            best_local_decoder = (
                copy.deepcopy(local_decoder.state_dict()) if local_decoder is not None else None
            )
            # Persist each new best immediately. Long reconstruction jobs must remain
            # diagnosable and recoverable before patience or the epoch budget expires.
            torch.save(best_agent, args.out / "latent_best.pt")
            torch.save(best_decoder, args.out / "decoder_best.pt")
            if best_local_decoder is not None:
                torch.save(best_local_decoder, args.out / "local_decoder_best.pt")
        else:
            stale += 1
        scheduler.step()
        if recovery_score >= args.target or (
            phase == "strict_recovery" and args.patience and stale >= args.patience
        ):
            break
    if best_agent is None or best_decoder is None:
        raise RuntimeError("no finite shared-latent checkpoint was produced")
    agent.load_state_dict(best_agent)
    decoder.load_state_dict(best_decoder)
    team = evaluate(datasets["team_holdout"])
    battle = evaluate(datasets["battle_holdout"])
    live_train = evaluate(live_train_dataset) if args.evaluate_live_train else None
    overfit = evaluate(fixed_overfit_dataset) if fixed_overfit_dataset is not None else None
    changed = sorted(name for name, value in best_agent.items() if _digest(value) != _digest(initial[name]))
    illegal = [name for name in changed if not name.startswith("shared_latent.")]
    if (args.direct_input_control or args.freeze_trunk) and changed:
        raise RuntimeError(f"direct-input control changed agent tensors: {changed[:10]}")
    if illegal:
        raise RuntimeError(f"training changed tensors outside the shared trunk: {illegal[:10]}")
    passed = min(float(team["minimum_recovery"]), float(battle["minimum_recovery"])) >= args.target
    report = {
        "status": "complete", "passed": passed, "target": args.target,
        "source": str(args.checkpoint), "data": str(args.data),
        "training_source": training_source,
        "replay_train_manifest": str(args.replay_train_manifest) if args.replay_train_manifest else None,
        "live_mix_every_replay_batches": args.live_mix_every,
        "decoder_source": str(args.decoder_checkpoint) if args.decoder_checkpoint else None,
        "decoder_input": "exact_control_vector" if args.direct_input_control else "shared_latent_only",
        "contextual_decoder": args.contextual_decoder,
        "flat_decoder": args.flat_decoder,
        "simple_flat_decoder": args.simple_flat_decoder,
        "linear_flat_decoder": args.linear_flat_decoder,
        "categorical_identity_heads": args.categorical_identity_heads,
        "categorical_all_identity_gate": args.categorical_all_identity_gate,
        "raw_input_shortcut": args.direct_input_control,
        "direct_input_control": args.direct_input_control,
        "trunk_frozen": args.freeze_trunk,
        "evaluation_only": args.epochs == 0,
        "tied_identity_decoder": args.tied_identity_decoder,
        "worst_field_weight": args.worst_field_weight,
        "rare_value_weight": args.rare_value_weight,
        "identity_embedding_weight": args.identity_embedding_weight,
        "tail_fraction": args.tail_fraction,
        "ordinal_ev_decoder": args.ordinal_ev_decoder,
        "ordinal_all_identities": args.ordinal_all_identities,
        "numeric_all_identity_gate": args.numeric_all_identity_gate,
        "cumulative_all_identities": args.cumulative_all_identities,
        "linear_id_values": args.linear_id_values,
        "linear_exact_values": args.linear_exact_values,
        "categorical_cross_entropy": args.categorical_cross_entropy,
        "code_scaled_identities": args.code_scaled_identities,
        "priority_raw_blocks": list(priority_raw_blocks),
        "priority_raw_weight": args.priority_raw_weight,
        "legality_weight": args.legality_weight,
        "active_weight": args.active_weight,
        "denoise_fraction": args.denoise_fraction,
        "denoise_weight": args.denoise_weight,
        "synthetic_coverage_every": args.synthetic_coverage_every,
        "synthetic_coverage_fraction": args.synthetic_coverage_fraction,
        "synthetic_all_learned_identities": args.synthetic_all_learned_identities,
        "overfit_first_batch": args.overfit_first_batch,
        "diagnostic_only": args.diagnostic_only,
        "lr_final_factor": args.lr_final_factor,
        "mse_warmup_epochs": args.mse_warmup_epochs,
        "history_reconstruction_required": False, "validation_teams_used": False,
        "future_information_used": False, "heuristic_answer_used": False,
        "agent_parameter_count": agent.n_params(),
        "shared_latent_parameter_count": sum(p.numel() for p in agent.shared_latent.parameters()),
        "decoder_parameter_count_training_only": sum(p.numel() for p in decoder.parameters()),
        "local_token_aux_weight": args.local_token_aux_weight,
        "local_decoder_parameter_count_training_only": (
            sum(p.numel() for p in local_decoder.parameters()) if local_decoder is not None else 0
        ),
        "changed_tensor_count": len(changed), "changed_only_shared_latent": not illegal,
        "epochs": rows, "team_holdout": team, "battle_holdout": battle,
        "live_train": live_train,
    }
    if overfit is not None:
        report["overfit_train"] = overfit
    torch.save(best_agent, args.out / "latent_best.pt")
    torch.save(best_decoder, args.out / "decoder_best.pt")
    if best_local_decoder is not None:
        torch.save(best_local_decoder, args.out / "local_decoder_best.pt")
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.out / "progress.json").write_text(
        json.dumps({"status": "complete", "passed": passed, "best_minimum_recovery": min(team["minimum_recovery"], battle["minimum_recovery"])}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)
    if not passed and not args.diagnostic_only:
        raise SystemExit(2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--replay-train-manifest", type=Path)
    parser.add_argument("--decoder-checkpoint", type=Path)
    parser.add_argument("--replay-shard-buffer", type=int, default=16)
    parser.add_argument("--live-mix-every", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--overfit-first-batch", action="store_true")
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument(
        "--evaluate-live-train",
        action="store_true",
        help="Report recovery on the authoritative live training split for train/holdout diagnosis.",
    )
    parser.add_argument("--direct-input-control", action="store_true")
    parser.add_argument("--freeze-trunk", action="store_true")
    parser.add_argument("--tied-identity-decoder", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--trunk-lr", type=float, default=1e-4)
    parser.add_argument("--decoder-lr", type=float, default=3e-4)
    parser.add_argument(
        "--local-token-aux-weight",
        type=float,
        default=0.0,
        help=(
            "Training-only weight for reconstructing each creature's exact facts from its own "
            "output token, preventing team-combination memorization without a deployed bypass."
        ),
    )
    parser.add_argument("--decoder-slot-dim", type=int, default=256)
    parser.add_argument("--contextual-decoder", action="store_true")
    parser.add_argument("--flat-decoder", action="store_true")
    parser.add_argument("--simple-flat-decoder", action="store_true")
    parser.add_argument("--linear-flat-decoder", action="store_true")
    parser.add_argument("--categorical-identity-heads", action="store_true")
    parser.add_argument("--categorical-all-identity-gate", action="store_true")
    parser.add_argument("--id-weight", type=float, default=2.0)
    parser.add_argument("--id-regression-weight", type=float, default=5.0)
    parser.add_argument("--raw-weight", type=float, default=1.0)
    parser.add_argument("--field-weight", type=float, default=1.0)
    parser.add_argument("--worst-field-weight", type=float, default=0.0)
    parser.add_argument("--rare-value-weight", type=float, default=0.0)
    parser.add_argument("--identity-embedding-weight", type=float, default=0.0)
    parser.add_argument("--tail-fraction", type=float, default=0.25)
    parser.add_argument("--ordinal-ev-decoder", action="store_true")
    parser.add_argument("--ordinal-all-identities", action="store_true")
    parser.add_argument("--numeric-all-identity-gate", action="store_true")
    parser.add_argument("--cumulative-all-identities", action="store_true")
    parser.add_argument("--linear-id-values", action="store_true")
    parser.add_argument("--linear-exact-values", action="store_true")
    parser.add_argument("--categorical-cross-entropy", action="store_true")
    parser.add_argument("--code-scaled-identities", action="store_true")
    parser.add_argument("--priority-raw-blocks", default="")
    parser.add_argument("--priority-raw-weight", type=float, default=1.0)
    parser.add_argument("--legality-weight", type=float, default=1.0)
    parser.add_argument("--active-weight", type=float, default=1.0)
    parser.add_argument("--denoise-fraction", type=float, default=0.0)
    parser.add_argument("--denoise-weight", type=float, default=0.0)
    parser.add_argument("--synthetic-coverage-every", type=int, default=0)
    parser.add_argument("--synthetic-coverage-fraction", type=float, default=0.25)
    parser.add_argument("--synthetic-all-learned-identities", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--lr-final-factor", type=float, default=1.0)
    parser.add_argument("--mse-warmup-epochs", type=int, default=0)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--numeric-tolerance", type=float, default=0.05)
    parser.add_argument("--target", type=float, default=0.95)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=20261069)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
