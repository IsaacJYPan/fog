"""Analytic control for the 2,048-number contextual-trunk information gate.

This is evaluation-only and is never a deployable encoder.  It establishes that every required
current-state input fits in the proposed width and that the held-out evaluator can recover it
exactly.  Learned trunks are still evaluated with independently trained decoders.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fog.config import load_config
from fog.models.agent import FogAgent
from fog.models.shared_latent import SharedBattleLatent
from fog.training.shared_latent_reconstruction import TensorObservationDataset


def pack_exact_current(
    batch: dict,
    embeds,
    *,
    latent_dim: int = 2048,
) -> torch.Tensor:
    """Pack required exact current facts into one fixed-width control vector."""
    normalized_ids = SharedBattleLatent.normalized_ids(batch["entity_ids"], embeds)
    entity = torch.cat([batch["entities"], normalized_ids], dim=-1).flatten(1)
    active = batch["field"].new_zeros(batch["field"].shape[0], 12)
    active.scatter_(1, batch["active_idx"].clamp(0, 11), 1.0)
    exact = torch.cat(
        [
            entity,
            batch["field"],
            active,
            batch["legality_mask"].to(batch["field"].dtype),
        ],
        dim=-1,
    )
    if exact.shape[-1] > latent_dim:
        raise ValueError(f"{exact.shape[-1]} exact values do not fit in width {latent_dim}")
    return torch.nn.functional.pad(exact, (0, latent_dim - exact.shape[-1]))


def unpack_exact_current(
    latent: torch.Tensor,
    embeds,
) -> dict[str, torch.Tensor]:
    """Invert :func:`pack_exact_current` without using any source side arguments."""
    raw_width = 140
    id_width = 17
    entity_width = raw_width + id_width
    cursor = 0
    entity = latent[:, cursor : cursor + 12 * entity_width].reshape(
        latent.shape[0], 12, entity_width
    )
    cursor += 12 * entity_width
    raw = entity[..., :raw_width]
    normalized_ids = entity[..., raw_width:]
    scales = normalized_ids.new_tensor(
        [
            embeds.species.num_embeddings - 1,
            embeds.item.num_embeddings - 1,
            embeds.ability.num_embeddings - 1,
            *([embeds.move.num_embeddings - 1] * 5),
            embeds.nature.num_embeddings - 1,
            *([embeds.ev.num_embeddings - 1] * 6),
            embeds.item.num_embeddings - 1,
            embeds.ability.num_embeddings - 1,
        ]
    ).clamp_min(1.0)
    ids = torch.round(normalized_ids * scales).long()
    field = latent[:, cursor : cursor + 51]
    cursor += 51
    active = latent[:, cursor : cursor + 12]
    cursor += 12
    legality = latent[:, cursor : cursor + 14] >= 0.5
    return {
        "entities": raw,
        "entity_ids": ids,
        "field": field,
        "active_idx": torch.stack(
            [active[:, :6].argmax(-1), active[:, 6:].argmax(-1) + 6], dim=-1
        ),
        "legality_mask": legality,
    }


def evaluate_split(dataset: TensorObservationDataset, embeds, *, batch_size: int) -> dict:
    rows = 0
    raw_columns = torch.ones(140, dtype=torch.bool)
    field_columns = torch.ones(51, dtype=torch.bool)
    id_fields = torch.ones(17, dtype=torch.bool)
    legal_rows = 0
    active_rows = 0
    for batch in dataset.batches(batch_size, shuffle=False, seed=0, device="cpu"):
        recovered = unpack_exact_current(pack_exact_current(batch, embeds), embeds)
        count = batch["entities"].shape[0]
        rows += count
        raw_columns &= (
            recovered["entities"].eq(batch["entities"]).reshape(-1, 140).all(0)
        )
        field_columns &= recovered["field"].eq(batch["field"]).all(0)
        id_fields &= (
            recovered["entity_ids"].eq(batch["entity_ids"]).reshape(-1, 17).all(0)
        )
        legal_rows += int(recovered["legality_mask"].eq(batch["legality_mask"]).all(-1).sum())
        active_rows += int(recovered["active_idx"].eq(batch["active_idx"]).all(-1).sum())
    result = {
        "rows": rows,
        "raw_column_exact": float(raw_columns.all()),
        "field_column_exact": float(field_columns.all()),
        "identity_field_exact": float(id_fields.all()),
        "legality_row_accuracy": legal_rows / max(1, rows),
        "active_row_accuracy": active_rows / max(1, rows),
    }
    result["minimum_recovery"] = min(
        result["raw_column_exact"],
        result["field_column_exact"],
        result["identity_field_exact"],
        result["legality_row_accuracy"],
        result["active_row_accuracy"],
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    agent = FogAgent.from_tables(load_config(args.config)).cpu().eval()
    report = {
        "status": "complete",
        "control_only": True,
        "deployable": False,
        "latent_dim": 2048,
        "packed_exact_values": 1961,
        "unused_dimensions": 87,
        "history_required": False,
        "validation_teams_used": False,
        "future_information_used": False,
        "splits": {},
    }
    for split in ("team_holdout", "battle_holdout"):
        dataset = TensorObservationDataset(args.data / f"{split}.pt")
        report["splits"][split] = evaluate_split(
            dataset,
            agent.backbone.embeds,
            batch_size=args.batch_size,
        )
    report["passed"] = all(
        result["minimum_recovery"] == 1.0 for result in report["splits"].values()
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
