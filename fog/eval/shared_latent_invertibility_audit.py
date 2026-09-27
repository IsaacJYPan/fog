"""Directly prove exact current inputs are recoverable from the shared learned vector."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fog.config import load_config
from fog.eval.encoder_information_probe_live import TensorObservationDataset


def run(args: argparse.Namespace) -> dict:
    from fog.models.agent import FogAgent

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    agent = FogAgent.from_tables(load_config(args.config)).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False))
    agent.eval()
    trunk = agent.shared_latent
    if trunk is None:
        raise ValueError("shared latent trunk is not enabled")
    entity_inverse = torch.linalg.pinv(trunk.exact_entity.weight.transpose(0, 1))
    field_inverse = torch.linalg.pinv(trunk.field.weight.transpose(0, 1))

    def audit(path: Path) -> dict:
        dataset = TensorObservationDataset(path)
        totals = {
            "rows": 0, "raw_n": 0, "raw_within": 0, "raw_max": 0.0,
            "id_n": 0, "id_exact": 0, "id_max": 0.0,
            "field_n": 0, "field_within": 0, "field_max": 0.0,
            "metadata_n": 0, "metadata_exact": 0,
        }
        with torch.no_grad():
            for batch in dataset.batches(args.batch_size, shuffle=False, seed=0, device=device):
                contexts = agent.encode_contexts(batch)
                latent = trunk(batch, contexts, agent.backbone.embeds)
                entity_tokens = latent[:, : trunk.entity_width].reshape(
                    latent.shape[0], 12, trunk.entity_dim
                )
                recovered_exact = entity_tokens[..., : trunk.exact_entity_dim] @ entity_inverse
                recovered_raw = recovered_exact[..., : batch["entities"].shape[-1]]
                recovered_ids = recovered_exact[..., batch["entities"].shape[-1] :]
                raw_error = (recovered_raw - batch["entities"]).abs()
                scales = batch["entity_ids"].new_tensor(
                    [
                        agent.backbone.embeds.species.num_embeddings - 1,
                        agent.backbone.embeds.item.num_embeddings - 1,
                        agent.backbone.embeds.ability.num_embeddings - 1,
                        *([agent.backbone.embeds.move.num_embeddings - 1] * 5),
                        agent.backbone.embeds.nature.num_embeddings - 1,
                        *([agent.backbone.embeds.ev.num_embeddings - 1] * 6),
                        agent.backbone.embeds.item.num_embeddings - 1,
                        agent.backbone.embeds.ability.num_embeddings - 1,
                    ], dtype=torch.float32,
                ).clamp_min(1)
                ids = torch.round(recovered_ids * scales).long().clamp_min(0)
                ids = torch.minimum(ids, scales.long())
                id_error = (ids - batch["entity_ids"]).abs()
                field_token = latent[:, trunk.field_offset : trunk.history_offset]
                recovered_field = field_token @ field_inverse
                field_error = (recovered_field - batch["field"]).abs()
                metadata = latent[:, trunk.metadata_offset : trunk.metadata_offset + 26]
                active = batch["field"].new_zeros(batch["field"].shape[0], 12)
                active.scatter_(1, batch["active_idx"].clamp(0, 11), 1.0)
                expected_metadata = torch.cat([active, batch["legality_mask"].float()], dim=-1)
                metadata_error = (metadata - expected_metadata).abs()
                totals["rows"] += int(latent.shape[0])
                totals["raw_n"] += raw_error.numel()
                totals["raw_within"] += int((raw_error <= args.tolerance).sum())
                totals["raw_max"] = max(totals["raw_max"], float(raw_error.max()))
                totals["id_n"] += id_error.numel()
                totals["id_exact"] += int((id_error == 0).sum())
                totals["id_max"] = max(totals["id_max"], float(id_error.max()))
                totals["field_n"] += field_error.numel()
                totals["field_within"] += int((field_error <= args.tolerance).sum())
                totals["field_max"] = max(totals["field_max"], float(field_error.max()))
                totals["metadata_n"] += metadata_error.numel()
                totals["metadata_exact"] += int((metadata_error == 0).sum())
        return {
            "rows": totals["rows"],
            "raw_within_tolerance": totals["raw_within"] / totals["raw_n"],
            "raw_max_abs_error": totals["raw_max"],
            "identity_exact_accuracy": totals["id_exact"] / totals["id_n"],
            "identity_max_index_error": totals["id_max"],
            "field_within_tolerance": totals["field_within"] / totals["field_n"],
            "field_max_abs_error": totals["field_max"],
            "active_and_legality_exact_accuracy": totals["metadata_exact"] / totals["metadata_n"],
        }

    entity_rank = int(torch.linalg.matrix_rank(trunk.exact_entity.weight.float()))
    field_rank = int(torch.linalg.matrix_rank(trunk.field.weight.float()))
    report = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "decoder": "analytic_pseudoinverse_of_learned_projection",
        "raw_shortcut": False,
        "entity_projection_shape": list(trunk.exact_entity.weight.shape),
        "entity_projection_rank": entity_rank,
        "field_projection_shape": list(trunk.field.weight.shape),
        "field_projection_rank": field_rank,
        "team_holdout": audit(args.data / "team_holdout.pt"),
        "battle_holdout": audit(args.data / "battle_holdout.pt"),
        "validation_teams_used": False,
        "history_reconstruction_required": False,
    }
    report["passed"] = (
        entity_rank == trunk.exact_entity.weight.shape[1]
        and field_rank == trunk.field.weight.shape[1]
        and all(
            split["raw_within_tolerance"] >= args.target
            and split["identity_exact_accuracy"] >= args.target
            and split["field_within_tolerance"] >= args.target
            and split["active_and_legality_exact_accuracy"] >= args.target
            for split in (report["team_holdout"], report["battle_holdout"])
        )
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--target", type=float, default=0.95)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
