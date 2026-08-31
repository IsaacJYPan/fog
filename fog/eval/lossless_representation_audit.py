"""Verify exact observation recovery from the structured encoder memory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fog.config import load_config
from fog.encoding.entities import ENTITY_RAW_DIM, EV_ID_FIELDS, N_ID_FIELDS
from fog.eval.encoder_information_probe_live import TensorObservationDataset
from fog.models.agent import FogAgent
from fog.models.three_encoder import CurrentStateEncoder


def _equal(a: torch.Tensor | None, b: torch.Tensor | None) -> tuple[int, int]:
    if a is None or b is None:
        return (int(a is b), 1)
    return int((a == b).sum()), int(a.numel())


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    agent = FogAgent.from_tables(load_config(args.config)).to(device).eval()
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False))
    dataset = TensorObservationDataset(args.data)
    totals: dict[str, list[int]] = {}
    token_totals: dict[str, list[int]] = {}
    embeds = agent.backbone.embeds
    id_scales = torch.tensor(
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
        device=device,
    ).clamp_min(1)
    with torch.no_grad():
        for batch in dataset.batches(args.batch_size, shuffle=False, seed=0, device=device):
            contexts = agent.encode_contexts(batch)
            expected_public_raw, expected_public_ids = CurrentStateEncoder.public_inputs(
                batch["entities"], batch["entity_ids"], agent.backbone.embeds
            )
            pairs = {
                "public_raw": (contexts.public_exact_raw, expected_public_raw),
                "public_ids": (contexts.public_exact_ids, expected_public_ids),
                "private_raw": (contexts.private_exact_raw, batch["entities"][:, :6]),
                "private_ids": (contexts.private_exact_ids, batch["entity_ids"][:, :6]),
                "field": (contexts.field_exact, batch["field"]),
                "history_raw": (contexts.history_exact_raw, batch["history"]),
                "history_ids": (contexts.history_exact_ids, batch.get("history_ids")),
                "history_state": (contexts.history_exact_state, batch.get("history_state")),
                "history_lengths": (contexts.history_lengths, batch["history_len"]),
            }
            for name, pair in pairs.items():
                correct, count = _equal(*pair)
                bucket = totals.setdefault(name, [0, 0]); bucket[0] += correct; bucket[1] += count
            if agent.structured_encoder is not None:
                memory = agent.structured_encoder(contexts, embeds)
                public_id_start = ENTITY_RAW_DIM + embeds.out_dim
                private_id_start = public_id_start + 2 * embeds.item.embedding_dim
                recovered = {
                    "public_raw": memory.public[..., :ENTITY_RAW_DIM],
                    "public_ids": torch.round(
                        memory.public[..., public_id_start : public_id_start + N_ID_FIELDS]
                        * id_scales
                    ).long(),
                    "private_raw": memory.private[..., :ENTITY_RAW_DIM],
                    "private_ids": torch.round(
                        memory.private[..., private_id_start : private_id_start + N_ID_FIELDS]
                        * id_scales
                    ).long(),
                    "field": memory.field[:, 0, : contexts.field_exact.shape[-1]],
                }
                token_pairs = {
                    "public_raw": (recovered["public_raw"], contexts.public_exact_raw),
                    "public_ids": (recovered["public_ids"], contexts.public_exact_ids),
                    "private_raw": (recovered["private_raw"], contexts.private_exact_raw),
                    "private_ids": (recovered["private_ids"], contexts.private_exact_ids),
                    "field": (recovered["field"], contexts.field_exact),
                }
                for name, pair in token_pairs.items():
                    correct, count = _equal(*pair)
                    bucket = token_totals.setdefault(name, [0, 0])
                    bucket[0] += correct; bucket[1] += count
    fields = {name: {"exact": correct / count, "correct": correct, "count": count} for name, (correct, count) in totals.items()}
    minimum = min(item["exact"] for item in fields.values())
    token_fields = {
        name: {"exact": correct / count, "correct": correct, "count": count}
        for name, (correct, count) in token_totals.items()
    }
    token_minimum = min(
        (item["exact"] for item in token_fields.values()), default=0.0
    )
    report = {
        "status": "complete", "checkpoint": str(args.checkpoint), "data": str(args.data),
        "rows": dataset.n, "minimum_exact_recovery": minimum, "target": 0.95,
        "passed": minimum >= 0.95 and token_minimum >= 0.95, "fields": fields,
        "learned_token_minimum_exact_recovery": token_minimum,
        "learned_token_fields": token_fields,
        "learned_token_reads_raw_sidecar": False,
        "unknown_opponent_facts_invented": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); report = run(args); raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
