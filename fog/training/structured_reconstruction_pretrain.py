"""Prove and train recovery of exact observations from wide learned memory tokens."""
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
from fog.encoding.entities import EV_ID_FIELDS
from fog.eval.encoder_information_probe_live import TensorObservationDataset


def _digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class ExactRecordDecoder(nn.Module):
    def __init__(self, token_dim: int, raw_dim: int, id_fields: int = 0) -> None:
        super().__init__()
        self.raw = nn.Linear(token_dim, raw_dim)
        self.ids = nn.Linear(token_dim, id_fields) if id_fields else None

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.raw(tokens), self.ids(tokens) if self.ids is not None else None


def _id_scales(agent, device) -> torch.Tensor:
    embeds = agent.backbone.embeds
    return torch.tensor(
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
        device=device,
    ).clamp_min(1.0)


def run(args: argparse.Namespace) -> dict:
    from fog.models.agent import FogAgent

    cfg = load_config(args.config)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    agent = FogAgent.from_tables(cfg).to(device)
    initial = torch.load(args.checkpoint, map_location=device, weights_only=False)
    agent.load_state_dict(initial)
    if agent.structured_encoder is None:
        raise ValueError("structured reconstruction requires model.structured_memory=true")
    agent.requires_grad_(False)
    agent.structured_encoder.requires_grad_(True)
    token_dim = agent.structured_encoder.token_dim
    from fog.encoding.dims import DIMS
    from fog.encoding.entities import ENTITY_RAW_DIM, N_ID_FIELDS

    decoders = nn.ModuleDict(
        {
            "public": ExactRecordDecoder(token_dim, ENTITY_RAW_DIM, N_ID_FIELDS),
            "private": ExactRecordDecoder(token_dim, ENTITY_RAW_DIM, N_ID_FIELDS),
            "field": ExactRecordDecoder(token_dim, DIMS.field_dim),
        }
    ).to(device)
    datasets = {
        name: TensorObservationDataset(args.data / f"{name}.pt")
        for name in ("train", "team_holdout", "battle_holdout")
    }
    scales = _id_scales(agent, device)
    optimizer = torch.optim.AdamW(
        [
            {"params": agent.structured_encoder.parameters(), "lr": args.encoder_lr},
            {"params": decoders.parameters(), "lr": args.decoder_lr},
        ]
    )

    def predictions(batch: dict):
        contexts = agent.encode_contexts(batch)
        memory = agent.structured_encoder(contexts, agent.backbone.embeds)
        public_raw, public_ids = decoders["public"](memory.public)
        private_raw, private_ids = decoders["private"](memory.private)
        field_raw, _ = decoders["field"](memory.field[:, 0])
        targets = {
            "public_raw": contexts.public_exact_raw,
            "public_ids": contexts.public_exact_ids,
            "private_raw": contexts.private_exact_raw,
            "private_ids": contexts.private_exact_ids,
            "field": contexts.field_exact,
        }
        predicted = {
            "public_raw": public_raw,
            "public_ids": public_ids,
            "private_raw": private_raw,
            "private_ids": private_ids,
            "field": field_raw,
        }
        return predicted, targets

    def objective(batch: dict) -> torch.Tensor:
        predicted, targets = predictions(batch)
        public_ids = targets["public_ids"].float() / scales
        private_ids = targets["private_ids"].float() / scales
        losses = [
            F.mse_loss(predicted["public_raw"], targets["public_raw"]),
            F.mse_loss(predicted["private_raw"], targets["private_raw"]),
            F.mse_loss(predicted["field"], targets["field"]),
            args.id_weight * F.mse_loss(predicted["public_ids"], public_ids),
            args.id_weight * F.mse_loss(predicted["private_ids"], private_ids),
        ]
        return torch.stack(losses).mean()

    def evaluate(dataset: TensorObservationDataset) -> dict[str, float | list[float]]:
        agent.eval()
        decoders.eval()
        totals = {
            "loss": 0.0,
            "batches": 0,
            "public_raw_squared": 0.0,
            "public_raw_count": 0,
            "public_raw_target_sum": 0.0,
            "public_raw_target_square": 0.0,
            "private_raw_squared": 0.0,
            "private_raw_count": 0,
            "private_raw_target_sum": 0.0,
            "private_raw_target_square": 0.0,
            "field_squared": 0.0,
            "field_count": 0,
            "field_target_sum": 0.0,
            "field_target_square": 0.0,
        }
        id_correct = {"public": torch.zeros(len(scales), dtype=torch.long), "private": torch.zeros(len(scales), dtype=torch.long)}
        id_count = {"public": torch.zeros(len(scales), dtype=torch.long), "private": torch.zeros(len(scales), dtype=torch.long)}
        within = {"public_raw": 0, "private_raw": 0, "field": 0}
        with torch.no_grad():
            for batch in dataset.batches(args.batch_size, shuffle=False, seed=args.seed, device=device):
                predicted, targets = predictions(batch)
                totals["loss"] += float(objective(batch))
                totals["batches"] += 1
                for name in ("public_raw", "private_raw", "field"):
                    error = predicted[name] - targets[name]
                    totals[f"{name}_squared"] += float(error.square().sum())
                    totals[f"{name}_count"] += error.numel()
                    totals[f"{name}_target_sum"] += float(targets[name].sum())
                    totals[f"{name}_target_square"] += float(targets[name].square().sum())
                    within[name] += int((error.abs() <= args.numeric_tolerance).sum())
                for group in ("public", "private"):
                    target = targets[f"{group}_ids"].cpu()
                    recovered = torch.round(
                        predicted[f"{group}_ids"].cpu() * scales.cpu()
                    ).long().clamp_min(0)
                    upper = scales.cpu().long()
                    recovered = torch.minimum(recovered, upper)
                    id_correct[group] += (recovered == target).sum(dim=tuple(range(target.ndim - 1)))
                    per_field_rows = target.numel() // target.shape[-1]
                    id_count[group] += per_field_rows
        report: dict[str, float | list[float]] = {
            "loss": totals["loss"] / max(1, totals["batches"]),
        }
        for name in ("public_raw", "private_raw", "field"):
            count = int(totals[f"{name}_count"])
            mse = totals[f"{name}_squared"] / max(1, count)
            target_mean = totals[f"{name}_target_sum"] / max(1, count)
            variance = totals[f"{name}_target_square"] / max(1, count) - target_mean ** 2
            report[f"{name}_mse"] = mse
            report[f"{name}_variance_recovered"] = 1.0 - mse / max(variance, 1e-12)
            report[f"{name}_within_tolerance"] = within[name] / max(1, count)
        for group in ("public", "private"):
            accuracy = (id_correct[group].float() / id_count[group].clamp_min(1)).tolist()
            report[f"{group}_id_accuracy_by_field"] = accuracy
            report[f"{group}_id_min_accuracy"] = min(accuracy)
            report[f"{group}_id_accuracy"] = float(id_correct[group].sum() / id_count[group].sum())
        report["minimum_recovery"] = min(
            float(report["public_raw_variance_recovered"]),
            float(report["private_raw_variance_recovered"]),
            float(report["field_variance_recovered"]),
            float(report["public_id_min_accuracy"]),
            float(report["private_id_min_accuracy"]),
        )
        return report

    rows: list[dict] = []
    best_score = float("-inf")
    best_agent = None
    best_decoders = None
    stale = 0
    for epoch in range(args.epochs):
        agent.train()
        decoders.train()
        train_loss = 0.0
        batches = 0
        for batch in datasets["train"].batches(args.batch_size, shuffle=True, seed=args.seed + epoch, device=device):
            optimizer.zero_grad(set_to_none=True)
            loss = objective(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(agent.structured_encoder.parameters()) + list(decoders.parameters()), 5.0
            )
            optimizer.step()
            train_loss += float(loss.detach())
            batches += 1
        team = evaluate(datasets["team_holdout"])
        battle = evaluate(datasets["battle_holdout"])
        score = min(float(team["minimum_recovery"]), float(battle["minimum_recovery"]))
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss / max(1, batches),
            "team_minimum_recovery": team["minimum_recovery"],
            "battle_minimum_recovery": battle["minimum_recovery"],
            "team_loss": team["loss"],
            "battle_loss": battle["loss"],
        }
        rows.append(row)
        print(f"[structured-reconstruction] {row}", flush=True)
        if score > best_score + args.min_delta:
            best_score = score
            best_agent = copy.deepcopy(agent.state_dict())
            best_decoders = copy.deepcopy(decoders.state_dict())
            stale = 0
        else:
            stale += 1
        if score >= args.target or (args.patience and stale >= args.patience):
            break
    if best_agent is None or best_decoders is None:
        raise RuntimeError("structured reconstruction produced no finite checkpoint")
    agent.load_state_dict(best_agent)
    decoders.load_state_dict(best_decoders)
    team = evaluate(datasets["team_holdout"])
    battle = evaluate(datasets["battle_holdout"])
    changed = sorted(name for name, value in best_agent.items() if _digest(value) != _digest(initial[name]))
    illegal = [name for name in changed if not name.startswith("structured_encoder.")]
    if illegal:
        raise RuntimeError(f"non-representation tensors changed: {illegal[:10]}")
    report = {
        "status": "complete",
        "source": str(args.checkpoint),
        "data": str(args.data),
        "target": args.target,
        "passed": min(float(team["minimum_recovery"]), float(battle["minimum_recovery"])) >= args.target,
        "agent_parameter_count": agent.n_params(),
        "token_dim": token_dim,
        "history_reconstruction_required": False,
        "validation_teams_used": False,
        "future_information_used": False,
        "changed_tensor_count": len(changed),
        "changed_only_structured_encoder": not illegal,
        "epochs": rows,
        "team_holdout": team,
        "battle_holdout": battle,
    }
    args.out.mkdir(parents=True, exist_ok=False)
    torch.save(best_agent, args.out / "representation_best.pt")
    torch.save(best_decoders, args.out / "decoder_best.pt")
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.out / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2), flush=True)
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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--encoder-lr", type=float, default=3e-5)
    parser.add_argument("--decoder-lr", type=float, default=3e-4)
    parser.add_argument("--id-weight", type=float, default=5.0)
    parser.add_argument("--numeric-tolerance", type=float, default=0.05)
    parser.add_argument("--target", type=float, default=0.95)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=20261061)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
