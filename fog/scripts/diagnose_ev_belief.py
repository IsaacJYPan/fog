"""Read-only per-stat EV target/prediction diagnostic for the belief model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fog.config import load_config_with_args
from fog.models.agent import FogAgent
from fog.training.dataset import StreamingReplayDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--compatible", action="store_true")
    args = parser.parse_args()

    cfg = load_config_with_args(args.config)
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    agent = FogAgent.from_tables(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    compatibility = agent.load_compatible_state_dict(state) if args.compatible else None
    if not args.compatible:
        agent.load_state_dict(state)
    agent.eval()
    dataset = StreamingReplayDataset(args.manifest, shard_buffer=16)
    n_classes = agent.belief.sizes["ev"]
    counts = torch.zeros(6, dtype=torch.long)
    correct = torch.zeros(6, dtype=torch.long)
    nll = torch.zeros(6, dtype=torch.float64)
    target_hist = torch.zeros(6, n_classes, dtype=torch.long)
    pred_hist = torch.zeros(6, n_classes, dtype=torch.long)

    with torch.no_grad():
        for batch in dataset.batches(
            args.batch_size, shuffle=False, seed=20261221, device=device
        ):
            out = agent(batch)
            logits = out.belief.logits["ev"]
            targets = batch["belief_targets"]["ev"]
            valid = batch["belief_valid"]["ev"]
            predictions = logits.argmax(dim=-1)
            log_probs = logits.log_softmax(dim=-1)
            for stat in range(6):
                mask = valid[:, :, stat] & (targets[:, :, stat] >= 0)
                if not bool(mask.any()):
                    continue
                truth = targets[:, :, stat][mask]
                pred = predictions[:, :, stat][mask]
                truth_log_prob = log_probs[:, :, stat].gather(
                    -1, targets[:, :, stat].clamp(min=0).unsqueeze(-1)
                ).squeeze(-1)[mask]
                counts[stat] += truth.numel()
                correct[stat] += (pred == truth).sum().cpu()
                nll[stat] += (-truth_log_prob).sum().cpu()
                target_hist[stat] += torch.bincount(
                    truth.cpu(), minlength=n_classes
                )
                pred_hist[stat] += torch.bincount(pred.cpu(), minlength=n_classes)

    names = ("hp", "atk", "def", "spa", "spd", "spe")
    stats = {}
    for index, name in enumerate(names):
        total = int(counts[index])
        target_top = torch.topk(target_hist[index], min(8, n_classes))
        pred_top = torch.topk(pred_hist[index], min(8, n_classes))
        stats[name] = {
            "count": total,
            "top1": float(correct[index] / max(1, total)),
            "nll": float(nll[index] / max(1, total)),
            "majority_target_accuracy": float(target_hist[index].max() / max(1, total)),
            "target_modes": [
                {"class": int(cls), "count": int(count)}
                for count, cls in zip(target_top.values, target_top.indices, strict=True)
            ],
            "prediction_modes": [
                {"class": int(cls), "count": int(count)}
                for count, cls in zip(pred_top.values, pred_top.indices, strict=True)
            ],
        }
    report = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "compatibility": compatibility,
        "stats": stats,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
