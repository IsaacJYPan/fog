"""Per-family hidden-information quality on a replay-level held-out manifest.

This is read-only evaluation. It reports top-1/top-3 accuracy, negative log likelihood, and
top-1 calibration for every hidden field family on exactly the censored rows used by replay BC.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


@dataclass
class FamilyAccumulator:
    bins: int = 15
    count: int = 0
    top1: int = 0
    top3: int = 0
    nll_sum: float = 0.0
    bin_count: np.ndarray = field(default_factory=lambda: np.zeros(15, dtype=np.int64))
    bin_confidence: np.ndarray = field(default_factory=lambda: np.zeros(15, dtype=np.float64))
    bin_correct: np.ndarray = field(default_factory=lambda: np.zeros(15, dtype=np.int64))

    def update(self, logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor) -> None:
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_targets = targets.reshape(-1)
        mask = valid.reshape(-1) & (flat_targets >= 0)
        if not bool(mask.any()):
            return
        selected = flat_logits[mask]
        labels = flat_targets[mask]
        finite_target = torch.isfinite(selected.gather(1, labels[:, None]).squeeze(1))
        selected = selected[finite_target]
        labels = labels[finite_target]
        if not len(labels):
            return
        probabilities = selected.softmax(-1)
        confidence, prediction = probabilities.max(-1)
        correct = prediction.eq(labels)
        k = min(3, probabilities.shape[-1])
        top3 = probabilities.topk(k, dim=-1).indices.eq(labels[:, None]).any(-1)
        target_probability = probabilities.gather(1, labels[:, None]).squeeze(1).clamp_min(1e-12)
        conf = confidence.detach().cpu().numpy()
        corr = correct.detach().cpu().numpy()
        indices = np.minimum((conf * self.bins).astype(np.int64), self.bins - 1)
        self.count += len(labels)
        self.top1 += int(correct.sum())
        self.top3 += int(top3.sum())
        self.nll_sum += float((-target_probability.log()).sum())
        np.add.at(self.bin_count, indices, 1)
        np.add.at(self.bin_confidence, indices, conf)
        np.add.at(self.bin_correct, indices, corr.astype(np.int64))

    def merge(self, other: FamilyAccumulator) -> None:
        self.count += other.count
        self.top1 += other.top1
        self.top3 += other.top3
        self.nll_sum += other.nll_sum
        self.bin_count += other.bin_count
        self.bin_confidence += other.bin_confidence
        self.bin_correct += other.bin_correct

    def report(self) -> dict:
        ece = 0.0
        for count, confidence, correct in zip(
            self.bin_count, self.bin_confidence, self.bin_correct
        ):
            if count:
                ece += count / max(1, self.count) * abs(correct / count - confidence / count)
        return {
            "count": self.count,
            "top1": self.top1 / self.count if self.count else None,
            "top3": self.top3 / self.count if self.count else None,
            "nll": self.nll_sum / self.count if self.count else None,
            "top1_ece": float(ece),
        }


def evaluate(agent, dataset, *, batch_size: int, device: torch.device) -> dict:
    families: dict[str, FamilyAccumulator] = {}
    agent.eval()
    with torch.no_grad():
        for batch in dataset.batches(batch_size, shuffle=False, seed=0, device=device):
            output = agent(batch).belief
            if output is None:
                raise RuntimeError("checkpoint has no hidden-information predictor")
            for family, logits in output.logits.items():
                accumulator = families.setdefault(family, FamilyAccumulator())
                accumulator.update(
                    logits, batch["belief_targets"][family], batch["belief_valid"][family]
                )
    damage = FamilyAccumulator()
    for family in ("item", "ability", "move", "nature", "ev", "tera"):
        if family in families:
            damage.merge(families[family])
    return {
        "families": {name: value.report() for name, value in families.items()},
        "damage_relevant_all_fields": damage.report(),
    }


def main() -> None:
    from fog.config import load_config_with_args
    from fog.models.agent import FogAgent
    from fog.training.dataset import ReplayDataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", required=True, help="LABEL=PATH")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config_with_args(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = ReplayDataset(args.manifest)
    report = {"manifest": str(args.manifest), "rows": len(dataset), "checkpoints": {}}
    for specification in args.checkpoint:
        label, path = specification.split("=", 1)
        agent = FogAgent.from_tables(cfg).to(device)
        agent.load_state_dict(torch.load(path, map_location=device))
        report["checkpoints"][label] = {"path": path, **evaluate(
            agent, dataset, batch_size=args.batch_size, device=device
        )}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[replay-belief-quality] report -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
