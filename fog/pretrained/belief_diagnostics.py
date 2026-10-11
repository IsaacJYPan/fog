"""Score verified decoder distributions without Monte Carlo sampling noise."""

from __future__ import annotations

import math

import torch


class DistributionMetrics:
    def __init__(self):
        self.families = {}

    def update(self, logits, targets, valid):
        for family, values in logits.items():
            selected = valid[family] & (targets[family] >= 0)
            if not bool(selected.any()):
                continue
            values, truth = values[selected].float(), targets[family][selected]
            if bool((truth >= values.shape[-1]).any()):
                raise ValueError("Verified diagnostic target outside vocabulary")
            logp = values.log_softmax(-1)
            true_logp = logp.gather(-1, truth[:, None]).squeeze(-1)
            if not bool(torch.isfinite(true_logp).all()):
                raise ValueError("Verified diagnostic target excluded by decoder constraints")
            probabilities = logp.exp()
            entropy = -(torch.where(torch.isfinite(logp), logp, 0) * probabilities).sum(-1)
            row = self.families.setdefault(
                family, dict(fields=0, correct=0, nll=0.0, normalized_nll=0.0, true_probability=0.0, entropy=0.0)
            )
            row["fields"] += len(truth)
            row["correct"] += int((values.argmax(-1) == truth).sum())
            nll = float(-true_logp.sum())
            row["nll"] += nll
            row["normalized_nll"] += nll / math.log(values.shape[-1])
            row["true_probability"] += float(true_logp.exp().sum())
            row["entropy"] += float(entropy.sum())

    def report(self):
        return {
            family: {
                "fields": row["fields"],
                "top1": row["correct"] / row["fields"],
                **{key: row[key] / row["fields"] for key in ("nll", "normalized_nll", "true_probability", "entropy")},
            }
            for family, row in self.families.items()
        }

    def merge(self, other):
        for family, values in other.families.items():
            row = self.families.setdefault(family, {key: 0 for key in values})
            for key, value in values.items():
                row[key] += value
