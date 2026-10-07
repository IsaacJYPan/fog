"""Sample-based hidden-field calibration and exchangeable joint-set accuracy."""

from __future__ import annotations

from collections import Counter

import numpy as np


def set_key(row):
    return (*row[:3], tuple(sorted(int(move) for move in row[3:7])), *row[7:])


class SampleMetrics:
    def __init__(self):
        self.fields = {}
        self.joint_count = self.joint_correct = self.recall_count = self.recall_correct = 0

    def update(self, samples, truth, verified, known):
        samples = np.asarray(samples)
        truth, verified, known = map(np.asarray, (truth, verified, known))
        if samples.ndim != 3 or samples.shape[1:] != (6, 15) or not len(samples):
            raise ValueError("Expected complete categorical team samples [draw,6,15]")
        scalar_families = {1: "item", 2: "ability", 7: "nature", 14: "tera"}
        scalar_families.update({column: "ev" for column in range(8, 14)})
        for slot in range(6):
            hidden = verified[slot] & ~known[slot] & (truth[slot] >= 0)
            for column, family in scalar_families.items():
                if hidden[column]:
                    self._field(family, samples[:, slot, column].tolist(), int(truth[slot, column]))
            if verified[slot, 3:7].all() and hidden[3:7].any():
                self._field(
                    "moveset",
                    [tuple(sorted(row)) for row in samples[:, slot, 3:7]],
                    tuple(sorted(truth[slot, 3:7])),
                )
            # Partial or contradictory submitted labels cannot count as exact sets.
            if verified[slot].all() and hidden.any():
                correct = sum(set_key(row) == set_key(truth[slot]) for row in samples[:, slot])
                self.joint_correct += correct
                self.joint_count += len(samples)
                self.recall_correct += bool(correct)
                self.recall_count += 1

    def _field(self, family, values, target):
        counts = Counter(values)
        modal = max(sorted(counts), key=counts.get)
        confidence = counts[modal] / len(values)
        rows = self.fields.setdefault(family, [])
        # These are empirical sampler marginals. Route trace probabilities are
        # probabilities of sampled paths, not exact unordered set likelihoods.
        brier = (
            sum((count / len(values)) ** 2 for count in counts.values())
            - 2 * counts[target] / len(values)
            + 1
        )
        rows.append((float(modal == target), confidence, brier, counts[target] / len(values)))

    def report(self):
        families = {}
        for name, rows in self.fields.items():
            values = np.asarray(rows)
            bins = np.minimum((values[:, 1] * 15).astype(int), 14)
            ece = sum(
                np.mean(bins == index)
                * abs(values[bins == index, 0].mean() - values[bins == index, 1].mean())
                for index in range(15)
                if np.any(bins == index)
            )
            families[name] = {
                "fields": len(rows),
                "empirical_top1": float(values[:, 0].mean()),
                "empirical_top1_ece": float(ece),
                "empirical_brier": float(values[:, 2].mean()),
                "sample_accuracy": float(values[:, 3].mean()),
            }
        return {
            "families": families,
            "joint_set_accuracy": self.joint_correct / self.joint_count
            if self.joint_count
            else None,
            "joint_sample_count": self.joint_count,
            "joint_recall_at_k": self.recall_correct / self.recall_count
            if self.recall_count
            else None,
            "eligible_set_contexts": self.recall_count,
            "calibration_scope": "Monte Carlo plug-in marginals at a fixed draw count; sampling error remains. Moves are an unordered set. Unverified labels are excluded.",
        }
