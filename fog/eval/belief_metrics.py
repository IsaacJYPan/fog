"""Belief-quality metrics (spec §9, §6 dashboard).

  * per-attribute top-1 / top-3 vs turn number (MUST be monotone increasing within games — flat means
    the history encoder is broken)
  * calibration (ECE)
  * on replays: accuracy at least 3 turns BEFORE the actual reveal (predicts early, not just reads log)
"""
from __future__ import annotations

import numpy as np


def expected_calibration_error(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Standard ECE.

    confidences [N] in [0,1] (top-1 prob), correct [N] bool. Returns weighted |acc − conf| over bins.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean()
        conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def accuracy_vs_turn(predictions, labels, turns, bucket: int = 5) -> dict[int, float]:
    """Top-1 accuracy bucketed by turn number (§6 dashboard: MUST be monotone increasing within
    games — a flat curve means the history encoder is broken).

    predictions/labels: [N] top-1 predicted class and true class; turns: [N] turn index. Returns
    {bucket_start_turn: accuracy}. Buckets of `bucket` turns.
    """
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    turns = np.asarray(turns)
    out: dict[int, float] = {}
    for b0 in range(0, int(turns.max()) + 1 if len(turns) else 1, bucket):
        m = (turns >= b0) & (turns < b0 + bucket)
        if m.any():
            out[b0] = float((predictions[m] == labels[m]).mean())
    return out


def pre_reveal_accuracy(predictions, labels, turns, reveal_turns, lead: int = 3) -> float:
    """Accuracy on predictions made at least `lead` turns BEFORE each attribute's actual reveal (§9:
    the belief net predicts early, not just reads the log).

    Every array is [N] and aligned per (attribute-instance, turn). `reveal_turns[i]` is the turn the
    i-th instance's attribute was actually revealed. Only rows with turn <= reveal_turn - lead count.
    """
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    turns = np.asarray(turns)
    reveal_turns = np.asarray(reveal_turns)
    m = turns <= (reveal_turns - lead)
    if not m.any():
        return float("nan")
    return float((predictions[m] == labels[m]).mean())
