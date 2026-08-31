from __future__ import annotations

import numpy as np

from fog.eval.search_trace import world_accuracy


def test_world_accuracy_separates_hidden_revealed_and_duplicates():
    truth = np.arange(90, dtype=np.int64).reshape(6, 15)
    revealed = np.full((6, 15), -1, dtype=np.int64)
    revealed[:, 0] = truth[:, 0]
    exact = truth.copy()
    wrong = truth.copy()
    wrong[:, 1:] += 1
    result = world_accuracy(np.stack([exact, wrong, wrong]), truth, revealed)

    assert result["hidden_fields"] == 84
    assert result["revealed_or_clamped_fields"] == 6
    assert result["exact_true_hidden_world_sampled"]
    assert result["unique_samples"] == 2
    assert result["duplicate_samples"] == 1
    assert result["revealed_correctness_per_sample"] == [1.0, 1.0, 1.0]
    assert result["hidden_correctness_max"] == 1.0
    assert result["hidden_correctness_min"] == 0.0


def test_world_accuracy_ignores_unknown_truth():
    truth = np.full((6, 15), -1, dtype=np.int64)
    revealed = truth.copy()
    worlds = np.zeros((2, 6, 15), dtype=np.int64)
    result = world_accuracy(worlds, truth, revealed)
    assert result["hidden_fields"] == 0
    assert result["hidden_correctness_mean"] is None
