from types import SimpleNamespace

import numpy as np

from fog.constants.schema import SET_N_ATTRS
from fog.scripts.collect_simple_belief_value import _game_arrays


def _fragment(value: float) -> dict:
    return {
        "entities": np.full((12, 140), value, dtype=np.float32),
        "entity_ids": np.zeros((12, 17), dtype=np.int64),
        "field": np.full(51, value, dtype=np.float32),
        "history": np.zeros((60, 108), dtype=np.float32),
        "history_ids": np.zeros((60, 17), dtype=np.int64),
        "history_state": np.zeros((60, 14), dtype=np.float32),
        "history_len": int(value),
        "active_idx": np.array([0, 6], dtype=np.int64),
        "legality_mask": np.ones(14, dtype=bool),
        "opp_revealed": {"tokens": np.full((6, SET_N_ATTRS), -1, dtype=np.int64)},
        "action_features": np.zeros((14, 34), dtype=np.float32),
    }


def test_simple_game_conversion_preserves_exact_truth_without_policy_labels():
    truth = np.arange(6 * SET_N_ATTRS, dtype=np.int64).reshape(6, SET_N_ATTRS)
    game = SimpleNamespace(
        outcome=1.0,
        belief_targets=truth,
        frags=[_fragment(1.0), _fragment(2.0)],
        actions=[1, 8],
    )
    arrays = _game_arrays(game)
    assert arrays["entities"].shape == (2, 12, 140)
    assert arrays["history"].shape == (2, 60, 108)
    assert arrays["history_ids"].shape == (2, 60, 17)
    assert arrays["history_state"].shape == (2, 60, 14)
    np.testing.assert_array_equal(arrays["belief_targets"][0], truth)
    np.testing.assert_array_equal(arrays["belief_targets"][1], truth)
    assert np.all(arrays["self_belief_targets"] == -1)
    assert not arrays["policy_valid"].any()
    assert np.all(arrays["own_set_provenance"] == 2)
    assert np.all(arrays["opponent_set_provenance"] == 0)
    np.testing.assert_array_equal(arrays["outcome"], np.ones(2, dtype=np.float32))
