"""Regression checks for the live/replay comparison boundary."""
import numpy as np

from fog.eval.replay_live_parity import compare_rows, keyed_rows
from fog.encoding.entities import ENTITY_RAW_DIM


def row(turn=1, action=0):
    return {"role": "p1", "turn": turn,
            "fields": {"entities": np.zeros((12, ENTITY_RAW_DIM), np.float32),
                       "label": np.asarray(action)}}


def test_alignment_retains_multiple_decisions_per_turn():
    rows = [row(), row(action=8), row(turn=2)]
    assert list(keyed_rows(rows)) == [("p1", 1, 0, 0), ("p1", 1, 0, 1), ("p1", 2, 0, 0)]
    report = compare_rows(rows, [rows[0], rows[2]])
    assert report["missing_replay"] == [["p1", 1, 0, 1]]
    assert report["missing_live"] == []
    assert report["aligned"] == 2


def test_named_blocks_count_whole_row_exactness_and_max_difference():
    from fog.encoding.entities import L

    first, second = row(), row()
    second["fields"]["entities"][0, L.move_pp] = 0.75
    report = compare_rows([first], [second])
    assert report["fields"]["entities.own.move_pp"]["exact_share"] == 0
    assert report["fields"]["entities.own.move_pp"]["max_difference"] == 0.75
    assert report["fields"]["entities.opponent.move_pp"]["exact_share"] == 1
    assert report["differences"][0]["decision"] == ["p1", 1, 0, 0]


def test_forward_hook_copies_actual_model_tensors():
    import torch
    from types import SimpleNamespace

    from fog.eval.replay_live_parity import RecordingSeat

    class Model(torch.nn.Module):
        def forward(self, batch):
            return SimpleNamespace(log_probs=torch.zeros((1, 14)))

    agent = Model()
    seat = RecordingSeat(agent)
    batch = {"entities": torch.randn(1, 12, ENTITY_RAW_DIM),
             "decision_phase": torch.tensor([1]),
             "opp_revealed": {"tokens": torch.full((1, 6, 15), -1), "masks": {}}}
    try:
        agent(batch)
        np.testing.assert_array_equal(seat.pending["entities"], batch["entities"][0])
        batch["entities"].zero_()
        assert seat.pending["entities"].any()
        assert seat.pending["decision_phase"] == 1
        assert (seat.pending["opp_tokens"] == -1).all()
    finally:
        seat.hook.remove()
