from __future__ import annotations

import numpy as np
import pytest

from fog.eval.replay_holdout_report import policy_summary, switch_diagnostics, value_calibration


def test_constant_half_prediction_has_brier_quarter():
    outcome = np.array([1.0, -1.0, 1.0, -1.0])
    result = value_calibration(np.zeros(4), outcome)
    assert result["brier"] == pytest.approx(0.25)
    assert result["value_mse"] == pytest.approx(1.0)
    # Everything sits in bin 5 ([0.5, 0.6)); predicted 0.5, observed 0.5.
    assert result["reliability"][5]["count"] == 4
    assert result["reliability"][5]["mean_predicted"] == pytest.approx(0.5)
    assert result["reliability"][5]["observed_win_rate"] == pytest.approx(0.5)
    assert result["ece"] == pytest.approx(0.0)


def test_hand_computed_brier_and_ece():
    # p = 0.9, 0.9, 0.15, 1.0 ; targets = 1, 0, 0, 1
    value = np.array([0.8, 0.8, -0.7, 1.0])
    outcome = np.array([1.0, -1.0, -1.0, 1.0])
    result = value_calibration(value, outcome)
    assert result["brier"] == pytest.approx((0.01 + 0.81 + 0.0225 + 0.0) / 4)
    table = result["reliability"]
    assert [row["count"] for row in table] == [0, 1, 0, 0, 0, 0, 0, 0, 0, 3]
    assert table[1]["mean_predicted"] == pytest.approx(0.15)
    assert table[1]["observed_win_rate"] == pytest.approx(0.0)
    assert table[9]["mean_predicted"] == pytest.approx((0.9 + 0.9 + 1.0) / 3)
    assert table[9]["observed_win_rate"] == pytest.approx(2 / 3)
    expected_ece = 0.25 * 0.15 + 0.75 * abs((2.8 / 3) - (2 / 3))
    assert result["ece"] == pytest.approx(expected_ece)
    assert table[0]["mean_predicted"] is None


def test_draw_is_half_and_values_are_clipped():
    result = value_calibration(np.array([1.5, -2.0]), np.array([0.0, 0.0]))
    # p clipped to 1 and 0, target 0.5 for both.
    assert result["brier"] == pytest.approx(0.25)
    assert result["reliability"][9]["count"] == 1
    assert result["reliability"][0]["count"] == 1


def test_empty_input_and_shape_mismatch():
    assert value_calibration(np.array([]), np.array([]))["brier"] is None
    with pytest.raises(ValueError):
        value_calibration(np.zeros(2), np.zeros(3))


def test_policy_summary_splits_moves_and_switches():
    nll = np.array([0.1, 0.5, 1.0, 2.0])
    correct = np.array([True, False, True, True])
    legal = np.array([2.0, 4.0, 3.0, 3.0])
    action = np.array([0, 3, 8, 13])
    result = policy_summary(nll, correct, legal, action)
    assert result["rows"] == 4
    assert result["top1"] == pytest.approx(0.75)
    assert result["avg_legal_actions"] == pytest.approx(3.0)
    assert result["top1_lift"] == pytest.approx(2.25)
    assert result["nll"] == pytest.approx(0.9)
    assert result["move"]["rows"] == 2 and result["move"]["top1"] == pytest.approx(0.5)
    assert result["switch"]["nll"] == pytest.approx(1.5)


def test_game_phase_uses_each_games_last_turn():
    from fog.eval.replay_holdout_report import game_phase

    # Game 0 lasts 9 turns, game 1 lasts 3; rows of one game need not be in order.
    turn = np.array([1, 3, 4, 6, 7, 9, 9, 1, 2, 3, 1])
    game = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0])
    assert game_phase(turn, game).tolist() == [0, 0, 1, 1, 2, 2, 2, 0, 1, 2, 0]


def test_switch_diagnostics_separate_when_and_where():
    # A model can put little total probability on switching while ranking the right
    # switch target first. Move rows and switch rows have separate denominators.
    probabilities = np.array([
        [0.5, 0.2, 0.05, 0.05, 0.05, 0.05, 0.025, 0.025, 0.04, 0.01],
        [0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.025, 0.075],
        [0.5, 0.05, 0.05, 0.05, 0.05, 0.05, 0.025, 0.025, 0.15, 0.05],
    ])
    action = np.array([8, 8, 0])
    switch_probability = probabilities[:, 8:].sum(-1)
    target_correct = probabilities[:, 8:].argmax(-1) + 8 == action
    result = switch_diagnostics(switch_probability, target_correct, action)
    assert result["human_switch_rows"] == 2
    assert result["human_move_rows"] == 1
    assert result["mean_switch_probability_on_human_switch"] == pytest.approx(0.075)
    assert result["mean_switch_probability_on_human_move"] == pytest.approx(0.2)
    assert result["switch_target_top1"] == pytest.approx(0.5)


def test_switch_diagnostics_empty_families_and_shape_errors():
    empty = switch_diagnostics(np.array([]), np.array([], dtype=bool), np.array([]))
    assert empty["human_switch_rows"] == empty["human_move_rows"] == 0
    assert empty["switch_target_top1"] is None
    assert empty["mean_switch_probability_on_human_move"] is None
    moves = switch_diagnostics(np.array([0.3]), np.array([False]), np.array([0]))
    assert moves["mean_switch_probability_on_human_switch"] is None
    assert moves["switch_target_top1"] is None
    with pytest.raises(ValueError, match="per row"):
        switch_diagnostics(np.ones(2), np.ones(1), np.ones(2))


def test_score_reports_switch_mass_and_targets_on_supervised_rows(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import polars as pl
    import torch

    from fog.eval import replay_holdout_report as report
    from fog.models.agent import FogAgent
    from fog.training.dataset import StreamingReplayDataset

    probabilities = torch.zeros(4, 14)
    probabilities[:, 0] = torch.tensor([0.9, 0.8, 0.8, 0.1])
    probabilities[:, 8] = torch.tensor([0.04, 0.05, 0.15, 0.8])
    probabilities[:, 9] = torch.tensor([0.06, 0.15, 0.05, 0.1])
    batch = {
        "action": torch.tensor([8, 9, 0, 8]),
        "policy_valid": torch.tensor([True, True, True, False]),
        "legality_mask": probabilities > 0,
        "outcome": torch.tensor([1.0, 1.0, -1.0, -1.0]),
        "field": torch.zeros(4, report._turn_column() + 1),
    }

    class Agent:
        def eval(self):
            return self

        def __call__(self, batch):
            return SimpleNamespace(log_probs=probabilities.log(), value=torch.zeros(4))

    monkeypatch.setattr(FogAgent, "from_tables", lambda cfg: Agent())
    monkeypatch.setattr(StreamingReplayDataset, "__init__", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(StreamingReplayDataset, "batches", lambda *args, **kwargs: iter([batch]))
    manifest = tmp_path / "manifest.parquet"
    pl.DataFrame({"id": ["game"], "n_rows": [4]}).write_parquet(manifest)
    result = report.score({}, manifest, None)
    assert result["rows"] == 4 and result["policy_rows"] == 3
    switches = result["switch_diagnostics"]
    assert switches["human_switch_rows"] == 2 and switches["human_move_rows"] == 1
    assert switches["mean_switch_probability_on_human_switch"] == pytest.approx(0.15)
    assert switches["mean_switch_probability_on_human_move"] == pytest.approx(0.2)
    assert switches["switch_target_top1"] == pytest.approx(0.5)
