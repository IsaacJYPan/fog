from __future__ import annotations

import numpy as np
import pytest

from fog.eval.replay_holdout_report import policy_summary, value_calibration


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
