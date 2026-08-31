from __future__ import annotations

from fog.training.outcome_heuristic_distill import advantage_weight, team_holdout


def test_advantage_weight_rewards_better_than_expected_outcomes():
    loss = advantage_weight(-1.0, 0.0)
    expected = advantage_weight(0.0, 0.0)
    win = advantage_weight(1.0, 0.0)
    assert 0.25 <= loss < expected < win <= 4.0


def test_advantage_weight_is_clipped():
    assert advantage_weight(1.0, -100.0) == 4.0
    assert advantage_weight(-1.0, 100.0) == 0.25


def test_team_holdout_is_deterministic_and_nonempty():
    first = team_holdout(171, 0.1, 20260803)
    second = team_holdout(171, 0.1, 20260803)
    assert first == second
    assert len(first) == 17
    assert all(0 <= index < 171 for index in first)
