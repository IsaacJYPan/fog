import pytest
import torch

from fog.eval.showdown_direct_selection import (
    MatchResult,
    greedy_action_with_tera_bonus,
    match_result_from_dict,
    matched_schedule,
    validate_evaluation_temperature,
)


def test_greedy_tera_bonus_is_local_and_preserves_illegal_actions():
    log_probs = torch.tensor([-0.2, -0.7, -1.0, -1.5, -2.0, float("-inf"), -3.0, -4.0,
                              -0.5, -1.1, -1.2, -1.3, -1.4, float("-inf")])
    assert greedy_action_with_tera_bonus(log_probs) == 0
    assert greedy_action_with_tera_bonus(log_probs, 2.0) == 4
    # The illegal Tera slot remains impossible even after a very large family bonus.
    only_illegal_tera_high = log_probs.clone()
    only_illegal_tera_high[4] = float("-inf")
    assert greedy_action_with_tera_bonus(only_illegal_tera_high, 100.0) in {6, 7}
    with pytest.raises(ValueError, match="non-negative"):
        greedy_action_with_tera_bonus(log_probs, -1.0)


def test_evaluation_temperature_validation_accepts_sampling_and_rejects_bad_values():
    assert validate_evaluation_temperature(0.0) == 0.0
    assert validate_evaluation_temperature(1.2) == pytest.approx(1.2)
    with pytest.raises(ValueError, match="finite and non-negative"):
        validate_evaluation_temperature(-0.1)
    with pytest.raises(ValueError, match="finite and non-negative"):
        validate_evaluation_temperature(float("nan"))


def test_matched_schedule_crosses_team_and_position_for_a():
    schedule = matched_schedule(pool_size=2, games=4, seed=3)
    assert len(schedule) == 4
    x, y = schedule[0].p1_team, schedule[0].p2_team
    assert [(g.p1_is_a, g.p1_team, g.p2_team) for g in schedule] == [
        (True, x, y),
        (True, y, x),
        (False, x, y),
        (False, y, x),
    ]
    # A uses each team once from each player position.
    a_roles_and_teams = [
        ("p1" if g.p1_is_a else "p2", g.p1_team if g.p1_is_a else g.p2_team)
        for g in schedule
    ]
    assert sorted(a_roles_and_teams) == sorted([("p1", x), ("p1", y), ("p2", x), ("p2", y)])


def test_matched_schedule_requires_complete_blocks():
    with pytest.raises(ValueError, match="multiple of four"):
        matched_schedule(pool_size=2, games=6, seed=0)


def test_match_result_excludes_draws_from_decisive_interval():
    result = MatchResult("a vs b", "a", "b", 12, completed_games=12,
                         a_wins=6, b_wins=4, draws=2)
    row = result.as_dict()
    assert row["decisive_games"] == 10
    assert row["a_win_rate_decisive"] == pytest.approx(0.6)


def test_match_result_checkpoint_roundtrip_ignores_derived_fields():
    original = MatchResult("a vs b", "a", "b", 12, completed_games=8,
                           a_wins=5, b_wins=3, turns=120, wall_s=4.5)
    restored = match_result_from_dict(original.as_dict())
    assert restored == original
