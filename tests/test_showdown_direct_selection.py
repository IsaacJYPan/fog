import pytest

from fog.eval.showdown_direct_selection import (
    MatchResult,
    match_result_from_dict,
    matched_schedule,
)


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
