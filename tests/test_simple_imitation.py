import pytest

from fog.training.simple_imitation import (
    assert_teacher_answer_is_not_an_input,
    split_by_acting_team,
)


def test_teacher_answer_and_action_features_are_rejected():
    rows = [{"team_index": 0, "frag": {"action_features": object()}, "action": 1}]
    with pytest.raises(ValueError, match="action features"):
        assert_teacher_answer_is_not_an_input(
            {"model": {"use_action_features": False, "simple_rule_prior_strength": 0.0}},
            rows,
        )


def test_model_configuration_must_disable_teacher_derived_inputs():
    with pytest.raises(ValueError, match="use_action_features"):
        assert_teacher_answer_is_not_an_input(
            {"model": {"use_action_features": True, "simple_rule_prior_strength": 0.0}}, []
        )
    with pytest.raises(ValueError, match="simple_rule_prior_strength"):
        assert_teacher_answer_is_not_an_input(
            {"model": {"use_action_features": False, "simple_rule_prior_strength": 1.0}}, []
        )


def test_whole_teams_do_not_cross_train_holdout_split():
    rows = [
        {"team_index": team, "frag": {}, "action": 0}
        for team in range(10)
        for _ in range(3)
    ]
    train, hold, hold_teams = split_by_acting_team(
        rows, holdout_fraction=0.2, seed=7
    )
    train_teams = {rows[int(i)]["team_index"] for i in train}
    held_teams = {rows[int(i)]["team_index"] for i in hold}
    assert held_teams == set(hold_teams)
    assert train_teams.isdisjoint(held_teams)
    assert train_teams | held_teams == set(range(10))
