from fog.eval.policy_disagreement_rollout import (
    _normalize_choice,
    _outcome_for_player,
    _sample_seed,
    select_simple_outcome_label,
    summarize_pairs,
)


def test_choice_normalization_matches_showdown_worker_contract():
    assert _normalize_choice("/choose move 1") == "move 1"
    assert _normalize_choice("/team 123456") == "team 123456"
    assert _normalize_choice(None) == ""


def test_sample_seeds_are_valid_and_step_specific():
    first = _sample_seed(7, 0)
    second = _sample_seed(7, 1)
    assert first.startswith("sodium,") and len(first.split(",")[1]) == 64
    assert first != second


def test_outcome_mapping_normalizes_showdown_and_poke_env_names():
    assert _outcome_for_player("Player A", "playera") == 1.0
    assert _outcome_for_player("Player B", "playera") == -1.0
    assert _outcome_for_player(None, "playera") == 0.0


def test_pair_summary_excludes_failed_pairs():
    states = [{"pairs": [
        {"policy_outcome": 1.0, "simple_outcome": -1.0},
        {"policy_outcome": -1.0, "simple_outcome": 1.0},
        {"policy_outcome": 1.0, "simple_outcome": 1.0},
        {"policy_outcome": None, "simple_outcome": None},
    ]}]
    result = summarize_pairs(states)
    assert result["paired_rollouts"] == 3
    assert result["policy_action_better"] == 1
    assert result["simple_action_better"] == 1
    assert result["tied"] == 1
    assert result["mean_policy_minus_simple_outcome"] == 0.0


def test_outcome_label_requires_unopposed_ordinary_move_evidence():
    simple_wins = [
        {"policy_outcome": -1.0, "simple_outcome": 1.0},
        {"policy_outcome": -1.0, "simple_outcome": 1.0},
        {"policy_outcome": 1.0, "simple_outcome": 1.0},
        {"policy_outcome": None, "simple_outcome": None},
    ]
    assert select_simple_outcome_label(0, 2, simple_wins, min_simple_better=2)
    assert not select_simple_outcome_label(8, 9, simple_wins, min_simple_better=2)
    opposed = simple_wins + [{"policy_outcome": 1.0, "simple_outcome": -1.0}]
    assert not select_simple_outcome_label(0, 2, opposed, min_simple_better=2)
