from fog.eval.value_continuation_target_control import (
    split_team_indices,
    summarize_value_targets,
)


def test_summary_separates_stochastic_floor_from_reducible_error():
    states = [
        {
            "value_prediction": 0.5,
            "progress_score": 0.25,
            "live_outcome": 1.0,
            "continuations": [{"outcome": value} for value in (1.0, -1.0, 1.0, 1.0)],
        },
        {
            "value_prediction": -0.5,
            "progress_score": -0.25,
            "live_outcome": -1.0,
            "continuations": [{"outcome": value} for value in (-1.0, -1.0, 1.0, -1.0)],
        },
    ]
    summary = summarize_value_targets(states)
    assert summary["states"] == 2
    assert summary["valid_continuations"] == 8
    assert summary["critic_mse_to_continuation_mean"] == 0.0
    assert summary["critic_mse_to_repeated_hard_outcomes"] > 0.0
    assert summary["unbiased_stochastic_outcome_variance"] > 0.0
    assert summary["critic_mse_to_live_single_outcome"] == 0.25
    assert summary["progress_states"] == 2
    assert summary["progress_mse_to_continuation_mean"] == 0.0625
    assert summary["progress_correlation_with_continuation_mean"] == 1.0


def test_summary_ignores_missing_or_single_continuation_states():
    summary = summarize_value_targets([
        {"value_prediction": 0.0, "continuations": [{"outcome": 1.0}]},
        {"value_prediction": None, "continuations": [{"outcome": 1.0}, {"outcome": -1.0}]},
    ])
    assert summary == {"states": 0, "valid_continuations": 0}


def test_team_split_is_disjoint_deterministic_and_complete():
    train = split_team_indices(17, holdout_fraction=0.2, seed=7, split="train")
    holdout = split_team_indices(17, holdout_fraction=0.2, seed=7, split="holdout")
    assert train == split_team_indices(17, holdout_fraction=0.2, seed=7, split="train")
    assert set(train).isdisjoint(holdout)
    assert sorted(train + holdout) == list(range(17))
    assert split_team_indices(17, holdout_fraction=0.2, seed=7, split="all") == list(range(17))
