"""Dependence and pooled calibration must survive the statistical handoff."""

from copy import deepcopy

import pytest

from fog.pretrained.grouped_belief_review import connected_groups, sampler_comparisons


def test_shared_rosters_join_entire_games_transitively():
    assert connected_groups([["a", "b"], ["b", "c"], ["c", "d"], ["x", "y"]]) == [[0, 1, 2], [3]]


def test_pooled_ece_and_identical_pairs_do_not_create_false_certainty():
    games = {}
    for name, correct, rosters in [
        ("a", 1, ["x", "y"]),
        ("b", 0, ["y", "z"]),
        ("c", 1, ["other", "last"]),
    ]:
        games[name] = {
            "team_rosters": rosters,
            "resampling_data": {
                "fields": {"item": [[correct, 0.5, 0.5, 0.5]]},
                "joint_correct": correct,
                "joint_count": 1,
                "recall_correct": correct,
                "recall_count": 1,
            },
        }
    methods = {
        name: {"by_game": deepcopy(games)}
        for name in [
            "previous_neural",
            "prior_only",
            "facts_residual",
            "native_residual",
            "expanded_existing_prior",
        ]
    }
    result = sampler_comparisons(methods, replicates=200)
    assert result["roster_groups"] == 2 and result["group_sizes"] == [2, 1]
    # The opposite errors in the same confidence bin cancel after pooling.
    assert result["summaries"]["prior_only"]["item/ece"]["point"] == pytest.approx(1 / 6)
    contrast = result["contrasts"]["native_residual_minus_prior_only"]["joint_recall_at_8"]
    assert contrast["point"] == 0 and contrast["low"] is None and contrast["high"] is None
    assert contrast["degenerate_bootstrap"]
    methods["native_residual"]["by_game"]["a"]["team_rosters"] = ["changed"]
    with pytest.raises(ValueError, match="roster groups"):
        sampler_comparisons(methods, replicates=200)
