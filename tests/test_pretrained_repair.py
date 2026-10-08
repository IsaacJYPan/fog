from copy import deepcopy
import pytest
from fog.scripts.pretrained_repair import validate_replacement, compatible_collection


def test_recovery_allows_new_battle_seed_but_keeps_assignment():
    original = {
        "opponent": "kakuna",
        "submitted_teams": {"p1": "team A", "p2": "team B"},
        "schedule": {
            "p1_is_a": True,
            "p1_team": 1,
            "p2_team": 2,
            "controller_seed": 123,
            "showdown_seed": 456,
        },
    }
    repaired = deepcopy(original)
    repaired["schedule"]["showdown_seed"] = 789
    validate_replacement(original, repaired)
    for field in ("p1_is_a", "p1_team", "p2_team", "controller_seed"):
        changed = deepcopy(repaired)
        changed["schedule"][field] = 999
        with pytest.raises(ValueError, match="assigned"):
            validate_replacement(original, changed)
    changed = deepcopy(repaired)
    changed["submitted_teams"]["p1"] = "different team"
    with pytest.raises(ValueError, match="assigned"):
        validate_replacement(original, changed)


def test_recovery_cannot_change_search_budget_or_learner_order():
    collection = {"foul_play_search_ms": 500, "learners": {"first": "aaa", "second": "bbb"}}
    compatible_collection(collection, deepcopy(collection))
    changed = dict(collection, foul_play_search_ms=1000)
    with pytest.raises(ValueError, match="recipe"):
        compatible_collection(collection, changed)
    changed = dict(collection, learners={"second": "bbb", "first": "aaa"})
    with pytest.raises(ValueError, match="ordering"):
        compatible_collection(collection, changed)
