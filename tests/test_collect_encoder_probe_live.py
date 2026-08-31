from fog.eval.collect_encoder_probe_live import split_name


def test_live_probe_split_keeps_teams_and_later_battles_separate() -> None:
    heldout = {2}
    assert split_name({"team": 2, "game": 2}, 5, heldout) == "team_holdout"
    assert split_name({"team": 2, "game": 7}, 5, heldout) == "team_holdout"
    assert split_name({"team": 1, "game": 1}, 5, heldout) == "train"
    assert split_name({"team": 1, "game": 6}, 5, heldout) == "battle_holdout"


def test_live_probe_split_supports_multiple_training_cycles() -> None:
    heldout = {2}
    assert split_name({"team": 1, "game": 1}, 5, heldout, 3) == "train"
    assert split_name({"team": 1, "game": 13}, 5, heldout, 3) == "train"
    assert split_name({"team": 1, "game": 15}, 5, heldout, 3) == "battle_holdout"
    assert split_name({"team": 2, "game": 18}, 5, heldout, 3) == "team_holdout"
