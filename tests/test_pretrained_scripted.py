from copy import deepcopy

import pytest

from fog.pretrained.scripted_worker import ScriptedPlayer
from fog.scripts.pretrained_compare import compare_evaluations


@pytest.fixture
def scripted_view(monkeypatch):
    import poke_env.battle
    from fog.pretrained import scripted_worker as worker

    updates = []
    monkeypatch.setattr(poke_env.battle, "Battle", lambda *args, **kwargs: object())
    monkeypatch.setattr(worker.LocalShowdownBattle, "_apply_view_update", lambda battle, messages, request: updates.append((messages, request)))
    monkeypatch.setattr(worker, "_choice", lambda kind, battle, request, rng: "move " + str(rng.randrange(2) + 1))
    return updates


def payload(tag="battle-test"):
    return {
        "op": "act", "battle_tag": tag, "username": "Opponent", "seed": 22,
        "choose": True, "messages": ["public-only"],
        "request": {"active": [{"moves": [{"id": "a"}, {"id": "b"}]}]},
    }


def test_scripted_opponent_only_receives_visible_view_and_own_request(scripted_view):
    player = ScriptedPlayer("simpleheuristics")
    message = payload()
    assert player.update(message)["choice"] in ("move 1", "move 2")
    assert scripted_view == [(message["messages"], message["request"])]
    corrupt = deepcopy(message)
    corrupt["request"]["hidden"] = {"opponent_team": "secret"}
    with pytest.raises(ValueError, match="Complete-team labels"):
        player.update(corrupt)
    assert len(scripted_view) == 1


def test_scripted_random_fallbacks_are_per_game_and_resume_order_independent(scripted_view):
    a, b = ScriptedPlayer("maxdamage"), ScriptedPlayer("maxdamage")
    first = [a.update(payload())["choice"] for _ in range(12)]
    for index in range(5):
        message = payload("earlier-" + str(index))
        b.update(message)
        b.update(dict(message, op="close"))
    assert [b.update(payload())["choice"] for _ in range(12)] == first
    b.update(dict(payload(), op="close"))
    assert not b.battles
    assert b.update(payload())["choice"] == first[0]


def test_preview_matches_neural_fixed_order_and_wait_does_not_choose(scripted_view):
    player = ScriptedPlayer("simpleheuristics")
    message = payload()
    message["request"] = {"teamPreview": True}
    assert player.update(message)["choice"] == "team 123456"
    message["request"] = {"wait": True}
    assert "choice" not in player.update(message)


def test_scripted_games_cannot_be_misread_as_kakuna_strength_comparison(tmp_path):
    import json
    from fog.pretrained.kakuna import KAKUNA_34_SHA256

    paths = []
    for arm in ("ordinary", "joint"):
        path = tmp_path / (arm + ".json")
        path.write_text(json.dumps({
            "protocol": {"stage": "development", "foulplay": False, "scripted_opponent": "simpleheuristics", "original_sha256": KAKUNA_34_SHA256, "candidate_sha256": arm},
            "failures": 0, "unresolved": 0, "rows": [],
        }))
        paths.append(path)
    with pytest.raises(ValueError, match="against unchanged Kakuna"):
        compare_evaluations({"models": ["ordinary", "joint"]}, *paths)


def test_incomplete_crossed_block_budget_is_rejected_before_starting_games(monkeypatch):
    import sys
    from fog.scripts.pretrained_evaluate import main

    monkeypatch.setattr(sys, "argv", [
        "pretrained_evaluate", "--candidate", "absent.pt", "--original", "absent.pt",
        "--native-python", "python", "--pools", "absent", "--out", "absent", "--games", "4",
    ])
    with pytest.raises(ValueError, match="at least eight games"):
        main()
