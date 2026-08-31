from types import SimpleNamespace

import pytest

from fog.training.progress_reward import battle_progress_score


def _mon(hp: float):
    return SimpleNamespace(current_hp_fraction=hp, fainted=hp <= 0)


def test_network_view_counts_unseen_opponents_as_healthy():
    battle = SimpleNamespace(
        team={f"mine{i}": _mon(1.0) for i in range(6)},
        opponent_team={"seen": _mon(1.0)},
    )
    assert battle_progress_score(battle) == pytest.approx(0.0)


def test_network_view_rewards_health_and_knockout_advantage():
    mine = {f"mine{i}": _mon(1.0) for i in range(6)}
    theirs = {f"opp{i}": _mon(1.0) for i in range(6)}
    theirs["opp0"] = _mon(0.0)
    battle = SimpleNamespace(team=mine, opponent_team=theirs)
    assert battle_progress_score(battle) == pytest.approx(1.0 / 6.0)


def test_internal_battle_score_uses_complete_sides_and_perspective():
    sides = {
        "p1": SimpleNamespace(pokemon=[_mon(1.0)] * 6),
        "p2": SimpleNamespace(pokemon=[_mon(0.5)] * 6),
    }
    battle = SimpleNamespace(side=lambda role: sides[role])
    assert battle_progress_score(battle, "p1") == pytest.approx(0.25)
    assert battle_progress_score(battle, "p2") == pytest.approx(-0.25)
