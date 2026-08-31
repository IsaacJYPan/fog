from __future__ import annotations

import random
from dataclasses import dataclass

from fog.env.actions import action_legality_mask, map_action_to_order, map_order_to_action
from fog.env.showdown_local_runner import _choice


@dataclass
class Move:
    id: str


@dataclass
class Mon:
    species: str
    moves: dict[str, Move] | None = None
    _last_request: dict | None = None
    name: str | None = None


class Battle:
    def __init__(self) -> None:
        self.moves = [Move(f"move{i}") for i in range(4)]
        self.team = {
            "p1: active": Mon("active", {m.id: m for m in self.moves}),
            **{f"p1: bench{i}": Mon(f"bench{i}") for i in range(5)},
        }
        self.active_pokemon = self.team["p1: active"]
        self.available_moves = list(self.moves)
        self.available_switches = list(self.team.values())[1:]
        self.can_tera = True
        self.opponent_team = {"p2: opp": Mon("opp")}
        self.opponent_active_pokemon = self.opponent_team["p2: opp"]


class Player:
    def create_order(self, target, **kwargs):
        name = getattr(target, "id", getattr(target, "species", str(target)))
        return ("order", name, kwargs)

    def choose_default_move(self):
        return ("default",)


@dataclass
class Order:
    order: object | None
    terastallize: bool = False


def test_action_to_order_mapping_all_14_slots():
    battle = Battle()
    player = Player()

    for idx in range(4):
        assert map_action_to_order(battle, idx, player) == ("order", f"move{idx}", {})

    for idx in range(4, 8):
        assert map_action_to_order(battle, idx, player) == (
            "order",
            f"move{idx - 4}",
            {"terastallize": True},
        )

    for idx in range(8, 13):
        assert map_action_to_order(battle, idx, player) == ("order", f"bench{idx - 8}", {})

    assert map_action_to_order(battle, 13, player) == ("default",)


def test_action_legality_mask_matches_mapping_slots():
    battle = Battle()
    mask = action_legality_mask(battle)
    assert all(mask[:13])
    assert not mask[13]


def test_action_legality_mask_forced_when_no_regular_action():
    battle = Battle()
    battle.available_moves = []
    battle.available_switches = []
    mask = action_legality_mask(battle)
    assert sum(mask) == 1
    assert mask[13]


def test_order_to_action_inverts_moves_tera_and_switches():
    battle = Battle()
    assert map_order_to_action(battle, Order(battle.moves[2])) == 2
    assert map_order_to_action(battle, Order(battle.moves[1], terastallize=True)) == 5
    assert map_order_to_action(battle, Order(battle.available_switches[3])) == 11


def test_order_to_action_rejects_unmatched_teacher_order():
    battle = Battle()
    try:
        map_order_to_action(battle, Order(Move("not-available")))
    except ValueError:
        pass
    else:
        raise AssertionError("unmatched scripted orders must not become policy labels")


def test_struggle_maps_to_default_slot_even_when_switching_is_legal():
    battle = Battle()
    struggle = Move("struggle")
    battle.available_moves = [struggle]
    mask = action_legality_mask(battle)
    assert mask[13]
    assert any(mask[8:13])
    assert map_order_to_action(battle, Order(struggle)) == 13


def test_switch_uses_authoritative_numeric_request_slot_for_numeric_nickname():
    battle = Battle()
    target = battle.team["p1: bench1"]
    target.species = "2 girls 1 cup"
    battle.last_request = {
        "side": {
            "pokemon": [
                {"ident": "p1: active"},
                {"ident": "p1: bench0"},
                {"ident": "p1: bench1"},
                {"ident": "p1: bench2"},
                {"ident": "p1: bench3"},
                {"ident": "p1: bench4"},
            ]
        }
    }

    # Project slot 1 is bench1 (the third actual party member). Sending its nickname would make
    # Showdown parse the leading "2" as party slot two; the numeric request position is exact.
    assert map_action_to_order(battle, 9, Player()) == ("order", "/choose switch 3", {})


def test_switch_uses_request_slot_when_available_target_is_rebuilt_object():
    battle = Battle()
    target = battle.team["p1: bench1"]
    target.species = "6666"
    battle.available_switches[1] = Mon("6666")
    battle.last_request = {
        "side": {
            "pokemon": [
                {"ident": "p1: active"},
                {"ident": "p1: bench0"},
                {"ident": "p1: bench1"},
                {"ident": "p1: bench2"},
                {"ident": "p1: bench3"},
                {"ident": "p1: bench4"},
            ]
        }
    }

    assert map_action_to_order(battle, 9, Player()) == ("order", "/choose switch 3", {})


def test_switch_uses_target_request_ident_when_illusion_alias_points_elsewhere():
    battle = Battle()
    rebuilt = Mon("6666", _last_request={"ident": "p1: 6666"})
    battle.team["p1: bench1"].species = "6666"
    battle.available_switches[1] = rebuilt
    battle.last_request = {
        "side": {
            "pokemon": [
                {"ident": "p1: active"},
                {"ident": "p1: bench0"},
                {"ident": "p1: 6666"},
                {"ident": "p1: bench2"},
                {"ident": "p1: bench3"},
                {"ident": "p1: bench4"},
            ]
        }
    }

    # Illusion reconciliation can leave the request ident absent from battle.team while the
    # offered Pokemon still retains its authoritative request row.
    assert "p1: 6666" not in battle.team
    assert map_action_to_order(battle, 9, Player()) == ("order", "/choose switch 3", {})


def test_switch_uses_unique_request_nickname_after_illusion_clears_last_request():
    battle = Battle()
    battle.team["p1: bench1"].species = "zoroarkhisui"
    rebuilt = Mon("zoroarkhisui", _last_request=None, name="6666")
    battle.available_switches[1] = rebuilt
    battle.last_request = {
        "side": {
            "pokemon": [
                {"ident": "p1: active"},
                {"ident": "p1: bench0"},
                {"ident": "p1: 6666"},
                {"ident": "p1: bench2"},
                {"ident": "p1: bench3"},
                {"ident": "p1: bench4"},
            ]
        }
    }

    assert map_action_to_order(battle, 9, Player()) == ("order", "/choose switch 3", {})


def test_local_runner_normalizes_scripted_controller_switch_to_request_slot():
    battle = Battle()
    target = battle.team["p1: bench1"]
    target.species = "zoroarkhisui"
    target.name = "6666"
    battle.last_request = {
        "side": {
            "pokemon": [
                {"ident": "p1: active"},
                {"ident": "p1: bench0"},
                {"ident": "p1: 6666"},
                {"ident": "p1: bench2"},
                {"ident": "p1: bench3"},
                {"ident": "p1: bench4"},
            ]
        }
    }

    class ScriptedSwitchOrder:
        order = target
        message = "/choose switch 6666"

    assert _choice(
        lambda _battle: ScriptedSwitchOrder(),
        battle,
        {"active": [{}]},
        random.Random(0),
    ) == "/choose switch 3"
