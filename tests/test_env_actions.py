from __future__ import annotations

from dataclasses import dataclass

from fog.env.actions import action_legality_mask, map_action_to_order


@dataclass
class Move:
    id: str


@dataclass
class Mon:
    species: str
    moves: dict[str, Move] | None = None


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
