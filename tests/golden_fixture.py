"""The scripted golden battle (spec §10): a fixed 5-turn gen9ou battle built from lightweight mocks.

Deterministic by construction — same fixture + same generated tables => byte-identical tensors.
Used by test_golden_battle.py to pin the ENTIRE encoding stack (entities/field/history/reveals/
tracker/mirror) across refactors.
"""
from __future__ import annotations

import numpy as np


class Move:
    def __init__(self, move_id: str, max_pp: int = 16, current_pp: int | None = None):
        self.id = move_id
        self.max_pp = max_pp
        self.current_pp = current_pp if current_pp is not None else max_pp


class Mon:
    def __init__(self, species, item, ability, moves, tera, base_stats, types, weight,
                 hp=1.0, status=None, status_counter=0, boosts=None, effects=None,
                 fainted=False, terastallized=False):
        self.species = species
        self.item = item
        self.ability = ability
        self.moves = {m: Move(m) for m in moves}
        self.tera_type = tera
        self.base_stats = dict(zip(("hp", "atk", "def", "spa", "spd", "spe"), base_stats))
        self.types = list(types)
        self.weight = weight
        self.current_hp_fraction = hp
        self.fainted = fainted
        self.status = status
        self.status_counter = status_counter
        self.boosts = boosts or {}
        self.effects = effects or {}
        self.terastallized = terastallized


class Obs:
    def __init__(self, events):
        self.events = events


class GoldenBattle:
    battle_tag = "battle-gen9ou-golden"
    player_role = "p1"

    def __init__(self):
        self.team = {
            "p1: Great Tusk": Mon("greattusk", "boosterenergy", "protosynthesis",
                                  ("headlongrush", "closecombat", "icespinner", "rapidspin"),
                                  "ground", (115, 131, 131, 53, 53, 87), ("ground", "fighting"), 320.0,
                                  hp=0.62),
            "p1: Kingambit": Mon("kingambit", "blackglasses", "supremeoverlord",
                                 ("suckerpunch", "kowtowcleave", "ironhead", "swordsdance"),
                                 "dark", (100, 135, 120, 60, 85, 50), ("dark", "steel"), 120.0,
                                 status="brn", boosts={"atk": 2}),
            "p1: Gholdengo": Mon("gholdengo", "choicescarf", "goodasgold",
                                 ("makeitrain", "shadowball", "trick", "focusblast"),
                                 "flying", (87, 60, 95, 133, 91, 84), ("steel", "ghost"), 30.0),
            "p1: Dragapult": Mon("dragapult", "choiceband", "clearbody",
                                 ("dragondarts", "uturn", "phantomforce", "suckerpunch"),
                                 "dragon", (88, 120, 75, 100, 75, 142), ("dragon", "ghost"), 50.0,
                                 hp=0.0, fainted=True),
            "p1: Slowking-Galar": Mon("slowkinggalar", "assaultvest", "regenerator",
                                      ("sludgebomb", "flamethrower", "psyshock", "icebeam"),
                                      "poison", (95, 65, 80, 110, 110, 30), ("poison", "psychic"), 79.5,
                                      hp=0.85),
            "p1: Zamazenta": Mon("zamazenta", "choiceband", "dauntlessshield",
                                 ("closecombat", "crunch", "stoneedge", "heavyslam"),
                                 "fighting", (92, 130, 115, 80, 115, 138), ("fighting",), 210.0),
        }
        # opponent: 2 revealed of 6 previewed
        self.opponent_team = {
            "p2: Great Tusk": Mon("greattusk", None, "protosynthesis",           # item knocked off
                                  ("headlongrush", "rapidspin"),
                                  None, (115, 131, 131, 53, 53, 87), ("ground", "fighting"), 320.0,
                                  hp=0.45, effects={"taunt": 2}),
            "p2: Gholdengo": Mon("gholdengo", "airballoon", "goodasgold",
                                 ("makeitrain",),
                                 "fairy", (87, 60, 95, 133, 91, 84), ("steel", "ghost"), 30.0,
                                 hp=0.9, terastallized=True),
        }
        self.teampreview_opponent_team = [
            Mon(s, None, None, (), None, (100,) * 6, ("normal",), 50.0)
            for s in ("greattusk", "gholdengo", "kingambit", "dragapult", "zamazenta",
                      "slowkinggalar")
        ]
        self.active_pokemon = self.team["p1: Great Tusk"]
        self.opponent_active_pokemon = self.opponent_team["p2: Gholdengo"]
        self.available_moves = [self.active_pokemon.moves["headlongrush"],
                                self.active_pokemon.moves["icespinner"]]
        self.available_switches = [self.team["p1: Kingambit"], self.team["p1: Gholdengo"],
                                   self.team["p1: Slowking-Galar"], self.team["p1: Zamazenta"]]
        self.can_tera = True
        self.force_switch = False
        self.turn = 6
        self.weather = {"sandstorm": 3}
        self.fields = {"grassyterrain": 4, "trickroom": 5}
        self.side_conditions = {"stealthrock": 1, "spikes": 2}
        self.opponent_side_conditions = {"stealthrock": 1, "reflect": 4, "tailwind": 5}
        self.won = None

        self.observations = {
            1: Obs([
                ["", "switch", "p1a: Great Tusk", "Great Tusk", "100/100"],
                ["", "switch", "p2a: Great Tusk", "Great Tusk", "100/100"],
                ["", "move", "p1a: Great Tusk", "Headlong Rush", "p2a: Great Tusk"],
                ["", "-damage", "p2a: Great Tusk", "55/100"],
                ["", "move", "p2a: Great Tusk", "Rapid Spin", "p1a: Great Tusk"],
                ["", "-damage", "p1a: Great Tusk", "88/100"],
            ]),
            2: Obs([
                ["", "-enditem", "p1a: Great Tusk", "Booster Energy"],
                ["", "move", "p1a: Great Tusk", "Ice Spinner", "p2a: Great Tusk"],
                ["", "-crit", "p2a: Great Tusk"],
                ["", "-damage", "p2a: Great Tusk", "20/100"],
                ["", "move", "p2a: Great Tusk", "Headlong Rush", "p1a: Great Tusk"],
                ["", "-supereffective", "p1a: Great Tusk"],
                ["", "-damage", "p1a: Great Tusk", "30/100"],
            ]),
            3: Obs([
                ["", "switch", "p1a: Slowking-Galar", "Slowking-Galar", "100/100"],
                ["", "-ability", "p1a: Slowking-Galar", "Regenerator"],
                ["", "move", "p2a: Great Tusk", "Headlong Rush", "p1a: Slowking-Galar"],
                ["", "-resisted", "p1a: Slowking-Galar"],
                ["", "-damage", "p1a: Slowking-Galar", "78/100"],
                ["", "-start", "p2a: Great Tusk", "move: Taunt"],
            ]),
            4: Obs([
                ["", "move", "p1a: Slowking-Galar", "Future Sight", "p2a: Great Tusk"],
                ["", "-start", "p1a: Slowking-Galar", "move: Future Sight"],
                ["", "switch", "p2a: Gholdengo", "Gholdengo", "100/100"],
                ["", "-item", "p2a: Gholdengo", "Air Balloon"],
            ]),
            5: Obs([
                ["", "-terastallize", "p2a: Gholdengo", "Fairy"],
                ["", "move", "p2a: Gholdengo", "Make It Rain", "p1a: Slowking-Galar"],
                ["", "-damage", "p1a: Slowking-Galar", "40/100"],
                ["", "move", "p1a: Slowking-Galar", "Sludge Bomb", "p2a: Gholdengo"],
                ["", "-immune", "p2a: Gholdengo"],
                ["", "-terastallize", "p1a: Slowking-Galar", "Poison"],
            ]),
        }


def build_golden_observation():
    """Feed the fixture through tracker + ledger + encoders exactly as the live player does."""
    from fog.constants.tables import TABLES
    from fog.encoding.events import build_snapshot, feed_ledger_from_events
    from fog.encoding.mirror import build_opponent_view_np
    from fog.encoding.observation import Observation
    from fog.encoding.reveals import RevealLedger, tokens_from_views
    from fog.encoding.tracker import BattleTracker
    from fog.env.player import _own_truth_tokens

    battle = GoldenBattle()
    tracker = BattleTracker(my_role="p1")
    ledger = RevealLedger()
    my_actions = {1: 0, 2: 2, 3: 10, 4: 1, 5: 0}
    for t in sorted(battle.observations):
        events = battle.observations[t].events
        tracker.feed_turn(t, events)
        feed_ledger_from_events(events, "p1", ledger, TABLES)
        tracker.record_decision(t, build_snapshot(battle, TABLES), my_actions[t])

    obs = Observation.from_battle(battle, tracker=tracker)
    my_keys = list(battle.team)
    my_species = [TABLES.species_index(m.species) for m in battle.team.values()]
    my_view_tokens = tokens_from_views(ledger.views(my_keys, my_species))
    mirror_frag, my_view = build_opponent_view_np(
        obs, my_view_tokens, _own_truth_tokens(battle, TABLES), TABLES
    )
    return obs, mirror_frag, my_view


def golden_arrays() -> dict[str, np.ndarray]:
    obs, mirror_frag, my_view = build_golden_observation()
    return {
        "entities": obs.entities,
        "entity_ids": obs.entity_ids,
        "field": obs.field,
        "history": obs.history,
        "legality_mask": obs.legality_mask.astype(np.int8),
        "active_idx": obs.active_idx,
        "opp_tokens": obs.opp_revealed["tokens"],
        "mirror_entities": mirror_frag["entities"],
        "mirror_entity_ids": mirror_frag["entity_ids"],
        "mirror_field": mirror_frag["field"],
        "my_view_tokens": my_view["tokens"],
        "my_truth_species": my_view["labels"]["species"],
    }
