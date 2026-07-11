"""Tests for FogPlayer per-battle state isolation (concurrency safety) and live team preview."""
from __future__ import annotations

import numpy as np
import torch

from fog.env.player import FogPlayer
from fog.training.rollout import RolloutBuffer


class _Obs:
    def __init__(self, events):
        self.events = events


class _Mon:
    def __init__(self, species="garchomp", item="leftovers", ability="roughskin",
                 moves=("earthquake", "dragonclaw"), tera=None):
        self.species = species
        self.item = item
        self.ability = ability
        self.moves = {m: object() for m in moves}
        self.tera_type = tera
        self.base_stats = {"hp": 108, "atk": 130, "def": 95, "spa": 80, "spd": 85, "spe": 102}
        self.types = ["dragon", "ground"]
        self.weight = 95.0
        self.current_hp_fraction = 1.0
        self.fainted = False
        self.status = None
        self.status_counter = 0
        self.boosts = {}
        self.effects = {}
        self.terastallized = False


class _Battle:
    def __init__(self, tag: str, n_mons: int = 6):
        self.battle_tag = tag
        self.player_role = "p1"
        self.turn = 1
        self.observations = {}
        self.team = {f"p1: Mon{i}": _Mon() for i in range(n_mons)}
        self.opponent_team = {}
        self.teampreview_opponent_team = []
        self.active_pokemon = None
        self.opponent_active_pokemon = None
        self.weather = {}
        self.fields = {}
        self.side_conditions = {}
        self.opponent_side_conditions = {}
        self.force_switch = False
        self.won = True


def test_per_battle_state_is_isolated():
    p = FogPlayer(agent=None)
    b1, b2 = _Battle("battle-a"), _Battle("battle-b")
    tag1, led1, trk1, fed1 = p._battle_state(b1)
    tag2, led2, trk2, fed2 = p._battle_state(b2)
    assert tag1 != tag2 and trk1 is not trk2 and led1 is not led2 and fed1 is not fed2

    trk1.feed_turn(1, [["", "faint", "p1a: Mon0"]])
    assert trk1.faints["p1"] == 1 and trk2.faints["p1"] == 0
    led1.record_item("p1: Mon0", 3)
    assert led2._items == {}

    # re-fetch returns the SAME per-battle objects (idempotent)
    _, led1b, trk1b, _ = p._battle_state(b1)
    assert led1b is led1 and trk1b is trk1

    # finishing one battle drops only its state
    p._battle_finished(b1)
    assert "battle-a" not in p._trackers and "battle-b" in p._trackers


def test_teampreview_without_agent_falls_back():
    p = FogPlayer(agent=None)
    assert p.choose_teampreview(_Battle("x")) == "/team 123456"


def test_teampreview_with_agent_orders_and_records():
    from fog.models.agent import FogAgent

    torch.manual_seed(0)
    agent = FogAgent.from_tables()
    p = FogPlayer(agent, temperature=0.0)
    buf = RolloutBuffer()
    p.attach_buffer(buf)
    b = _Battle("preview-battle")
    order = p.choose_teampreview(b)
    assert order.startswith("/team ")
    digits = order.split(" ", 1)[1]
    assert sorted(digits) == list("123456")            # a permutation, lead chosen by the head
    # the preview pick was recorded as a bandit decision for this game
    gid = p._game_ids["preview-battle"]
    g = buf._games[gid]
    assert g.preview is not None and 0 <= g.preview["slot"] < 6
    assert np.isfinite(g.preview["log_prob"])
