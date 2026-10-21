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


def test_private_state_truth_provider_is_immutable_tracker_source():
    from fog.constants.tables import TABLES

    p = FogPlayer(agent=None)
    battle = _Battle("immutable-private-truth")
    submitted = np.full((6, 15), -1, dtype=np.int64)
    species = TABLES.species_index("cinderace")
    submitted[0, 0] = species
    submitted[0, 2] = TABLES.ability_index("blaze")
    p.private_state_truth_provider = lambda _battle: submitted.copy()

    # A temporary in-battle ability change must not replace the submitted private state.
    battle.team["p1: Mon0"].species = "cinderace"
    battle.team["p1: Mon0"].ability = "shielddust"
    _, _, tracker, _ = p._battle_state(battle)

    assert tracker.exact_private_truth[species][2] == TABLES.ability_index("blaze")
    assert tracker.exact_private_truth[species][2] != TABLES.ability_index("shielddust")


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


def test_mirror_failure_can_batch_with_prior_decision(monkeypatch):
    from fog.encoding import mirror
    from fog.encoding.dims import DIMS
    from fog.encoding.observation import Observation

    battle = _Battle("mirror-failure")
    obs = Observation.from_battle(battle)
    obs.history = np.full((2, DIMS.turn_dim), 0.25, dtype=np.float32)
    obs.history_ids = np.zeros((2, DIMS.hist_id_fields), dtype=np.int64)
    obs.history_state = np.zeros((2, DIMS.hist_state), dtype=np.float32)
    obs.history_len = 2
    player = FogPlayer()
    buffer = RolloutBuffer()
    player.attach_buffer(buffer)
    recorded = player._mirror(battle, obs, None)
    assert recorded is not None
    assert player.record_mirror is True

    def fail(*args, **kwargs):
        raise RuntimeError("missing mirror data")

    monkeypatch.setattr(mirror, "build_opponent_view_np", fail)
    fallback = player._mirror(battle, obs, None)
    assert player.record_mirror is True
    assert "missing mirror data" in player.mirror_last_error
    assert fallback is not None
    player.phi_hat = object()
    assert player._leakage_ce(fallback, obs, None) is None

    game = buffer.start_game()
    for fragment, view in (recorded, fallback):
        buffer.add_decision(
            game,
            {"history": obs.history, "opp_view": {**fragment, "my_view": view}},
            action=0,
            log_prob=0.0,
            value=0.0,
        )
    buffer.end_game(game, outcome=1.0)
    batch = buffer.to_batch()
    assert batch["opp_view"]["entities"].shape[0] == 2
    assert batch["opp_view"]["history_len"][1] == 0
    assert not batch["opp_view"]["history"][1].any()
    assert any(valid[0].any() for valid in batch["self_belief_valid"].values())
    assert all(not valid[1].any() for valid in batch["self_belief_valid"].values())

    first_attempt = FogPlayer()
    first_attempt.attach_buffer(RolloutBuffer())
    assert first_attempt._mirror(battle, obs, None) is None
    assert first_attempt.record_mirror is False
    assert player.record_mirror is True


def test_replacing_probe_and_weights_reaches_mirror_measurement():
    from fog.encoding.observation import Observation

    class Probe:
        def __init__(self, ce):
            self.ce = ce
            self.weights = None

        def leakage_ce(self, batch, view, *, family_weights):
            assert batch["entities"].shape[0] == 1
            assert (view["labels"]["species"] >= 0).any()
            self.weights = family_weights
            return torch.tensor([self.ce])

    battle = _Battle("replacement-probe")
    obs = Observation.from_battle(battle)
    player = FogPlayer()
    original = Probe(2.5)
    player.phi_hat = original
    player.leak_family_weights = {"species": 0.5}
    # A probe can request a mirror without an attached rollout buffer.
    recorded = player._mirror(battle, obs, None)
    assert player._leakage_ce(recorded, obs, None) == 2.5
    assert original.weights == {"species": 0.5}

    replacement = Probe(1.25)
    player.phi_hat = replacement
    player.leak_family_weights = {"item": 2.0}
    assert player._leakage_ce(recorded, obs, None) == 1.25
    assert replacement.weights == {"item": 2.0}
    assert player.diagnostic_errors == {}
