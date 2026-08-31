"""Tests for the §7b/§7c readability eval driver (server-free parts)."""
from __future__ import annotations

import numpy as np
import torch

from fog.models.agent import FogAgent
from fog.models.probe import TrainableProbe
from tests.test_leakage_pipeline import _filled_buffer


def test_readability_curves_shapes():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    agent = FogAgent()
    probe = TrainableProbe.from_agent(agent)
    batch = _filled_buffer(rng, games=3, decisions=4).to_batch()

    from fog.eval.readability import readability_curves

    out = readability_curves(probe, batch, minibatch=8)
    assert set(out) == {"curve", "per_family_ce", "mean_ce"}
    assert np.isfinite(out["mean_ce"]) and out["mean_ce"] >= 0
    assert set(out["per_family_ce"]) == {
        "species", "item", "ability", "move", "nature", "ev", "tera",
    }
    assert all(np.isfinite(v) for v in out["curve"].values())


class _MockObs:
    def __init__(self, events):
        self.events = events


class _MockMove:
    pass


class _MockMon:
    def __init__(self, item, moves):
        self.item = item
        self.moves = {m: _MockMove() for m in moves}


class _MockBattle:
    player_role = "p1"

    def __init__(self):
        # turn 1: lead moves; turn 2: tera + same move again (lock evidence); turn 3: second move
        self.observations = {
            1: _MockObs([
                ["", "move", "p1a: Dragapult", "Dragon Darts", "p2a: Corviknight"],
                ["", "move", "p2a: Corviknight", "Roost", "p2a: Corviknight"],
            ]),
            2: _MockObs([
                ["", "-terastallize", "p1a: Dragapult", "Dragon"],
                ["", "move", "p1a: Dragapult", "Dragon Darts", "p2a: Corviknight"],
            ]),
            3: _MockObs([
                ["", "move", "p1a: Dragapult", "U-turn", "p2a: Corviknight"],
            ]),
        }
        self.team = {
            "p1: Dragapult": _MockMon("choiceband", ["dragondarts", "uturn", "suckerpunch", "phantomforce"]),
        }


def test_case_studies_extraction():
    from fog.eval.readability import CaseStudies

    cs = CaseStudies()
    cs.extend_from_battle(_MockBattle())
    assert cs.tera_turns == [2]
    # choice mon: first move turn 1, repeated (no switch between) turn 2 -> delay 1
    assert cs.choice_delays == [1]
    # reveal order: dragondarts (slot 0) then uturn (slot 1)
    assert cs.reveal_orders == [(0, 1)]
    rep = cs.report()
    assert rep["choice_lock_delay_mean"] == 1.0
    assert rep["tera_never_frac"] == 0.0
    assert np.isfinite(rep["move_reveal_entropy"])
