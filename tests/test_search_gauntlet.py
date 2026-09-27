"""Search-arm comparisons, simulation budgets, and paired battle schedules."""
from __future__ import annotations

import random
import types

import pytest

from fog.env.engine_battle import HAVE_ENGINE
from fog.eval.search_gauntlet import (
    ArmPlayer,
    MatchRow,
    _team_schedule,
    gate_report,
    run_search_match,
)
from tests.test_joint_search import P_ME, P_ME_VARIANT, P_OPP


# ------------------------------------------------------------------------------------- pure logic
def test_team_schedule_paired_swaps_teams():
    pool = ["A", "B", "C"]
    rng = random.Random(0)
    sched = _team_schedule(pool, 6, rng, paired=True)
    assert len(sched) == 6
    for g1, g2 in zip(sched[0::2], sched[1::2]):
        assert g1 == (g2[1], g2[0]), "each matchup must be replayed with teams swapped"
    # odd n truncates, never exceeds
    assert len(_team_schedule(pool, 5, random.Random(1), paired=True)) == 5
    assert len(_team_schedule(pool, 5, random.Random(1), paired=False)) == 5


def test_match_row_accounting():
    row = MatchRow(label="x", wins=3, games=10, decisions=50, sims=800, failures=2,
                   turns=250, truncated=1)
    assert row.sims_per_decision == pytest.approx(16.0)
    d = row.as_dict()
    assert d["point"] == pytest.approx(0.3)
    assert d["low"] < 0.3 < d["high"]
    assert d["turns"] == 250 and d["truncated"] == 1
    assert "truncated 1" in row.line()
    assert MatchRow(label="empty").sims_per_decision == 0.0  # no div-by-zero


def _row(kind, arm, opponent, wins, games):
    r = MatchRow(label=f"{arm} vs {opponent}", wins=wins, games=games, decisions=1, sims=1)
    r.meta = {"kind": kind, "arm": arm, "opponent": opponent}
    return r


def test_gate_report_orderings():
    # joint clearly above both, everywhere -> CI pass
    rows = [
        _row("gauntlet", "joint", "SetOracle", 700, 1000),
        _row("gauntlet", "determinized", "SetOracle", 400, 1000),
        _row("gauntlet", "policy", "SetOracle", 300, 1000),
        _row("h2h", "joint", "determinized", 700, 1000),
        _row("h2h", "joint", "policy", 700, 1000),
    ]
    gate = gate_report(rows)
    assert gate["pass_point"] and gate["pass_ci"]
    assert len(gate["checks"]) == 4

    # point-ahead but CI-overlapping -> point pass, CI fail (the "more games" verdict)
    rows[0].wins = 420
    gate = gate_report(rows)
    assert not gate["pass_ci"]
    assert gate["pass_point"]

    # joint below determinized head-to-head -> the negative result, reported not raised
    rows[3].wins = 300
    gate = gate_report(rows)
    assert not gate["pass_point"]

    # no recognizable rows -> no vacuous pass
    empty = gate_report([_row("gauntlet", "policy", "MaxDamage", 5, 10)])
    assert not empty["pass_point"] and not empty["pass_ci"] and empty["checks"] == []


def test_arm_player_validation():
    plain = types.SimpleNamespace(oracle="none")
    with pytest.raises(ValueError, match="unknown arm"):
        ArmPlayer("mcts", plain)
    with pytest.raises(ValueError, match="needs the team pool"):
        ArmPlayer("joint", plain, pool=None)
    with pytest.raises(ValueError, match="needs det_k"):
        ArmPlayer("determinized", plain)
    with pytest.raises(ValueError, match="model.oracle"):
        ArmPlayer("policy", plain, oracle_truth=True)
    with pytest.raises(ValueError, match="not a §5 arm"):
        ArmPlayer("determinized", types.SimpleNamespace(oracle="set"),
                  det_k=4, oracle_truth=True)


# ------------------------------------------------------------------------------- engine-backed

engine = pytest.mark.skipif(not HAVE_ENGINE, reason="poke-engine not installed")

POOL = [P_ME, P_ME_VARIANT, P_OPP]


@pytest.fixture(scope="module")
def posterior_agent():
    from fog.models.agent import FogAgent

    a = FogAgent.from_tables({"model": {"use_posterior_features": True}})
    a.eval()
    return a


@pytest.fixture(scope="module")
def oracle_agent():
    from fog.models.agent import FogAgent

    a = FogAgent.from_tables({"model": {"oracle": "set", "use_posterior_features": True}})
    a.eval()
    return a


def _scripted_opponent(seed=1):
    from fog.env.engine_scripted import EngineMaxDamagePolicy

    return EngineMaxDamagePolicy(seed=seed)


@engine
@pytest.mark.parametrize("arm", ["policy", "determinized", "joint"])
def test_arm_rows_end_to_end(arm, posterior_agent):
    """Every arm plays complete legal engine games; search arms account sims and clean pool
    positions simulate without failures."""
    player = ArmPlayer(arm, posterior_agent, pool=POOL, det_k=2, iters=2, seed=3)
    row = run_search_match(player, _scripted_opponent(), POOL, 2,
                           label=f"{arm} vs MaxDamage", seed=5, max_turns=60)
    assert row.games == 2 and row.failed_games == 0
    assert 0 <= row.wins <= 2
    assert row.decisions > 0
    if arm == "policy":
        assert row.sims == 0
    else:
        assert row.sims > 0, "search arms must account their simulation budget"
        assert row.sims_per_decision > 1.0


@engine
def test_reading_opponent_plays_and_h2h_counters(posterior_agent, oracle_agent):
    """The set-oracle reader completes games (truth injection exercised every decision), and an
    ArmPlayer opponent's budget counters land in row.meta."""
    reader = ArmPlayer("policy", oracle_agent, seed=7, oracle_truth=True)
    row = run_search_match(ArmPlayer("policy", posterior_agent, seed=8), reader, POOL, 2,
                           label="policy vs SetOracle", seed=9, max_turns=60)
    assert row.games == 2 and row.failed_games == 0
    assert reader.decisions > 0

    joint = ArmPlayer("joint", posterior_agent, pool=POOL, iters=2, seed=10)
    det = ArmPlayer("determinized", posterior_agent, det_k=2, seed=11)
    h2h = run_search_match(joint, det, POOL, 2, label="joint vs determinized [h2h]",
                           seed=12, max_turns=60, paired=True)
    assert h2h.games == 2
    assert h2h.sims > 0 and h2h.meta["opp_sims"] > 0
    assert h2h.meta["opp_sims_per_decision"] > 1.0
