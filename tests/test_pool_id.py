"""Pool-id belief evaluation: the game-level gate bookkeeping, candidate scoring
mechanics, and the end-to-end engine run."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from fog.env.engine_battle import HAVE_ENGINE
from fog.eval.pool_id import GATE_TURN, aggregate_pool_id
from tests.test_joint_search import P_ME, P_ME_VARIANT, P_OPP


# ------------------------------------------------------------------------------- aggregation
def test_aggregate_gate_reads_last_decision_by_turn_4():
    # game 0: wrong at t1, RIGHT at t4 (last <=4 wins) -> correct
    # game 1: right at t2, wrong at t3, then right only at t9 (past the gate) -> wrong
    # game 2: no decision <= 4 -> excluded from the gate
    records = [
        (0, 1, False), (0, 4, True), (0, 9, True),
        (1, 2, True), (1, 3, False), (1, 9, True),
        (2, 7, True),
    ]
    r = aggregate_pool_id(records, games=3, failed=0, pool_size=8)
    assert r["gate_turn"] == GATE_TURN
    assert r["gate_games"] == 2
    assert r["gate_accuracy"] == pytest.approx(0.5)
    assert not r["gate_passed"]
    assert r["decisions"] == 7 and r["chance"] == pytest.approx(1 / 8)
    assert r["identifiable_decisions"] == 0
    assert np.isnan(r["identifiable_top1_accuracy"])
    assert r["ambiguous_decisions"] == 0
    assert np.isnan(r["ambiguous_top1_accuracy"])

    # curve buckets: turn 9 lands in t9, turns >=10 merge into 10+
    records += [(0, 15, True)]
    r = aggregate_pool_id(records, games=3, failed=0, pool_size=8)
    assert 10 in r["curve_by_turn"]
    assert r["curve_by_turn"][1] == 0.0 and r["curve_by_turn"][4] == 1.0


def test_aggregate_no_gate_games_is_not_a_pass():
    r = aggregate_pool_id([(0, 9, True)], games=1, failed=0, pool_size=4)
    assert r["gate_games"] == 0
    assert np.isnan(r["gate_accuracy"])
    assert not r["gate_passed"]


def test_aggregate_reports_uniquely_identifiable_subset():
    records = [
        (0, 2, True, 1),
        (0, 3, False, 1),
        (1, 2, False, 4),
    ]
    r = aggregate_pool_id(records, games=2, failed=0, pool_size=8)
    assert r["identifiable_decisions"] == 2
    assert r["identifiable_top1_accuracy"] == pytest.approx(0.5)
    assert r["ambiguous_decisions"] == 1
    assert r["ambiguous_top1_accuracy"] == 0.0
    assert r["ambiguous_uniform_accuracy"] == pytest.approx(0.25)
    assert r["ambiguous_excess_accuracy"] == pytest.approx(-0.25)


def test_aggregate_reports_ambiguous_candidate_collapse():
    records = [
        (0, 1, True, 4, 2, 2),
        (0, 2, False, 3, 2, 0),
        (1, 1, False, 8, 5, 0),
    ]
    r = aggregate_pool_id(records, games=2, failed=0, pool_size=8)
    assert r["ambiguous_prediction_counts"]["0"] == 2
    assert r["ambiguous_prediction_counts"]["2"] == 1
    assert r["ambiguous_true_counts"]["2"] == 2
    assert r["ambiguous_true_counts"]["5"] == 1


# ------------------------------------------------------------------------------- engine-backed
engine = pytest.mark.skipif(not HAVE_ENGINE, reason="poke-engine not installed")
showdown_direct = pytest.mark.skipif(
    shutil.which("node") is None
    or not (Path(__file__).parents[1] / "server" / "pokemon-showdown" / "dist").is_dir(),
    reason="requires Node and the pinned built Pokemon Showdown",
)


@pytest.fixture(scope="module")
def agent():
    from fog.models.agent import FogAgent

    a = FogAgent.from_tables({})
    a.eval()
    return a


@engine
def test_pool_id_logprobs_mechanics(agent):
    """Scores are per-candidate joint log-probs: finite for every legal pool candidate,
    deterministic, and DIFFERENT across candidates that differ in hidden attributes."""
    import torch

    from fog.encoding.observation import Observation
    from fog.env.engine_battle import EngineBattle
    from fog.env.engine_runner import _stack_obs
    from fog.env.team_truth import truth_tokens_from_paste
    from fog.eval.pool_id import pool_id_logprobs

    battle = EngineBattle(P_OPP, P_ME, seed=0, battle_tag="poolid-test")
    obs = Observation.from_battle(battle.view("p1"), tracker=battle.trackers["p1"])
    batch = _stack_obs([obs], torch.device("cpu"))
    cands = np.stack([truth_tokens_from_paste(p, battle.tables)
                      for p in (P_ME, P_ME_VARIANT)])

    lp = pool_id_logprobs(agent, batch, cands)
    assert lp.shape == (2,)
    assert np.isfinite(lp).all(), "legal pool candidates must never score -inf"
    assert lp[0] != lp[1], "hidden-set variants must not score identically"
    np.testing.assert_array_equal(lp, pool_id_logprobs(agent, batch, cands))


@engine
def test_pool_id_midgame_reveal_at_nonfirst_paste_position(agent):
    """The runs/pool_id_m15 flat-curve bug, pinned: a revealed move is clamped into subslot 0 of
    the observation, but its PASTE position in the true candidate is later (Knock Off is Great
    Tusk's 4th move in P_ME). Pre-fix, the paste-order gather hit the move-distinctness -inf and
    eliminated the true candidate; post-fix the candidate grid is aligned to the clamp and stays
    finite, while a candidate whose moveset genuinely lacks the revealed move (P_ME_VARIANT
    carries Stealth Rock instead) scores exactly -inf."""
    import torch

    from fog.encoding.observation import Observation
    from fog.env.engine_battle import EngineBattle
    from fog.env.engine_runner import _stack_obs
    from fog.env.team_truth import truth_tokens_from_paste
    from fog.eval.pool_id import pool_id_logprobs

    battle = EngineBattle(P_OPP, P_ME, seed=0, battle_tag="poolid-align")
    obs = Observation.from_battle(battle.view("p1"), tracker=battle.trackers["p1"])
    batch = _stack_obs([obs], torch.device("cpu"))
    cands = np.stack([truth_tokens_from_paste(p, battle.tables)
                      for p in (P_ME, P_ME_VARIANT)])

    knock_off = int(cands[0, 0, 3 + 3])  # P_ME slot 0, paste subslot 3 (non-first)
    assert knock_off >= 0
    assert knock_off not in cands[1, 0, 3:7].tolist()  # variant runs Stealth Rock instead
    toks = batch["opp_revealed"]["tokens"]
    assert int(toks[0, 0, 3]) == -1  # turn 0: no move revealed yet
    toks[0, 0, 3] = knock_off        # inject the reveal at its clamp position (subslot 0)

    lp = pool_id_logprobs(agent, batch, cands)
    assert np.isfinite(lp[0]), "true candidate must survive a non-first-paste-position reveal"
    assert lp[1] == -np.inf, "a candidate whose moveset lacks the revealed move is inconsistent"


@engine
def test_pool_id_joint_and_order_sensitivity(agent):
    """T7: the joint (headline) scorer survives a mid-game reveal at a non-first paste position (true
    candidate finite and top-1 over the hidden-differing variant), and the order-sensitivity probe
    emits sane values (finite std, stability in [0, 1])."""
    import torch

    from fog.encoding.observation import Observation
    from fog.env.engine_battle import EngineBattle
    from fog.env.engine_runner import _stack_obs
    from fog.env.team_truth import truth_tokens_from_paste
    from fog.eval.pool_id import pool_id_joint_logprobs, pool_id_order_sensitivity

    battle = EngineBattle(P_OPP, P_ME, seed=0, battle_tag="poolid-joint")
    obs = Observation.from_battle(battle.view("p1"), tracker=battle.trackers["p1"])
    batch = _stack_obs([obs], torch.device("cpu"))
    cands = np.stack([truth_tokens_from_paste(p, battle.tables) for p in (P_ME, P_ME_VARIANT)])

    knock_off = int(cands[0, 0, 3 + 3])           # P_ME Great Tusk Knock Off, paste subslot 3
    batch["opp_revealed"]["tokens"][0, 0, 3] = knock_off   # revealed, clamped into subslot 0

    lp = pool_id_joint_logprobs(agent, batch, cands)
    assert lp.shape == (2,)
    assert np.isfinite(lp[0]), "true candidate must survive a non-first-paste-position reveal"
    assert lp[1] == -np.inf, "the variant lacks the revealed move -> inconsistent -> -inf"
    # deterministic under the canonical (default) order
    np.testing.assert_array_equal(lp, pool_id_joint_logprobs(agent, batch, cands))

    os = pool_id_order_sensitivity(agent, batch, cands, true_index=0, m_orders=4, seed=0)
    assert np.isfinite(os["true_logprob_std"]) and os["true_logprob_std"] >= 0.0
    assert 0.0 <= os["argmax_stability"] <= 1.0


def test_slot_order_guard_fires_on_shuffled_paste():
    from fog.env.team_truth import assert_uniform_slot_order

    a = np.full((6, 15), -1, dtype=np.int64)
    a[:, 0] = [1, 2, 3, 4, 5, 6]
    b = a.copy()
    b[:, 0] = [2, 1, 3, 4, 5, 6]  # same species multiset, different slot order
    c = np.full((6, 15), -1, dtype=np.int64)
    c[:, 0] = [7, 8, 9, 10, 11, 12]  # distinct multiset: exempt
    assert_uniform_slot_order(np.stack([a, a, c]))  # must not raise
    with pytest.raises(ValueError, match="different slot orders"):
        assert_uniform_slot_order(np.stack([a, b]))


@engine
def test_run_pool_id_end_to_end(agent):
    from fog.eval.pool_id import run_pool_id

    report = run_pool_id(agent, [P_ME, P_ME_VARIANT, P_OPP], games=2, seed=0, max_turns=8)
    assert report["games"] == 2 and report["failed_games"] == 0
    assert report["decisions"] > 0
    assert report["curve_by_turn"], "at least one turn bucket must be populated"
    assert all(0.0 <= v <= 1.0 for v in report["curve_by_turn"].values())
    assert isinstance(report["gate_passed"], bool)


@showdown_direct
def test_run_pool_id_direct_end_to_end(agent):
    from fog.env.teams import load_pool
    from fog.eval.pool_id import run_pool_id_direct

    report = run_pool_id_direct(
        agent,
        load_pool("data/teams/pool_a"),
        games=1,
        seed=7,
        max_turns=2,
        os_sample=0,
        score_orders=1,
        showdown_root="server/pokemon-showdown",
    )
    assert report["backend"] == "showdown_direct"
    assert report["games"] == 1 and report["failed_games"] == 0
    assert report["decisions"] > 0
