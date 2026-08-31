"""v2 decision logger (fog/search/pb_log.py): the mechanical pool filter's exact/subset
semantics, and a whole logged game through pb_search_action with per-battle belief-table updates
and outcome backfill."""
from __future__ import annotations

import numpy as np
import pytest

from fog.env.engine_battle import HAVE_ENGINE

pytestmark = pytest.mark.skipif(not HAVE_ENGINE, reason="poke-engine not installed")

ATTACKER = """Hydreigon @ Choice Specs
Ability: Levitate
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Draco Meteor
- Dark Pulse
- Flash Cannon
- Fire Blast
"""

WEAK2 = """Great Tusk @ Heavy-Duty Boots
Level: 5
Ability: Protosynthesis
EVs: 4 HP
Serious Nature
- Headlong Rush
- Rapid Spin
- Knock Off
- Ice Spinner

Kingambit @ Black Glasses
Level: 5
Ability: Supreme Overlord
EVs: 252 HP / 252 Atk / 4 SpD
Adamant Nature
- Kowtow Cleave
- Sucker Punch
- Iron Head
- Swords Dance
"""


def test_keep_mask_exact_and_subset_semantics():
    from fog.search.pb_log import keep_mask_from_reveals

    cand = np.arange(2 * 6 * 15, dtype=np.int64).reshape(2, 6, 15)
    revealed = np.full((6, 15), -1, dtype=np.int64)
    np.testing.assert_array_equal(keep_mask_from_reveals(revealed, cand), [True, True])
    # exact attribute: a revealed item must match
    revealed[0, 1] = cand[0, 0, 1]
    np.testing.assert_array_equal(keep_mask_from_reveals(revealed, cand), [True, False])
    # move subset: revealed move present ANYWHERE in the slot's move set keeps the candidate,
    # regardless of which subslot it was packed into (order-free by design)
    revealed = np.full((6, 15), -1, dtype=np.int64)
    revealed[1, 3] = cand[1, 1, 6]              # candidate 1's LAST move, revealed FIRST
    mask = keep_mask_from_reveals(revealed, cand)
    assert mask[1] and not mask[0]


def test_initial_booster_absence_is_mechanical_evidence():
    """A turn-one automatic activation is observable in both directions: seeing it keeps only
    candidates that predict it, while its absence eliminates a Booster Energy lead. A positive-
    reveal-only token grid cannot express the second fact, so the dedicated entry check must."""
    from fog.env.engine_battle import EngineBattle
    from fog.env.teams import load_pool
    from fog.search.evidence import keep_mask_from_initial_drive_activation
    from fog.search.own_states import StablePoolOwnStates

    pool = load_pool("data/teams/pool_a")
    candidates = StablePoolOwnStates(pool).tokens

    without = EngineBattle(pool[1], pool[1], seed=0, battle_tag="entry-without")
    keep_without = keep_mask_from_initial_drive_activation(without, "p1", candidates)
    np.testing.assert_array_equal(
        keep_without, [False, True, True, True, True, True, True, False]
    )

    with_booster = EngineBattle(pool[0], pool[1], seed=0, battle_tag="entry-with")
    keep_with = keep_mask_from_initial_drive_activation(with_booster, "p1", candidates)
    np.testing.assert_array_equal(
        keep_with, [True, False, False, False, False, False, False, True]
    )


def test_whole_game_logged_with_outcome_backfill(tmp_path):
    from types import SimpleNamespace

    from fog.env.engine_battle import EngineBattle
    from fog.models.agent import FogAgent
    from fog.search.pb_log import play_and_log_pb_games

    class StubOwn:
        def states(self, battle, role):
            t = battle.truth_tokens(role)
            return SimpleNamespace(tokens=np.stack([t, t]),
                                   weights=np.array([0.5, 0.5]), true_index=0, k=2)

    agent = FogAgent.from_tables({})
    proto = EngineBattle(ATTACKER, WEAK2, seed=0, battle_tag="pblog-proto")
    opp_pool = np.stack([proto.truth_tokens("p2")])

    def factory(game_id, seed):
        return EngineBattle(ATTACKER, WEAK2, seed=seed, battle_tag=f"pblog-{game_id}")

    out = str(tmp_path / "pb_log_v2.pt")
    rows = play_and_log_pb_games(
        agent, factory, StubOwn(), opp_pool, n_games=1, out_path=out, seed=5,
        max_decisions=20, search_kwargs={"iters": 24, "n_queried": 2})

    assert len(rows) >= 1, "at least the opening decision goes through the solver"
    for row in rows:
        assert row["outcome"] == 1.0, "Hydreigon wins; outcome backfilled onto every row"
        np.testing.assert_allclose(row["action_dist"].sum(), 1.0)
        assert row["action_dist"][row["action"]] > 0
        assert row["sims_run"] > 0 and row["j_used"] == 1
        assert row["rho"].shape == (2, 1)
        assert row["sigma1"].shape[0] == 2
        assert row["obs"] is not None and row["true_index"] == 0
        assert len(row["queried_beliefs"]) <= 2
        # Phase-G value-training fields: j* found (opp team is the sole pool member), and the token
        # grids index rho's axes (own_tokens is [K,6,15], opp_tokens is [J,6,15]).
        assert row["opp_true_index"] == 0
        assert row["own_tokens"].shape == (row["rho"].shape[0], 6, 15)
        assert row["opp_tokens"].shape == (row["rho"].shape[1], 6, 15)
        # G1 fields: mid-solve snapshot posteriors stored for the value trainer
        assert 1 <= row["sample_iteration"] <= row["iterations"]
        assert len(row["sample_beliefs"]) <= 2
        for lab1, lab2, post in row["sample_beliefs"]:
            assert lab1 in row["labels1"] and lab2 in row["labels2"]
            assert post.shape == row["rho"].shape

    import torch

    saved = torch.load(out, weights_only=False)
    assert saved["meta"]["version"] == 2
    assert saved["meta"]["opp_pool_tokens"].shape == (1, 6, 15)
    assert len(saved["rows"]) == len(rows)


# Three level-5 Pokemon: the learner's lead is knocked out but two live bench remain, so the
# runner must resolve a MULTI-choice forced replacement mid-game (F2). Before F2 this raised
# inside build_pb_grid; the lone-Hydreigon whole-game test above can never reach it.
WEAK3 = WEAK2 + """
Dragonite @ Heavy-Duty Boots
Level: 5
Ability: Multiscale
EVs: 4 HP
Serious Nature
- Extreme Speed
- Earthquake
- Dragon Dance
- Roost
"""


def test_learner_faint_logs_one_sided_replacement_rows(tmp_path):
    """F2 end-to-end: a game where the learner (p1) faints with 2+ live bench plays through the
    runner without raising, and its forced replacement is logged as a solved one-sided row (the
    synthetic observe column, a valid sampled replacement) with the belief update applied."""
    from types import SimpleNamespace

    from fog.env.engine_battle import EngineBattle
    from fog.models.agent import FogAgent
    from fog.search.pb_grid import OBSERVE_LABEL
    from fog.search.pb_log import play_and_log_pb_games

    class StubOwn:
        # WEAK3 is the LEARNER here (p1); the opponent is the lone strong Hydreigon.
        def states(self, battle, role):
            t = battle.truth_tokens(role)
            return SimpleNamespace(tokens=np.stack([t, t]),
                                   weights=np.array([0.5, 0.5]), true_index=0, k=2)

    agent = FogAgent.from_tables({})
    proto = EngineBattle(WEAK3, ATTACKER, seed=0, battle_tag="pblog-repl-proto")
    opp_pool = np.stack([proto.truth_tokens("p2")])

    def factory(game_id, seed):
        return EngineBattle(WEAK3, ATTACKER, seed=seed, battle_tag=f"pblog-repl-{game_id}")

    rows = play_and_log_pb_games(
        agent, factory, StubOwn(), opp_pool, n_games=1, seed=5,
        max_decisions=40, search_kwargs={"iters": 16, "n_queried": 2})

    one_sided = [r for r in rows if r["labels2"] == [OBSERVE_LABEL]]
    assert one_sided, "the learner faints and owes a solved forced replacement (F2)"
    for r in one_sided:
        assert r["sigma2"].shape == (r["rho"].shape[1], 1)
        np.testing.assert_allclose(r["sigma2"], 1.0)
        assert r["sims_run"] > 0
        np.testing.assert_allclose(r["action_dist"].sum(), 1.0)
        assert r["action_dist"][r["action"]] > 0, "sampled from the solved replacement support"
        assert r["outcome"] in (-1.0, 0.0, 1.0)


def test_keep_mask_unspecified_candidate_attribute_cannot_be_contradicted():
    """G2 fix: a candidate sheet's -1 (paste-unspecified attribute — tera type in practice)
    matches ANY revealed value; a specified mismatching value still eliminates."""
    from fog.search.pb_log import keep_mask_from_reveals

    cand = np.zeros((2, 6, 15), dtype=np.int64)
    cand[0, 0, 14] = -1                     # candidate 0: no Tera Type line in the paste
    cand[1, 0, 14] = 5                      # candidate 1: tera type 5
    revealed = np.full((6, 15), -1, dtype=np.int64)
    revealed[0, 14] = 3                     # we terastallized: engine default type 3 goes public
    np.testing.assert_array_equal(keep_mask_from_reveals(revealed, cand), [True, False])


def test_stable_pool_own_states_fixed_axes():
    """G2: the persistent table needs a battle-lifetime-stable own axis; the stable source
    returns the WHOLE pool every turn (no reveal filtering) with the true index located."""
    from fog.env.engine_battle import EngineBattle
    from fog.search.own_states import StablePoolOwnStates

    src = StablePoolOwnStates([ATTACKER, WEAK2])
    b = EngineBattle(ATTACKER, WEAK2, seed=0, battle_tag="stable-own")
    s = src.states(b, "p1")
    assert s.k == 2 and s.true_index == 0
    np.testing.assert_allclose(s.weights, [0.5, 0.5])
    s2 = src.states(b, "p2")
    assert s2.true_index == 1

    off_pool = EngineBattle(ATTACKER, WEAK2, seed=0, battle_tag="stable-own-off")
    with pytest.raises(RuntimeError, match="not in this pool"):
        StablePoolOwnStates([WEAK2]).states(off_pool, "p1")


def test_opening_solve_cache_reuses_the_matchup_solve(tmp_path):
    """L1: the second game of the same matchup reuses the stored opening solve exactly (one
    cache hit), the game still completes, and rows stay schema-valid."""
    from types import SimpleNamespace

    from fog.env.engine_battle import EngineBattle
    from fog.models.agent import FogAgent
    from fog.search.pb_log import OpeningSolveCache, play_and_log_pb_games

    class StubOwn:
        def states(self, battle, role):
            t = battle.truth_tokens(role)
            return SimpleNamespace(tokens=np.stack([t, t]),
                                   weights=np.array([0.5, 0.5]), true_index=0, k=2)

    agent = FogAgent.from_tables({})
    proto = EngineBattle(ATTACKER, WEAK2, seed=0, battle_tag="pblog-cache-proto")
    opp_pool = np.stack([proto.truth_tokens("p2")])

    def factory(game_id, seed):
        return EngineBattle(ATTACKER, WEAK2, seed=seed, battle_tag=f"pblog-cache-{game_id}")

    cache = OpeningSolveCache()
    rows = play_and_log_pb_games(
        agent, factory, StubOwn(), opp_pool, n_games=2, seed=5,
        max_decisions=20, search_kwargs={"iters": 16, "n_queried": 2},
        solve_cache=cache)
    assert cache.hits >= 1, "game 2's opening is the same matchup — it must hit"
    assert cache.misses >= 1
    games = {r["game_id"] for r in rows}
    assert games == {0, 1}
    for row in rows:
        np.testing.assert_allclose(row["action_dist"].sum(), 1.0)
        assert row["action_dist"][row["action"]] > 0
        assert row["truncated"] is False
        assert row["outcome"] in (-1.0, 0.0, 1.0)
        # G2 kept maps ride on every row (no truncation configured -> identity maps)
        np.testing.assert_array_equal(row["own_kept"], np.arange(row["rho"].shape[0]))
        np.testing.assert_array_equal(row["opp_kept"], np.arange(row["rho"].shape[1]))
        assert row["dropped_own_mass"] == pytest.approx(0.0)
        assert row["dropped_opp_mass"] == pytest.approx(0.0)


def test_keep_mask_consumed_item_reveals_no_item_without_eliminating():
    """G2 fix (found by the scale smoke): the census records item STATE, so a consumed / popped /
    knocked-off item reveals as <NO_ITEM> — which must not eliminate a candidate whose sheet has
    an item (Air Balloon in POOL-A). A revealed REAL item still eliminates mismatched sheets."""
    from fog.search.pb_log import keep_mask_from_reveals

    no_item = 250
    cand = np.zeros((2, 6, 15), dtype=np.int64)
    cand[0, 0, 1] = 6                        # sheet: Air Balloon
    cand[1, 0, 1] = 33                       # sheet: Choice Specs
    revealed = np.full((6, 15), -1, dtype=np.int64)
    revealed[0, 1] = no_item                 # balloon popped: current item is <NO_ITEM>
    np.testing.assert_array_equal(
        keep_mask_from_reveals(revealed, cand, no_item_token=no_item), [True, True])
    revealed[0, 1] = 6                       # a real revealed item still eliminates
    np.testing.assert_array_equal(
        keep_mask_from_reveals(revealed, cand, no_item_token=no_item), [True, False])


def test_selfplay_both_sides_solve_and_update(tmp_path):
    """I.2: solver-vs-solver self-play — both sides' decisions are solved rows over shared pool
    axes, outcomes are mirrored per role, and each side's observed action carries the acting
    solve's likelihood (no scripted opponent anywhere in the game)."""
    import torch

    from fog.env.engine_battle import EngineBattle
    from fog.models.agent import FogAgent
    from fog.search.own_states import StablePoolOwnStates
    from fog.search.pb_log import play_and_log_pb_selfplay_games

    agent = FogAgent.from_tables({})
    pool = [ATTACKER, WEAK2]
    src = StablePoolOwnStates(pool)

    def factory(game_id, seed):
        return EngineBattle(ATTACKER, WEAK2, seed=seed, battle_tag=f"pbsp-{game_id}")

    out = str(tmp_path / "selfplay.pt")
    rows = play_and_log_pb_selfplay_games(
        agent, factory, src, src.tokens, n_games=1, out_path=out, seed=3,
        max_decisions=60, search_kwargs={"iters": 16, "n_queried": 2})

    by_role = {"p1": [r for r in rows if r["role"] == "p1"],
               "p2": [r for r in rows if r["role"] == "p2"]}
    assert by_role["p1"] and by_role["p2"], "both sides log solved decisions (2x data)"
    by_turn = {r: {row["turn"]: row for row in by_role[r]} for r in by_role}
    shared_turns = sorted(set(by_turn["p1"]) & set(by_turn["p2"]))
    assert shared_turns, "at least one simultaneous decision must exercise the shared solve"
    for turn in shared_turns:
        p1, p2 = by_turn["p1"][turn], by_turn["p2"][turn]
        np.testing.assert_allclose(p1["sigma2"], p2["sigma1"])
        np.testing.assert_allclose(p1["sigma1"], p2["sigma2"])
        np.testing.assert_allclose(p1["rho"].T, p2["rho"])
    outcomes = {r: {row["outcome"] for row in by_role[r]} for r in by_role}
    assert outcomes["p1"] == {1.0} and outcomes["p2"] == {-1.0}, \
        "Hydreigon side wins; outcomes are mirrored per role"
    for row in rows:
        np.testing.assert_allclose(row["action_dist"].sum(), 1.0)
        assert row["action_dist"][row["action"]] > 0
        assert row["sims_run"] > 0
        # shared axes: each side's true own index is its pool position; the opponent's true
        # index on the row's kept axis resolves (nothing was truncated at this pool size)
        assert row["true_index"] in (0, 1) and row["opp_true_index"] in (0, 1)
        assert row["truncated"] is False
        assert "action_label_col" in row
    assert any("value_obs" in row for row in rows), "played public leaves are logged for value"

    saved = torch.load(out, weights_only=False)
    assert saved["meta"]["mode"] == "selfplay"


def test_selfplay_requires_shared_axes():
    from types import SimpleNamespace

    from fog.search.pb_log import play_and_log_pb_selfplay_games

    src = SimpleNamespace(tokens=np.zeros((2, 6, 15), dtype=np.int64))
    with pytest.raises(ValueError, match="shared candidate space"):
        play_and_log_pb_selfplay_games(
            None, None, src, np.ones((2, 6, 15), dtype=np.int64))


def test_loop_generation_smoke(tmp_path):
    """I.5: one full generation — solver self-play -> distill v2 -> belief refresh — runs
    in-process at smoke scale, mutates the agent, and reports the §15.4 gates it ran (and
    names the ones it did not)."""
    from fog.models.agent import FogAgent
    from fog.scripts.pb_loop import run_generation

    agent = FogAgent.from_tables({"model": {"use_posterior_features": True}})
    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "attacker.txt").write_text(ATTACKER)
    (pool / "weak2.txt").write_text(WEAK2)
    report, next_net = run_generation(
        agent, str(pool), tmp_path / "gen_00", games=3, workers=1, seed=0,
        flush_every=0, value_net_path=None,
        search_kwargs={"iters": 12, "n_queried": 2},
        distill_kwargs={"epochs": 2, "batch_size": 8, "holdout_frac": 0.5},
        refresh_kwargs={"games": 4, "holdout_games": 2, "epochs": 1,
                        "engine_batch": 2, "minibatch_size": 64})
    assert report["rows"] > 0
    assert next_net is None
    assert (tmp_path / "gen_00" / "agent.pt").exists()
    assert (tmp_path / "gen_00" / "report.json").exists()
    g = report["gates"]
    assert isinstance(g["kl_improved_within_generation"], bool)
    assert isinstance(g["refresh_belief_ce_not_worse"], bool)
    assert "not_run" in g["strength_vs_fixed_gauntlet"], "unran gates are named, never faked"
