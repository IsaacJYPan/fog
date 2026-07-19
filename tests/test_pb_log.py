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

    cand = np.arange(2 * 6 * 9, dtype=np.int64).reshape(2, 6, 9)
    revealed = np.full((6, 9), -1, dtype=np.int64)
    np.testing.assert_array_equal(keep_mask_from_reveals(revealed, cand), [True, True])
    # exact attribute: a revealed item must match
    revealed[0, 1] = cand[0, 0, 1]
    np.testing.assert_array_equal(keep_mask_from_reveals(revealed, cand), [True, False])
    # move subset: revealed move present ANYWHERE in the slot's move set keeps the candidate,
    # regardless of which subslot it was packed into (order-free by design)
    revealed = np.full((6, 9), -1, dtype=np.int64)
    revealed[1, 3] = cand[1, 1, 6]              # candidate 1's LAST move, revealed FIRST
    mask = keep_mask_from_reveals(revealed, cand)
    assert mask[1] and not mask[0]


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
        # grids index rho's axes (own_tokens is [K,6,9], opp_tokens is [J,6,9]).
        assert row["opp_true_index"] == 0
        assert row["own_tokens"].shape == (row["rho"].shape[0], 6, 9)
        assert row["opp_tokens"].shape == (row["rho"].shape[1], 6, 9)
        # G1 fields: mid-solve snapshot posteriors stored for the value trainer
        assert 1 <= row["sample_iteration"] <= row["iterations"]
        assert len(row["sample_beliefs"]) <= 2
        for lab1, lab2, post in row["sample_beliefs"]:
            assert lab1 in row["labels1"] and lab2 in row["labels2"]
            assert post.shape == row["rho"].shape

    import torch

    saved = torch.load(out, weights_only=False)
    assert saved["meta"]["version"] == 2
    assert saved["meta"]["opp_pool_tokens"].shape == (1, 6, 9)
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
