"""pb_search_action (fog/search/pb_search.py, PROPOSAL_IMPL_PLAN Phase F): the assembled
two-sided decision. Pins: the API has NO temperature (settled 16.2), the executed action is
sampled from the true state's averaged row and stays inside the legality mask, persistent-table
mode uses (and validates) the table's rho, the adaptive sampled mode terminates, trivial
decisions skip the grid entirely, and the whole call is reproducible given its seeds."""
from __future__ import annotations

import inspect
import random

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

DEFENDER = """Tyranitar @ Leftovers
Ability: Sand Stream
EVs: 252 HP / 4 Atk / 252 SpD
Careful Nature
- Stealth Rock
- Rock Blast
- Earthquake
- Thunder Wave
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


class _StubOwnStates:
    """K duplicate truth worlds, true state at index 0 — the own_source contract."""

    def __init__(self, k: int = 2):
        self.k = k

    def states(self, battle, role):
        from types import SimpleNamespace

        t = battle.truth_tokens(role)
        return SimpleNamespace(tokens=np.stack([t] * self.k),
                               weights=np.full(self.k, 1.0 / self.k),
                               true_index=0, k=self.k)


def _battle(p2: str = DEFENDER, seed: int = 0):
    from fog.env.engine_battle import EngineBattle

    return EngineBattle(ATTACKER, p2, seed=seed, battle_tag=f"pbsearch-test-{seed}")


@pytest.fixture(scope="module")
def agent():
    from fog.models.agent import FogAgent

    return FogAgent.from_tables({})


def test_api_has_no_temperature_and_no_greedy_mode():
    from fog.search.pb_search import pb_search_action

    params = inspect.signature(pb_search_action).parameters
    assert "temperature" not in params, "settled decision 16.2: exact sampling only"
    assert "greedy" not in params


def test_persistent_mode_end_to_end(agent):
    from fog.search.pb_search import pb_search_action

    b = _battle()
    opp = np.stack([b.truth_tokens("p2")])
    r = pb_search_action(agent, b, "p1", own_source=_StubOwnStates(2), opp_tokens=opp,
                         seed=7, iters=48, n_queried=4)
    mask = b.action_mask("p1")
    assert mask[r.action], "the executed action must be legal on the REAL battle"
    assert r.action_dist[r.action] > 0, "sampled from the solved distribution's support"
    np.testing.assert_allclose(r.action_dist.sum(), 1.0)
    assert (r.action_dist[~mask] == 0).all()
    np.testing.assert_allclose(r.sigma1.sum(axis=1), 1.0)
    np.testing.assert_allclose(r.sigma2.sum(axis=1), 1.0)
    assert r.sims_run > 0 and r.j_used == 1 and r.failed_cells == 0
    assert r.pruned_mass_max <= 0.2
    assert len(r.queried_beliefs) == 4
    for lab1, lab2, post in r.queried_beliefs:
        assert lab1 in r.labels1 and lab2 in r.labels2
        assert post.shape == r.rho.shape
        assert post.sum() == pytest.approx(1.0) or post.sum() == pytest.approx(0.0)
    # G1: the random-iteration snapshot posteriors ride alongside (§9.3 mitigation 2)
    assert 1 <= r.sample_iteration <= r.iterations
    assert len(r.sample_beliefs) == 4
    for lab1, lab2, post in r.sample_beliefs:
        assert lab1 in r.labels1 and lab2 in r.labels2
        assert post.shape == r.rho.shape
        assert post.sum() == pytest.approx(1.0) or post.sum() == pytest.approx(0.0)


def test_selective_depth_two_expands_only_likely_public_leaves(agent):
    """A real >1-ply solve retains the root grid, expands one high-reach leaf, and re-solves."""
    from fog.search.pb_search import pb_search_action

    battle = _battle(seed=17)
    opponent = np.stack([battle.truth_tokens("p2")])
    result = pb_search_action(
        agent, battle, "p1", own_source=_StubOwnStates(1), opp_tokens=opponent,
        seed=18, iters=24, search_depth=2, depth2_max_nodes=1,
        depth2_min_reach=0.0, depth2_child_iters=24,
    )
    assert result.search_depth == 2
    assert result.depth2_expanded == 1
    assert result.depth2_child_sims > 0
    assert result.sims_run > result.depth2_child_sims
    assert 0.0 < result.depth2_reach_mass <= 1.0
    assert not result.depth2_failures
    assert battle.action_mask("p1")[result.action]


def test_table_mode_uses_and_validates_the_table(agent):
    from fog.search.belief_table import JointBeliefTable
    from fog.search.pb_search import pb_search_action

    b = _battle()
    opp = np.stack([b.truth_tokens("p2")])
    table = JointBeliefTable(np.log(np.array([[0.75], [0.25]])))
    r = pb_search_action(agent, b, "p1", own_source=_StubOwnStates(2), opp_tokens=opp,
                         table=table, seed=1, iters=32)
    np.testing.assert_allclose(r.rho, table.normalized())

    with pytest.raises(ValueError, match="does not index"):
        pb_search_action(agent, b, "p1", own_source=_StubOwnStates(3), opp_tokens=opp,
                         table=table, iters=8)
    with pytest.raises(ValueError, match="stable candidates"):
        pb_search_action(agent, b, "p1", own_source=_StubOwnStates(2), table=table, iters=8)


def test_reproducible_given_seed(agent):
    from fog.search.pb_search import pb_search_action

    results = []
    for _ in range(2):
        b = _battle(seed=9)
        opp = np.stack([b.truth_tokens("p2")])
        results.append(pb_search_action(agent, b, "p1", own_source=_StubOwnStates(2),
                                        opp_tokens=opp, rng=random.Random(3), iters=32))
    assert results[0].action == results[1].action
    np.testing.assert_array_equal(results[0].sigma1, results[1].sigma1)
    np.testing.assert_array_equal(results[0].action_dist, results[1].action_dist)


def test_trivial_forced_decision_skips_the_grid(agent):
    """A pending forced replacement with one legal choice is executed directly: no simulation,
    no solver — and no behavioral likelihood for the belief update (proposal §14)."""
    from fog.search.pb_leaves import commit_pair_branches
    from fog.search.pb_search import pb_search_action

    b = _battle(p2=WEAK2)
    draco = next(i for i, m in enumerate(b.active("p1").moves)
                 if str(m.id).lower().replace(" ", "") == "dracometeor")
    outcomes, _ = commit_pair_branches(b, "p1", draco,
                                       int(np.flatnonzero(b.action_mask("p2"))[0]))
    leaf = next(o.leaf for o in outcomes if o.leaf is not None
                and tuple(o.leaf.pending_sides()) == ("p2",))
    legal = np.flatnonzero(leaf.action_mask("p2"))
    assert len(legal) == 1, "fixture: the only choice is switching in the last Pokemon"
    r = pb_search_action(agent, leaf, "p2", own_source=_StubOwnStates(1))
    assert r.action == int(legal[0])
    assert r.sims_run == 0 and r.iterations == 0
    assert r.sample_iteration == -1 and r.sample_beliefs == [], "no solve ran, no snapshot"


def test_one_sided_replacement_is_solved_and_sampled(agent):
    """F2: a forced replacement with two live bench Pokemon is a real one-sided solve — the result
    carries the synthetic single opponent column, sigma2 is the [J, 1] ones vector, and the
    executed replacement is sampled exactly from the solved (possibly mixed) sigma1 row."""
    from fog.search.pb_grid import OBSERVE_LABEL
    from fog.search.pb_leaves import commit_pair_branches
    from fog.search.pb_search import pb_search_action

    weak3 = WEAK2 + """
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
    from fog.env.engine_battle import EngineBattle

    b = EngineBattle(weak3, ATTACKER, seed=1, battle_tag="pbsearch-repl")
    draco = next(i for i, m in enumerate(b.active("p2").moves)
                 if str(m.id).lower().replace(" ", "") == "dracometeor")
    outcomes, _ = commit_pair_branches(
        b, "p1", int(np.flatnonzero(b.action_mask("p1"))[0]), draco)
    leaf = next(o.leaf for o in outcomes if o.leaf is not None
                and tuple(o.leaf.pending_sides()) == ("p1",))
    assert len(np.flatnonzero(leaf.action_mask("p1"))) >= 2, "fixture: a multi-choice replacement"

    opp = np.stack([leaf.truth_tokens("p2")] * 2)
    r = pb_search_action(agent, leaf, "p1", own_source=_StubOwnStates(2), opp_tokens=opp,
                         seed=7, iters=32)
    assert r.labels2 == [OBSERVE_LABEL]
    assert r.sigma2.shape == (2, 1)
    np.testing.assert_allclose(r.sigma2, 1.0)
    assert r.sims_run > 0 and r.iterations > 0, "a real solve ran (not the trivial fast path)"
    assert leaf.action_mask("p1")[r.action], "the executed replacement is legal"
    assert r.action_dist[r.action] > 0
    np.testing.assert_allclose(r.action_dist.sum(), 1.0)
    np.testing.assert_allclose(r.sigma1.sum(axis=1), 1.0)


def test_adaptive_sampled_mode_terminates_and_grows(agent):
    from fog.search.pb_search import pb_search_action

    b = _battle()
    truth = b.truth_tokens("p2")

    calls = []

    def opp_source(battle, role, j, *, seed):
        calls.append(j)
        toks = np.stack([truth] * j)                 # no dedup: j grows as requested
        return toks, np.log(np.full(j, 1.0 / j))

    r = pb_search_action(agent, b, "p1", own_source=_StubOwnStates(1), opp_source=opp_source,
                         seed=2, iters=32, j_start=1, j_max=2, j_tol=0.05)
    assert calls[0] == 1
    assert r.j_used in (1, 2)
    assert len(calls) <= 2, "identical duplicate worlds must stabilize immediately"


def test_persistent_truncation_zero_mass_always_dropped(agent):
    """G2: zero-mass opponent candidates are dropped from the solve even without top_k/top_j —
    exact (their strategies carry no weight) and load-bearing for fork validity."""
    from fog.search.belief_table import JointBeliefTable
    from fog.search.pb_search import pb_search_action

    b = _battle()
    truth = b.truth_tokens("p2")
    opp = np.stack([truth, truth, truth])
    with np.errstate(divide="ignore"):
        log_prior = np.log(np.array([[0.45, 0.3, 0.0], [0.15, 0.1, 0.0]]))
        table = JointBeliefTable(log_prior)
    r = pb_search_action(agent, b, "p1", own_source=_StubOwnStates(2), opp_tokens=opp,
                         table=table, seed=3, iters=24)
    np.testing.assert_array_equal(r.opp_kept, [0, 1])
    assert r.j_used == 2 and r.sigma2.shape[0] == 2
    assert r.dropped_opp_mass == pytest.approx(0.0), "zero mass dropped is zero mass lost"
    np.testing.assert_allclose(r.rho.sum(), 1.0)


def test_persistent_truncation_top_j_and_refuses_private_truth_dependent_top_k(agent):
    """Opponent trimming is public; own trimming may not force the hidden truth into support."""
    from fog.search.belief_table import JointBeliefTable
    from fog.search.pb_search import pb_search_action

    b = _battle()
    truth = b.truth_tokens("p2")
    opp = np.stack([truth, truth, truth])
    # own margins [0.25, 0.75] (true index 0 is the LIGHTEST); opp margins [0.5, 0.3, 0.2]
    table = JointBeliefTable(np.log(np.outer([0.25, 0.75], [0.5, 0.3, 0.2])))
    with pytest.raises(ValueError, match="leaks that truth"):
        pb_search_action(agent, b, "p1", own_source=_StubOwnStates(2), opp_tokens=opp,
                         table=table, seed=3, iters=24, top_k=1, top_j=2)

    r = pb_search_action(agent, b, "p1", own_source=_StubOwnStates(2), opp_tokens=opp,
                         table=table, seed=3, iters=24, top_k=None, top_j=2)
    np.testing.assert_array_equal(r.own_kept, [0, 1])
    np.testing.assert_array_equal(r.opp_kept, [0, 1])
    assert r.true_index == 0 and r.sigma1.shape[0] == 2
    assert r.dropped_own_mass == pytest.approx(0.0)
    assert r.dropped_opp_mass == pytest.approx(0.2)
    assert r.rho.shape == (2, 2)
    np.testing.assert_allclose(r.rho.sum(), 1.0)
    assert r.own_tokens.shape == (2, 6, 15) and r.opp_tokens.shape == (2, 6, 15)
    mask = b.action_mask("p1")
    assert mask[r.action]


def test_truncation_is_persistent_mode_only(agent):
    from fog.search.pb_search import pb_search_action

    b = _battle()
    with pytest.raises(ValueError, match="persistent-mode"):
        pb_search_action(agent, b, "p1", own_source=_StubOwnStates(1),
                         opp_source=lambda bb, r, j, *, seed: (None, None),
                         top_j=4, iters=8)


def test_policy_init_warm_start(agent):
    """I.3: policy_init=True initializes both sides' strategies from the per-world policy
    distributions — the solve still runs, the result stays valid, and the mapping helper puts
    zero mass on unavailable labels with rows summing to one."""
    from fog.search.pb_grid import build_pb_grid
    from fog.search.pb_search import _policy_inits, pb_search_action

    b = _battle()
    opp = np.stack([b.truth_tokens("p2")])
    own = _StubOwnStates(2).states(b, "p1")
    grid = build_pb_grid(b, "p1", own.tokens, opp, seed=0)
    init1, init2 = _policy_inits(agent, b, "p1", grid, own.tokens, opp, random.Random(0))
    assert init1.shape == grid.avail1.shape and init2.shape == grid.avail2.shape
    np.testing.assert_allclose(init1.sum(axis=1), 1.0)
    np.testing.assert_allclose(init2.sum(axis=1), 1.0)
    assert (init1[~grid.avail1] == 0).all() and (init2[~grid.avail2] == 0).all()

    r = pb_search_action(agent, b, "p1", own_source=_StubOwnStates(2), opp_tokens=opp,
                         seed=7, iters=32, policy_init=True)
    assert b.action_mask("p1")[r.action]
    np.testing.assert_allclose(r.sigma1.sum(axis=1), 1.0)
    np.testing.assert_allclose(r.sigma2.sum(axis=1), 1.0)


def test_value_net_switch_drives_the_solve(agent):
    """G5 plumbing: value_net=PBValueNet routes leaf values through the per-pair-table bridge
    (pb_value_fn) instead of the interim head; the decision stays valid. The interim head stays
    the default baseline arm (value_net=None)."""
    from fog.models.pb_value import PBValueNet
    from fog.search.pb_search import pb_search_action

    b = _battle()
    opp = np.stack([b.truth_tokens("p2")])
    net = PBValueNet.from_tables(d=32, n_rounds=1)
    r = pb_search_action(agent, b, "p1", own_source=_StubOwnStates(2), opp_tokens=opp,
                         seed=7, iters=24, value_net=net, value_recompute_every=4)
    assert b.action_mask("p1")[r.action]
    np.testing.assert_allclose(r.sigma1.sum(axis=1), 1.0)
    np.testing.assert_allclose(r.sigma2.sum(axis=1), 1.0)
    assert r.sims_run > 0
