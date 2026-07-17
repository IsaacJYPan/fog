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
