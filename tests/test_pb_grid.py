"""build_pb_grid (fog/search/pb_grid.py, PROPOSAL_IMPL_PLAN Phase C): the K x J x A x B branch
table. Pins: budget accounting, probability bookkeeping, label spaces for BOTH sides, the
availability factorization, duplicate-world invariance (identical candidate worlds must produce
identical branch tables — the shared-opponent-particles property made testable), censored leaves,
one-sided-root rejection, determinism, and the Phase-E value-assembly contract."""
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

DEFENDER = """Tyranitar @ Leftovers
Ability: Sand Stream
EVs: 252 HP / 4 Atk / 252 SpD
Careful Nature
- Stealth Rock
- Rock Blast
- Earthquake
- Thunder Wave
"""


def _battle(seed: int = 0):
    from fog.env.engine_battle import EngineBattle

    return EngineBattle(ATTACKER, DEFENDER, seed=seed, battle_tag=f"pbgrid-test-{seed}")


def _truth_grid(battle, k_dup: int, j_dup: int):
    own = np.stack([battle.truth_tokens("p1")] * k_dup)
    opp = np.stack([battle.truth_tokens("p2")] * j_dup)
    return own, opp


@pytest.fixture(scope="module")
def grid_2x2():
    from fog.search.pb_grid import build_pb_grid

    b = _battle()
    own, opp = _truth_grid(b, 2, 2)
    return build_pb_grid(b, "p1", own, opp, seed=11)


def test_shapes_labels_and_budget_accounting(grid_2x2):
    from fog.search.pb_grid import public_label

    g = grid_2x2
    n_k, n_a = g.avail1.shape
    n_j, n_b = g.avail2.shape
    assert (n_k, n_j) == (2, 2)
    assert g.branch_prob.shape[:4] == (2, 2, n_a, n_b)
    # every label round-trips through the local action map
    b = _battle()
    for k in range(n_k):
        for a in np.flatnonzero(g.avail1[k]):
            assert public_label(b, "p1", int(g.local_action1[k, a])) == g.labels1[a]
    for j in range(n_j):
        for bcol in np.flatnonzero(g.avail2[j]):
            assert public_label(b, "p2", int(g.local_action2[j, bcol])) == g.labels2[bcol]
    # budget: one commit per available (k, j, a, b)
    want = sum(int(g.avail1[k].sum()) * int(g.avail2[j].sum())
               for k in range(n_k) for j in range(n_j))
    assert g.sims_run == want
    assert not g.failed.any()


def test_branch_probabilities_sum_to_one_on_available_cells(grid_2x2):
    g = grid_2x2
    cell_mass = g.branch_prob.sum(axis=-1)                          # [K, J, A, B]
    available = (g.avail1[:, None, :, None] & g.avail2[None, :, None, :]) & ~g.failed
    np.testing.assert_allclose(cell_mass[available], 1.0, atol=1e-9)
    np.testing.assert_allclose(cell_mass[~available], 0.0)
    assert (g.pruned_mass[available] <= 0.2 + 1e-12).all()
    # exactly one of terminal / leaf per existing branch
    exists = g.branch_prob > 0
    is_terminal = ~np.isnan(g.terminal_value)
    has_leaf = g.leaf_index >= 0
    assert ((is_terminal ^ has_leaf) == exists).all()


def test_identical_candidate_worlds_produce_identical_tables(grid_2x2):
    """Both own-worlds and both opponent-worlds are the same team here, so every derived table
    must be invariant across them: chance enumeration is deterministic given the state, and the
    fresh per-fork rng seeds (settled 21) must not leak into branch structure."""
    g = grid_2x2
    np.testing.assert_array_equal(g.avail1[0], g.avail1[1])
    np.testing.assert_array_equal(g.avail2[0], g.avail2[1])
    np.testing.assert_allclose(g.branch_prob[0], g.branch_prob[1])
    np.testing.assert_allclose(g.branch_prob[:, 0], g.branch_prob[:, 1])
    np.testing.assert_array_equal(np.isnan(g.terminal_value[0]), np.isnan(g.terminal_value[1]))


def test_leaves_are_censored_observations(grid_2x2):
    g = grid_2x2
    assert len(g.leaf_obs) > 0
    assert max(int(g.leaf_index.max()), -1) == len(g.leaf_obs) - 1
    obs = g.leaf_obs[0]
    tokens = np.asarray(obs.opp_revealed["tokens"])
    assert (tokens == -1).any(), \
        "a one-turn leaf must still hide most opponent attributes (fog semantics, 19.12)"


def test_value_assembly_contract(grid_2x2):
    """Phase E's u-tensor assembly: branch-weighted mix of terminal payoffs and stubbed leaf
    values, no NaN on available cells, bounded by the payoff range."""
    g = grid_2x2
    v_leaf = np.full(len(g.leaf_obs), 0.5)
    per_branch = np.where(g.leaf_index >= 0, v_leaf[np.maximum(g.leaf_index, 0)],
                          np.where(np.isnan(g.terminal_value), 0.0, g.terminal_value))
    u = (g.branch_prob * per_branch).sum(axis=-1)
    available = (g.avail1[:, None, :, None] & g.avail2[None, :, None, :]) & ~g.failed
    assert np.isfinite(u[available]).all()
    assert (np.abs(u[available]) <= 1.0 + 1e-9).all()


def test_grid_rejects_one_sided_roots():
    from fog.search.pb_grid import build_pb_grid
    from fog.search.pb_leaves import commit_pair_branches

    weak2 = """Great Tusk @ Heavy-Duty Boots
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
    from fog.env.engine_battle import EngineBattle

    b = EngineBattle(ATTACKER, weak2, seed=0, battle_tag="pbgrid-onesided")
    draco = next(i for i, m in enumerate(b.active("p1").moves)
                 if str(m.id).lower().replace(" ", "") == "dracometeor")
    outcomes, _ = commit_pair_branches(b, "p1", draco,
                                       int(np.flatnonzero(b.action_mask("p2"))[0]))
    leaf = next(o.leaf for o in outcomes if o.leaf is not None
                and tuple(o.leaf.pending_sides()) == ("p2",))
    own, opp = _truth_grid(leaf, 1, 1)
    with pytest.raises(ValueError, match="simultaneous"):
        build_pb_grid(leaf, "p1", own, opp)


def test_grid_deterministic_given_seed():
    from fog.search.pb_grid import build_pb_grid

    tables = []
    for _ in range(2):
        b = _battle(seed=4)
        own, opp = _truth_grid(b, 1, 1)
        g = build_pb_grid(b, "p1", own, opp, seed=9)
        tables.append((g.branch_prob.copy(), g.pruned_mass.copy(), g.sims_run))
    np.testing.assert_array_equal(tables[0][0], tables[1][0])
    np.testing.assert_array_equal(tables[0][1], tables[1][1])
    assert tables[0][2] == tables[1][2]
