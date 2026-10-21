"""Scalar-head value assembly for public-belief search.

Check branch weighting, posterior-dependent values, static caching for blind heads,
failed-cell fallback, and integration with the regret solver on an engine grid.
"""
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


@pytest.fixture(scope="module")
def grid_k2():
    """K=2 duplicate own worlds x J=1 — enough for own-margin features to vary with sigma1."""
    from fog.env.engine_battle import EngineBattle
    from fog.search.pb_grid import build_pb_grid

    b = EngineBattle(ATTACKER, DEFENDER, seed=0, battle_tag="pbval-test")
    own = np.stack([b.truth_tokens("p1")] * 2)
    opp = np.stack([b.truth_tokens("p2")] * 1)
    return build_pb_grid(b, "p1", own, opp, seed=2)


def _uniform_strategies(grid):
    s1 = grid.avail1 / grid.avail1.sum(axis=1, keepdims=True)
    s2 = grid.avail2 / grid.avail2.sum(axis=1, keepdims=True)
    return s1.astype(np.float64), s2.astype(np.float64)


def _rho(grid):
    n_k, n_j = grid.avail1.shape[0], grid.avail2.shape[0]
    return np.full((n_k, n_j), 1.0 / (n_k * n_j))


def test_blind_agent_static_u_and_single_head_pass(grid_k2):
    from fog.models.agent import FogAgent
    from fog.search.pb_values import make_grid_values

    agent = FogAgent.from_tables({})                                  # posterior_dim == 0
    values_fn, stats = make_grid_values(agent, grid_k2, _rho(grid_k2))
    assert stats.leaves == len(grid_k2.leaf_obs) > 0
    assert stats.failed_cells == 0
    s1, s2 = _uniform_strategies(grid_k2)
    u1 = values_fn(s1, s2)
    u2 = values_fn(np.roll(s1, 1, axis=0), s2)            # different strategies, same cached u
    assert stats.head_passes == 1, "a posterior-blind head cannot react: compute once, reuse"
    assert u2 is u1
    available = (grid_k2.avail1[:, None, :, None] & grid_k2.avail2[None, :, None, :])
    assert np.isfinite(u1[available]).all()
    assert (np.abs(u1[available]) <= 1.0 + 1e-9).all()    # tanh values x branch mix stay bounded
    # duplicate own worlds under identical features -> identical value rows
    np.testing.assert_allclose(u1[0], u1[1], atol=1e-6)


def test_aware_agent_values_move_with_the_strategies(grid_k2):
    """The coupling channel end-to-end: same cached leaves, different current strategies,
    different induced posterior, different u."""
    import torch

    from fog.models.agent import FogAgent
    from fog.search.pb_values import make_grid_values

    torch.manual_seed(0)
    agent = FogAgent.from_tables({"model": {"use_posterior_features": True}})
    values_fn, stats = make_grid_values(agent, grid_k2, _rho(grid_k2))
    s1, s2 = _uniform_strategies(grid_k2)
    u_uniform = values_fn(s1, s2)
    # a separating strategy: own-world 0 plays the first label, own-world 1 the second
    sep = np.zeros_like(s1)
    cols = [int(c) for c in np.flatnonzero(grid_k2.avail1[0])[:2]]
    sep[0, cols[0]] = 1.0
    sep[1, cols[1]] = 1.0
    u_sep = values_fn(sep, s2)
    assert stats.head_passes == 2, "an aware head re-runs per call"
    assert not np.allclose(u_uniform, u_sep), \
        "posterior features must couple leaf values to the current strategies"


def test_failed_cells_take_the_fallback(grid_k2):
    import copy

    from fog.models.agent import FogAgent
    from fog.search.pb_values import make_grid_values

    grid = copy.copy(grid_k2)
    grid.failed = grid_k2.failed.copy()
    a = int(np.flatnonzero(grid.avail1[0])[0])
    b = int(np.flatnonzero(grid.avail2[0])[0])
    grid.failed[0, 0, a, b] = True
    values_fn, stats = make_grid_values(FogAgent.from_tables({}), grid, _rho(grid), fallback_value=0.123)
    u = values_fn(*_uniform_strategies(grid))
    assert u[0, 0, a, b] == pytest.approx(0.123)
    assert stats.failed_cells == 1


def test_solves_end_to_end_through_the_two_sided_solver(grid_k2):
    """The simulation-to-solver chain on a real engine grid: build values, run the solver, get valid
    averaged strategies for BOTH sides and a finite root value."""
    from fog.models.agent import FogAgent
    from fog.search.pb_values import make_grid_values
    from fog.search.two_sided import two_sided_regret_solve

    values_fn, _ = make_grid_values(FogAgent.from_tables({}), grid_k2, _rho(grid_k2))
    sol = two_sided_regret_solve(values_fn, _rho(grid_k2), grid_k2.avail1, grid_k2.avail2,
                                 iters=96, min_iters=32, check_every=16)
    np.testing.assert_allclose(sol.sigma1.sum(axis=1), 1.0)
    np.testing.assert_allclose(sol.sigma2.sum(axis=1), 1.0)
    assert (sol.sigma1[~grid_k2.avail1] == 0).all()
    assert (sol.sigma2[~grid_k2.avail2] == 0).all()
    assert np.isfinite(sol.root_value)
