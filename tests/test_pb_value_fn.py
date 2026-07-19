"""Phase-G value bridge: PBValueNet -> solver `values_fn` (proposal §9.2, §10 steps 6-7).

Engine-free: a hand-built PBGrid stand-in plus a fake backbone (`head_inputs` returns precomputed
per-leaf s_board/c_dyn), with `_stack_obs`/`_device` monkeypatched so no battle is needed. The real
PBValueNet does the actual work, so the test checks the BRIDGE's assembly — leaf->(a,b) indexing,
the per-leaf posterior re-weighting, branch/terminal/failed handling, and the refresh cadence — not
the network internals (those are tests/test_pb_value.py).
"""
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from fog.models.pb_value import PBValueNet, _POS_FAMILY, N_ATTRS
from fog.search.two_sided import pair_posteriors

SIZES = {"species": 32, "item": 12, "ability": 16, "move": 40, "spread": 8, "tera": 6}
STATE, DYN, D = 24, 20, 16


@dataclass
class FakeGrid:
    branch_prob: np.ndarray
    terminal_value: np.ndarray
    leaf_index: np.ndarray
    leaf_obs: list
    failed: np.ndarray


class FakeAgent:
    """`head_inputs(chunk)` returns the precomputed s_board/c_dyn rows for the chunk's leaf ids."""
    def __init__(self, s_all, c_all):
        self.s_all, self.c_all = s_all, c_all

    def eval(self):
        return self

    def head_inputs(self, chunk):
        ids = chunk["ids"]
        return self.s_all[ids], self.c_all[ids]


def _rand_teams(n, gen):
    cols = [torch.randint(0, SIZES[_POS_FAMILY[p]], (n, 1), generator=gen) for p in range(N_ATTRS)]
    return torch.cat(cols, dim=1).reshape(n, 6, 9)


def _build(monkeypatch):
    import fog.env.engine_runner as er
    monkeypatch.setattr(er, "_device", lambda _agent: torch.device("cpu"), raising=False)
    monkeypatch.setattr(er, "_stack_obs", lambda obs, device: {"ids": torch.tensor(obs, dtype=torch.long)},
                        raising=False)

    K, J, A, B, M = 2, 2, 2, 2, 2
    gen = torch.Generator().manual_seed(3)
    branch_prob = np.zeros((K, J, A, B, M))
    terminal_value = np.full((K, J, A, B, M), np.nan)
    leaf_index = np.full((K, J, A, B, M), -1, dtype=np.int64)
    leaf_obs = []
    rng = np.random.default_rng(0)
    for k in range(K):
        for j in range(J):
            for a in range(A):
                for b in range(B):
                    branch_prob[k, j, a, b, 0] = 0.4                  # terminal branch
                    branch_prob[k, j, a, b, 1] = 0.6                  # leaf branch
                    terminal_value[k, j, a, b, 0] = rng.uniform(-1, 1)
                    leaf_index[k, j, a, b, 1] = len(leaf_obs)
                    leaf_obs.append(len(leaf_obs))                    # leaf id == its own obs
    failed = np.zeros((K, J, A, B), dtype=bool)
    failed[1, 0, 1, 1] = True                                        # exercise the fallback path
    grid = FakeGrid(branch_prob, terminal_value, leaf_index, leaf_obs, failed)

    n_leaves = len(leaf_obs)
    s_all = torch.randn(n_leaves, STATE, generator=gen)
    c_all = torch.randn(n_leaves, DYN, generator=gen)
    agent = FakeAgent(s_all, c_all)
    net = PBValueNet(state_dim=STATE, dyn_dim=DYN, d=D, n_rounds=2, sizes=SIZES).eval()
    own = _rand_teams(K, gen)
    opp = _rand_teams(J, gen)
    rho = np.array([[0.4, 0.1], [0.2, 0.3]])
    return grid, agent, net, own, opp, rho, s_all, c_all


def _independent_u(grid, net, own, opp, rho, s_all, c_all, sigma1, sigma2, fallback):
    """Recompute u by an independent path (the bridge's own math, written out here)."""
    post = pair_posteriors(rho, sigma1, sigma2)                       # [A, B, K, J]
    pos = np.argwhere(grid.leaf_index >= 0)
    leaf_ab = np.zeros((len(grid.leaf_obs), 2), dtype=np.int64)
    leaf_ab[grid.leaf_index[tuple(pos.T)]] = pos[:, [2, 3]]
    post_leaf = post[leaf_ab[:, 0], leaf_ab[:, 1]]                    # [L, K, J]
    with torch.no_grad():
        V = net(s_all, c_all, own, opp, torch.as_tensor(post_leaf, dtype=torch.float32)).numpy()
    scalar = (post_leaf * V).sum(axis=(1, 2))                        # [L]
    fill = np.where(np.isnan(grid.terminal_value), 0.0, grid.terminal_value)
    per_branch = np.where(grid.leaf_index >= 0, scalar[np.maximum(grid.leaf_index, 0)], fill)
    u = (grid.branch_prob * per_branch).sum(axis=-1)
    u[grid.failed] = fallback
    return u


def test_reference_mode_matches_independent_recompute(monkeypatch):
    """recompute_every=1 (the exact single-query reference) equals the independent recomputation, at
    two different strategy points — i.e. the bridge tracks the posterior correctly."""
    from fog.search.pb_value_fn import make_pb_value_grid_values
    grid, agent, net, own, opp, rho, s_all, c_all = _build(monkeypatch)
    values_fn, stats = make_pb_value_grid_values(
        agent, net, grid, rho, own.numpy(), opp.numpy(), recompute_every=1, fallback_value=0.25)

    for seed in (1, 2):
        g = torch.Generator().manual_seed(seed)
        s1 = torch.rand(2, 2, generator=g).numpy(); s1 /= s1.sum(1, keepdims=True)
        s2 = torch.rand(2, 2, generator=g).numpy(); s2 /= s2.sum(1, keepdims=True)
        u = values_fn(s1, s2)
        u_exp = _independent_u(grid, net, own, opp, rho, s_all, c_all, s1, s2, 0.25)
        assert u.shape == (2, 2, 2, 2)
        assert np.allclose(u, u_exp, atol=1e-5)
    assert stats.failed_cells == 1


def test_reweighting_tracks_posterior_between_refreshes(monkeypatch):
    """With a large N (only the first call refreshes the table), u STILL changes as the strategies
    move, because the cheap re-weighting recomputes each leaf's posterior every call — this is the
    coupling channel that survives without a network pass."""
    from fog.search.pb_value_fn import make_pb_value_grid_values
    grid, agent, net, own, opp, rho, *_ = _build(monkeypatch)
    values_fn, stats = make_pb_value_grid_values(
        agent, net, grid, rho, own.numpy(), opp.numpy(), recompute_every=1000)

    g = torch.Generator().manual_seed(7)
    s1a = torch.rand(2, 2, generator=g).numpy(); s1a /= s1a.sum(1, keepdims=True)
    s2a = torch.rand(2, 2, generator=g).numpy(); s2a /= s2a.sum(1, keepdims=True)
    s1b = torch.rand(2, 2, generator=g).numpy(); s1b /= s1b.sum(1, keepdims=True)
    s2b = torch.rand(2, 2, generator=g).numpy(); s2b /= s2b.sum(1, keepdims=True)

    u_a = values_fn(s1a, s2a)
    u_b = values_fn(s1b, s2b)
    assert stats.net_refreshes == 1                                  # only the first call refreshed
    assert stats.reweightings == 2
    assert not np.allclose(u_a, u_b)                                 # yet the values moved


def test_refresh_cadence_counts(monkeypatch):
    """One network pass per `recompute_every` calls (calls 0, N, 2N, ...)."""
    from fog.search.pb_value_fn import make_pb_value_grid_values
    grid, agent, net, own, opp, rho, *_ = _build(monkeypatch)
    values_fn, stats = make_pb_value_grid_values(
        agent, net, grid, rho, own.numpy(), opp.numpy(), recompute_every=4)
    s = np.full((2, 2), 0.5)
    for _ in range(9):
        values_fn(s, s)
    assert stats.reweightings == 9
    assert stats.net_refreshes == 3                                  # calls 0, 4, 8


def test_all_terminal_grid_has_no_leaves(monkeypatch):
    """A grid where every branch is terminal (no leaves) must not crash the value bridge — the leaf
    lookup is skipped and u is just the terminal payoffs. Regression for the empty-leaf IndexError
    that also lived in the interim pb_values.make_grid_values."""
    from fog.search.pb_value_fn import make_pb_value_grid_values
    grid, agent, net, own, opp, rho, *_ = _build(monkeypatch)
    K, J, A, B, M = grid.branch_prob.shape
    # rebuild as all-terminal: every branch terminal, no leaf observations
    grid.leaf_index = np.full((K, J, A, B, M), -1, dtype=np.int64)
    grid.terminal_value = np.where(grid.branch_prob > 0, 0.3, np.nan)
    grid.leaf_obs = []
    values_fn, stats = make_pb_value_grid_values(
        agent, net, grid, rho, own.numpy(), opp.numpy(), recompute_every=1)
    u = values_fn(np.full((2, 2), 0.5), np.full((2, 2), 0.5))
    assert u.shape == (2, 2, 2, 2)
    # non-failed cells: branch-prob-weighted terminal value (branches sum to 1.0 here); the one
    # failed cell falls back to fallback_value (0.0 by default).
    assert np.allclose(u[~grid.failed], 0.3)
    assert u[grid.failed] == 0.0
    assert stats.leaves == 0 and stats.net_refreshes == 0    # no leaves -> no network passes


def test_drives_the_two_sided_solver(monkeypatch):
    """The whole Phase-G value path plugs into two_sided_regret_solve: the solver calls values_fn
    every iteration, the network-backed u drives regret updates, and it returns a valid solution.
    This is the end-to-end wiring the interim path was a placeholder for."""
    from fog.search.pb_value_fn import make_pb_value_grid_values
    from fog.search.two_sided import two_sided_regret_solve
    grid, agent, net, own, opp, rho, *_ = _build(monkeypatch)
    values_fn, stats = make_pb_value_grid_values(
        agent, net, grid, rho, own.numpy(), opp.numpy(), recompute_every=8)
    avail1 = np.ones((2, 2), dtype=bool)
    avail2 = np.ones((2, 2), dtype=bool)
    sol = two_sided_regret_solve(values_fn, rho, avail1, avail2, iters=64, min_iters=32)
    assert sol.sigma1.shape == (2, 2) and sol.sigma2.shape == (2, 2)
    assert np.allclose(sol.sigma1.sum(1), 1.0) and np.allclose(sol.sigma2.sum(1), 1.0)
    assert stats.net_refreshes >= 1 and stats.reweightings == sol.iterations


def test_exact_and_approx_agree_at_first_call(monkeypatch):
    """N=1 and N=8 must give the identical u on the FIRST call (both refresh at call 0); they only
    diverge later, which is the drift the small-pool gate measures."""
    from fog.search.pb_value_fn import make_pb_value_grid_values
    grid, agent, net, own, opp, rho, *_ = _build(monkeypatch)
    exact, _ = make_pb_value_grid_values(agent, net, grid, rho, own.numpy(), opp.numpy(),
                                         recompute_every=1)
    approx, _ = make_pb_value_grid_values(agent, net, grid, rho, own.numpy(), opp.numpy(),
                                          recompute_every=8)
    g = torch.Generator().manual_seed(5)
    s1 = torch.rand(2, 2, generator=g).numpy(); s1 /= s1.sum(1, keepdims=True)
    s2 = torch.rand(2, 2, generator=g).numpy(); s2 /= s2.sum(1, keepdims=True)
    assert np.allclose(exact(s1, s2), approx(s1, s2), atol=1e-6)
