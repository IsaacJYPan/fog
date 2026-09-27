"""Drift gate mechanism tests (proposal §9.2 / §15.3; fog/eval/pb_value_drift.py).

Engine-free, same hand-built grid + fake backbone as tests/test_pb_value_fn.py. With a random net
these are MECHANISM tests: they prove the metric responds to real table drift and that the exact arm
is self-consistent — the gate-quality NUMBERS need a trained checkpoint (documented in the module).
"""
import numpy as np
import torch

from fog.eval.pb_value_drift import format_report, reweight_staleness_curve, solution_gap
from fog.models.pb_value import _POS_FAMILY, N_ATTRS, PBValueNet

SIZES = {
    "species": 32, "item": 12, "ability": 16, "move": 40,
    "nature": 8, "ev": 8, "tera": 6,
}
STATE, DYN, D = 24, 20, 16


class _FakeGrid:
    def __init__(self, branch_prob, terminal_value, leaf_index, leaf_obs, failed):
        self.branch_prob, self.terminal_value = branch_prob, terminal_value
        self.leaf_index, self.leaf_obs, self.failed = leaf_index, leaf_obs, failed
        self.leaf_public_keys = [""] * len(leaf_obs)
        for k, j, a, b, m in np.argwhere(leaf_index >= 0):
            self.leaf_public_keys[leaf_index[k, j, a, b, m]] = f"{a}:{b}"


class _FakeAgent:
    def __init__(self, s_all, c_all):
        self.s_all, self.c_all = s_all, c_all

    def eval(self):
        return self

    def head_inputs(self, chunk):
        ids = chunk["ids"]
        return self.s_all[ids], self.c_all[ids]


def _rand_teams(n, gen):
    cols = [torch.randint(0, SIZES[_POS_FAMILY[p]], (n, 1), generator=gen) for p in range(N_ATTRS)]
    return torch.cat(cols, dim=1).reshape(n, 6, 15)


def _build(monkeypatch):
    import fog.env.engine_runner as er
    monkeypatch.setattr(er, "_device", lambda _a: torch.device("cpu"), raising=False)
    monkeypatch.setattr(er, "_stack_obs",
                        lambda obs, device: {"ids": torch.tensor(obs, dtype=torch.long)},
                        raising=False)
    K = J = A = B = 2
    M = 2
    gen = torch.Generator().manual_seed(3)
    branch_prob = np.zeros((K, J, A, B, M))
    terminal_value = np.full((K, J, A, B, M), np.nan)
    leaf_index = np.full((K, J, A, B, M), -1, dtype=np.int64)
    leaf_obs = []
    for k in range(K):
        for j in range(J):
            for a in range(A):
                for b in range(B):
                    branch_prob[k, j, a, b, 0] = 0.4
                    branch_prob[k, j, a, b, 1] = 0.6
                    # type-SEPARATING payoff: own type k prefers action a==k, opponent type j
                    # prefers b==j. Separated strategies make the action reveal the type, so the
                    # action-conditioned posterior actually MOVES across iterations — the precondition
                    # for any table drift to exist. A symmetric (type-blind) game would sit at zero
                    # drift trivially, which measures nothing.
                    terminal_value[k, j, a, b, 0] = (1.5 * (1.0 if a == k else -1.0)
                                                     + 0.5 * (1.0 if b != j else -1.0))
                    leaf_index[k, j, a, b, 1] = len(leaf_obs)
                    leaf_obs.append(len(leaf_obs))
    grid = _FakeGrid(branch_prob, terminal_value, leaf_index, leaf_obs,
                     np.zeros((K, J, A, B), dtype=bool))
    n = len(leaf_obs)
    agent = _FakeAgent(torch.randn(n, STATE, generator=gen), torch.randn(n, DYN, generator=gen))
    net = PBValueNet(state_dim=STATE, dyn_dim=DYN, d=D, n_rounds=2, sizes=SIZES).eval()
    own = _rand_teams(K, gen).numpy()
    opp = _rand_teams(J, gen).numpy()
    rho = np.array([[0.4, 0.1], [0.2, 0.3]])
    avail = np.ones((2, 2), dtype=bool)
    return agent, net, grid, rho, own, opp, avail


def test_staleness_curve_responds_to_drift(monkeypatch):
    agent, net, grid, rho, own, opp, avail = _build(monkeypatch)
    curve = reweight_staleness_curve(agent, net, grid, rho, own, opp, avail, avail,
                                     staleness=(1, 2, 4, 8), iters=48)
    assert set(curve) == {1, 2, 4, 8}
    for r in curve.values():
        assert np.isfinite(r["mean"]) and np.isfinite(r["max"])
        assert r["mean"] >= 0.0 and r["max"] >= r["mean"]
    # the belief genuinely moves the table across iterations, so staleness produces real error
    # (a value blind to rho would give an all-zero curve — the §5.1 degenerate case).
    assert max(r["max"] for r in curve.values()) > 1e-6
    # a more stale table is (weakly) worse: the mean error grows with staleness.
    means = [curve[s]["mean"] for s in (1, 2, 4, 8)]
    assert means == sorted(means)


def test_solution_gap_reference_is_self_consistent(monkeypatch):
    agent, net, grid, rho, own, opp, avail = _build(monkeypatch)
    gap = solution_gap(agent, net, grid, rho, own, opp, avail, avail, Ns=(1, 2, 4, 16), iters=48)
    # N=1 is the reference compared to itself: exactly zero, deterministically.
    assert gap[1]["sigma1_tv"] == 0.0 and gap[1]["sigma2_tv"] == 0.0
    assert gap[1]["root_value_gap"] == 0.0
    for r in gap.values():
        assert np.isfinite(r["sigma1_tv"]) and r["sigma1_tv"] >= 0.0
        assert r["iterations"] == 48                                  # tol=0 -> full length
    # fewer refreshes as N grows (the whole point: N=1 refreshes every iter, N=16 far fewer)
    assert gap[1]["net_refreshes"] == 48
    assert gap[16]["net_refreshes"] < gap[2]["net_refreshes"] < gap[1]["net_refreshes"]
    assert gap[16]["net_refreshes"] == 3                             # calls 0,16,32


def test_report_formats(monkeypatch):
    agent, net, grid, rho, own, opp, avail = _build(monkeypatch)
    curve = reweight_staleness_curve(agent, net, grid, rho, own, opp, avail, avail,
                                     staleness=(1, 4), iters=32)
    gap = solution_gap(agent, net, grid, rho, own, opp, avail, avail, Ns=(1, 4), iters=32)
    text = format_report(curve, gap)
    assert "staleness" in text and "N=" in text and "sigma1_tv" in text
