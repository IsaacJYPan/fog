"""Phase-4 distillation (SEARCH_PLAN §7): the KL core, the deployment/PBS input asymmetry, both
scopes actually move toward search play on held-out games, and belief nets never train here."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from fog.models.agent import FogAgent
from fog.training.distill import distill, kl_by_turn, kl_search_policy

AWARE_CFG = {"model": {"use_posterior_features": True}}


# ---------------------------------------------------------------------------------- pure math
def test_kl_search_policy_hand_computed():
    p = torch.tensor([[0.5, 0.5, 0.0]])
    q = torch.log(torch.tensor([[0.25, 0.25, 0.5]]))
    # KL = 0.5 log(0.5/0.25) * 2 = log 2
    kl = kl_search_policy(p, q)
    assert kl.item() == pytest.approx(float(np.log(2)), rel=1e-5)
    # identical distributions -> 0; zero-mass columns contribute nothing even at -inf log-prob
    q2 = torch.log(torch.tensor([[0.5, 0.5, 1e-30]])).clamp(min=-1e9)
    assert kl_search_policy(p, q2).item() == pytest.approx(0.0, abs=1e-5)


def test_kl_search_policy_grad_finite_at_masked_actions():
    """-inf log-probs on actions search never plays must not NaN the gradient (the ppo entropy
    lesson: clamp before multiplying, never rely on 0 * -inf)."""
    logits = torch.tensor([[1.0, 0.0, -2.0]], requires_grad=True)
    mask = torch.tensor([[True, True, False]])
    log_q = torch.log_softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)
    p = torch.tensor([[0.7, 0.3, 0.0]])
    kl_search_policy(p, log_q).sum().backward()
    assert torch.isfinite(logits.grad[:, :2]).all()
    assert not torch.isnan(logits.grad).any()


def test_kl_by_turn_buckets():
    kl = np.array([1.0, 2.0, 3.0, 4.0])
    turns = np.array([1, 5, 6, 30])
    out = kl_by_turn(kl, turns)
    assert out["t1-5"] == pytest.approx(1.5)
    assert out["t6-10"] == pytest.approx(3.0)
    assert out["t21+"] == pytest.approx(4.0)
    assert "t11-20" not in out  # empty buckets are omitted, not zero-filled


# --------------------------------------------------------------------------------- distill core
def _fake_log(agent, n_games=6, rows_per_game=8, seed=0):
    """A synthetic decision log in the posterior_log row format. The search target is a FUNCTION
    OF THE OBSERVATION (a strong ±3 signal in field[0] picks action 0 vs 1), so matching it on
    HELD-OUT games requires learning, not memorizing — random per-row targets would make holdout
    KL improvement impossible by construction and the improvement asserts vacuous or flaky."""
    from fog.encoding.dims import DIMS
    from fog.encoding.entities import ENTITY_RAW_DIM, N_ID_FIELDS

    rng = np.random.default_rng(seed)
    n = n_games * rows_per_game
    n_species = agent.sizes["n_species"]
    legal = np.zeros((n, 14), dtype=bool)
    legal[:, [0, 1, 8]] = True
    sig = rng.integers(0, 2, n)
    target = np.where(sig == 1, 0, 1)
    field = rng.normal(size=(n, DIMS.field_dim)).astype(np.float32)
    field[:, 0] = np.where(sig == 1, 3.0, -3.0)
    dist = np.zeros((n, 14), dtype=np.float32)
    dist[np.arange(n), target] = 0.9
    for a in (0, 1, 8):
        dist[:, a] += 0.1 / 3
    tokens = np.full((n, 6, 15), -1, dtype=np.int64)
    tokens[:, :, 0] = rng.integers(0, n_species, (n, 6))
    return {
        "entities": rng.normal(size=(n, 12, ENTITY_RAW_DIM)).astype(np.float32),
        "entity_ids": np.zeros((n, 12, N_ID_FIELDS), dtype=np.int64),
        "field": field,
        "history": rng.normal(size=(n, 4, DIMS.turn_dim)).astype(np.float32),
        "active_idx": np.tile([0, 6], (n, 1)).astype(np.int64),
        "legality_mask": legal,
        "opp_tokens": tokens,
        "posterior_features": rng.random((n, 4)).astype(np.float32),
        "search_dist": dist,
        "action": target.astype(np.int64),
        "turn": rng.integers(1, 30, n).astype(np.int64),
        "outcome": rng.choice([-1.0, 1.0], n).astype(np.float32),
        "game": np.repeat(np.arange(n_games), rows_per_game).astype(np.int64),
    }


@pytest.fixture(scope="module")
def aware_agent_factory():
    def make():
        torch.manual_seed(11)
        a = FogAgent.from_tables(AWARE_CFG)
        a.eval()
        return a
    return make


@pytest.mark.parametrize("scope", ["heads", "full"])
def test_distill_improves_holdout_kl(scope, aware_agent_factory):
    """Both scopes must move the policy toward search play on HELD-OUT games (targets here are
    learnable from the observation, so a real optimizer step shows up as lower KL)."""
    agent = aware_agent_factory()
    data = _fake_log(agent)
    report = distill(agent, data, scope=scope, holdout_frac=0.34,
                     epochs=6, lr=1e-3, batch_size=16, seed=0)
    assert report["improved"], (
        f"{scope}: held-out KL {report['holdout_before']['kl']:.4f} -> "
        f"{report['holdout_after']['kl']:.4f}"
    )
    assert report["holdout_after"]["top1_agreement"] >= report["holdout_before"]["top1_agreement"]
    assert set(report["holdout_after"]["kl_by_turn"]) == set(report["holdout_before"]["kl_by_turn"])


def test_distill_scope_freezes_correctly(aware_agent_factory):
    """heads scope: trunk and belief nets byte-identical after training. full scope: backbone
    moves, belief/self-belief still frozen (their retraining is the §7 consistency step)."""
    def snapshot(agent, module):
        return [p.detach().clone() for p in getattr(agent, module).parameters()]

    for scope, trunk_moves in (("heads", False), ("full", True)):
        agent = aware_agent_factory()
        data = _fake_log(agent)
        before = {m: snapshot(agent, m) for m in ("backbone", "belief", "self_belief")}
        distill(agent, data, scope=scope, holdout_frac=0.34, epochs=2, lr=1e-3,
                batch_size=16, seed=0)
        moved = {m: any(not torch.equal(a, b) for a, b in zip(before[m], snapshot(agent, m)))
                 for m in before}
        assert moved["backbone"] == trunk_moves, f"{scope}: backbone frozen-ness wrong"
        assert not moved["belief"] and not moved["self_belief"], (
            f"{scope}: belief nets must never train in distill (retrained separately, §7)"
        )


def test_distill_rejects_blind_and_oracle_agents():
    blind = FogAgent({})
    with pytest.raises(ValueError, match="posterior-aware"):
        distill(blind, {"outcome": np.zeros(4, dtype=np.float32),
                        "game": np.zeros(4, dtype=np.int64)})
    with pytest.raises(ValueError, match="scope"):
        distill(blind, {}, scope="everything")
    oracle = FogAgent({"model": {"oracle": "set", "use_posterior_features": True}})
    with pytest.raises(ValueError, match="non-oracle"):
        distill(oracle, {})


def test_distill_deterministic(aware_agent_factory):
    def run():
        agent = aware_agent_factory()
        r = distill(agent, _fake_log(agent), scope="heads", holdout_frac=0.34,
                    epochs=2, lr=1e-3, batch_size=16, seed=3)
        return r["holdout_after"]["kl"], r["train_loss_last"]

    assert run() == run()
