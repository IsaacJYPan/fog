"""Exact-vs-sampled agreement (SEARCH_PLAN §6): aggregation math and the paired-measurement
protocol on real engine games."""
from __future__ import annotations

import pytest

from fog.eval.source_agreement import aggregate_agreement


# ------------------------------------------------------------------------------- aggregation
def test_aggregate_agreement_per_k_and_buckets():
    # (turn, k, agree, exact_states, sampled_states)
    records = [
        (1, 8, True, 4, 8), (3, 8, False, 4, 8), (12, 8, True, 2, 8),
        (2, 16, True, 4, 16),
    ]
    r = aggregate_agreement(records, (8, 16, 32), games=2, failed=0)
    k8 = r["per_k"][8]
    assert k8["decisions"] == 3
    assert k8["agreement"] == pytest.approx(2 / 3)
    assert k8["agreement_by_turn"]["t1-5"] == pytest.approx(0.5)
    assert k8["agreement_by_turn"]["t11-20"] == pytest.approx(1.0)
    assert k8["mean_exact_states"] == pytest.approx(10 / 3)
    assert r["per_k"][16]["agreement"] == 1.0
    assert r["per_k"][32] == {"decisions": 0}  # unmeasured K reported as empty, not invented


# ------------------------------------------------------------------------------- engine-backed
from fog.env.engine_battle import HAVE_ENGINE
from tests.test_joint_search import P_ME, P_ME_VARIANT, P_OPP

engine = pytest.mark.skipif(not HAVE_ENGINE, reason="poke-engine not installed")


@engine
def test_run_agreement_end_to_end():
    """The paired protocol runs on real games: every measured decision produces one record per K,
    agreement is a valid rate, and the exact arm's state counts are bounded by the pool size."""
    from fog.eval.source_agreement import run_agreement
    from fog.models.agent import FogAgent

    agent = FogAgent.from_tables({"model": {"use_posterior_features": True}})
    agent.eval()
    pool = [P_ME, P_ME_VARIANT, P_OPP]

    report = run_agreement(agent, pool, ks=(2, 4), games=2, iters=2, seed=0, max_turns=8)
    assert report["games"] == 2 and report["failed_games"] == 0
    k2, k4 = report["per_k"][2], report["per_k"][4]
    assert k2["decisions"] == k4["decisions"] > 0, "each K measures every reference decision"
    for r in (k2, k4):
        assert 0.0 <= r["agreement"] <= 1.0
        assert r["mean_exact_states"] <= len(pool)
        assert r["mean_sampled_states"] >= 1.0
