import numpy as np
import pytest

from fog.eval.stats import BlockSPRT

N_MATCHES = 2000


def _simulate(p, seed, n_matches=N_MATCHES, max_blocks=1000):
    rng = np.random.default_rng(seed)
    scores = rng.binomial(4, p, size=(n_matches, max_blocks)) / 4.0
    return [BlockSPRT.evaluate(row.tolist(), max_blocks=max_blocks) for row in scores]


def _rate(results, decision):
    return sum(r.decision == decision for r in results) / len(results)


def test_type_one_error_at_p0():
    res = _simulate(0.50, seed=1)
    assert _rate(res, "accept_h1") <= 0.07


def test_type_two_error_at_p1():
    res = _simulate(0.55, seed=2)
    assert _rate(res, "accept_h0") <= 0.07


def test_strong_candidate_accepts_quickly():
    res = _simulate(0.60, seed=3, n_matches=500)
    assert _rate(res, "accept_h1") >= 0.98
    assert np.mean([r.blocks for r in res]) < 500


def test_no_decision_before_min_blocks():
    t = BlockSPRT(min_blocks=100)
    for _ in range(99):
        assert t.update(1.0).decision == "continue"
    assert t.update(1.0).decision == "accept_h1"


def test_constant_scores_do_not_crash():
    d = BlockSPRT.evaluate([0.5] * 1000)
    assert d.decision in {"accept_h0", "accept_h1", "max_reached"}
    d = BlockSPRT.evaluate([0.0] * 200)
    assert d.decision == "accept_h0" and d.blocks == 100


def test_max_reached():
    d = BlockSPRT.evaluate([0.5, 0.55] * 50, min_blocks=10, max_blocks=10, alpha=0.01, beta=0.01)
    assert d.decision in {"max_reached", "accept_h0", "accept_h1"} and d.blocks == 10


@pytest.mark.parametrize("kw", [dict(p0=0.6, p1=0.5), dict(p0=0.0), dict(p1=1.0),
                                dict(alpha=0.5), dict(beta=0.0), dict(min_blocks=10, max_blocks=5)])
def test_invalid_params(kw):
    with pytest.raises(ValueError):
        BlockSPRT(**kw)


def test_invalid_score():
    with pytest.raises(ValueError):
        BlockSPRT().update(1.5)
