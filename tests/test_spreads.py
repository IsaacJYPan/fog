"""Tests for the Gen 9 stat formula and weighted spread clustering."""
from __future__ import annotations

import numpy as np

from fog.constants.build_spreads import (
    MERGE_THRESHOLD,
    cluster_species_spreads,
    nature_modified_stats,
    parse_chaos_spread,
)


def test_stat_formula_known_values():
    base100 = np.array([100, 100, 100, 100, 100, 100])
    # 252 EVs / 31 IV / level 100, neutral: (2*100+31+63) = 294 -> HP 404, others 299
    stats = nature_modified_stats(base100, np.array([252] * 6), "hardy")
    assert stats[0] == 404
    assert all(s == 299 for s in stats[1:])
    # 0 EVs: 231 -> HP 341, others 236
    stats0 = nature_modified_stats(base100, np.zeros(6, dtype=int), "serious")
    assert stats0[0] == 341
    assert all(s == 236 for s in stats0[1:])


def test_stat_formula_nature_and_garchomp_speed():
    # Jolly Garchomp, 252 Spe: base 102 -> (2*102+31+63) = 298 -> +5 = 303 -> *1.1 = 333 (well-known)
    garchomp = np.array([108, 130, 95, 80, 85, 102])
    evs = np.array([0, 0, 0, 0, 0, 252])
    jolly = nature_modified_stats(garchomp, evs, "Jolly")
    assert jolly[5] == 333
    # Jolly drops SpA by 10%: 2*80+31 = 191 -> 191 -> +5 = 196 -> *0.9 = 176
    assert jolly[3] == 176
    # neutral for comparison
    serious = nature_modified_stats(garchomp, evs, "serious")
    assert serious[5] == 303


def test_shedinja_hp():
    shed = np.array([1, 90, 45, 30, 30, 40])
    assert nature_modified_stats(shed, np.zeros(6, dtype=int), "hardy")[0] == 1


def test_parse_chaos_spread():
    nature, evs = parse_chaos_spread("Modest:0/0/0/252/4/252")
    assert nature == "modest"
    assert list(evs) == [0, 0, 0, 252, 4, 252]


def _spread(nature, evs, w):
    return {"nature": nature, "evs": evs, "usage_weight": w}


def test_clustering_separates_and_ranks():
    base = np.array([100, 100, 100, 100, 100, 100])
    observed = [
        _spread("adamant", [0, 252, 0, 0, 4, 252], 0.50),   # physical
        _spread("adamant", [4, 252, 0, 0, 0, 252], 0.10),   # ~same physical (4 EV shuffle)
        _spread("modest", [252, 0, 0, 252, 4, 0], 0.30),    # bulky special
    ]
    reps = cluster_species_spreads(observed, base, max_clusters=16, seed=0)
    assert len(reps) == 2  # the two adamant spreads merge (<4 points apart everywhere)
    assert reps[0]["nature"] == "adamant" and abs(reps[0]["weight"] - 0.6) < 1e-9
    assert reps[1]["nature"] == "modest" and abs(reps[1]["weight"] - 0.3) < 1e-9
    # representative is the highest-weight observed spread, not a centroid
    assert reps[0]["evs"] == [0, 252, 0, 0, 4, 252]
    # representative stats match a direct computation exactly
    direct = nature_modified_stats(base, np.array(reps[0]["evs"]), reps[0]["nature"])
    assert reps[0]["stats"] == [int(v) for v in direct]


def test_clustering_respects_max_clusters_and_reproduces_stats():
    rng = np.random.default_rng(1)
    base = np.array([80, 120, 70, 60, 70, 110])
    natures = ["adamant", "jolly", "impish", "careful", "timid"]
    observed = []
    for i in range(40):
        evs = np.zeros(6, dtype=int)
        evs[rng.integers(0, 6)] = 252
        evs[rng.integers(0, 6)] = 252
        observed.append(_spread(natures[i % 5], list(evs), float(rng.random())))
    reps = cluster_species_spreads(observed, base, max_clusters=8, seed=0)
    assert 1 <= len(reps) <= 8
    for r in reps:
        direct = nature_modified_stats(base, np.array(r["evs"]), r["nature"])
        assert np.abs(np.array(r["stats"]) - direct).max() < MERGE_THRESHOLD


def test_clustering_deterministic():
    base = np.array([90, 90, 90, 90, 90, 90])
    observed = [
        _spread("adamant", [0, 252, 0, 0, 4, 252], 0.4),
        _spread("timid", [0, 0, 0, 252, 4, 252], 0.4),
        _spread("impish", [252, 4, 252, 0, 0, 0], 0.2),
    ]
    a = cluster_species_spreads(observed, base, seed=3)
    b = cluster_species_spreads(observed, base, seed=3)
    assert a == b
