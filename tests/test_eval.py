"""Tests for the eval logic: Elo fit, belief/leakage aggregations, closure, gauntlet CIs."""
from __future__ import annotations

import math

import numpy as np

from fog.eval.belief_metrics import (
    accuracy_vs_turn,
    expected_calibration_error,
    pre_reveal_accuracy,
)
from fog.eval.elo import RoundRobin, expected_score
from fog.eval.gauntlet import GauntletResult
from fog.eval.leakage_metrics import (
    choice_lock_reveal_delay,
    move_reveal_entropy,
    readability_curve,
    tera_timing_distribution,
)
from fog.eval.oracle_gap import closure


def test_elo_orders_by_strength():
    # A beats B beats C consistently
    rr = RoundRobin(["A", "B", "C"], games_per_pair=100)
    rr.record("A", "B", 70, 100)
    rr.record("A", "C", 90, 100)
    rr.record("B", "C", 70, 100)
    ratings = rr.fit_ratings()
    assert ratings["A"] > ratings["B"] > ratings["C"]
    # expected score consistent with ordering
    assert expected_score(ratings["A"], ratings["C"]) > 0.5


def test_elo_run_uses_play_fn():
    rr = RoundRobin(["X", "Y"], games_per_pair=50)
    ratings = rr.run(lambda a, b, n: (40, n) if a == "X" else (n // 2, n))
    assert ratings["X"] > ratings["Y"]


def test_closure_formula():
    assert closure(0.6, 0.5, 0.7) == 0.5
    assert math.isnan(closure(0.6, 0.5, 0.5))


def test_accuracy_vs_turn_buckets():
    preds = np.array([1, 1, 0, 1, 1, 1])
    labels = np.array([1, 0, 0, 1, 1, 1])
    turns = np.array([0, 1, 2, 5, 6, 7])
    acc = accuracy_vs_turn(preds, labels, turns, bucket=5)
    assert acc[0] == 2 / 3  # turns 0,1,2: correct on 0 and 2
    assert acc[5] == 1.0    # turns 5,6,7 all correct


def test_pre_reveal_accuracy():
    preds = np.array([1, 1, 1])
    labels = np.array([1, 1, 0])
    turns = np.array([2, 5, 2])
    reveal = np.array([8, 8, 4])  # lead=3: rows with turn<=reveal-3 -> row0(2<=5),row2(2<=1?no)
    acc = pre_reveal_accuracy(preds, labels, turns, reveal, lead=3)
    # eligible: row0 (2<=5 yes), row1 (5<=5 yes), row2 (2<=1 no) -> preds[0,1] vs labels -> 2/2
    assert acc == 1.0


def test_leakage_aggregations():
    curve = readability_curve([(0, 5.0), (1, 4.0), (3, 2.0), (4, 1.0)], bucket=3)
    assert set(curve.keys()) == {0, 3}
    assert curve[0] == 4.5 and curve[3] == 1.5
    hist = tera_timing_distribution([1, 1, 5, 0], max_turn=10)
    assert hist[1] == 2 and hist[5] == 1 and hist[0] == 1
    assert choice_lock_reveal_delay([2, 4]) == 3.0
    assert math.isnan(choice_lock_reveal_delay([]))
    # move reveal entropy sums over reveal positions: identical single-move orders -> 0
    assert move_reveal_entropy([(0,), (0,), (0,)]) == 0.0
    # differing first-revealed move -> positive entropy
    assert move_reveal_entropy([(0, 1), (1, 0)]) > 0.0


def test_ece_and_gauntlet_ci():
    conf = np.array([0.9, 0.8, 0.6])
    correct = np.array([True, True, False])
    assert 0.0 <= expected_calibration_error(conf, correct) <= 1.0
    g = GauntletResult("MaxDamage", 1300, 2000)
    iv = g.winrate
    assert iv.low < iv.point < iv.high
    assert "insufficient" not in g.line()
    assert "insufficient" in GauntletResult("X", 5, 10).line()
