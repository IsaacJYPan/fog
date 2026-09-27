"""Tests for evaluation statistics."""
from __future__ import annotations

import numpy as np

from fog.eval.belief_metrics import expected_calibration_error
from fog.eval.oracle_gap import closure
from fog.eval.stats import wilson_interval


def test_wilson_center_and_bounds():
    iv = wilson_interval(500, 1000)
    assert abs(iv.point - 0.5) < 1e-9
    assert 0.46 < iv.low < 0.5 < iv.high < 0.54
    # narrower with more games
    wide = wilson_interval(50, 100)
    assert (wide.high - wide.low) > (iv.high - iv.low)


def test_wilson_edge_cases():
    assert wilson_interval(0, 0).point != wilson_interval(0, 0).point  # nan
    perfect = wilson_interval(100, 100)
    assert perfect.high <= 1.0 + 1e-9 and perfect.low < 1.0


def test_ece_perfectly_calibrated():
    # confidence == accuracy in every bin -> ECE ~ 0
    conf = np.array([0.9] * 100)
    correct = np.array([True] * 90 + [False] * 10)
    assert expected_calibration_error(conf, correct, n_bins=10) < 0.02


def test_closure_fraction():
    assert abs(closure(0.6, 0.5, 0.7) - 0.5) < 1e-9
    assert closure(0.5, 0.5, 0.5) != closure(0.5, 0.5, 0.5)  # nan on zero headroom
