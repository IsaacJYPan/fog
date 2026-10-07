import random

import pytest

from integrations.foul_play_sampling import first_success


def test_skipped_sweeps_match_first_success_probabilities():
    rng = random.Random(21)
    draws = [first_success([0.2, 0.4], 0, rng) for _ in range(20000)]
    # P(first=0 | some success) = .2 / (.2 + .8*.4).
    assert abs(draws.count(0) / len(draws) - 0.2 / 0.52) < 0.01
    assert first_success([0.0, 1.0], 0, rng) == 1
    assert first_success([1.0, 1.0], 1, rng) == 1


def test_zero_and_tiny_rates_do_not_require_rejection_iterations():
    rng = random.Random(22)
    assert first_success([0.0, 0.0], 0, rng) is None
    draws = [first_success([1e-30, 2e-30], 0, rng) for _ in range(10000)]
    assert abs(draws.count(0) / len(draws) - 1 / 3) < 0.02
    with pytest.raises(ValueError, match="finite probabilities"):
        first_success([float("nan")], 0, rng)
