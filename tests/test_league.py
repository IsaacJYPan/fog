import random
from collections import Counter
from pathlib import Path

import pytest

from fog.training.league import League


def test_league_rejects_invalid_fraction_sum():
    with pytest.raises(ValueError, match="must equal 1"):
        League(latest_frac=0.5, past_frac=0.3, scripted_frac=0.3)


def test_league_without_past_checkpoint_uses_scripted_fallback():
    league = League(scripted=("MaxDamage", "SimpleHeuristics"))
    rng = random.Random(7)
    assert {
        league.sample_opponent(rng) for _ in range(100)
    } == {"MaxDamage", "SimpleHeuristics"}


def test_targeted_scripted_weights_preserve_frozen_selfplay():
    checkpoint = Path("frozen.pt")
    league = League(
        latest_frac=0.2,
        past_frac=0.2,
        scripted_frac=0.6,
        latest_ckpt=checkpoint,
        past_ckpts=[checkpoint],
        scripted=("SimpleHeuristics",) * 3 + ("MaxDamage",),
    )
    rng = random.Random(19)
    counts = Counter(league.sample_opponent(rng) for _ in range(20_000))

    assert 0.42 < counts["SimpleHeuristics"] / 20_000 < 0.48
    assert 0.12 < counts["MaxDamage"] / 20_000 < 0.18
    assert 0.37 < counts[checkpoint] / 20_000 < 0.43


def test_warmup_opponent_precedes_configured_mixture():
    checkpoint = Path("starting-model.pt")
    league = League(
        latest_frac=0.0,
        past_frac=0.5,
        scripted_frac=0.5,
        latest_ckpt=checkpoint,
        past_ckpts=[checkpoint],
        scripted=("SimpleHeuristics",),
        warmup_opponent="SimpleHeuristics",
        warmup_decisions=50_000,
    )
    rng = random.Random(23)

    assert {
        league.sample_opponent(rng, decisions_seen=n)
        for n in (0, 10_000, 49_999)
    } == {"SimpleHeuristics"}

    counts = Counter(
        league.sample_opponent(rng, decisions_seen=50_000)
        for _ in range(20_000)
    )
    assert 0.47 < counts["SimpleHeuristics"] / 20_000 < 0.53
    assert 0.47 < counts[checkpoint] / 20_000 < 0.53


def test_warmup_requires_an_opponent():
    with pytest.raises(ValueError, match="warmup_opponent"):
        League(warmup_decisions=1)
