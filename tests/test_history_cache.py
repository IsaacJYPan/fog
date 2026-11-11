"""The per-turn history cache must be invisible: cached and uncached encodings are bit-identical at
every decision of real engine games, including on forked (cloned) trackers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fog.env.engine_battle import HAVE_ENGINE

pytestmark = pytest.mark.skipif(not HAVE_ENGINE, reason="poke-engine not installed")

POOL_DIR = Path(__file__).resolve().parents[1] / "data" / "teams" / "pool_b"


@pytest.fixture(scope="module")
def pool():
    from fog.env.teams import load_pool

    return load_pool(POOL_DIR)


def _encode(view, tracker, role, use_cache):
    from fog.constants.tables import TABLES
    from fog.encoding.history import encode_history, encode_history_sidecars

    hist, n = encode_history(view, TABLES, perspective=role, tracker=tracker, use_cache=use_cache)
    ids, state = encode_history_sidecars(
        view, TABLES, perspective=role, tracker=tracker, use_cache=use_cache
    )
    return hist, n, ids, state


def _assert_equivalent(view, tracker, role) -> None:
    cached = _encode(view, tracker, role, True)
    again = _encode(view, tracker, role, True)  # second call is served from the cache
    fresh = _encode(view, tracker, role, False)
    for a, b, c in zip(cached, again, fresh):
        if isinstance(a, np.ndarray):
            assert a.dtype == c.dtype and a.shape == c.shape
            assert np.array_equal(a, c) and np.array_equal(b, c)
            assert a.flags.writeable  # callers get their own arrays
        else:
            assert a == b == c


def _play(battle, policies, *, check, max_steps=400, record=True):
    from fog.constants.tables import TABLES
    from fog.encoding.events import record_history_decision

    decisions = 0
    steps = 0
    while not battle.finished and steps < max_steps:
        moves = {}
        for role in battle.pending_sides():
            view = battle.view(role)
            if check:
                _assert_equivalent(view, battle.trackers[role], role)
                decisions += 1
            idx = policies[role].choose(battle, role)
            if record:
                record_history_decision(battle.trackers[role], battle.turn, view, TABLES, idx)
            moves[role] = battle.engine_action(role, idx)
        battle.step(moves)
        steps += 1
    return decisions


def test_cached_history_matches_uncached_over_full_games(pool):
    from fog.env.engine_battle import EngineBattle
    from fog.env.engine_scripted import EngineMaxDamagePolicy, EngineRandomPolicy

    total = 0
    for game in range(3):
        battle = EngineBattle(
            pool[(2 * game) % len(pool)], pool[(2 * game + 1) % len(pool)],
            seed=game, battle_tag=f"hist-cache-{game}",
        )
        policies = {"p1": EngineRandomPolicy(seed=10 + game), "p2": EngineMaxDamagePolicy(seed=20 + game)}
        total += _play(battle, policies, check=True)
        assert battle.finished
        assert battle.turn > 3
        for role in ("p1", "p2"):
            cache = battle.trackers[role]._history_record_cache
            assert cache, "cache never populated"
            for entry in cache.values():
                assert not entry[3].flags.writeable
    assert total > 50


def test_cache_hits_skip_reparsing(pool, monkeypatch):
    import fog.encoding.events as events
    from fog.env.engine_battle import EngineBattle
    from fog.env.engine_scripted import EngineRandomPolicy

    battle = EngineBattle(pool[0], pool[1], seed=7, battle_tag="hist-cache-hits")
    policies = {"p1": EngineRandomPolicy(seed=1), "p2": EngineRandomPolicy(seed=2)}
    _play(battle, policies, check=False, max_steps=8)
    role = "p1"
    view, tracker = battle.view(role), battle.trackers[role]
    _encode(view, tracker, role, True)  # warm

    calls = {"rec": 0, "ids": 0}
    real_rec, real_ids = events.parse_turn_events, events.parse_turn_event_ids

    def count_rec(*a, **k):
        calls["rec"] += 1
        return real_rec(*a, **k)

    def count_ids(*a, **k):
        calls["ids"] += 1
        return real_ids(*a, **k)

    monkeypatch.setattr(events, "parse_turn_events", count_rec)
    monkeypatch.setattr(events, "parse_turn_event_ids", count_ids)
    hist, n, _, _ = _encode(view, tracker, role, True)
    assert n > 0 and calls == {"rec": 0, "ids": 0}
    _encode(view, tracker, role, False)
    assert calls == {"rec": n, "ids": n}


def test_cloned_tracker_cache(pool):
    from fog.env.engine_battle import EngineBattle
    from fog.env.engine_scripted import EngineMaxDamagePolicy, EngineRandomPolicy

    battle = EngineBattle(pool[2], pool[3], seed=11, battle_tag="hist-cache-clone")
    policies = {"p1": EngineRandomPolicy(seed=3), "p2": EngineMaxDamagePolicy(seed=4)}
    forks_checked = 0
    steps = 0
    while not battle.finished and steps < 400:
        if steps % 4 == 2:
            before = {r: dict(t._history_record_cache) for r, t in battle.trackers.items()}
            fork = battle.clone_for_step(seed=steps)
            for role, tracker in fork.trackers.items():
                assert tracker._history_record_cache is not battle.trackers[role]._history_record_cache
            fork_policies = {"p1": EngineRandomPolicy(seed=steps), "p2": EngineRandomPolicy(seed=steps + 1)}
            forks_checked += _play(fork, fork_policies, check=True, max_steps=3, record=False)
            # The fork's lookups/insertions never leak into the original's cache.
            for role, tracker in battle.trackers.items():
                assert tracker._history_record_cache == before[role]
        moves = {}
        for role in battle.pending_sides():
            view = battle.view(role)
            _assert_equivalent(view, battle.trackers[role], role)
            from fog.constants.tables import TABLES
            from fog.encoding.events import record_history_decision

            idx = policies[role].choose(battle, role)
            record_history_decision(battle.trackers[role], battle.turn, view, TABLES, idx)
            moves[role] = battle.engine_action(role, idx)
        battle.step(moves)
        steps += 1
    assert forks_checked > 10
