"""EngineMCTS: usage-prior world sampler, root aggregation and the engine-backend controller."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fog.env.engine_battle import HAVE_ENGINE

POOL = Path(__file__).resolve().parents[1] / "data" / "teams" / "pool_b"

engine_only = pytest.mark.skipif(not HAVE_ENGINE, reason="poke-engine not installed")


@pytest.fixture(scope="module")
def pool():
    from fog.env.teams import load_pool

    return load_pool(POOL)


@pytest.fixture(scope="module")
def tables():
    from fog.constants.tables import TABLES

    return TABLES


def _partial_reveal(truth: np.ndarray, tables) -> np.ndarray:
    """A public grid with a mix of revealed facts, a knocked-off item and unknown species."""
    grid = np.full_like(truth, -1)
    grid[:4, 0] = truth[:4, 0]
    grid[0, 1] = truth[0, 1]
    grid[0, 2] = truth[0, 2]
    grid[0, 3:5] = truth[0, 3:5]
    grid[1, 3] = truth[1, 5]
    grid[1, 14] = truth[1, 14]
    grid[2, 1] = tables.item_index("<NO_ITEM>")
    return grid


# ------------------------------------------------------------------------------------- sampler
def test_sampler_shapes_weights_and_reveals(pool, tables):
    from fog.env.team_truth import truth_tokens_from_paste
    from fog.search.evidence import keep_mask_from_reveals
    from fog.search.usage_worlds import sample_usage_worlds

    truth = truth_tokens_from_paste(pool[0], tables)
    grid = _partial_reveal(truth, tables)
    tokens, weights = sample_usage_worlds(grid, 8, seed=5, dedupe=False)
    assert tokens.shape == (8, 6, 15) and tokens.dtype == np.int64
    assert weights.shape == (8,) and np.isclose(weights.sum(), 1.0)
    no_item = tables.item_index("<NO_ITEM>")
    assert keep_mask_from_reveals(grid, tokens, no_item_token=no_item).all()
    for world in tokens:
        assert (world[:, [0, 1, 2, 7, 8, 9, 10, 11, 12, 13, 14]] >= 0).all()
        species = world[:, 0].tolist()
        assert len(set(species)) == 6, "species clause: no duplicate species"
        assert species[:4] == grid[:4, 0].tolist()
        assert world[0, 1] == grid[0, 1] and world[0, 2] == grid[0, 2]
        assert world[1, 14] == grid[1, 14]
        assert {int(grid[0, 3]), int(grid[0, 4])} <= set(world[0, 3:7].tolist())
        assert int(grid[1, 3]) in set(world[1, 3:7].tolist())
        assert all(0 <= s < tables.n_species for s in species)

    deduped, dweights = sample_usage_worlds(grid, 8, seed=5)
    assert deduped.shape[1:] == (6, 15) and 1 <= len(deduped) <= 8
    assert np.isclose(dweights.sum(), 1.0) and (dweights > 0).all()


def test_sampler_is_deterministic_under_seed(pool, tables):
    from fog.env.team_truth import truth_tokens_from_paste
    from fog.search.usage_worlds import sample_usage_worlds

    grid = _partial_reveal(truth_tokens_from_paste(pool[1], tables), tables)
    a, wa = sample_usage_worlds(grid, 6, seed=11)
    b, wb = sample_usage_worlds(grid, 6, seed=11)
    c, _ = sample_usage_worlds(grid, 6, seed=12)
    assert np.array_equal(a, b) and np.array_equal(wa, wb)
    assert not (a.shape == c.shape and np.array_equal(a, c))


@engine_only
def test_sampler_on_live_battle_reveals(pool, tables):
    from fog.env.engine_battle import EngineBattle
    from fog.env.engine_scripted import EngineMaxDamagePolicy
    from fog.search.evidence import keep_mask_from_reveals
    from fog.search.usage_worlds import public_reveal_grid, sample_usage_worlds

    battle = EngineBattle(pool[0], pool[1], seed=2)
    policy = EngineMaxDamagePolicy(seed=0)
    for _ in range(8):
        if battle.finished:
            break
        battle.step({r: battle.engine_action(r, policy.choose(battle, r))
                     for r in battle.pending_sides()})
    grid = public_reveal_grid(battle, "p1")
    assert (grid[:, 3:7] >= 0).any(), "some opponent move should be public by now"
    tokens, _ = sample_usage_worlds(grid, 4, seed=0)
    no_item = tables.item_index("<NO_ITEM>")
    assert keep_mask_from_reveals(grid, tokens, no_item_token=no_item).all()
    for world in tokens:  # every world builds an engine fork
        battle.fork_for_search("p1", world, seed=0)


# --------------------------------------------------------------------------------- aggregation
def test_aggregate_root_weights_worlds_by_kept_visits():
    from fog.search.engine_mcts import WorldSearch, aggregate_root

    worlds = [
        WorldSearch(weight=3.0, visits={"a": 75, "b": 25}, total_visits=100, kept_visits=100,
                    dropped_visits=0, mean_score=0.8),
        WorldSearch(weight=1.0, visits={"b": 10, "c": 30}, total_visits=50, kept_visits=40,
                    dropped_visits=10, mean_score=0.4),
        WorldSearch(weight=5.0, visits={}, total_visits=10, kept_visits=0,
                    dropped_visits=10, mean_score=0.5),   # nothing kept: ignored
    ]
    root = aggregate_root(worlds)
    assert root.policy["a"] == pytest.approx(0.75 * 0.75)
    assert root.policy["b"] == pytest.approx(0.75 * 0.25 + 0.25 * 0.25)
    assert root.policy["c"] == pytest.approx(0.25 * 0.75)
    assert sum(root.policy.values()) == pytest.approx(1.0)
    assert root.value == pytest.approx(0.75 * 0.8 + 0.25 * 0.4)
    with pytest.raises(ValueError):
        aggregate_root(worlds[2:])


def test_normalize_choice():
    from fog.search.engine_mcts import is_tera_choice, normalize_choice

    assert normalize_choice("switch Great Tusk") == "switch greattusk"
    assert normalize_choice("STEALTHROCK") == "stealthrock"
    assert normalize_choice("EARTHQUAKE-tera") == "earthquake-tera"
    assert is_tera_choice("earthquake-tera") and not is_tera_choice("switch terapagos")


# ---------------------------------------------------------------------------------- controller
@engine_only
def test_choice_map_tracks_tera_and_switches(pool):
    from fog.env.engine_battle import EngineBattle
    from fog.search.engine_mcts import engine_choice_map

    battle = EngineBattle(pool[0], pool[1], seed=1)
    before = engine_choice_map(battle, "p1")
    assert any(key.endswith("-tera") for key in before)
    assert all(8 <= idx <= 12 for key, idx in before.items() if key.startswith("switch "))
    tera_idx = min(idx for key, idx in before.items() if key.endswith("-tera"))
    battle.step({"p1": battle.engine_action("p1", tera_idx),
                 "p2": battle.engine_action("p2", 0)})
    assert battle.tera_used["p1"]
    after = engine_choice_map(battle, "p1")
    assert not any(key.endswith("-tera") for key in after)


@engine_only
def test_controller_plays_only_legal_actions(pool):
    from fog.env.engine_battle import EngineBattle
    from fog.env.engine_scripted import EngineMaxDamagePolicy
    from fog.search.engine_mcts import EngineMCTSPolicy

    mcts = EngineMCTSPolicy(k=2, iterations=150, seed=0)
    calls = 0
    for game, (a, b) in enumerate([(0, 1), (2, 3), (4, 5)]):
        mcts_role = "p1" if game % 2 == 0 else "p2"
        battle = EngineBattle(pool[a], pool[b], seed=game)
        opp = EngineMaxDamagePolicy(seed=game)
        for _ in range(1500):
            if battle.finished:
                break
            moves = {}
            for role in battle.pending_sides():
                if role == mcts_role:
                    idx = mcts.choose(battle, role)
                    calls += 1
                    assert battle.action_mask(role)[idx]
                    if mcts.last_policy is not None:
                        assert mcts.last_policy[idx] == mcts.last_policy.max()
                        assert np.isclose(mcts.last_policy.sum(), 1.0)
                        assert not (mcts.last_policy * ~battle.action_mask(role)).any()
                else:
                    idx = opp.choose(battle, role)
                moves[role] = battle.engine_action(role, idx)
            battle.step(moves)
        assert battle.finished
    c = mcts.counters
    assert c["decisions"] == calls > 0
    assert c["searched"] + c["single_option"] + c["fallbacks"] == c["decisions"]
    assert c["fallbacks"] == 0 and c["worlds_dropped"] == 0
    assert c["worlds_sampled"] >= c["searched"]


@engine_only
def test_controller_falls_back_when_every_world_fails(pool, monkeypatch):
    from fog.env.engine_battle import EngineBattle
    from fog.search.engine_mcts import EngineMCTSPolicy

    battle = EngineBattle(pool[0], pool[1], seed=0)

    def broken(*args, **kwargs):
        raise RuntimeError("world build failed")

    monkeypatch.setattr(battle, "fork_for_search", broken)
    mcts = EngineMCTSPolicy(k=3, iterations=50, seed=0)
    idx = mcts.choose(battle, "p1")
    assert battle.action_mask("p1")[idx]
    assert mcts.counters["fallbacks"] == 1
    assert mcts.counters["worlds_dropped"] == mcts.counters["worlds_sampled"] > 0
    assert mcts.last_policy is None and mcts.last_value is None
    assert idx == mcts.fallback_policy.choose(battle, "p1")
