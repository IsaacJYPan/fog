"""Batched direct-Showdown rollouts record what the sequential rollout path records."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "gen9ou_typed_token_selfplay.yaml"
POOL_DIR = ROOT / "data" / "teams" / "pool_b"
SHOWDOWN_BUILT = (ROOT / "server" / "pokemon-showdown" / "dist" / "sim" / "index.js").exists()

pytestmark = pytest.mark.skipif(not SHOWDOWN_BUILT, reason="pinned Showdown build missing")

_FRAG_ARRAYS = (
    "entities", "entity_ids", "field", "history", "history_ids", "history_state",
    "history_len", "active_idx", "legality_mask", "action_features", "turn",
)


def _probe(games: int, seed: int, batch: int | None) -> dict:
    from fog.eval.showdown_rollout_probe import run_probe

    return run_probe(str(CONFIG), str(POOL_DIR), games=games, seed=seed, batch=batch)


@pytest.fixture(scope="module")
def batch8_and_reference():
    return _probe(8, 5, 8), _probe(8, 5, 1)


def test_batch_one_matches_sequential_path_exactly():
    sequential = _probe(2, 3, None)
    batched = _probe(2, 3, 1)
    assert sequential["learner_decisions"] > 0
    assert batched["decisions"] == sequential["decisions"]
    assert batched["digest"] == sequential["digest"]


def test_batch_eight_completes_with_legal_actions_and_consistent_buffer(batch8_and_reference):
    out, _ = batch8_and_reference
    buf, stats = out["buffer"], out["stats"]
    assert len(buf._games) == 8
    assert stats["forward_rows"] > stats["forwards"]  # decisions really were batched
    keys = None
    for game in buf._games.values():
        n = len(game.actions)
        assert n > 0
        assert len(game.frags) == len(game.log_probs) == len(game.values) == n
        assert game.outcome in (-1.0, 0.0, 1.0)
        for frag, action, log_prob in zip(game.frags, game.actions, game.log_probs, strict=True):
            keys = keys or set(frag)
            assert set(frag) == keys
            assert bool(frag["legality_mask"][action])
            assert np.isfinite(log_prob)
            assert frag["behavior_log_probs"][action] == pytest.approx(log_prob)


def test_batch_eight_matches_one_by_one_on_the_same_observations(batch8_and_reference):
    """Same observations give the same log-probs and values up to float reduction order."""
    out, reference = batch8_and_reference
    compared = 0
    for gid, game in out["buffer"]._games.items():
        ref = reference["buffer"]._games[gid]
        for i, (frag, ref_frag) in enumerate(zip(game.frags, ref.frags, strict=False)):
            same_obs = all(np.array_equal(frag[k], ref_frag[k]) for k in _FRAG_ARRAYS)
            if not same_obs:
                break
            # Recorded log-prob is of the action each run sampled; compare full rows instead.
            assert np.allclose(
                frag["behavior_log_probs"], ref_frag["behavior_log_probs"], atol=1e-5
            )
            assert game.values[i] == pytest.approx(ref.values[i], abs=1e-5)
            compared += 1
            if game.actions[i] != ref.actions[i]:  # a float-level tie may flip one sample
                break
    assert compared > 100


def test_per_battle_state_does_not_leak_between_battles():
    import tempfile

    import torch

    from fog.config import load_config
    from fog.env.teams import load_pool
    from fog.eval.showdown_rollout_probe import build_worker, run_batched
    from fog.training.direct_parallel import WorkerGame

    pool = load_pool(str(POOL_DIR))
    worker, agent = build_worker(load_config(CONFIG), pool)
    rng = random.Random(11)
    assigned = [
        WorkerGame(0, n, rng.getrandbits(62), rng.randrange(len(pool)), rng.randrange(len(pool)))
        for n in range(4)
    ]
    expected = {g.battle_tag: (pool[g.p1_index], pool[g.p2_index]) for g in assigned}
    seen: dict[str, set] = {}
    adapter = worker.adapter
    private, final = adapter.private_state_truth_provider, adapter.truth_provider

    def check_private(battle):
        seen.setdefault(battle.battle_tag, set()).add(("p1", worker.current_private_state))
        return private(battle)

    def check_final(battle):
        seen.setdefault(battle.battle_tag, set()).add(("p2", worker.current_opponent))
        return final(battle)

    adapter.private_state_truth_provider = check_private
    adapter.truth_provider = check_final
    try:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "opponent.pt"
            torch.save(agent.state_dict(), ckpt)
            opponent = worker._opponent_controller(str(ckpt))
            buf, _, stats = run_batched(worker, (assigned, str(ckpt), 0.0), 4)
    finally:
        worker.close()
    assert stats["forward_rows"] > stats["forwards"]
    for tag, (p1, p2) in expected.items():
        assert seen[tag] == {("p1", p1), ("p2", p2)}
    assert len(buf._games) == 4 and all(g.outcome is not None for g in buf._games.values())
    for player in (adapter, opponent):
        assert not player._trackers and not player._ledgers and not player._fed_turns
        assert not player._game_ids and not player._announced_battles


def test_both_seats_learner_share_one_forward():
    import torch

    from fog.config import load_config
    from fog.env.player import FogPlayer
    from fog.env.showdown_batch_runner import BatchedShowdownRunner, BattleSpec
    from fog.env.showdown_direct import ShowdownDirectClient
    from fog.env.showdown_local_runner import bind_fog_player
    from fog.env.teams import load_pool, to_packed
    from fog.models.agent import FogAgent
    from fog.training.rollout import RolloutBuffer

    torch.manual_seed(0)
    agent = FogAgent.from_tables(load_config(CONFIG)).eval()
    buf = RolloutBuffer()
    seats = [bind_fog_player(FogPlayer(agent)) for _ in range(2)]
    for seat in seats:
        seat.attach_buffer(buf)
    pool = load_pool(str(POOL_DIR))
    specs = [
        BattleSpec(
            options={
                "formatid": "gen9ou",
                "seed": f"sodium,{n + 1:064x}",
                "strictChoices": True,
                "p1": {"name": "A", "team": to_packed(pool[n])},
                "p2": {"name": "B", "team": to_packed(pool[n + 1])},
            },
            battle_tag=f"battle-both-{n}",
            seed=n + 1,
        )
        for n in range(2)
    ]
    with ShowdownDirectClient() as client:
        runner = BatchedShowdownRunner(client, *seats, batch_size=2, max_turns=6)
        results = runner.play(specs)
    assert len(results) == 2
    assert runner.stats["forwards"] < runner.stats["forward_rows"]
    assert len(buf._games) == 4  # one rollout game per seat per battle
