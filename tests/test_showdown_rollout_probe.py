"""Direct-Showdown self-play rollouts skip belief heads without changing anything recorded."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "gen9ou_typed_token_selfplay.yaml"
POOL_DIR = ROOT / "data" / "teams" / "pool_b"
SHOWDOWN_BUILT = (ROOT / "server" / "pokemon-showdown" / "dist" / "sim" / "index.js").exists()


def test_truth_provider_and_probe_do_not_force_belief_heads():
    import torch

    from fog.config import load_config
    from fog.env.player import FogPlayer
    from fog.models.agent import FogAgent

    torch.manual_seed(0)
    player = FogPlayer(FogAgent.from_tables(load_config(CONFIG)))
    assert not player._act_needs_belief()
    player.private_state_truth_provider = lambda _battle: None
    player.phi_hat = object()
    assert not player._act_needs_belief()
    player.supervised_label_provider = lambda *_args: 0
    assert player._act_needs_belief()


@pytest.mark.skipif(not SHOWDOWN_BUILT, reason="pinned Showdown build missing")
def test_rollout_digest_identical_with_and_without_belief_heads(monkeypatch):
    from fog.env.player import FogPlayer
    from fog.eval.showdown_rollout_probe import run_probe

    skipped = run_probe(str(CONFIG), str(POOL_DIR), games=2, seed=3)
    monkeypatch.setattr(FogPlayer, "_act_needs_belief", lambda self: True)
    full = run_probe(str(CONFIG), str(POOL_DIR), games=2, seed=3)
    assert skipped["learner_decisions"] > 0
    assert skipped["decisions"] == full["decisions"]
    assert skipped["digest"] == full["digest"]


@pytest.mark.skipif(not SHOWDOWN_BUILT, reason="pinned Showdown build missing")
def test_worker_batch_of_one_records_what_the_sequential_loop_records():
    from fog.eval.showdown_rollout_probe import run_probe

    sequential = run_probe(str(CONFIG), str(POOL_DIR), games=2, seed=5)
    batched = run_probe(str(CONFIG), str(POOL_DIR), games=2, seed=5, batch=1)
    assert sequential["learner_decisions"] > 0
    assert batched["stats"]["forwards"] > 0
    assert batched["digest"] == sequential["digest"]


@pytest.mark.skipif(not SHOWDOWN_BUILT, reason="pinned Showdown build missing")
def test_private_actor_is_refreshed_from_the_shared_actor_each_chunk(monkeypatch):
    """`env.actor_device` keeps a copy of the actor; it must act with the shared weights."""
    import torch

    from fog.eval import showdown_rollout_probe as probe
    from fog.training import ppo

    shared = probe.run_probe(str(CONFIG), str(POOL_DIR), games=2, seed=7, batch=2)
    monkeypatch.setattr(ppo, "_needs_private_actor", lambda device: True)
    built = {}
    original = probe.build_worker

    def build_and_scramble(cfg, pool, *, threads=1):
        worker, agent = original(cfg, pool, threads=threads)
        assert worker.actor is not worker.shared_actor
        with torch.no_grad():
            for parameter in worker.actor.parameters():
                parameter.add_(1.0)  # a stale copy would change every recorded value
        built["worker"] = worker
        return worker, agent

    monkeypatch.setattr(probe, "build_worker", build_and_scramble)
    private = probe.run_probe(str(CONFIG), str(POOL_DIR), games=2, seed=7, batch=2)
    worker = built["worker"]
    for name, value in worker.shared_actor.state_dict().items():
        assert torch.equal(worker.actor.state_dict()[name], value), name
    assert private["digest"] == shared["digest"]
