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
