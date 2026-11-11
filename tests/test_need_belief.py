"""`FogAgent.forward(need_belief=False)` must skip belief heads without changing log_probs/value."""
from __future__ import annotations

from pathlib import Path

import pytest

from fog.env.engine_battle import HAVE_ENGINE

pytestmark = pytest.mark.skipif(not HAVE_ENGINE, reason="poke-engine not installed")

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "gen9ou_typed_token_selfplay.yaml"
POOL_DIR = ROOT / "data" / "teams" / "pool_b"
BELIEF_FIELDS = ("belief", "self_belief", "belief_chain", "self_belief_chain")


@pytest.fixture(scope="module")
def agent_and_batches():
    import torch

    from fog.config import load_config
    from fog.env.engine_runner import EngineRunner
    from fog.env.engine_scripted import EngineRandomPolicy
    from fog.env.teams import load_pool
    from fog.models.agent import FogAgent
    from fog.training.rollout import RolloutBuffer

    torch.manual_seed(0)
    agent = FogAgent.from_tables(load_config(CONFIG))
    agent.eval()
    batches: list[dict] = []

    def capture(_module, args, kwargs):
        batches.append(dict(args[0]))

    handle = agent.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        buf = RolloutBuffer(gamma=1.0, lam=0.95, beta=0.0)
        runner = EngineRunner(agent, buf, pool=load_pool(POOL_DIR), batch_size=2, seed=0)
        runner.play_games(EngineRandomPolicy(seed=5), 2, max_decisions=12)
    finally:
        handle.remove()
    batches = [b for b in batches if b.get("opp_revealed") is not None][:6]
    assert batches, "runner produced no batches with opp_revealed"
    return agent, batches


def test_need_belief_off_is_bit_identical(agent_and_batches):
    import torch

    agent, batches = agent_and_batches
    with torch.no_grad():
        for batch in batches:
            torch.manual_seed(1)
            on = agent(batch)
            torch.manual_seed(1)
            off = agent(batch, need_belief=False)
            assert on.belief is not None
            assert torch.equal(on.log_probs, off.log_probs)
            assert torch.equal(on.value, off.value)
            for name in BELIEF_FIELDS:
                assert getattr(off, name) is None, name


def test_default_unchanged_without_opp_revealed(agent_and_batches):
    import torch

    agent, batches = agent_and_batches
    batch = {k: v for k, v in batches[0].items() if k not in ("opp_revealed", "opp_view")}
    with torch.no_grad():
        torch.manual_seed(2)
        default = agent(batch)
        torch.manual_seed(2)
        explicit = agent(batch, need_belief=True)
        torch.manual_seed(2)
        off = agent(batch, need_belief=False)
    assert torch.equal(default.log_probs, explicit.log_probs)
    assert torch.equal(default.value, explicit.value)
    assert torch.equal(default.log_probs, off.log_probs)
    assert torch.equal(default.value, off.value)
    for name in BELIEF_FIELDS:
        assert getattr(default, name) is None, name


def test_compute_belief_chain_conflict_raises(agent_and_batches):
    agent, batches = agent_and_batches
    with pytest.raises(ValueError):
        agent({**batches[0], "compute_belief_chain": True}, need_belief=False)
