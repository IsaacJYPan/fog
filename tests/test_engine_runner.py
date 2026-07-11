"""EngineRunner + engine_scripted tests (SIMULATION_PLAN §4): batched self-play must feed the
SAME RolloutBuffer/ppo_update contract as the Showdown path. Skipped without poke-engine."""
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


@pytest.fixture(scope="module")
def agent():
    from fog.models.agent import FogAgent

    a = FogAgent.from_tables({})
    a.eval()
    return a


def test_scripted_policies_always_legal(pool):
    from fog.env.engine_battle import EngineBattle
    from fog.env.engine_scripted import EngineMaxDamagePolicy, EngineRandomPolicy

    battle = EngineBattle(pool[0], pool[1], seed=0, battle_tag="scripted-legal")
    rand, maxd = EngineRandomPolicy(seed=1), EngineMaxDamagePolicy(seed=2)
    steps = 0
    while not battle.finished and steps < 400:
        moves = {}
        for role in battle.pending_sides():
            policy = rand if role == "p1" else maxd
            idx = policy.choose(battle, role)
            assert battle.action_mask(role)[idx], f"{policy.name} chose illegal action {idx}"
            moves[role] = battle.engine_action(role, idx)
        battle.step(moves)
        steps += 1
    assert battle.finished


def test_maxdamage_prefers_power(pool):
    from fog.env.engine_battle import EngineBattle
    from fog.env.engine_scripted import EngineMaxDamagePolicy

    battle = EngineBattle(pool[0], pool[1], seed=3, battle_tag="maxdmg")
    policy = EngineMaxDamagePolicy(seed=0)
    idx = policy.choose(battle, "p1")
    assert 0 <= idx < 4
    from fog.constants.tables import TABLES

    active = battle.active("p1")
    chosen_power = TABLES.moves["features"][str(active.moves[idx].id).lower()].get("base_power") or 0
    mask = battle.action_mask("p1")
    for slot in range(4):
        if mask[slot]:
            p = TABLES.moves["features"].get(str(active.moves[slot].id).lower(), {}).get("base_power") or 0
            assert chosen_power >= p


def test_runner_batch_matches_showdown_contract(pool, agent):
    import torch

    from fog.env.engine_runner import EngineRunner
    from fog.env.engine_scripted import EngineRandomPolicy
    from fog.training.ppo import ppo_update
    from fog.training.rollout import RolloutBuffer

    torch.manual_seed(0)
    buf = RolloutBuffer(gamma=1.0, lam=0.95, beta=0.0)
    runner = EngineRunner(agent, buf, pool=pool, batch_size=4, seed=0)
    stats = runner.play_games(EngineRandomPolicy(seed=5), 4)
    assert stats["games"] == 4 and stats["decisions"] > 20

    batch = buf.to_batch(torch.device("cpu"))
    # the exact C4 keys the Showdown path produces — ppo_update must see no difference
    for key in ("entities", "entity_ids", "field", "history", "active_idx", "legality_mask",
                "opp_revealed", "opp_view", "actions", "old_log_probs", "advantages", "returns",
                "belief_targets", "belief_valid", "self_belief_targets", "preview"):
        assert key in batch, f"missing batch key {key}"
    n = batch["entities"].shape[0]
    assert n == stats["decisions"]
    # belief truth is exact by construction on the engine path: every decision has species labels
    assert bool((batch["belief_targets"]["species"] >= 0).all())
    assert bool(batch["belief_valid"]["species"].all())

    train_agent = agent.train()
    opt = torch.optim.AdamW(train_agent.parameters(), lr=1e-4)
    metrics = ppo_update(train_agent, opt, batch, {"ppo": {"epochs": 1}})
    train_agent.eval()
    assert np.isfinite(metrics["loss"])
    assert metrics["belief_loss"] > 0 and metrics["self_belief_loss"] > 0


def test_runner_frozen_opponent(pool, agent):
    import torch

    from fog.env.engine_runner import EngineRunner, FrozenAgentPolicy
    from fog.models.agent import FogAgent
    from fog.training.rollout import RolloutBuffer

    torch.manual_seed(1)
    frozen = FogAgent.from_tables({})
    buf = RolloutBuffer()
    runner = EngineRunner(agent, buf, pool=pool, batch_size=2, seed=2)
    stats = runner.play_games(FrozenAgentPolicy(frozen, name="past-self"), 2)
    assert stats["games"] == 2
    assert stats["decisions"] > 0
    batch = buf.to_batch(torch.device("cpu"))
    assert batch["entities"].shape[0] == stats["decisions"]
