"""Cross-simulator divergence gauntlet (ARCHITECTURE.md §3.2).

PPO will farm any mechanic poke-engine gets wrong: a policy that beats MaxDamage on the engine but
not on Showdown has learned the ENGINE, not Pokemon. This module plays the same matchup on both
backends and reports the win-rate gap; the engine training loop halts when |gap| exceeds
`cross_sim.halt_gap` (default 0.03 — the plan §7 rule).

The Showdown half needs the local server (`node pokemon-showdown start --no-security`). A cross-sim
check that silently skips would void the control, so a missing server RAISES: either start the
server before an engine training run with `cross_sim.every_decisions > 0`, or set it to 0 and
accept that the run has no online divergence control (fuzz + final Showdown fine-tune still apply).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fog.models.agent import FogAgent


def cross_sim_gauntlet(agent: "FogAgent", cfg: dict) -> dict:
    """Play `cross_sim.games` vs the same scripted opponent on BOTH backends with the live agent.

    Returns {"engine_wr", "showdown_wr", "gap", "games", "opponent"}; gap = engine - showdown.
    Restores the agent's train/eval mode. Raises on an unreachable server (module docstring).
    """
    import torch

    from fog.env.engine_runner import EngineRunner
    from fog.env.engine_scripted import SCRIPTED_ENGINE_POLICIES
    from fog.env.teams import load_pool

    env_cfg = cfg.get("env", {})
    cross = cfg.get("cross_sim", {})
    n = int(cross.get("games", 200))
    fmt = env_cfg.get("battle_format", "gen9ou")
    opponent = str(cross.get("opponent", "MaxDamage"))
    if opponent not in SCRIPTED_ENGINE_POLICIES:
        raise ValueError(
            f"cross_sim.opponent {opponent!r} has no engine port "
            f"(available: {sorted(SCRIPTED_ENGINE_POLICIES)})"
        )
    pool = load_pool(env_cfg["team_pool"])
    # greedy on both sides: the comparison wants the policy's modal behavior, not sampling noise
    temperature = float(cfg.get("eval", {}).get("temperature", 0.0))
    seed = int(cross.get("seed", 7))

    was_training = agent.training
    agent.eval()
    try:
        runner = EngineRunner(
            agent, None, pool=pool,
            batch_size=int(cross.get("engine_batch", 32)),
            seed=seed, temperature=temperature,
        )
        stats = runner.play_games(SCRIPTED_ENGINE_POLICIES[opponent](seed=seed + 1), n)
        engine_wr = stats["wins"] / max(1, stats["games"])

        # Showdown half through the standard eval harness (temp checkpoint -> run_match)
        from pathlib import Path

        from fog.eval.harness import run_match

        ckpt = Path(cfg.get("log", {}).get("dir", "runs")) / "checkpoints" / "cross_sim_eval.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(agent.state_dict(), ckpt)
        try:
            wins, finished = run_match(
                str(ckpt), opponent, n, fmt,
                team_pool=pool, cfg=cfg, device=next(agent.parameters()).device,
            )
        except Exception as exc:  # server down, poke-env missing, ...
            raise RuntimeError(
                "cross-sim gauntlet could not run its Showdown half (is the local server up? "
                "`node pokemon-showdown start --no-security`). A skipped check would void the "
                "divergence control, so this is fatal; set cross_sim.every_decisions=0 to opt "
                f"out explicitly. Underlying error: {exc!r}"
            ) from exc
        showdown_wr = wins / max(1, finished)
    finally:
        agent.train(was_training)

    return {
        "engine_wr": engine_wr,
        "showdown_wr": showdown_wr,
        "gap": engine_wr - showdown_wr,
        "games": n,
        "opponent": opponent,
    }
