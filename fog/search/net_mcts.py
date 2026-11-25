"""EngineMCTS with the network's policy fused at the root (a first, cheap network-in-search step).

At each decision EngineMCTS runs unchanged (``K`` usage-sampled worlds, wall-clock budget) and
returns its root visit distribution ``p``. The network reads the same position through the normal
censored observation and gives its policy ``pi``. The move played is

    argmax_a  log p(a) + alpha * log pi(a)        over legal actions,

so ``alpha = 0`` is plain EngineMCTS and larger ``alpha`` lets the network re-rank moves the
search left close. When every world fails, the network's argmax is played.

The network sees only our censored view (``Observation.from_battle``); the search's sampled worlds
never reach it. The network forward runs inside the same wall-clock turn as the search but is not
charged against ``budget_ms``; it costs a few tens of milliseconds on CPU.

Factory for ``showdown_direct_selection``::

    --controller hyb=fog.search.net_mcts:showdown_controller \
    --controller-kwargs 'hyb={"checkpoint": "ck/rl_cont.pt", "alpha": 1.0, "budget_ms": 200}'
"""
from __future__ import annotations

from typing import Any

import numpy as np

_EPS = 1e-6


def fuse(search_policy: np.ndarray | None, net_log_probs: np.ndarray, mask: np.ndarray,
         alpha: float) -> int:
    """Legal argmax of ``log p + alpha * log pi``; the network's argmax when search gave nothing."""
    mask = np.asarray(mask, dtype=bool)
    net = np.where(mask, np.asarray(net_log_probs, dtype=np.float64), -np.inf)
    if search_policy is None or not np.any(np.asarray(search_policy)[mask] > 0):
        return int(np.argmax(net))
    p = np.asarray(search_policy, dtype=np.float64)
    score = np.log(np.clip(p, _EPS, None)) + float(alpha) * np.maximum(net, np.log(_EPS))
    score = np.where(mask, score, -np.inf)
    return int(np.argmax(score))


def showdown_controller(*, checkpoint: str, config: str = "configs/search_distill_cpu.yaml",
                        alpha: float = 1.0, k: int = 8, budget_ms: int = 1600,
                        threads: int = 1, seed: int = 0, device: str = "cpu") -> Any:
    import torch

    from fog.config import load_config_with_args
    from fog.env.player import FogPlayer
    from fog.env.showdown_local_runner import bind_fog_player
    from fog.search.engine_mcts import ShowdownEngineMCTSController
    from fog.training.value_finetune import load_posterior_agent

    cfg = load_config_with_args(config)
    agent = load_posterior_agent(cfg, checkpoint, torch.device(device))
    agent.eval()
    player = FogPlayer(agent, temperature=0.0)
    mcts = ShowdownEngineMCTSController(k=k, budget_ms=budget_ms, threads=threads, seed=seed)
    stats = {"fused_decisions": 0, "net_changed_choice": 0, "search_fallbacks": 0}

    def override(battle: Any, obs: Any, out: Any) -> int:
        mask = np.asarray(obs.legality_mask, dtype=bool)
        net = out.log_probs[0].detach().float().cpu().numpy()
        if mask.sum() == 1:
            return int(np.flatnonzero(mask)[0])
        role = getattr(battle, "player_role", None) or "p1"
        searched = mcts.search.choose(battle, mcts._tracker(battle), role)
        policy = mcts.search.last_policy
        action = fuse(policy, net, mask, alpha)
        stats["fused_decisions"] += 1
        stats["search_fallbacks"] += int(policy is None)
        stats["net_changed_choice"] += int(policy is not None and action != searched)
        return action

    player.action_override = override
    finished = player._battle_finished

    def battle_finished(battle: Any) -> None:
        finished(battle)
        mcts._battle_finished(battle)

    def take_game_counters() -> dict[str, Any]:
        counters = mcts.take_game_counters()
        counters.update(stats)
        for key in stats:
            stats[key] = 0
        return counters

    player._battle_finished = battle_finished
    player.take_game_counters = take_game_counters
    player.name = f"NetMCTS(alpha={alpha})"
    return bind_fog_player(player)
