"""Throughput probe for self-play rollouts on the ``showdown_direct`` backend.

Plays N games in one process with the same rollout worker PPO uses (one persistent Node
simulator, batch-1 CPU inference). The learner is randomly initialised from the config; the
opponent is a frozen copy of it loaded from a temporary checkpoint, as in league self-play.

    python -m fog.eval.showdown_rollout_probe --games 16
    python -m cProfile -o probe.prof -m fog.eval.showdown_rollout_probe --games 16
"""

from __future__ import annotations

import argparse
import random
import tempfile
import time
from pathlib import Path


def build_worker(cfg: dict, pool: list[str], *, threads: int = 1):
    import torch

    from fog.models.agent import FogAgent
    from fog.models.probe import FrozenProbe
    from fog.training.ppo import _ShowdownRolloutWorker

    torch.manual_seed(int(cfg.get("seed", 0)))
    agent = FogAgent.from_tables(cfg).cpu().eval()
    leak = cfg.get("leakage", {})
    beta = float(leak.get("beta", 0.0))
    phi = FrozenProbe.from_agent(agent).cpu().eval() if beta > 0 else None
    ppo = cfg.get("ppo", {})
    settings = (
        float(ppo.get("gamma", 1.0)),
        float(ppo.get("lam", 0.95)),
        beta,
        bool(leak.get("clip_delta", True)),
        float(cfg.get("reward", {}).get("progress_coef", 0.0)),
    )
    supervision = str(cfg.get("env", {}).get("belief_supervision", "full_truth"))
    worker = _ShowdownRolloutWorker(
        0, cfg, agent, phi, pool, settings, supervision, {}, threads, 2
    )
    return worker, agent


def _feed(h, value) -> None:
    import numpy as np
    import torch

    if isinstance(value, dict):
        for key in sorted(value, key=str):
            h.update(repr(key).encode())
            _feed(h, value[key])
    elif isinstance(value, (list, tuple)):
        h.update(f"seq{len(value)}".encode())
        for item in value:
            _feed(h, item)
    elif isinstance(value, torch.Tensor):
        _feed(h, value.detach().cpu().numpy())
    elif isinstance(value, np.ndarray):
        h.update(f"{value.dtype}{value.shape}".encode())
        h.update(np.ascontiguousarray(value).tobytes())
    else:
        h.update(repr(value).encode())


def rollout_digest(buf) -> str:
    """Hash of every recorded learner observation, action, log-prob, value and outcome."""
    import hashlib

    h = hashlib.sha256()
    for gid in sorted(buf._games):
        g = buf._games[gid]
        _feed(h, (g.frags, g.actions, g.log_probs, g.values, g.leak_deltas, g.leak_ce))
        _feed(h, (g.progress_scores, g.final_progress_score, g.final_leak_ce))
        _feed(h, (g.preview, g.outcome, g.bonus))
    return h.hexdigest()


def run_probe(
    config: str, pool_dir: str, games: int, seed: int, threads: int = 1
) -> dict[str, float]:
    import torch

    from fog.config import load_config
    from fog.env.teams import load_pool
    from fog.training.direct_parallel import WorkerGame

    cfg = load_config(config)
    pool = load_pool(pool_dir)
    worker, agent = build_worker(cfg, pool, threads=threads)
    rng = random.Random(seed)
    assigned = [
        WorkerGame(0, n, rng.getrandbits(62), rng.randrange(len(pool)), rng.randrange(len(pool)))
        for n in range(games)
    ]
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "opponent.pt"
        torch.save(agent.state_dict(), ckpt)
        opponent = worker._opponent_controller(str(ckpt))
        opponent_calls = [0]
        inner_act = opponent.act

        def counted_act(battle):
            opponent_calls[0] += 1
            return inner_act(battle)

        opponent.act = counted_act
        try:
            start = time.perf_counter()
            buf, wins = worker.run((assigned, str(ckpt), 0.0))
            elapsed = time.perf_counter() - start
        finally:
            worker.close()
    learner = sum(len(g.actions) for g in buf._games.values())
    decisions = learner + opponent_calls[0]
    return {
        "games": float(games),
        "decisions": float(decisions),
        "learner_decisions": float(learner),
        "digest": rollout_digest(buf),
        "seconds": elapsed,
        "decisions_per_s": decisions / elapsed,
        "games_per_s": games / elapsed,
        "p1_wins": float(wins),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/gen9ou_typed_token_selfplay.yaml")
    parser.add_argument("--pool", default="data/teams/pool_b")
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    out = run_probe(args.config, args.pool, args.games, args.seed, args.threads)
    print(
        f"games={int(out['games'])} decisions={int(out['decisions'])} "
        f"seconds={out['seconds']:.2f} decisions/s={out['decisions_per_s']:.2f} "
        f"games/s={out['games_per_s']:.3f} p1_wins={int(out['p1_wins'])} "
        f"learner_decisions={int(out['learner_decisions'])} digest={out['digest'][:16]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
