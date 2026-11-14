"""Throughput probe for self-play rollouts on the ``showdown_direct`` backend.

Plays N games in one process with the same rollout worker PPO uses (one persistent Node
simulator, batch-1 CPU inference). With ``--batch N`` the same worker's controllers play through
``fog.env.showdown_batch_runner`` instead: N concurrent battles, one batched forward per wave. The learner is randomly initialised from the config; the
opponent is a frozen copy of it loaded from a temporary checkpoint, as in league self-play.

    python -m fog.eval.showdown_rollout_probe --games 16
    python -m fog.eval.showdown_rollout_probe --games 32 --batch 8
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


def run_batched(worker, payload, batch_size: int):
    """`_ShowdownRolloutWorker.run` with `env.rollout_batch` set to `batch_size`."""
    worker.rollout_batch = int(batch_size)
    local_buf, wins = worker.run(payload, batched=True)
    return local_buf, wins, worker.batch_stats


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
    config: str,
    pool_dir: str,
    games: int,
    seed: int,
    threads: int = 1,
    batch: int | None = None,
    opponent_name: str = "self",
) -> dict:
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
        spec = str(ckpt) if opponent_name == "self" else opponent_name
        opponent_calls = [0]
        if opponent_name == "self":
            opponent = worker._opponent_controller(spec)
            inner_act = opponent.act

            def counted_act(battle):
                opponent_calls[0] += 1
                return inner_act(battle)

            opponent.act = counted_act
        stats = None
        try:
            start = time.perf_counter()
            if batch is None:
                buf, wins = worker.run((assigned, spec, 0.0))
            else:
                buf, wins, stats = run_batched(worker, (assigned, spec, 0.0), batch)
            elapsed = time.perf_counter() - start
        finally:
            worker.close()
    learner = sum(len(g.actions) for g in buf._games.values())
    # The batched runner does not call `act` for batched decisions; count its p2 move choices.
    # Against a scripted opponent only the learner's decisions run the network and are counted.
    if opponent_name != "self":
        opponent = 0
    else:
        opponent = opponent_calls[0] if stats is None else stats["move_choices"][1]
    decisions = learner + opponent
    return {
        "buffer": buf,
        "stats": stats,
        "games": float(games),
        "decisions": float(decisions),
        "learner_decisions": float(learner),
        "digest": rollout_digest(buf),
        "seconds": elapsed,
        "decisions_per_s": decisions / elapsed,
        "games_per_s": games / elapsed,
        "p1_wins": float(wins),
    }


def run_pool_probe(
    config: str,
    pool_dir: str,
    games: int,
    seed: int,
    *,
    workers: int,
    threads: int = 1,
    batch: int = 1,
    device: str = "cpu",
    opponent: str = "self",
) -> dict:
    """Collection rate of the trainer's own worker processes (`PersistentProcessPool`).

    One untimed game per worker first (it loads the opponent and warms the simulator), then
    `games` games split across the workers as the trainer splits a chunk. `opponent` is ``self``
    (a frozen copy of the learner, so both seats run the network) or a scripted controller name.
    """
    import torch

    from fog.config import load_config
    from fog.env.teams import load_pool
    from fog.models.agent import FogAgent
    from fog.training.direct_parallel import PersistentProcessPool, WorkerGame, partition_games
    from fog.training.ppo import _ShowdownRolloutWorker

    cfg = load_config(config)
    cfg.setdefault("env", {})["rollout_batch"] = int(batch)
    cfg["env"]["actor_device"] = str(device)
    pool = load_pool(pool_dir)
    torch.manual_seed(int(cfg.get("seed", 0)))
    shared_actor = FogAgent.from_tables(cfg).cpu().eval()
    shared_actor.share_memory()
    ppo = cfg.get("ppo", {})
    leak = cfg.get("leakage", {})
    settings = (
        float(ppo.get("gamma", 1.0)),
        float(ppo.get("lam", 0.95)),
        0.0,
        bool(leak.get("clip_delta", True)),
        float(cfg.get("reward", {}).get("progress_coef", 0.0)),
    )
    supervision = str(cfg.get("env", {}).get("belief_supervision", "full_truth"))
    rng = random.Random(seed)

    def schedule(count: int, first: int) -> list:
        return [
            WorkerGame(-1, first + n, rng.getrandbits(62), rng.randrange(len(pool)),
                       rng.randrange(len(pool)))
            for n in range(count)
        ]

    with tempfile.TemporaryDirectory() as tmp:
        spec = opponent
        if opponent == "self":
            spec = str(Path(tmp) / "opponent.pt")
            torch.save(shared_actor.state_dict(), spec)
        with PersistentProcessPool(
            workers, _ShowdownRolloutWorker, cfg, shared_actor, None, pool, settings,
            supervision, {}, threads, 2,
        ) as process_pool:
            warm = partition_games(schedule(workers, 0), workers)
            process_pool.map([(assigned, spec, 0.0) for assigned in warm])
            timed = partition_games(schedule(games, workers), workers)
            start = time.perf_counter()
            results = process_pool.map([(assigned, spec, 0.0) for assigned in timed])
            elapsed = time.perf_counter() - start
    learner = sum(len(g.actions) for buf, _ in results for g in buf._games.values())
    return {
        "games": float(games),
        "learner_decisions": float(learner),
        "seconds": elapsed,
        "learner_decisions_per_s": learner / elapsed,
        "games_per_s": games / elapsed,
        "p1_wins": float(sum(wins for _, wins in results)),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/gen9ou_typed_token_selfplay.yaml")
    parser.add_argument("--pool", default="data/teams/pool_b")
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--batch", type=int, default=None,
        help="play N concurrent battles with one batched forward per wave",
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="measure the trainer's worker processes instead of one in-process worker",
    )
    parser.add_argument("--device", default="cpu", help="env.actor_device for --workers")
    parser.add_argument(
        "--opponent", default="self",
        help="self (a frozen copy of the learner), MaxDamage, SimpleHeuristics or Random",
    )
    args = parser.parse_args(argv)
    if args.workers:
        out = run_pool_probe(
            args.config, args.pool, args.games, args.seed, workers=args.workers,
            threads=args.threads, batch=args.batch or 1, device=args.device,
            opponent=args.opponent,
        )
        print(
            f"workers={args.workers} batch={args.batch or 1} device={args.device} "
            f"opponent={args.opponent} games={int(out['games'])} "
            f"learner_decisions={int(out['learner_decisions'])} seconds={out['seconds']:.2f} "
            f"learner_decisions/s={out['learner_decisions_per_s']:.2f} "
            f"games/s={out['games_per_s']:.3f} p1_wins={int(out['p1_wins'])}",
            flush=True,
        )
        return
    out = run_probe(
        args.config, args.pool, args.games, args.seed, args.threads, args.batch, args.opponent
    )
    print(
        f"games={int(out['games'])} decisions={int(out['decisions'])} "
        f"seconds={out['seconds']:.2f} decisions/s={out['decisions_per_s']:.2f} "
        f"games/s={out['games_per_s']:.3f} p1_wins={int(out['p1_wins'])} "
        f"learner_decisions={int(out['learner_decisions'])} digest={out['digest'][:16]}",
        flush=True,
    )
    if out["stats"] is not None:
        stats = out["stats"]
        rows = stats["forward_rows"] / max(1, stats["forwards"])
        split = " ".join(f"{k}={v:.2f}" for k, v in stats["seconds"].items())
        print(f"forwards={stats['forwards']} rows/forward={rows:.1f} seconds: {split}", flush=True)


if __name__ == "__main__":
    main()
