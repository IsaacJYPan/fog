"""Vectorized self-play battle runner plus the M0 throughput probe."""
from __future__ import annotations

import argparse
import asyncio
import inspect
import time
from dataclasses import dataclass
from typing import Any

@dataclass
class ThroughputResult:
    num_workers: int
    num_games: int
    seconds: float

    @property
    def battles_per_sec(self) -> float:
        return self.num_games / self.seconds if self.seconds > 0 else float("inf")


def _server_configuration(server_url: str) -> Any:
    """Build the poke-env server configuration while tolerating minor version API drift."""
    try:
        from poke_env import LocalhostServerConfiguration, ServerConfiguration
    except ImportError:  # pragma: no cover - depends on optional poke-env install
        try:
            from poke_env.ps_client.server_configuration import (  # type: ignore
                LocalhostServerConfiguration,
                ServerConfiguration,
            )
        except ImportError as exc:
            raise RuntimeError(
                "poke-env is required for the M0 throughput probe. Install the project dev "
                'dependencies first, for example `pip install -e ".[dev]"`.'
            ) from exc

    if "localhost" in server_url or "127.0.0.1" in server_url:
        return LocalhostServerConfiguration

    auth_url = "https://play.pokemonshowdown.com/action.php?"
    return ServerConfiguration(server_url, auth_url)


async def _close_player(player: Any) -> None:
    client = getattr(player, "ps_client", None)
    if client is None:
        return
    for name in ("stop_listening", "close"):
        fn = getattr(client, name, None)
        if fn is None:
            continue
        result = fn()
        if inspect.isawaitable(result):
            await result
        return


async def _run_random_games(
    num_workers: int,
    num_games: int,
    server_url: str,
    battle_format: str,
) -> ThroughputResult:
    """Run RandomPlayer-vs-RandomPlayer games with poke-env concurrency."""
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1")
    if num_games < 1:
        raise ValueError("num_games must be >= 1")

    try:
        from poke_env.player import RandomPlayer
    except ImportError as exc:  # pragma: no cover - depends on optional poke-env install
        raise RuntimeError(
            "poke-env is required for the M0 throughput probe. Install the project dev "
            'dependencies first, for example `pip install -e ".[dev]"`.'
        ) from exc

    server_configuration = _server_configuration(server_url)
    player_kwargs = {
        "battle_format": battle_format,
        "server_configuration": server_configuration,
        "max_concurrent_battles": num_workers,
    }
    p1 = RandomPlayer(**player_kwargs)
    p2 = RandomPlayer(**player_kwargs)

    start = time.perf_counter()
    try:
        await p1.battle_against(p2, n_battles=num_games)
    finally:
        await asyncio.gather(_close_player(p1), _close_player(p2))
    return ThroughputResult(num_workers, num_games, time.perf_counter() - start)


def throughput_probe(cfg: dict) -> list[ThroughputResult]:
    """Sweep worker counts and report battles/sec (M0 gate)."""
    run = cfg.get("run", {})
    sweep = run.get("worker_sweep", [cfg["env"]["num_workers"]])
    num_games = run.get("num_games", 1000)
    server_url = cfg["env"]["server_url"]
    battle_format = cfg["env"]["battle_format"]
    results = []
    for n in sweep:
        res = asyncio.run(_run_random_games(n, num_games, server_url, battle_format))
        results.append(res)
        print(f"[M0] workers={n:2d}  games={num_games}  {res.battles_per_sec:.2f} battles/sec")
    return results


def main() -> None:
    from fog.config import load_config

    ap = argparse.ArgumentParser(description="FOG battle runner / M0 throughput probe.")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    kind = cfg.get("run", {}).get("kind")
    if kind == "throughput_probe":
        results = throughput_probe(cfg)
        gate = cfg.get("gate", {}).get("min_battles_per_sec", 3.0)
        best = max(r.battles_per_sec for r in results)
        status = "PASS" if best >= gate else "FAIL - fix throughput before model code"
        print(f"[M0] best={best:.2f} battles/sec  gate={gate}  -> {status}")
    else:
        raise NotImplementedError(f"runner does not handle run.kind={kind!r} yet.")


if __name__ == "__main__":
    main()
