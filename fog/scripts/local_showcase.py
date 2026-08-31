"""Play a paced, watchable battle on a real local Pokemon Showdown server every few minutes.

Training remains on the fast in-memory simulator.  This process is deliberately separate and uses
the network server only for human-visible showcase battles.  It writes ``latest_battle.json`` as
soon as the first decision arrives so a monitor can publish the room URL while the battle is live,
and can optionally save a replay when the battle finishes.
"""
from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


async def _close_player(player: Any) -> None:
    client = getattr(player, "ps_client", None)
    if client is None:
        return
    for name in ("stop_listening", "close"):
        method = getattr(client, name, None)
        if method is None:
            continue
        result = method()
        if inspect.isawaitable(result):
            await result
        return


def _checkpoint_from_pointer(default: Path, pointer: Path | None) -> Path:
    if pointer is None or not pointer.is_file():
        return default
    try:
        value = json.loads(pointer.read_text(encoding="utf-8"))["checkpoint"]
        candidate = Path(value)
        return candidate if candidate.is_file() else default
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return default


async def run(args: argparse.Namespace) -> None:
    import torch
    from poke_env import LocalhostServerConfiguration
    from poke_env.player import SimpleHeuristicsPlayer

    from fog.config import load_config
    from fog.env.player import FogPlayer, unique_account
    from fog.env.teams import load_pool, make_pool_teambuilder
    from fog.models.agent import FogAgent

    args.out.mkdir(parents=True, exist_ok=True)
    replay_dir = args.out / "replays"
    if args.save_replays:
        replay_dir.mkdir(exist_ok=True)
    latest_path = args.out / "latest_battle.json"
    history_path = args.out / "battle_history.jsonl"
    cfg = load_config(args.config)
    teams = load_pool(args.team_pool or cfg["env"]["team_pool"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = FogAgent.from_tables(cfg).to(device)
    adapter = FogPlayer(model, temperature=0.0)
    adapter.decision_delay_seconds = max(0.0, args.turn_delay)

    current_checkpoint: Path | None = None
    active_record: dict[str, Any] = {}

    def battle_started(battle: Any) -> None:
        tag = str(getattr(battle, "battle_tag", "battle"))
        active_record.clear()
        active_record.update(
            {
                "status": "live",
                "battle_tag": tag,
                "watch_url": f"http://localhost:{args.port}/{tag}",
                "started_at": _utc_now(),
                "checkpoint": str(current_checkpoint),
                "opponent": "SimpleHeuristics",
                "note": "This battle uses the real local Pokemon Showdown network server; training does not.",
            }
        )
        _atomic_json(latest_path, active_record)
        print(f"[showcase] live {active_record['watch_url']}", flush=True)

    adapter.on_battle_start = battle_started
    player = adapter.build(
        battle_format=cfg["env"].get("battle_format", "gen9ou"),
        server_configuration=LocalhostServerConfiguration,
        account_configuration=unique_account("fogwatch"),
        team=make_pool_teambuilder(teams, seed=args.seed),
        max_concurrent_battles=1,
    )
    opponent = SimpleHeuristicsPlayer(
        battle_format=cfg["env"].get("battle_format", "gen9ou"),
        server_configuration=LocalhostServerConfiguration,
        account_configuration=unique_account("simplewatch"),
        team=make_pool_teambuilder(teams, seed=args.seed + 1),
        max_concurrent_battles=1,
    )

    try:
        battle_number = 0
        while args.count <= 0 or battle_number < args.count:
            cycle_started = time.monotonic()
            checkpoint = _checkpoint_from_pointer(args.checkpoint, args.latest_pointer)
            if checkpoint != current_checkpoint:
                state = torch.load(checkpoint, map_location=device, weights_only=False)
                try:
                    model.load_state_dict(state)
                except RuntimeError:
                    model.load_compatible_state_dict(state)
                model.eval()
                current_checkpoint = checkpoint
                print(f"[showcase] loaded {checkpoint}", flush=True)

            player.reset_battles()
            opponent.reset_battles()
            active_record.clear()
            await player.battle_against(opponent, n_battles=1)
            battle_number += 1
            if active_record:
                tag = active_record["battle_tag"]
                battle = player.battles.get(tag)
                active_record.update(
                    {
                        "status": "complete",
                        "completed_at": _utc_now(),
                        "won": None if battle is None else battle.won,
                        "turns": None if battle is None else int(battle.turn),
                    }
                )
                if battle is not None and args.save_replays:
                    replay_path = replay_dir / f"{tag}.html"
                    player.save_replay(tag, replay_path)
                    active_record["replay"] = str(replay_path.resolve())
                _atomic_json(latest_path, active_record)
                with history_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(active_record) + "\n")
                print(
                    f"[showcase] complete tag={tag} won={active_record['won']} "
                    f"turns={active_record['turns']}",
                    flush=True,
                )

            if args.count > 0 and battle_number >= args.count:
                break
            remaining = max(0.0, args.interval - (time.monotonic() - cycle_started))
            await asyncio.sleep(remaining)
    finally:
        await asyncio.gather(_close_player(player), _close_player(opponent))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--latest-pointer", type=Path)
    parser.add_argument("--team-pool", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--interval", type=float, default=450.0, help="seconds between battle starts")
    parser.add_argument("--turn-delay", type=float, default=3.0, help="seconds before each model move")
    parser.add_argument("--count", type=int, default=0, help="0 means run until stopped")
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--save-replays",
        action="store_true",
        help="save completed visible battles as local HTML files (disabled by default)",
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
