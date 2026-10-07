"""Play native Gen9OU challenges or a fixed rated confirmation with secrets in environment variables."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import re
import statistics
import urllib.request

from fog.eval.stats import wilson_interval
from fog.pretrained.kakuna import INTERFACE_VERSION, KAKUNA_34_SHA256, sha256
from fog.pretrained.live import LiveController, build_live_player
from fog.pretrained.process import DecisionProcess
from fog.pretrained.runtime import interpreter_path, native_environment


def profile(username):
    userid = re.sub(r"[^a-z0-9]", "", username.lower())
    url = "https://pokemonshowdown.com/users/" + userid + ".json"
    with urllib.request.urlopen(url, timeout=15) as response:
        data = json.load(response)
    return {
        "utc": datetime.now(UTC).isoformat(),
        "source": url,
        "gen9ou": data.get("ratings", {}).get("gen9ou"),
    }


async def play(args):
    from poke_env import (
        AccountConfiguration,
        LocalhostServerConfiguration,
        ShowdownServerConfiguration,
    )
    from fog.scripts.local_showcase import _close_player

    digest = sha256(args.checkpoint)
    if args.mode == "ladder":
        gate = json.loads(args.confirmation.read_text()) if args.confirmation else {}
        if args.games < 200 or not gate.get("passed") or gate.get("candidate_sha256") != digest:
            raise ValueError(
                "Rated confirmation requires a passed, hash-matched local strength gate and at least 200 games"
            )
        if args.server != "public":
            raise ValueError("Rated confirmation uses the public Gen9OU ladder")
    args.out.mkdir(parents=True, exist_ok=True)
    descriptor = {
        "checkpoint_sha256": digest,
        "interface": INTERFACE_VERSION,
        "games": args.games,
        "mode": args.mode,
        "server": args.server,
        "team_sha256": sha256(args.team),
        "temperature": 1.0,
        "decision_deadline_seconds": 5.0,
    }
    path = args.out / "protocol.json"
    if path.exists():
        raise FileExistsError("Choose a new live run folder; do not mix rated cohorts or models")
    path.write_text(json.dumps(descriptor, indent=2) + "\n")
    username = os.environ.get("FOG_SHOWDOWN_USERNAME")
    if not username:
        raise ValueError("Set FOG_SHOWDOWN_USERNAME; credentials belong outside Git")
    command = [
        interpreter_path(args.native_python),
        "-m",
        "fog.pretrained.native_worker",
        "--checkpoint",
        str(args.checkpoint.resolve()),
        "--device",
        args.device,
    ]
    if digest != KAKUNA_34_SHA256:
        command.append("--trained")
    worker = DecisionProcess(
        command, cwd=Path.cwd(), log=args.out / "native.log", env=native_environment(Path.cwd())
    )
    worker.start()
    controller = LiveController(worker, args.out)
    player = build_live_player(
        controller,
        account_configuration=AccountConfiguration(
            username, os.environ.get("FOG_SHOWDOWN_PASSWORD")
        ),
        battle_format="gen9ou",
        server_configuration=ShowdownServerConfiguration
        if args.server == "public"
        else LocalhostServerConfiguration,
        team=args.team.read_text(encoding="utf-8"),
    )
    rows, profiles = [], []
    try:
        for index in range(args.games):
            if worker.process is None:
                await asyncio.to_thread(worker.start)
            if args.mode == "ladder":
                await player.ladder(1)
            else:
                await player.send_challenges(args.opponent, n_challenges=1)
            battle = list(player.battles.values())[-1]
            rows.append(await controller.finish(battle))
            if args.mode == "ladder" and (index + 1) % 20 == 0:
                try:
                    profiles.append(await asyncio.to_thread(profile, username))
                except Exception as error:
                    profiles.append({"error": type(error).__name__ + ": " + str(error)})
            ratings = [row["elo"] for row in rows[-50:] if row["elo"] is not None]
            report = {
                "protocol": descriptor,
                "games": len(rows),
                "win_rate_interval": asdict(
                    wilson_interval(sum(row["won"] is True for row in rows), len(rows))
                ),
                "last_50_elo": {
                    "median": statistics.median(ratings),
                    "min": min(ratings),
                    "last": ratings[-1],
                }
                if ratings
                else None,
                "ladder_snapshots": profiles,
                "failures": sum(row["failures"] for row in rows),
                "fallbacks": sum(row["fallbacks"] for row in rows),
                "uncertainty": "Wilson win-rate interval is descriptive; Elo/GXE are observed service estimates. Glicko rprd is retained in snapshots.",
                "promotion": False,
            }
            (args.out / "RESULTS.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(rows[-1]), flush=True)
    finally:
        await _close_player(player)
        await asyncio.to_thread(worker.close, force=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--native-python", type=Path, required=True)
    parser.add_argument("--team", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=["challenge", "ladder"], default="challenge")
    parser.add_argument("--server", choices=["local", "public"], default="local")
    parser.add_argument("--opponent")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--confirmation", type=Path)
    args = parser.parse_args()
    if args.mode == "challenge" and not args.opponent:
        parser.error("Challenge mode requires --opponent")
    asyncio.run(play(args))


if __name__ == "__main__":
    main()
