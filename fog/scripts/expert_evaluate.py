"""CPU playing-strength screen for a fixed, independently initialized FOG pilot."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from fog.env.showdown_direct import ShowdownDirectClient
from fog.eval.showdown_direct_selection import matched_schedule
from fog.eval.stats import paired_block_interval
from fog.pretrained.process import DecisionProcess
from fog.pretrained.runtime import native_environment
from fog.scripts.pretrained_evaluate import play
from fog.training.teacher_replay import digest


def evaluate(config, checkpoint, pool, out, *, games=256, native_python=None, native_checkpoint=None, native_games=128):
    if out.exists() or games < 8 or games % 4 or native_games < 8 or native_games % 4:
        raise ValueError("Use a new evaluation identity and complete four-game blocks")
    out.mkdir(parents=True)
    root = Path.cwd()
    teams = json.loads(pool.read_text())["teams"]
    candidate = DecisionProcess([sys.executable, "-m", "fog.scripts.expert_player_worker", "--config", str(config),
                                 "--checkpoint", str(checkpoint), "--temperature", "0"],
                                cwd=root, log=out / "candidate.log", deadline=5.0)
    native = None
    if native_python is not None:
        native = DecisionProcess([str(native_python), "-m", "fog.pretrained.native_worker", "--checkpoint", str(native_checkpoint),
                                  "--temperature", "0"], cwd=root, log=out / "native.log", env=native_environment(root), deadline=5.0)
    settings = {"checkpoint_sha256": digest(checkpoint), "pool_sha256": digest(pool), "seed": 20261013,
                "games_per_scripted_opponent": games, "native_games": native_games if native else 0,
                "temperature": 0.0, "native_temperature": 0.0,
                "preview_rule": "fixed original order for both neural agents", "deadline_seconds": 5.0,
                "native_sha256": digest(native_checkpoint) if native else None,
                "final_cohort_used": False, "automatic_promotion": False}
    (out / "PROTOCOL.json").write_text(json.dumps(settings, indent=2) + "\n")
    results = {}
    try:
        candidate.start()
        if native:
            native.start()
        with ShowdownDirectClient() as client:
            for name in ("simpleheuristics", "maxdamage", *(["kakuna34"] if native else [])):
                opponent = native if name == "kakuna34" else DecisionProcess(
                    [sys.executable, "-m", "fog.pretrained.scripted_worker", "--kind", name],
                    cwd=root, log=out / (name + ".log"), deadline=5.0)
                if opponent is not native:
                    opponent.start()
                rows = []
                try:
                    schedule = matched_schedule(len(teams), native_games if opponent is native else games, settings["seed"])
                    for index, game in enumerate(schedule):
                        row = play(client, candidate, opponent, teams, game, tag=f"expert-{name}-{index}", own_team_players=(candidate,))
                        row.update(index=index, schedule=asdict(game))
                        rows.append(row)
                        (out / (name + "-progress.json")).write_text(json.dumps({"rows": rows}) + "\n")
                        if row["status"] != "complete" or row["failures"] or row["fallbacks"]:
                            raise ValueError("CPU screen encountered an unresolved game, failure or fallback")
                        if (index + 1) % 16 == 0:
                            print(json.dumps({"opponent": name, "games": index + 1, "score": sum(r["score"] for r in rows) / len(rows)}), flush=True)
                finally:
                    if opponent is not native:
                        opponent.close()
                interval = paired_block_interval([row["score"] for row in rows])
                results[name] = {"games": len(rows), "wins": sum(row["score"] == 1 for row in rows),
                                 "losses": sum(row["score"] == 0 for row in rows), "draws": sum(row["score"] == 0.5 for row in rows),
                                 "interval": asdict(interval), "rows": rows}
        result = {"complete": True, "protocol": settings, "opponents": results, "gpu_requested": False,
                  "playing_strength_confirmation": False, "automatic_promotion": False}
        (out / "RESULTS.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        candidate.close()
        if native:
            native.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--native-python", type=Path)
    parser.add_argument("--native-checkpoint", type=Path)
    parser.add_argument("--native-games", type=int, default=128)
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint, args.pool, args.out, games=args.games,
             native_python=args.native_python, native_checkpoint=args.native_checkpoint, native_games=args.native_games)


if __name__ == "__main__":
    main()
