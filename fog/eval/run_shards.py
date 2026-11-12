"""Run one ``showdown_direct_selection`` command as parallel shards and merge them.

One evaluation process plays one game at a time, so a 1,000-game row at 1.6 s per move takes
about half a day. This runs the same row as several processes with different seeds, each
resumable, and pools them with ``fog.eval.merge_selection_reports``:

    python -m fog.eval.run_shards --shards 3 --games 1000 --out-dir runs/g1/simpleheuristics -- \
      --pool data/teams/gen9ou_public_v1/validation.json --device cpu \
      --controller mcts=fog.search.engine_mcts:showdown_controller --opponent simpleheuristics

Everything after ``--`` goes to ``fog.eval.showdown_direct_selection`` unchanged; do not pass
``--seed``, ``--out``, ``--games``, ``--resume`` or ``--sequential`` there. ``--games`` is the total and
is split across shards in whole four-game blocks. Re-running the same command resumes each
unfinished shard from its last complete block and skips finished ones, so an interrupted run
loses at most a few games.

A searcher's time budget is wall-clock: keep ``--shards`` at or below the number of idle cores
minus one, or the search gets fewer iterations than its budget says.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

_RESERVED = ("--seed", "--out", "--games", "--resume", "--sequential")


def split_games(games: int, shards: int) -> list[int]:
    """Split a game total into per-shard counts of whole four-game blocks."""
    if games < 4 or games % 4:
        raise ValueError("games must be a positive multiple of four")
    if shards < 1 or shards > games // 4:
        raise ValueError("shards must be between 1 and the number of four-game blocks")
    blocks, extra = divmod(games // 4, shards)
    return [4 * (blocks + (1 if index < extra else 0)) for index in range(shards)]


def shard_commands(passthrough: list[str], out_dir: Path, games: int, shards: int,
                   base_seed: int) -> list[tuple[Path, list[str]]]:
    """``(report path, argv)`` for each shard that still has work to do."""
    for flag in _RESERVED:
        if any(arg == flag or arg.startswith(flag + "=") for arg in passthrough):
            raise ValueError(f"{flag} is set by run_shards; remove it from the passed arguments")
    commands = []
    for index, shard_games in enumerate(split_games(games, shards)):
        report = out_dir / f"shard_{index}.json"
        argv = [sys.executable, "-m", "fog.eval.showdown_direct_selection", *passthrough,
                "--games", str(shard_games), "--seed", str(base_seed + index),
                "--out", str(report)]
        if report.exists():
            if json.loads(report.read_text(encoding="utf-8")).get("status") == "complete":
                continue
            argv.append("--resume")
        commands.append((report, argv))
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--games", type=int, required=True, help="total games per row")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1000, help="seed of the first shard")
    parser.add_argument("passthrough", nargs=argparse.REMAINDER,
                        help="arguments for showdown_direct_selection, after --")
    args = parser.parse_args()
    passthrough = args.passthrough[1:] if args.passthrough[:1] == ["--"] else args.passthrough
    args.out_dir.mkdir(parents=True, exist_ok=True)
    commands = shard_commands(passthrough, args.out_dir, args.games, args.shards, args.seed)
    running = []
    for report, argv in commands:
        log = report.with_suffix(".log").open("a", encoding="utf-8")
        print(f"[run-shards] starting {report.name}: {shlex.join(argv[1:])}", flush=True)
        running.append((report, subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT),
                        log))
    failed = []
    for report, process, log in running:
        code = process.wait()
        log.close()
        if code != 0:
            failed.append(f"{report.name} (exit {code}, see {report.with_suffix('.log')})")
    if failed:
        raise SystemExit("[run-shards] shards failed; re-run the same command to resume: "
                         + ", ".join(failed))

    from fog.eval.merge_selection_reports import main as merge_main

    reports = [str(args.out_dir / f"shard_{index}.json") for index in range(args.shards)]
    shard_command = ("python -m fog.eval.run_shards "
                     + shlex.join(sys.argv[1:]))
    sys.argv = ["merge_selection_reports", *reports, "--out", str(args.out_dir / "merged.json"),
                "--command", shard_command]
    merge_main()


if __name__ == "__main__":
    main()
