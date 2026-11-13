"""Import, convert, split, train, and score a resumable replay blueprint run.

Run from the repository root. Keep the entire work directory, including raw replays,
converted shards, and training state, when moving to another notebook session.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

REVISION = "792a86b5cdbcab9696032e947483b218453a0d4f"
IMPORT_FILTERS = {"format": "gen9ou", "min_rating": 1500,
                  "after": 1_706_745_600, "before": None}
DEFAULT_HOLDOUT = Path("data/replay_splits/p2_holdout_2000_ids.txt")
TRAIN_SIZES = (20_000, 100_000)
MODULE = "fog.scripts.run_replay_blueprint"


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _manifest_counts(path: Path, *, require_shards: bool = False) -> tuple[int, int] | None:
    import polars as pl

    try:
        frame = pl.read_parquet(path)
        rows_column = "n_rows" if "n_rows" in frame.columns else "rows"
        rows = int(frame[rows_column].sum())
        if not len(frame) or rows <= 0:
            return None
        if require_shards and not all(Path(p).is_file() for p in frame["shard"]):
            return None
        return len(frame), rows
    except (OSError, ValueError, pl.exceptions.PolarsError):
        return None


def _command(module: str, *arguments: object) -> list[str]:
    return [sys.executable, "-u", "-m", module, *(str(argument) for argument in arguments)]


def _run_command(command: list[str], output: Path) -> None:
    """Append output across attempts and stop the child process group on interruption."""
    print(shlex.join(command), flush=True)
    with output.open("a", encoding="utf-8") as stream:
        stream.write(f"\n$ {shlex.join(command)}\n")
        stream.flush()
        child = subprocess.Popen(
            command, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=(os.name == "posix"),
        )
        try:
            code = child.wait()
        except BaseException:
            if os.name == "posix":
                os.killpg(child.pid, signal.SIGTERM)
            else:
                child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(child.pid, signal.SIGKILL)
                else:
                    child.kill()
                child.wait()
            raise
    if code:
        raise subprocess.CalledProcessError(code, command)


def _train_complete(training: Path, epochs: int) -> bool:
    if not all((training / "checkpoints" / name).is_file()
               for name in ("bc_best.pt", "bc_final.pt", "bc_state.pt")):
        return False
    try:
        with (training / "metrics.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        return bool(rows) and int(rows[-1]["epoch"]) >= epochs
    except (OSError, ValueError, KeyError):
        return False


def _report_complete(path: Path) -> bool:
    report = _read_json(path)
    return all(key in report for key in ("rows", "policy", "value"))


def _relocate_manifests(work: Path, state: dict) -> None:
    """Rebase only runner-owned shard paths when restoring to a different directory."""
    import polars as pl

    old_root = Path(state.get("work_root", work.resolve()))
    old_argument = Path(state.get("work_dir", work))
    if old_root == work.resolve() and old_argument == work:
        return
    for directory in (work / "decisions", work / "split"):
        for manifest in directory.glob("*.parquet"):
            frame = pl.read_parquet(manifest)
            if "shard" not in frame.columns:
                continue
            rebased = []
            for value in frame["shard"]:
                path = Path(value)
                try:
                    relative = path.relative_to(old_root if path.is_absolute() else old_argument)
                except ValueError:
                    rebased.append(value)
                else:
                    rebased.append(str(work / relative))
            if rebased != frame["shard"].to_list():
                temporary = manifest.with_suffix(".parquet.tmp")
                frame.with_columns(pl.Series("shard", rebased)).write_parquet(temporary)
                temporary.replace(manifest)
    state["work_root"] = str(work.resolve())
    state["work_dir"] = str(work)


def run(args: argparse.Namespace) -> Path:
    """Execute missing stages; existing scripts own the data and training semantics."""
    from fog.config import load_config

    if args.chronological_holdout_games is None and not args.holdout_ids.is_file():
        raise FileNotFoundError(
            f"fixed holdout IDs are missing: {args.holdout_ids}. Recover the brief 01 "
            "holdout IDs before a scale run; --chronological-holdout-games is only "
            "an alternative holdout for a smoke run."
        )
    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but PyTorch cannot use a CUDA GPU")

    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    logs = work / "output"
    logs.mkdir(exist_ok=True)
    state_path = work / "run.json"
    settings = {key: (str(value) if isinstance(value, Path) else value)
                for key, value in vars(args).items() if key != "work_dir"}
    settings["parquet"] = [str(path) for path in args.parquet]
    settings["resolved_config"] = load_config(args.config)
    settings["holdout_contents"] = (args.holdout_ids.read_text(encoding="utf-8")
                                     if args.chronological_holdout_games is None else None)
    state = _read_json(state_path)
    if state and state.get("arguments") != settings:
        changed = sorted(key for key in settings if state.get("arguments", {}).get(key)
                         != settings[key])
        raise ValueError(
            f"runner arguments changed ({', '.join(changed)}); use the same command "
            "to resume or choose a new --work-dir"
        )
    if not state:
        try:
            commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                                    capture_output=True, text=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            commit = "unknown"
        state = {"arguments": settings, "commit": commit,
                 "command": shlex.join([sys.executable, "-m", MODULE, *sys.argv[1:]]),
                 "work_root": str(work.resolve()), "work_dir": str(work),
                 "steps": {}}
        _save_json(state_path, state)
    _relocate_manifests(work, state)
    _save_json(state_path, state)

    def stage(name, complete, commands, *, prepare=None):
        previous = state["steps"].get(name, {})
        if previous.get("status") == "running":
            # A hard session stop cannot run the timing finally block. Preserve that
            # limitation instead of counting only the successful retry as total wall time.
            previous["interrupted_wall_seconds_unknown"] = True
        if complete():
            print(f"[{name}] output complete; skipping", flush=True)
            entry = state["steps"].setdefault(name, {"seconds": None})
            entry["status"] = "complete"
            _save_json(state_path, state)
            return
        if prepare is not None:
            prepare()
        entry = state["steps"].setdefault(name, {"seconds": 0.0})
        entry["status"] = "running"
        entry.setdefault("commands", [])
        _save_json(state_path, state)
        started = time.perf_counter()
        try:
            for command in commands():
                entry["commands"].append(shlex.join(command))
                _save_json(state_path, state)
                _run_command(command, logs / f"{name}.txt")
            if not complete():
                raise RuntimeError(f"{name} exited without complete output; see {logs / name}.txt")
        except BaseException:
            entry["status"] = "interrupted_or_failed"
            raise
        else:
            entry["status"] = "complete"
        finally:
            entry["seconds"] = float(entry.get("seconds") or 0.0) + time.perf_counter() - started
            _save_json(state_path, state)

    raw = work / "raw"
    imported = raw / ".metamon" / "import.json"
    prior_import = _read_json(imported)
    if prior_import and (prior_import.get("revision") != REVISION
                         or prior_import.get("filters") != IMPORT_FILTERS):
        raise ValueError("existing import revision/filters differ from the pinned replay recipe; "
                         "choose a new --work-dir")

    def import_complete():
        metadata = _read_json(imported)
        kept_games = int(metadata.get("kept_games", 0))
        if not kept_games or sum(1 for _ in raw.glob("gen9ou-*.json")) < kept_games:
            return False
        return (metadata.get("status") == "target_reached"
                and int(metadata.get("kept_games", 0)) >= args.games) or (
            metadata.get("status") == "selected_shards_exhausted"
            and bool(state["steps"].get("import", {}).get("exhausted"))
        )

    def import_commands():
        base = ["--out", raw, "--cache", work / "cache", "--revision", REVISION,
                "--min-rating", 1500, "--after", "2024-02-01", "--max-games", args.games]
        if args.parquet:
            local = [argument for path in args.parquet for argument in ("--parquet", path)]
            yield _command("fog.scripts.import_metamon_replays", *base, *local)
            state["steps"]["import"]["exhausted"] = True
            return
        # The importer can continue its cursor when the shard limit is increased. Grow
        # that limit until enough games pass the filters instead of downloading all 46.
        shards = 1
        while True:
            yield _command("fog.scripts.import_metamon_replays", *base,
                           "--max-shards", shards, "--max-download-mb", args.max_download_mb)
            if import_complete():
                return
            if shards >= args.max_shards:
                state["steps"]["import"]["exhausted"] = True
                return
            shards = min(args.max_shards, shards * 2)

    stage("import", import_complete, import_commands)
    kept = int(_read_json(imported).get("kept_games", 0))
    if kept < args.games:
        print(f"Import exhausted the selected source shards: {kept}/{args.games} games", flush=True)

    converted = work / "decisions"
    manifest = converted / "manifest.parquet"

    def conversion_complete():
        quality = _read_json(converted / "quality_report.json")
        counts = _manifest_counts(manifest, require_shards=True)
        return (quality.get("status") == "complete" and counts is not None
                and counts[0] == quality.get("parsed_replays")
                and quality.get("source_replays_considered") == kept)

    def prepare_conversion():
        # The converter refuses any existing manifest, even if interrupted before its
        # quality report. Its deterministic shard names can safely be written again.
        manifest.unlink(missing_ok=True)
        (converted / "quality_report.json").unlink(missing_ok=True)

    stage("convert", conversion_complete, lambda: [_command(
        "fog.scripts.convert_replay_decisions", "--raw", raw, "--out", converted,
        "--imputations", 1, "--workers", args.workers,
    )], prepare=prepare_conversion)

    split = work / "split"
    training_manifest = split / "train_manifest.parquet"
    holdout_manifest = split / "holdout_manifest.parquet"

    def split_complete():
        counts = _manifest_counts(training_manifest)
        return (counts is not None and _manifest_counts(holdout_manifest) is not None
                and all((split / f"train_{size}.parquet").is_file()
                        for size in TRAIN_SIZES if size <= counts[0]))

    def prepare_split():
        for name in ("train_manifest.parquet", "holdout_manifest.parquet",
                     *(f"train_{size}.parquet" for size in TRAIN_SIZES)):
            (split / name).unlink(missing_ok=True)

    holdout_args = (["--chronological-holdout-games", args.chronological_holdout_games]
                    if args.chronological_holdout_games is not None
                    else ["--holdout-ids", args.holdout_ids])
    stage("split", split_complete, lambda: [_command(
        "fog.scripts.split_replay_manifest", "--manifest", manifest, "--out", split,
        *holdout_args, "--train-sizes", *TRAIN_SIZES, "--skip-oversized-train-sizes",
    )], prepare=prepare_split)
    training_counts = _manifest_counts(training_manifest)
    holdout_counts = _manifest_counts(holdout_manifest)
    assert training_counts is not None and holdout_counts is not None
    if training_counts[0] < 100_000:
        print(f"Training on all {training_counts[0]} eligible games; fewer than 100000 remain",
              flush=True)

    training = work / "training"
    stage("train", lambda: _train_complete(training, args.epochs), lambda: [_command(
        "fog.training.bc", "--config", args.config, "--set",
        f"device={args.device}", f"bc.manifest={training_manifest}",
        f"bc.holdout_manifest={holdout_manifest}", f"bc.batch_size={args.batch_size}",
        f"bc.epochs={args.epochs}", "bc.resume=true", "bc.selection_metric=action_ce",
        "bc.streaming=true", "bc.early_stopping_patience=0", f"log.dir={training}",
    )])

    results = work / "results"
    results.mkdir(exist_ok=True)
    best = training / "checkpoints" / "bc_best.pt"
    for name, checkpoint_args in (("best", ["--checkpoint", best]),
                                  ("untrained", ["--untrained"])):
        report = results / f"holdout-{name}.json"
        stage(f"score-{name}", lambda report=report: _report_complete(report),
              lambda report=report, checkpoint_args=checkpoint_args: [_command(
                  "fog.eval.replay_holdout_report", "--config", args.config,
                  "--manifest", holdout_manifest, "--out", report,
                  "--batch-size", args.batch_size, *checkpoint_args,
              )])
    shutil.copy2(training / "metrics.csv", results / "metrics.csv")
    shutil.copy2(best, results / "bc_best.pt")
    if (logs / "train.txt").is_file():
        with (logs / "train.txt").open(encoding="utf-8") as stream:
            tail = "".join(deque(stream, maxlen=100))
    else:
        tail = "Training output was not retained with the existing checkpoints.\n"
    (results / "train-tail.txt").write_text(tail, encoding="utf-8")
    source_counts = _manifest_counts(manifest)
    summary = [f"commit: {state['commit']}", f"command: {state['command']}",
               f"revision: {REVISION}", f"imported games: {kept}",
               f"converted games/rows: {source_counts}",
               f"training games/rows: {training_counts}",
               f"holdout games/rows: {holdout_counts}",
               f"excluded newer or quarantined games: {kept - sum((training_counts[0], holdout_counts[0]))}"]
    for name, entry in state["steps"].items():
        seconds = entry.get("seconds")
        elapsed = f"{seconds:.3f} s" if seconds is not None else "not measured (existing output)"
        if entry.get("interrupted_wall_seconds_unknown"):
            elapsed += "; earlier interrupted attempt wall time was not recorded"
        summary.append(f"{name}: {entry['status']}; wall-clock {elapsed}")
        summary.extend(f"  {command}" for command in entry.get("commands", []))
    (results / "run-summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Results: {results}; retain {work} to resume in another session", flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=112_000)
    parser.add_argument("--work-dir", type=Path, default=Path("runs/blueprint_scale"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--config", type=Path, default=Path("configs/replay_blueprint_cpu.yaml"))
    holdout = parser.add_mutually_exclusive_group()
    holdout.add_argument("--holdout-ids", type=Path, default=DEFAULT_HOLDOUT)
    holdout.add_argument("--chronological-holdout-games", type=int,
                         help="alternative smoke-test holdout; does not reproduce the fixed holdout")
    parser.add_argument("--max-shards", type=int, default=46)
    parser.add_argument("--max-download-mb", type=int, default=6000)
    parser.add_argument("--parquet", type=Path, action="append", default=[],
                        help="local source shard for offline imports; repeatable")
    args = parser.parse_args()
    for name in ("games", "batch_size", "epochs", "workers", "max_shards", "max_download_mb"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.chronological_holdout_games is not None and not (
        0 < args.chronological_holdout_games < args.games
    ):
        parser.error("--chronological-holdout-games must be positive and less than --games")

    def terminate(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    run(args)


if __name__ == "__main__":
    main()
