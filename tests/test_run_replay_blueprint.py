from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import polars as pl
import pytest

from fog.scripts import run_replay_blueprint as runner


def _args(tmp_path, **changes):
    values = dict(games=6, work_dir=tmp_path / "run", device="cpu", batch_size=256,
                  epochs=2, workers=1, config=Path("configs/replay_blueprint_cpu.yaml"),
                  holdout_ids=tmp_path / "absent.txt", chronological_holdout_games=2,
                  max_shards=46, max_download_mb=6000, parquet=[], prepare_only=False)
    values.update(changes)
    return argparse.Namespace(**values)


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _import_metadata(**changes):
    return {"status": "target_reached", "kept_games": 6,
            "revision": runner.REVISION, "filters": runner.IMPORT_FILTERS, **changes}


def _stub_scripts(monkeypatch, args, *, interrupt_train=False):
    """Use the real splitter with stubbed import/model work to test orchestration."""
    calls = []
    training_attempts = 0

    def execute(command, output):
        nonlocal training_attempts
        calls.append(command)
        module = command[3]
        output.write_text("script output\n", encoding="utf-8")
        work = args.work_dir
        if module.endswith("import_metamon_replays"):
            raw = work / "raw"
            raw.mkdir(parents=True, exist_ok=True)
            for index in range(args.games):
                (raw / f"gen9ou-{index}.json").write_text("{}")
            _json(raw / ".metamon/import.json", _import_metadata(kept_games=args.games))
        elif module.endswith("convert_replay_decisions"):
            assert not (work / "decisions/manifest.parquet").exists()
            directory = work / "decisions/shards"
            directory.mkdir(parents=True, exist_ok=True)
            shards = [directory / f"game-{index}.npz" for index in range(args.games)]
            for path in shards:
                path.write_bytes(b"fixture")
            pl.DataFrame({"id": [f"game-{index}" for index in range(args.games)],
                          "n_rows": [3] * args.games, "shard": [str(p) for p in shards],
                          "uploadtime": list(range(args.games))}).write_parquet(
                              work / "decisions/manifest.parquet")
            _json(work / "decisions/quality_report.json", {
                "status": "complete", "parsed_replays": args.games,
                "source_replays_considered": args.games,
            })
        elif module.endswith("split_replay_manifest"):
            from fog.scripts import split_replay_manifest

            with monkeypatch.context() as context:
                context.setattr(sys, "argv", [module, *command[4:]])
                split_replay_manifest.main()
        elif module.endswith("training.bc"):
            assert "bc.resume=true" in command and "bc.selection_metric=action_ce" in command
            assert "bc.streaming=true" in command and "bc.early_stopping_patience=0" in command
            training_attempts += 1
            training = work / "training"
            checkpoints = training / "checkpoints"
            checkpoints.mkdir(parents=True, exist_ok=True)
            state = checkpoints / "bc_state.pt"
            if training_attempts == 1 and interrupt_train:
                state.write_bytes(b"epoch-one-state")
                (training / "metrics.csv").write_text("epoch,eval_action_ce\n0,2\n1,1.8\n")
                raise subprocess.CalledProcessError(-15, command)
            if interrupt_train:
                assert state.read_bytes() == b"epoch-one-state"
            for name in ("bc_state.pt", "bc_final.pt", "bc_best.pt"):
                (checkpoints / name).write_bytes(b"model")
            (training / "metrics.csv").write_text("epoch,eval_action_ce\n0,2\n1,1.8\n2,1.7\n")
        elif module.endswith("replay_holdout_report"):
            _json(Path(command[command.index("--out") + 1]),
                  {"rows": 6, "policy": {"nll": 1.7}, "value": {"brier": 0.25}})
        else:
            raise AssertionError(command)

    monkeypatch.setattr(runner, "_run_command", execute)
    return calls


def test_runner_resumes_incomplete_training_and_collects_results(tmp_path, monkeypatch):
    args = _args(tmp_path)
    calls = _stub_scripts(monkeypatch, args, interrupt_train=True)
    with pytest.raises(subprocess.CalledProcessError):
        runner.run(args)
    assert (args.work_dir / "training/checkpoints/bc_state.pt").read_bytes() == b"epoch-one-state"
    before = len(calls)
    results = runner.run(args)
    resumed_modules = [command[3] for command in calls[before:]]
    assert resumed_modules == ["fog.training.bc", "fog.eval.replay_holdout_report",
                               "fog.eval.replay_holdout_report"]
    assert {path.name for path in results.iterdir()} == {
        "metrics.csv", "bc_best.pt", "holdout-best.json", "holdout-untrained.json",
        "run-summary.txt", "train-tail.txt",
    }
    assert "training games/rows: (4, 12)" in (results / "run-summary.txt").read_text()
    assert "holdout games/rows: (2, 6)" in (results / "run-summary.txt").read_text()
    before = len(calls)
    assert runner.run(args) == results
    assert len(calls) == before
    state = json.loads((args.work_dir / "run.json").read_text())
    assert all(entry["status"] == "complete" for entry in state["steps"].values())


def test_runner_refuses_missing_fixed_holdout_before_starting_import(tmp_path, monkeypatch):
    args = _args(tmp_path, chronological_holdout_games=None)
    calls = _stub_scripts(monkeypatch, args)
    with pytest.raises(FileNotFoundError, match="fixed holdout IDs are missing"):
        runner.run(args)
    assert calls == []
    assert not args.work_dir.exists()


def test_prepare_only_skips_cuda_check_and_resumes_without_repreparing(tmp_path, monkeypatch):
    args = _args(tmp_path, prepare_only=True, device="cuda")
    calls = _stub_scripts(monkeypatch, args)
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert runner.run(args) == args.work_dir / "split"
    assert [command[3] for command in calls] == [
        "fog.scripts.import_metamon_replays", "fog.scripts.convert_replay_decisions",
        "fog.scripts.split_replay_manifest",
    ]
    assert not (args.work_dir / "training").exists()
    assert not (args.work_dir / "results").exists()
    before = len(calls)
    args.prepare_only = False
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert runner.run(args) == args.work_dir / "results"
    assert [command[3] for command in calls[before:]] == [
        "fog.training.bc", "fog.eval.replay_holdout_report", "fog.eval.replay_holdout_report",
    ]


def test_runner_refuses_changed_training_arguments(tmp_path, monkeypatch):
    args = _args(tmp_path)
    _stub_scripts(monkeypatch, args)
    runner.run(args)
    args.epochs = 3
    with pytest.raises(ValueError, match="arguments changed.*epochs"):
        runner.run(args)


def test_runner_refuses_same_path_recipe_edit(tmp_path, monkeypatch):
    config = tmp_path / "recipe.yaml"
    config.write_text(f"extends: {Path('configs/replay_blueprint_cpu.yaml').resolve()}\n")
    args = _args(tmp_path, config=config)
    _stub_scripts(monkeypatch, args)
    runner.run(args)
    with config.open("a") as stream:
        stream.write("seed: 7\n")
    with pytest.raises(ValueError, match="arguments changed.*resolved_config"):
        runner.run(args)


def test_runner_reports_unrecorded_wall_time_after_hard_interruption(tmp_path, monkeypatch):
    args = _args(tmp_path)
    calls = _stub_scripts(monkeypatch, args)
    runner.run(args)
    state_path = args.work_dir / "run.json"
    state = json.loads(state_path.read_text())
    state["steps"]["train"]["status"] = "running"
    _json(state_path, state)
    before = len(calls)
    result = runner.run(args)
    assert len(calls) == before
    assert "earlier interrupted attempt wall time was not recorded" in (
        result / "run-summary.txt").read_text()


def test_runner_refuses_same_path_holdout_edit(tmp_path, monkeypatch):
    ids = tmp_path / "holdout.txt"
    ids.write_text("game-4\ngame-5\n")
    args = _args(tmp_path, chronological_holdout_games=None, holdout_ids=ids)
    _stub_scripts(monkeypatch, args)
    runner.run(args)
    ids.write_text("game-3\ngame-5\n")
    with pytest.raises(ValueError, match="arguments changed.*holdout_contents"):
        runner.run(args)


@pytest.mark.parametrize("change", [{"revision": "different"},
                                    {"filters": {**runner.IMPORT_FILTERS, "min_rating": 1650}}])
def test_runner_refuses_wrong_preexisting_corpus(tmp_path, monkeypatch, change):
    args = _args(tmp_path)
    _json(args.work_dir / "raw/.metamon/import.json", _import_metadata(**change))
    calls = _stub_scripts(monkeypatch, args)
    with pytest.raises(ValueError, match="existing import revision/filters"):
        runner.run(args)
    assert calls == []


def test_runner_restarts_incomplete_conversion_and_split(tmp_path, monkeypatch):
    args = _args(tmp_path)
    calls = _stub_scripts(monkeypatch, args)
    runner.run(args)
    (args.work_dir / "decisions/quality_report.json").unlink()
    (args.work_dir / "split/holdout_manifest.parquet").unlink()
    before = len(calls)
    runner.run(args)
    assert [command[3] for command in calls[before:]] == [
        "fog.scripts.convert_replay_decisions", "fog.scripts.split_replay_manifest",
    ]


def test_runner_restores_owned_shard_paths_after_moving_work_dir(tmp_path, monkeypatch):
    args = _args(tmp_path)
    calls = _stub_scripts(monkeypatch, args)
    runner.run(args)
    moved = tmp_path / "restored"
    args.work_dir.rename(moved)
    args.work_dir = moved
    before = len(calls)
    runner.run(args)
    assert len(calls) == before
    for manifest in [moved / "decisions/manifest.parquet", *moved.glob("split/*.parquet")]:
        assert all(Path(path).is_file() for path in pl.read_parquet(manifest)["shard"])


def test_runner_progressively_imports_more_shards_and_stops_at_target(tmp_path, monkeypatch):
    args = _args(tmp_path)
    calls = _stub_scripts(monkeypatch, args)
    execute = runner._run_command
    shard_limits = []

    def import_until_second_shard(command, output):
        execute(command, output)
        if command[3].endswith("import_metamon_replays"):
            shards = int(command[command.index("--max-shards") + 1])
            shard_limits.append(shards)
            if shards == 1:
                _json(args.work_dir / "raw/.metamon/import.json", _import_metadata(
                    status="selected_shards_exhausted", kept_games=3))

    monkeypatch.setattr(runner, "_run_command", import_until_second_shard)
    runner.run(args)
    assert shard_limits == [1, 2]
    assert len([command for command in calls if command[3].endswith("training.bc")]) == 1


def test_runner_forwards_local_source_and_records_exhaustion(tmp_path, monkeypatch):
    args = _args(tmp_path, parquet=[tmp_path / "fixture.parquet"])
    calls = _stub_scripts(monkeypatch, args)
    execute = runner._run_command

    def exhausted(command, output):
        execute(command, output)
        if command[3].endswith("import_metamon_replays"):
            _json(args.work_dir / "raw/.metamon/import.json", _import_metadata(
                status="selected_shards_exhausted", kept_games=6))

    monkeypatch.setattr(runner, "_run_command", exhausted)
    runner.run(args)
    assert "--parquet" in calls[0]
    assert str(args.parquet[0]) in calls[0]
    before = len(calls)
    runner.run(args)
    assert len(calls) == before


def test_runner_rejects_unavailable_requested_gpu(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="cannot use a CUDA GPU"):
        runner.run(_args(tmp_path, device="cuda"))


@pytest.mark.skipif(os.name != "posix", reason="interruption test sends a POSIX termination signal")
def test_runner_end_to_end_offline_interrupts_and_resumes(tmp_path):
    """Real importer, converter, trainer, and scorer, using generated protocol records."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    log = """|gametype|singles
|player|p1|Alice|1|1800
|player|p2|Bob|2|1800
|teamsize|p1|1
|teamsize|p2|1
|gen|9
|tier|[Gen 9] OU
|clearpoke
|poke|p1|Garchomp|
|poke|p2|Corviknight|
|teampreview
|start
|switch|p1a: Chomp|Garchomp|100/100
|switch|p2a: Bird|Corviknight|100/100
|turn|1
|move|p1a: Chomp|Earthquake|p2a: Bird
|-immune|p2a: Bird
|move|p2a: Bird|Roost|p2a: Bird
|turn|2
|move|p1a: Chomp|Earthquake|p2a: Bird
|-immune|p2a: Bird
|move|p2a: Bird|Roost|p2a: Bird
|win|Alice
"""
    source = tmp_path / "synthetic.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"id": f"gen9ou-fixture-{index}", "formatid": "gen9ou", "rating": "1800",
         "uploadtime": 1_707_000_000 + index, "players": ["Alice", "Bob"], "log": log}
        for index in range(8)
    ]), source)
    work = tmp_path / "pipeline"
    command = [sys.executable, "-u", "-m", runner.MODULE, "--games", "8",
               "--work-dir", str(work), "--device", "cpu", "--batch-size", "8",
               "--epochs", "2", "--workers", "1", "--chronological-holdout-games", "2",
               "--parquet", str(source)]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    state_path = work / "training/checkpoints/bc_state.pt"
    with (tmp_path / "attempts.txt").open("w") as output:
        child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, env=env)
        try:
            deadline = time.monotonic() + 60
            while not state_path.exists():
                assert child.poll() is None, (tmp_path / "attempts.txt").read_text()
                assert time.monotonic() < deadline, "runner did not checkpoint within 60 seconds"
                time.sleep(0.01)
            child.send_signal(signal.SIGTERM)
            assert child.wait(timeout=15) == 143
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=15)
        first = torch.load(state_path, map_location="cpu", weights_only=False)
        assert first["epoch"] == 1
        assert not (work / "training/checkpoints/bc_final.pt").exists()
        completed = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT,
                                   env=env, timeout=60)
    assert completed.returncode == 0, (tmp_path / "attempts.txt").read_text()
    resumed = torch.load(state_path, map_location="cpu", weights_only=False)
    assert resumed["epoch"] == 2
    assert [row["epoch"] for row in resumed["history"]] == [0, 1, 2]
    assert "resume: completed epoch 1/2" in (work / "output/train.txt").read_text()
    assert runner._report_complete(work / "results/holdout-best.json")
    assert runner._report_complete(work / "results/holdout-untrained.json")
    assert runner._manifest_counts(work / "split/train_manifest.parquet")[0] == 6
    assert runner._manifest_counts(work / "split/holdout_manifest.parquet")[0] == 2
