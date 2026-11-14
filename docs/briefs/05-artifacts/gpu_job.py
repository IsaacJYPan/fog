"""Private replay training job. Set SPEC in the submitted copy; prepare data on CPU first."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from collections import deque
from pathlib import Path

# Replace only this literal in each submitted copy and retain that copy as a text artifact.
SPEC = {"commit": "SET_TO_PUSHED_CODE_COMMIT", "run": "G3", "games": 3000,
        "epochs": 18, "batch_size": 1024, "stop_after_epoch": 2}
ROOT = Path("/kaggle/working/fog")
WORK = Path("/kaggle/working/replay-work")
OUTPUT = Path("/kaggle/working/results")


def command(*args, **kwargs):
    subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def restore_prepared(input_root: Path, work: Path):
    """Kaggle datasets expand uploaded archives; kernel outputs may retain them."""
    archives = list(input_root.rglob("replay-work.tar.gz"))
    if archives:
        assert len(archives) == 1, "expected one prepared work archive"
        work.mkdir(exist_ok=True)
        with tarfile.open(archives[0]) as archive:
            archive.extractall(work, filter="data")
        return
    roots = [path.parent for path in input_root.rglob("run.json")
             if (path.parent / "decisions/manifest.parquet").is_file()
             and (path.parent / "split/holdout_manifest.parquet").is_file()]
    assert len(roots) == 1, f"expected one expanded prepared dataset, found {len(roots)}"
    shutil.copytree(roots[0], work, dirs_exist_ok=True)


def worker():
    import polars as pl
    import torch

    from fog.config import load_config_with_args
    from fog.scripts.run_replay_blueprint import _relocate_manifests
    from fog.training import bc

    assert torch.cuda.is_available(), "CUDA GPU required"
    state = json.loads((WORK / "run.json").read_text())
    _relocate_manifests(WORK, state)
    (WORK / "run.json").write_text(json.dumps(state, indent=2))
    train = WORK / "split" / f"train_{SPEC['games']}.parquet"
    holdout = WORK / "split/holdout_manifest.parquet"
    assert train.is_file(), f"prepare nested training manifest on CPU: {train.name}"
    expected = set((ROOT / "data/replay_splits/p2_holdout_2000_ids.txt").read_text().split())
    h = pl.read_parquet(holdout)
    assert set(h["id"].cast(pl.String)) == expected
    assert len(h) == 2000 and h["n_rows"].sum() == 104037
    training = WORK / SPEC["run"]
    overrides = [
        "device=cuda", f"bc.manifest={train}", f"bc.holdout_manifest={holdout}",
        f"bc.batch_size={SPEC['batch_size']}", f"bc.epochs={SPEC['epochs']}",
        "bc.resume=true", "bc.selection_metric=action_ce", "bc.keep_best_value=true",
        "bc.streaming=true", "bc.early_stopping_patience=0", f"log.dir={training}",
    ]
    cfg = load_config_with_args(ROOT / "configs/replay_blueprint_cpu.yaml", overrides)
    timings = []
    original_eval, original_save = bc._fixed_eval, bc._atomic_save

    def timed_eval(*args, **kwargs):
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = original_eval(*args, **kwargs)
        torch.cuda.synchronize()
        timings.append({"seconds": time.perf_counter() - started, "metrics": result})
        return result

    def stop_after_save(state, path):
        original_save(state, path)
        if state["epoch"] == SPEC.get("stop_after_epoch"):
            raise InterruptedError("deliberate completed-epoch stop for cross-job continuation")

    bc._fixed_eval, bc._atomic_save = timed_eval, stop_after_save
    OUTPUT.mkdir(exist_ok=True)
    metrics = training / "metrics.csv"
    if metrics.exists():
        shutil.copy2(metrics, OUTPUT / "metrics-before.csv")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    status = "failed"
    try:
        bc.train(cfg)
        status = "complete"
    except InterruptedError:
        status = "deliberately_interrupted"
    finally:
        torch.cuda.synchronize()
        runtime = {"spec": SPEC, "status": status, "torch": torch.__version__,
                   "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0),
                   "visible_devices": torch.cuda.device_count(),
                   "training_wall_seconds": time.perf_counter() - started,
                   "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
                   "validation_passes": timings,
                   "command": ["python", "-m", "fog.training.bc", "--config",
                               "configs/replay_blueprint_cpu.yaml", "--set", *overrides]}
        (OUTPUT / "training-runtime.json").write_text(json.dumps(runtime, indent=2))
        if metrics.exists():
            shutil.copy2(metrics, OUTPUT / "metrics-after.csv")
    if status == "complete":
        for name in ("bc_best", "bc_best_value"):
            command(sys.executable, "-m", "fog.eval.replay_holdout_report", "--config",
                    ROOT / "configs/replay_blueprint_cpu.yaml", "--manifest", holdout,
                    "--checkpoint", training / f"checkpoints/{name}.pt", "--out",
                    OUTPUT / f"holdout-{name}.json", "--batch-size", SPEC["batch_size"],
                    "--set", "device=cuda", cwd=ROOT)


def main():
    started = time.perf_counter()
    OUTPUT.mkdir(exist_ok=True)
    install_seconds = None
    status = "failed"
    try:
        command("git", "clone", "https://github.com/IsaacJYPan/fog", ROOT)
        command("git", "checkout", SPEC["commit"], cwd=ROOT)
        command("git", "submodule", "update", "--init", "--recursive", cwd=ROOT)
        command("npm", "--prefix", ROOT / "server/pokemon-showdown", "ci", "--ignore-scripts")
        command("npm", "--prefix", ROOT / "server/pokemon-showdown", "run", "build")
        command(sys.executable, "-m", "fog.scripts.kaggle_setup", cwd=ROOT)
        install_seconds = time.perf_counter() - started
        restore_prepared(Path("/kaggle/input"), WORK)
        # A continuation may attach the preceding private job's saved work archive.
        resumes = list(Path("/kaggle/input").rglob("replay-resume.tar.gz"))
        assert len(resumes) <= 1
        if resumes:
            with tarfile.open(resumes[0]) as archive:
                archive.extractall(WORK, filter="data")
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        with (OUTPUT / "train.txt").open("w") as log:
            command(ROOT / ".venv/bin/python", Path(__file__).resolve(), "--worker",
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
        status = "returned"
    finally:
        elapsed = time.perf_counter() - started
        (OUTPUT / "job-runtime.json").write_text(json.dumps({
            "spec": SPEC, "status": status, "install_seconds": install_seconds,
            "job_wall_seconds_before_export": elapsed,
            "note": "Add export and platform session overhead from the returned session status."
        }, indent=2))
        if WORK.exists():
            with tarfile.open("/kaggle/working/replay-resume.tar.gz", "w:gz") as archive:
                for path in WORK.iterdir():
                    archive.add(path, arcname=path.name)
        log = OUTPUT / "train.txt"
        if log.exists():
            with log.open() as stream:
                (OUTPUT / "tail.txt").write_text("".join(deque(stream, maxlen=60)))
        # Kaggle exports every file in its working directory. Retain the resume archive
        # and text results, rather than exporting a second data copy and the installed venv.
        if Path("/kaggle/working/replay-resume.tar.gz").is_file():
            shutil.rmtree(WORK)
            shutil.rmtree(ROOT)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        worker()
    else:
        main()
