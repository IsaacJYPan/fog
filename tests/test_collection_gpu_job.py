"""The measurement job must bootstrap on an empty hosted temporary filesystem."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch


def test_gpu_job_creates_missing_temporary_root(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "docs/briefs/06-artifacts/gpu_job.py"
    spec = importlib.util.spec_from_file_location("gpu_job", source)
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)
    root = tmp_path / "missing-temp/fog"
    output = tmp_path / "output"

    def mapped_path(value):
        return {"/kaggle/temp/fog": root, "/kaggle/working/06-artifacts": output}.get(
            str(value), Path(value),
        )

    class CompletedProcess:
        def wait(self, timeout=None):
            return 0

        def terminate(self):
            pass

    calls = []

    def popen(command, *, cwd=None, stdout=None, **kwargs):
        calls.append(command)
        if cwd is not None and not Path(cwd).is_dir():
            raise FileNotFoundError(cwd)
        if command[:2] == ["git", "clone"]:
            root.mkdir()
        if "fog.eval.showdown_rollout_probe" in command:
            stdout.write("workers=2 batch=32 device=cuda opponent=self games=512 "
                         "learner_decisions=1 seconds=1.00 learner_decisions/s=1.00 "
                         "games/s=512.000 p1_wins=0\n")
        return CompletedProcess()

    monkeypatch.setattr(job, "Path", mapped_path)
    monkeypatch.setattr(job.subprocess, "Popen", popen)
    monkeypatch.setattr(job.subprocess, "check_output", lambda *args, **kwargs: "GPU, 16 GiB\n")
    monkeypatch.setenv("FOG_COMMIT", "0" * 40)
    monkeypatch.setenv("FOG_STAGE", "collection")
    monkeypatch.delenv("FOG_PLATFORM_TIMEOUT", raising=False)
    job.main()
    assert root.is_dir()
    assert len([command for command in calls if "fog.eval.showdown_rollout_probe" in command]) == 16
    assert json.loads((output / "budget.json").read_text())["status"] == "complete"


def test_timing_hooks_preserve_native_trainer_phase_metrics(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from fog.training import ppo
    from fog.training.dashboard import TrainingDashboard
    from fog.training.direct_parallel import PersistentProcessPool

    source = Path(__file__).resolve().parents[1] / "docs/briefs/06-artifacts/training_timing.py"
    spec = importlib.util.spec_from_file_location("training_timing", source)
    timing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(timing)
    result = [(SimpleNamespace(_games={0: SimpleNamespace(actions=[0, 1])}), 1)]
    monkeypatch.setattr(PersistentProcessPool, "map", lambda *_a, **_kw: result)
    monkeypatch.setattr(ppo, "ppo_update", lambda *_a, **_kw: {
        "optimizer_steps": 1, "optimizer_rows": 4,
    })
    monkeypatch.setattr(ppo, "_maybe_chunk_update", lambda *_a, **_kw: None)
    monkeypatch.setattr(TrainingDashboard, "log", lambda _self, _step, metrics: dict(metrics))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.delenv("FOG_TIMING_OUTPUT", raising=False)
    timing.install(tmp_path / "events.jsonl")
    PersistentProcessPool.map(None, [])
    agent = torch.nn.Linear(1, 1)
    ppo.ppo_update(agent, None, {"actions": torch.zeros(4)}, {
        "ppo": {"minibatch_size": 4},
    })
    native = {"collection_seconds": 101.0, "update_seconds": 202.0,
              "optimization_seconds": 303.0, "collection_decisions_per_sec": 4.0,
              "optimization_rows_per_sec": 5.0}
    logged = TrainingDashboard.log(None, 4, dict(native))
    assert all(logged[key] == value for key, value in native.items())
    assert logged["timing_collection_rows"] == 2
    assert logged["timing_optimized_rows"] == 4
    assert "timing_optimizer_seconds" in logged


def test_measurement_resume_discards_only_work_after_saved_state(tmp_path):
    import csv

    source = Path(__file__).resolve().parents[1] / "docs/briefs/06-artifacts/gpu_job.py"
    spec = importlib.util.spec_from_file_location("gpu_job", source)
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)
    events = [{"kind": "collection", "rows": 4}, {"kind": "update", "rows": 4},
              {"kind": "update_phase", "rows": 4}, {"kind": "collection", "rows": 6},
              {"kind": "update", "rows": 6}, {"kind": "update_phase", "rows": 6}]
    prefix = "".join(json.dumps(row) + "\n" for row in events[:3])
    (tmp_path / "training-timing.jsonl").write_text(
        prefix + "".join(json.dumps(row) + "\n" for row in events[3:]),
    )
    (tmp_path / "training").mkdir()
    metrics = tmp_path / "training/metrics.csv"
    metrics.write_text("step,decisions,loss\n4,4,1.0\n10,10,0.5\n")
    job.trim_measurements(tmp_path, 4)
    assert (tmp_path / "training-timing.jsonl").read_text() == prefix
    assert list(csv.DictReader(metrics.open())) == [{"step": "4", "decisions": "4", "loss": "1.0"}]
    assert len((tmp_path / "timing-before-resume.jsonl").read_text().splitlines()) == 6
