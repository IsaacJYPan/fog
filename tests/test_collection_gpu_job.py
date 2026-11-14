"""The measurement job must bootstrap on an empty hosted temporary filesystem."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


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
