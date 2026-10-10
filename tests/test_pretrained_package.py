import pytest
import json

from fog.pretrained.kakuna import sha256
from fog.scripts import pretrained_experiment as workflow
from fog.scripts.pretrained_experiment import verify_published_records, write_notebook_runner


def test_changed_replay_invalidates_previously_audited_manifest(tmp_path):
    replay = tmp_path / "game.json.lz4"
    replay.write_bytes(b"frozen replay")
    manifest = {"records": [{"path": replay.name, "sha256": sha256(replay)}]}
    verify_published_records(tmp_path, manifest)
    replay.write_bytes(b"changed replay")
    with pytest.raises(ValueError, match="Audited published replay changed"):
        verify_published_records(tmp_path, manifest)


def test_notebook_watchdog_is_bound_before_reading_inputs(tmp_path):
    script = tmp_path / "run.py"
    write_notebook_runner(script, 1800)
    code = script.read_text()
    compile(code, str(script), "exec")
    assert "FROZEN_TIMEOUT_SECONDS = 1800" in code
    assert "timeout=FROZEN_TIMEOUT_SECONDS" in code


def test_explicit_cpu_preflight_never_reserves_or_requests_gpu(tmp_path, monkeypatch):
    folder = tmp_path / "cpu-smoke"
    folder.mkdir()
    (folder / "experiment.json").write_text(
        json.dumps(
            {
                "cpu_preflight": True,
                "stage": "train",
                "timeout_seconds": 1800,
            }
        )
    )
    (folder / "state.json").write_text(
        json.dumps(
            {
                "stage": "train",
                "phase": "uploaded",
                "kernel": "owner/cpu-smoke",
                "dataset": "owner/cpu-smoke",
            }
        )
    )
    monkeypatch.setattr(workflow, "verify", lambda _: None)
    monkeypatch.setattr(workflow, "handoff", lambda *_: None)

    def no_reservation(*args, **kwargs):
        pytest.fail("CPU preflight attempted to reserve GPU hours")

    monkeypatch.setattr(workflow, "reserve_gpu", no_reservation)
    commands = []

    def cli(*args):
        commands.append(args)
        if args[:2] == ("datasets", "status"):
            return json.dumps({"status": "ready"})
        assert args[:2] == ("kernels", "push")
        assert "--accelerator" not in args
        return "Kernel version 1 successfully pushed."

    monkeypatch.setattr(workflow, "cli", cli)
    workflow.submit(folder)
    assert len(commands) == 2
    assert json.loads((folder / "state.json").read_text())["phase"] == "submitted"
