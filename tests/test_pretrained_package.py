import pytest

from fog.pretrained.kakuna import sha256
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
