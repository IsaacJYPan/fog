import hashlib
import json
import zipfile

import pytest

from fog.scripts.teacher_pilot_queue import resolve_training


def source():
    return {"minimum_updates": 500, "experiment": {"format": "fog.teacher-pilot-experiment", "version": 1,
            "id": "fog-teacher-train-test", "stage": "train", "device": "cuda",
            "data_sha256": "data", "initial_sha256": "initial"}}


def archive(folder, *, bad_hash=False, gpu_verified=True):
    settings = source()
    checkpoint, state = b"portable weights", b"full training state"
    contract = {"passed": True, "cpu_cuda_verified": gpu_verified, "cpu_cuda_max_error": 1e-5, "tolerance": 1e-3}
    report = {"complete": True, "device": "cuda", "updates": 500,
              "checkpoint_sha256": hashlib.sha256(checkpoint).hexdigest(),
              "full_state_sha256": hashlib.sha256(state).hexdigest(),
              "starting_contract": contract, "final_contract": contract,
              "data_sha256": "data", "initial_sha256": "initial"}
    items = {"experiment.json": json.dumps(settings["experiment"]).encode(),
             "pipeline_state.json": json.dumps({"status": "complete", "error": None, "returncode": 0}).encode(),
             "training/RESULTS.json": json.dumps(report).encode(), "training/latest.pt": checkpoint,
             "training/latest.state.pt": state}
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in items.items()}
    if bad_hash:
        hashes["training/latest.pt"] = "0" * 64
    folder.mkdir()
    with zipfile.ZipFile(folder / "fog-experiment-output.zip", "w") as zipped:
        for name, value in items.items():
            zipped.writestr(name, value)
        zipped.writestr("output_hashes.json", json.dumps(hashes))
    return settings


def test_missing_gpu_output_waits_without_creating_evaluation(tmp_path):
    assert resolve_training(source(), tmp_path, tmp_path / "result") is None
    assert not (tmp_path / "result").exists()


def test_fixed_verified_output_resolves_once(tmp_path):
    settings = archive(tmp_path / "input")
    result = resolve_training(settings, tmp_path / "input", tmp_path / "result")
    assert result["updates"] == 500 and result["verified_exported_files"] == 5
    assert (tmp_path / "result/training/latest.state.pt").read_bytes() == b"full training state"
    with pytest.raises(FileExistsError):
        resolve_training(settings, tmp_path / "input", tmp_path / "result")


@pytest.mark.parametrize("changes,match", [({"bad_hash": True}, "hash mismatch"), ({"gpu_verified": False}, "probability contract")])
def test_bad_source_cannot_start_cpu_play(tmp_path, changes, match):
    settings = archive(tmp_path / "input", **changes)
    with pytest.raises(ValueError, match=match):
        resolve_training(settings, tmp_path / "input", tmp_path / "result")
    assert not (tmp_path / "result").exists()


def test_old_or_non_cuda_source_is_rejected(tmp_path):
    settings = source()
    settings["experiment"]["device"] = "cpu"
    with pytest.raises(ValueError, match="fixed CUDA"):
        resolve_training(settings, tmp_path, tmp_path / "result")
