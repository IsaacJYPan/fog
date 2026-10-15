import hashlib
import json
import zipfile

import pytest

from fog.scripts.curriculum_evaluation import fixed_checkpoints, resolve


def fixture(tmp_path, *, steps=480, error=None, continuation=False, anchor="anchor", cuda_resume=True):
    expected = {"format": "fog.curriculum-pilot", "version": 1, "device": "cuda", "stage": "train",
                "id": "fixed", "max_batches": 60, "optimized_positions": 30720,
                "optimizer_steps": 480, "condition": "backward", "initial_sha256": "start"}
    source = {"experiment": expected, "expected_training_identity": "recipe"}
    contract = {"passed": True, "cpu_cuda_verified": True, "tolerance": .001, "cpu_cuda_max_error": .0001}
    contents = {"training/latest.pt": b"weights", "training/latest.state.pt": b"full state"}
    sha = lambda value: hashlib.sha256(value).hexdigest()  # noqa: E731
    report = {"complete": True, "device": "cuda", "batches": 60, "optimized_positions": 30720,
              "optimizer_steps": steps, "condition": "backward", "identity": "recipe",
              "starting_contract": contract, "final_contract": dict(contract, cpu_cuda_max_error=error or .0001),
              "checkpoint_sha256": sha(contents["training/latest.pt"]), "full_state_sha256": sha(contents["training/latest.state.pt"])}
    if continuation:
        expected.update(parent_state_sha256="parent", total_optimizer_steps=484, total_optimized_positions=30728,
                        original_anchor_sha256="anchor", actor_device="cpu")
        report.update(lineage={"parent_state_sha256": "parent"}, total_optimizer_steps=484,
                      total_optimized_positions=30728, anchor_sha256=anchor, actor_device="cpu")
        contents["ACTOR_PREFLIGHT.json"] = json.dumps({"passed": True, "actor_real_collection_passed": True,
            "cuda_full_state_resume_verified": cuda_resume, "cuda_resume_checks": {"weights": cuda_resume},
            "cpu_cuda_max_error": .0001, "actor_device": "cpu", "selected_best_mean_time_ratio": 1.0}).encode()
    contents.update({"experiment.json": json.dumps(expected).encode(), "training/RESULTS.json": json.dumps(report).encode(),
                     "pipeline_state.json": json.dumps({"status": "complete", "error": None, "returncode": 0}).encode()})
    inventory = {name: sha(value) for name, value in contents.items()}
    with zipfile.ZipFile(tmp_path / "fog-experiment-output.zip", "w") as archive:
        for name, value in contents.items():
            archive.writestr(name, value)
        archive.writestr("output_hashes.json", json.dumps(inventory))
    return source


def test_cpu_evaluation_verifies_complete_matched_exposure_and_extracts_only_needed_model(tmp_path):
    source = fixture(tmp_path)
    path = resolve(source, tmp_path, tmp_path / "verified")
    assert path.read_bytes() == b"weights"
    assert not (path.parent / "latest.state.pt").exists()
    assert (tmp_path / "verified/SOURCE_RECEIPT.json").is_file()


def test_cpu_evaluation_waits_without_attached_output(tmp_path):
    source = {"experiment": {"format": "fog.curriculum-pilot", "version": 1, "device": "cuda", "stage": "train"}}
    assert resolve(source, tmp_path, tmp_path / "out") is None


@pytest.mark.parametrize("settings", [{"steps": 479}, {"error": .002}])
def test_cpu_evaluation_rejects_unequal_exposure_or_failed_probability_contract(tmp_path, settings):
    source = fixture(tmp_path, **settings)
    with pytest.raises(ValueError):
        resolve(source, tmp_path, tmp_path / "out")


def test_continuation_accepts_only_complete_verified_cuda_resume_and_original_anchor(tmp_path):
    source = fixture(tmp_path, continuation=True)
    assert resolve(source, tmp_path, tmp_path/"out").is_file()


@pytest.mark.parametrize("settings", [{"anchor": "changed"}, {"cuda_resume": False}])
def test_continuation_rejects_changed_anchor_or_unverified_cuda_resume(tmp_path, settings):
    source = fixture(tmp_path, continuation=True, **settings)
    with pytest.raises(ValueError):
        resolve(source, tmp_path, tmp_path/"out")


def test_fixed_controls_are_individually_hash_verified(tmp_path):
    path = tmp_path/"control.pt"
    path.write_bytes(b"weights")
    sha = hashlib.sha256(b"weights").hexdigest()
    descriptor = {"initial_sha256": sha, "files": {"control.pt": sha},
                  "fixed_checkpoints": {"starting": {"path": "control.pt", "sha256": sha}}}
    assert fixed_checkpoints(descriptor, tmp_path) == {"starting": path}
    path.write_bytes(b"different weights")
    with pytest.raises(ValueError, match="hash differs"):
        fixed_checkpoints(descriptor, tmp_path)


def test_fixed_control_cannot_write_an_external_output_directory(tmp_path):
    descriptor = {"initial_sha256": "sha", "files": {},
                  "fixed_checkpoints": {"../escape": {"path": "control.pt", "sha256": "sha"}}}
    with pytest.raises(ValueError, match="output label"):
        fixed_checkpoints(descriptor, tmp_path)
