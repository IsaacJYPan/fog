import hashlib
import json
import zipfile

import pytest

from fog.scripts.curriculum_evaluation import resolve


def fixture(tmp_path, *, steps=480, error=None):
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
