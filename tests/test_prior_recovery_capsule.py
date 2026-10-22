"""Frozen CPU diagnostic inputs cannot change identities or escape their archive."""

import hashlib
import json
import zipfile

import pytest

from copy import deepcopy

from fog.scripts.compare_prior_recovery import compare_probability_probes
from fog.scripts.prior_recovery_kaggle_worker import extract_bundle


def capsule(tmp_path, *, name="input.txt", wrong_hash=False, device="cpu"):
    content = b"fixed input"
    descriptor = {"format": "fog.prior-recovery-experiment", "version": 1,
                  "device": device, "stage": "evaluate", "files": {
                      name: "0" * 64 if wrong_hash else hashlib.sha256(content).hexdigest()}}
    target = tmp_path / "bundle.bin"
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr("experiment.json", json.dumps(descriptor))
        archive.writestr(name, content)
    return target, descriptor


def test_exact_input_inventory_and_hash_are_verified(tmp_path):
    bundle, descriptor = capsule(tmp_path)
    assert extract_bundle(bundle, tmp_path / "work") == descriptor
    assert (tmp_path / "work/input.txt").read_bytes() == b"fixed input"


@pytest.mark.parametrize("settings", [
    {"name": "../escape.txt"}, {"wrong_hash": True}, {"device": "cuda"},
])
def test_rejects_unsafe_corrupt_or_accelerated_capsules(tmp_path, settings):
    bundle, _ = capsule(tmp_path, **settings)
    with pytest.raises(ValueError):
        extract_bundle(bundle, tmp_path / "work")
    assert not (tmp_path / "escape.txt").exists()


def test_cpu_probe_requires_finite_matched_coverage_and_actual_agreement():
    reference = {"torch": "2.13.0", "contexts": 1,
                 "probabilities": {"candidate": {"item": [[0.2, 0.8]]}}}
    actual = deepcopy(reference)
    actual["torch"] = "2.11.0"
    proof = compare_probability_probes(reference, actual)
    assert proof["passed"] and not proof["actual_CUDA_test"]
    for bad in [float("nan"), float("inf"), 0.1]:
        actual["probabilities"]["candidate"]["item"][0][0] = bad
        with pytest.raises(ValueError):
            compare_probability_probes(reference, actual)
    actual = deepcopy(reference)
    actual["probabilities"]["candidate"]["item"] = [0.2, 0.8]
    with pytest.raises(ValueError, match="shape"):
        compare_probability_probes(reference, actual)
    actual["probabilities"] = {}
    with pytest.raises(ValueError, match="conditions"):
        compare_probability_probes(reference, actual)
