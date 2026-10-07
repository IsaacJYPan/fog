import json

import pytest

from fog.pretrained.kakuna import KAKUNA_34_SHA256
from fog.scripts.pretrained_compare import compare_evaluations, matched_reports


def test_same_update_count_with_different_data_schedule_is_rejected(tmp_path):
    experiments, results = [], []
    for condition in ("ordinary", "joint"):
        experiment = {
            "condition": condition,
            "phase": "pilot",
            "matched_recipe_sha256": "recipe",
            "matched_inputs_sha256": "inputs",
            "timeout_seconds": 3600,
            "updates": 1,
        }
        path = tmp_path / (condition + ".json")
        path.write_text(json.dumps(experiment))
        experiments.append(path)
        root = tmp_path / condition
        (root / "training").mkdir(parents=True)
        (root / "experiment.json").write_text(json.dumps(experiment))
        (root / "training/RESULTS.json").write_text(json.dumps({"completed_updates": 1}))
        (root / "training/native_000001.pt").write_bytes(b"model")
        (root / "training/metrics.jsonl").write_text(
            json.dumps(
                {
                    "update": 1,
                    "seconds": 2,
                    "schedule_sha256": condition,
                    "counts": {"positions": 40},
                }
            )
            + "\n"
        )
        results.append(root)
    with pytest.raises(ValueError, match="different data"):
        matched_reports(*experiments, *results)


def test_development_contrast_rejects_changed_inference_settings(tmp_path):
    paths = []
    for condition in ("ordinary", "joint"):
        protocol = {
            "stage": "development",
            "foulplay": False,
            "original_sha256": KAKUNA_34_SHA256,
            "candidate_sha256": condition,
            "pool_sha256": "dev",
            "interface": "same",
            "sources_sha256": "code",
            "seed": 4,
            "temperature": 1 if condition == "ordinary" else 0,
            "deadline_seconds": 5,
        }
        path = tmp_path / (condition + ".json")
        path.write_text(
            json.dumps({"protocol": protocol, "failures": 0, "unresolved": 0, "rows": []})
        )
        paths.append(path)
    with pytest.raises(ValueError, match="temperature"):
        compare_evaluations({"models": ["ordinary", "joint"]}, *paths)
