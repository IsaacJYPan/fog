from copy import deepcopy
import hashlib
import json
import zipfile

import pytest

from fog.pretrained.kakuna import KAKUNA_34_SHA256, sha256
from fog.scripts import pretrained_evaluation_queue as queue
from fog.scripts.pretrained_kaggle_worker import export_outputs
from fog.scripts import pretrained_kaggle_worker as worker


def make_sources(root, *, outputs=True, change=None):
    fixture = root / "inputs/contract/inputs.pt"
    fixture.parent.mkdir(parents=True)
    fixture.write_bytes(b"fixture")
    recipe = {
        "version": 1,
        "stage": "development",
        "games": 256,
        "smoke_games": 8,
        "seed": 12,
        "training_sources": {},
    }
    for arm in queue.ARMS:
        config = {"condition": arm, "updates": 2}
        paired = {key: value for key, value in config.items() if key != "condition"}
        experiment = {
            "id": "train-" + arm,
            "stage": "train",
            "condition": arm,
            "device": "cuda",
            "phase": "pilot",
            "updates": 2,
            "timeout_seconds": 3600,
            "matched_recipe_sha256": hashlib.sha256(
                json.dumps(paired, sort_keys=True).encode()
            ).hexdigest(),
            "matched_inputs_sha256": "same-inputs",
            "files": {"inputs/contract/inputs.pt": sha256(fixture)},
        }
        descriptor = root / (arm + "-experiment.json")
        descriptor.write_text(json.dumps(experiment))
        recipe["training_sources"][arm] = {
            "experiment": descriptor.name,
            "experiment_sha256": sha256(descriptor),
            "kernel": "owner/train-" + arm + "-run",
        }
        if not outputs:
            continue
        result = root / "mounted" / arm / "results"
        (result / "training").mkdir(parents=True)
        values = {
            "experiment.json": experiment,
            "pipeline_state.json": {"status": "complete"},
            "training/recipe.json": config,
            "training/RESULTS.json": {
                "complete": True,
                "completed_updates": 2,
                "planned_updates": 2,
                "condition": arm,
                "promotion": False,
                "identity": hashlib.sha256(
                    json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            },
        }
        native = (arm + "-weights").encode()
        native_sha = hashlib.sha256(native).hexdigest()
        for name, checkpoint_sha in (
            ("contract/contract.json", KAKUNA_34_SHA256),
            ("contract-trained/cpu/contract.json", native_sha),
            ("contract-trained/cuda/contract.json", native_sha),
        ):
            values[name] = {
                "passed": True,
                "weights_unchanged": True,
                "checkpoint_sha256": checkpoint_sha,
                "samples_sha256": sha256(fixture),
                "tolerance": 1e-3,
                "cpu_cuda_verified": True,
                "records": [
                    {"cache_max_absolute_error": 1e-5, "cpu_cuda_max_absolute_error": 1e-5}
                ],
            }
        if change:
            change(values, arm)
        for name, value in values.items():
            path = result / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
        for name in ("native_final.pt", "native_000002.pt"):
            (result / "training" / name).write_bytes(native)
        (result / "training/auxiliary_000002.pt").write_bytes(b"heads")
        (result / "training/latest.state.pt").write_bytes(b"optimizer bytes")
        (result / "contract-trained/cpu/probabilities.pt").write_bytes(b"probabilities")
        rows = [
            {
                "update": i,
                "schedule_sha256": str(i),
                "counts": {"positions": 10},
                "seconds": 0.1,
                "unclipped_gradient_norm": 0.5,
                "losses": {"primary": 1.0},
            }
            for i in (1, 2)
        ]
        (result / "training/metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
        export_outputs(result, result.parent / "fog-experiment-output.zip")
    return recipe


def test_missing_training_output_waits_without_resolving_candidates(tmp_path):
    recipe = make_sources(tmp_path, outputs=False)
    assert queue.resolve_queue(recipe, tmp_path / "mounted", tmp_path, tmp_path / "output") is None
    state = json.loads((tmp_path / "output/queue/STATE.json").read_text())
    assert state["status"] == "waiting"
    assert state["missing_experiments"] == ["train-joint", "train-ordinary"]
    assert not (tmp_path / "evaluation.resolved.json").exists()


def test_valid_pair_resolves_hashes_and_equal_schedules_without_optimizer_copies(tmp_path):
    recipe = make_sources(tmp_path)
    result = queue.resolve_queue(recipe, tmp_path / "mounted", tmp_path, tmp_path / "output")
    assert result["matched"]["updates"] == 2
    assert "training_sources" not in result
    for row in result["conditions"].values():
        assert sha256(tmp_path / row["native"]) == row["native_sha256"]
    assert not list((tmp_path / "inputs/resolved").rglob("latest.state.pt"))
    audit = json.loads((tmp_path / "output/queue/RESOLVED.json").read_text())
    assert all(row["all_export_hashes_verified"] for row in audit["training_sources"].values())


def test_queue_pilot_waits_for_cloud_activation_before_reading_large_exports(tmp_path, monkeypatch):
    recipe = make_sources(tmp_path)
    recipe.update(
        pilot_only=True,
        trigger_sources=["owner/control"],
        activation_marker={"filename": "activation.json", "value": "a" * 32},
    )
    monkeypatch.setattr(
        queue, "verify_source", lambda *_args, **_kwargs: pytest.fail("Read early exports")
    )
    assert queue.resolve_queue(recipe, tmp_path / "mounted", tmp_path, tmp_path / "output") is None
    assert (
        json.loads((tmp_path / "output/queue/STATE.json").read_text())["awaiting_activation"]
        == "activation.json"
    )


@pytest.mark.parametrize("failure", ["incomplete", "wrong_identity", "cuda_contract"])
def test_failed_or_wrong_gpu_output_never_reaches_evaluation(tmp_path, failure):
    def change(values, arm):
        if arm != "joint":
            return
        if failure == "incomplete":
            values["training/RESULTS.json"]["completed_updates"] = 1
        elif failure == "wrong_identity":
            values["experiment.json"] = dict(values["experiment.json"], id="another-run")
        else:
            values["contract-trained/cuda/contract.json"]["records"][0][
                "cpu_cuda_max_absolute_error"
            ] = 0.01

    recipe = make_sources(tmp_path, change=change)
    if failure == "wrong_identity":
        # A wrong frozen descriptor in an attached source must fail, not look absent.
        expected = json.loads((tmp_path / "joint-experiment.json").read_text())
        with pytest.raises(ValueError, match="different frozen experiment"):
            queue.verify_source(
                tmp_path / "mounted/joint/fog-experiment-output.zip", expected, tmp_path / "bad"
            )
    else:
        with pytest.raises(ValueError):
            queue.resolve_queue(recipe, tmp_path / "mounted", tmp_path, tmp_path / "output")
    assert not (tmp_path / "evaluation.resolved.json").exists()


def test_corrupted_member_even_unused_optimizer_is_rejected(tmp_path):
    recipe = make_sources(tmp_path)
    path = tmp_path / "mounted/joint/fog-experiment-output.zip"
    with zipfile.ZipFile(path) as original:
        members = {row.filename: original.read(row.filename) for row in original.infolist()}
    members["training/latest.state.pt"] = b"corrupted state"
    with zipfile.ZipFile(path, "w") as archive:
        for name, contents in members.items():
            archive.writestr(name, contents)
    with pytest.raises(ValueError, match="hash mismatch"):
        queue.resolve_queue(recipe, tmp_path / "mounted", tmp_path, tmp_path / "output")


def test_unmatched_optimization_schedule_is_rejected(tmp_path):
    recipe = make_sources(tmp_path)
    path = tmp_path / "mounted/joint/results/training/metrics.jsonl"
    path.write_text(path.read_text().replace('"positions": 10', '"positions": 9'))
    result = path.parents[1]
    export_outputs(result, result.parent / "fog-experiment-output.zip")
    with pytest.raises(ValueError, match="different data"):
        queue.resolve_queue(recipe, tmp_path / "mounted", tmp_path, tmp_path / "output")


def test_queue_cannot_spend_confirmation_cohorts(tmp_path):
    recipe = deepcopy(make_sources(tmp_path, outputs=False))
    recipe["stage"] = "confirmation"
    with pytest.raises(ValueError, match="development"):
        queue.validate_queue(recipe, tmp_path)


def test_waiting_pipeline_does_not_install_environments_or_play_games(tmp_path, monkeypatch):
    work, out = tmp_path / "work", tmp_path / "output"
    work.mkdir()
    (work / "evaluation.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        worker,
        "prepare_inputs",
        lambda *_: {
            "stage": "evaluate",
            "device": "cpu",
            "timeout_seconds": 60,
            "training_sources": {"ordinary": {}, "joint": {}},
        },
    )
    monkeypatch.setattr(queue, "resolve_queue", lambda *_: None)
    monkeypatch.setattr(
        worker, "execute", lambda *_args, **_kwargs: pytest.fail("Waiting job ran work")
    )
    worker.run_pipeline(tmp_path / "mounted", work, out, 60)
    assert json.loads((out / "pipeline_state.json").read_text())["status"] == "waiting"
    assert not (out / "environment").exists()
