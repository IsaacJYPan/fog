import json
import zipfile

import pytest

from fog.scripts import kaggle_experiment as workflow


def test_prepare_seals_private_distinct_resources_and_parent(tmp_path, monkeypatch):
    from fog.scripts import kaggle_bundle
    monkeypatch.setattr(workflow, "ROOT", tmp_path)
    monkeypatch.setattr(workflow, "HOME", tmp_path / "runs/experiments")
    workflow.write_json(tmp_path / "configs/kaggle_results.json", {"kernel": "owner/original"})
    worker = tmp_path / "fog/scripts/kaggle_worker.py"
    worker.parent.mkdir(parents=True)
    worker.write_text("print('worker')")
    parent = tmp_path / "selected.pt"
    parent.write_bytes(b"weights")
    job = tmp_path / "recipe.yaml"
    job.write_text("safety: {auto_refresh: true}\ntraining: {}\ninitial_checkpoint: old.pt\n")
    def bundle(recipe, path):
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("fog/input.txt", "frozen input")
    monkeypatch.setattr(kaggle_bundle, "bundle_for_kaggle", bundle)
    monkeypatch.setattr(workflow.subprocess, "check_output", lambda *a, **k: "commit\n")
    folder = workflow.prepare(job, "test", trial=True, parent=parent)
    metadata = workflow.read(folder / "notebook/kernel-metadata.json")
    dataset = workflow.read(folder / "dataset/dataset-metadata.json")
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["id"] != dataset["id"]
    assert metadata["dataset_sources"] == [dataset["id"]]
    assert workflow.read(folder / "experiment.json")["parent"] == "selected.pt"
    workflow.verify(folder)


def experiment(tmp_path, phase="prepared"):
    workflow.write_json(tmp_path / "state.json", {"phase": phase, "kernel": "owner/fog-test"})
    workflow.write_json(tmp_path / "seal.json", {})
    return tmp_path


def test_uncertain_submission_cannot_launch_duplicate(tmp_path, monkeypatch):
    folder = experiment(tmp_path, "submitting")
    monkeypatch.setattr(workflow, "cli", lambda *args: pytest.fail("must not contact Kaggle"))
    with pytest.raises(RuntimeError, match="uncertain"):
        workflow.submit(folder)


def test_superseded_package_cannot_be_submitted(tmp_path, monkeypatch):
    folder = experiment(tmp_path, "superseded")
    monkeypatch.setattr(workflow, "cli", lambda *args: pytest.fail("must not contact Kaggle"))
    with pytest.raises(RuntimeError, match="superseded"):
        workflow.submit(folder)


def test_submission_records_intent_before_network_failure(tmp_path, monkeypatch):
    folder = experiment(tmp_path, "uploaded")
    def cli(*args):
        if args[:2] == ("datasets", "status"):
            return '{"status": "ready"}'
        assert workflow.read(folder / "state.json")["phase"] == "submitting"
        assert args[-4:] == ("--accelerator", "NvidiaTeslaT4", "--timeout", "7200")
        raise ConnectionError("ambiguous network failure")
    monkeypatch.setattr(workflow, "cli", cli)
    with pytest.raises(ConnectionError):
        workflow.submit(folder)
    assert workflow.read(folder / "state.json")["phase"] == "submitting"


def test_processing_dataset_does_not_launch_gpu(tmp_path, monkeypatch):
    folder = experiment(tmp_path, "uploaded")
    calls = []
    def cli(*args):
        calls.append(args)
        return '{"status": "processing"}'
    monkeypatch.setattr(workflow, "cli", cli)
    workflow.submit(folder)
    assert len(calls) == 1
    assert workflow.read(folder / "state.json")["phase"] == "uploaded"


@pytest.mark.parametrize("message,phase", [
    ("Kernel push error: Maximum batch GPU session count of 2 reached.", "uploaded"),
    ("Kernel push error: unexpected server response", "submitting"),
])
def test_zero_exit_push_error_is_not_recorded_as_submitted(tmp_path, monkeypatch, message, phase):
    folder = experiment(tmp_path, "uploaded")
    monkeypatch.setattr(workflow, "HOME", tmp_path / "home")
    monkeypatch.setattr(workflow, "cli", lambda *args: (
        '{"status": "ready"}' if args[:2] == ("datasets", "status") else message
    ))
    with pytest.raises(RuntimeError, match="Kernel push error"):
        workflow.submit(folder)
    state = workflow.read(folder / "state.json")
    assert state["phase"] == phase
    assert state["submission"] == message
    assert "url" not in state


def test_capacity_rejection_allows_later_single_retry(tmp_path, monkeypatch):
    folder = experiment(tmp_path, "uploaded")
    monkeypatch.setattr(workflow, "HOME", tmp_path / "home")
    pushes = []

    def cli(*args):
        if args[:2] == ("datasets", "status"):
            return '{"status": "ready"}'
        pushes.append(args)
        if len(pushes) == 1:
            return "Kernel push error: Maximum batch GPU session count of 2 reached."
        return "submitted"

    monkeypatch.setattr(workflow, "cli", cli)
    with pytest.raises(RuntimeError, match="Maximum batch"):
        workflow.submit(folder)
    workflow.submit(folder)
    assert len(pushes) == 2
    assert workflow.read(folder / "state.json")["phase"] == "submitted"
    with pytest.raises(RuntimeError, match="Already submitted"):
        workflow.submit(folder)
    assert len(pushes) == 2


def test_modified_frozen_inputs_rejected(tmp_path):
    path = tmp_path / "input"
    path.write_text("original")
    workflow.write_json(tmp_path / "seal.json", {"input": workflow.digest(path)})
    path.write_text("changed")
    with pytest.raises(ValueError, match="Frozen experiment changed"):
        workflow.verify(tmp_path)


def test_wrong_experiment_output_is_not_imported(tmp_path, monkeypatch):
    folder = experiment(tmp_path, "submitted")
    workflow.write_json(folder / "experiment.json", {"id": "expected"})
    monkeypatch.setattr(workflow, "client", lambda: tmp_path / "kaggle.exe")
    def download(command, **kwargs):
        with zipfile.ZipFile(workflow.Path(command[-1]) / "fog-experiment-output.zip", "w") as z:
            z.writestr("experiment.json", json.dumps({"id": "wrong"}))
    monkeypatch.setattr(workflow.subprocess, "run", download)
    with pytest.raises(ValueError, match="different experiment"):
        workflow.collect(folder)
    assert not (folder / "output.zip").exists()
    assert workflow.read(folder / "state.json")["phase"] == "submitted"


def test_renamed_team_does_not_evade_holdout_guard():
    team = "Pikachu @ Light Ball\nAbility: Static\n- Thunderbolt\n- Protect\n"
    renamed = "Sparky (Pikachu) @ Light Ball\nAbility: Static\n- Protect\n- Thunderbolt\n"
    with pytest.raises(ValueError, match="overlaps"):
        workflow.ensure_disjoint([team], [renamed])
    workflow.ensure_disjoint([team], ["Raichu\nAbility: Static\n- Thunderbolt\n"])


@pytest.mark.parametrize("value", [True, 599, 43201, 21600.5, "21600"])
def test_invalid_runtime_is_rejected(value):
    with pytest.raises(ValueError, match="timeout_seconds"):
        workflow.runtime_seconds({"kaggle": {"timeout_seconds": value}})


def test_submission_uses_frozen_runtime(tmp_path, monkeypatch):
    folder = experiment(tmp_path, "uploaded")
    workflow.write_json(folder / "experiment.json", {"timeout_seconds": 21600})
    monkeypatch.setattr(workflow, "HOME", tmp_path / "home")
    def cli(*args):
        if args[:2] == ("datasets", "status"):
            return '{"status": "ready"}'
        assert args[-2:] == ("--timeout", "21600")
        return "submitted"
    monkeypatch.setattr(workflow, "cli", cli)
    workflow.submit(folder)
    assert workflow.read(folder / "state.json")["phase"] == "submitted"
