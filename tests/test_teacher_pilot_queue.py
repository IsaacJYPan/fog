import hashlib
import json
import shutil
import zipfile

import pytest

from fog.scripts.teacher_pilot_queue import resolve_training, verified_inventory


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


def dispatch_fixture(tmp_path, monkeypatch, *, cpu_pipeline="waiting"):
    from fog.scripts import teacher_pilot_experiment as workflow

    settings = archive(tmp_path / "input")
    settings["kernel"] = "owner/train"
    training, cpu = tmp_path / "training", tmp_path / "cpu"
    training.mkdir()
    cpu.mkdir()
    (training / "outputs").mkdir()
    exported = training / "outputs/source.zip"
    shutil.copy2(tmp_path / "input/fog-experiment-output.zip", exported)
    workflow.write_json(training / "experiment.json", settings["experiment"])
    workflow.write_json(training / "state.json", {"kernel": "owner/train", "output_archive": str(exported)})
    workflow.write_json(training / "seal.json", {})
    workflow.write_json(cpu / "seal.json", {})
    workflow.write_json(cpu / "experiment.json", {"stage": "evaluate", "device": "cpu", "timeout_seconds": 21600,
                                                 "training_source": settings})
    workflow.write_json(cpu / "notebook/kernel-metadata.json", {"is_private": True, "enable_gpu": False,
                                                               "kernel_sources": ["owner/train"]})
    waiting = cpu / "results/initial"
    workflow.write_json(waiting / "pipeline_state.json", {"status": cpu_pipeline, "returncode": 0, "error": None})
    workflow.write_json(cpu / "state.json", {"phase": "collected", "gpu_requested": False,
                                           "kernel": "owner/cpu", "run_id": 123, "result": str(waiting)})
    monkeypatch.setattr(workflow, "directory", lambda name: training)
    monkeypatch.setattr(workflow, "collect", lambda folder: None)
    return workflow, cpu


def test_dispatch_waits_for_gpu_without_submitting(tmp_path, monkeypatch):
    workflow, cpu = dispatch_fixture(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(workflow, "cli", lambda *args: calls.append(args) or "KernelWorkerStatus.RUNNING")
    workflow.dispatch_evaluation(cpu)
    assert calls == [("kernels", "status", "owner/train")]
    assert not (cpu / "cpu-dispatches").exists()


def test_dispatch_checks_source_and_preserves_waiting_result(tmp_path, monkeypatch):
    workflow, cpu = dispatch_fixture(tmp_path, monkeypatch)
    before = (cpu / "results/initial/pipeline_state.json").read_bytes()
    pushes = []

    def cli(*args):
        if args[1] == "status":
            return "KernelWorkerStatus.COMPLETE"
        assert args == ("kernels", "push", "-p", str(cpu / "notebook"), "--timeout", "21600")
        assert json.loads((cpu / "state.json").read_text())["phase"] == "submitting"
        pushes.append(args)
        return "Kernel version 2 successfully pushed."

    monkeypatch.setattr(workflow, "cli", cli)
    workflow.dispatch_evaluation(cpu)
    state = json.loads((cpu / "state.json").read_text())
    assert len(pushes) == 1 and state["phase"] == "submitted"
    assert state["source_attachment_pending"] is False
    assert (cpu / "results/initial/pipeline_state.json").read_bytes() == before
    receipt = json.loads(next((cpu / "cpu-dispatches").glob("*.json")).read_text())
    assert receipt["verified_source"]["updates"] == 500
    assert receipt["verified_source"]["gpu_contracts_passed"] is True
    assert receipt["gpu_requested"] is False and receipt["previous_waiting_run_id"] == 123


@pytest.mark.parametrize("response", [
    "Kernel push error: ambiguous server response",
    "The following are not valid kernel sources: ['owner/train']\nKernel version 2 successfully pushed.",
])
def test_uncertain_cpu_dispatch_cannot_be_retried(tmp_path, monkeypatch, response):
    workflow, cpu = dispatch_fixture(tmp_path, monkeypatch)
    pushes = []

    def cli(*args):
        if args[1] == "status":
            return "KernelWorkerStatus.COMPLETE"
        pushes.append(args)
        return response

    monkeypatch.setattr(workflow, "cli", cli)
    with pytest.raises(RuntimeError, match="uncertain"):
        workflow.dispatch_evaluation(cpu)
    assert json.loads((cpu / "state.json").read_text())["phase"] == "submitting"
    with pytest.raises(ValueError, match="uncertain"):
        workflow.dispatch_evaluation(cpu)
    assert len(pushes) == 1


def test_completed_cpu_screen_cannot_be_repeated_automatically(tmp_path, monkeypatch):
    workflow, cpu = dispatch_fixture(tmp_path, monkeypatch, cpu_pipeline="complete")
    monkeypatch.setattr(workflow, "cli", lambda *args: "KernelWorkerStatus.COMPLETE" if args[1] == "status" else pytest.fail("duplicate push"))
    with pytest.raises(ValueError, match="already ran"):
        workflow.dispatch_evaluation(cpu)
    assert not (cpu / "cpu-dispatches").exists()


def test_safe_directory_entries_do_not_reject_valid_gpu_bytes(tmp_path):
    settings = archive(tmp_path / "input")
    with zipfile.ZipFile(tmp_path / "input/fog-experiment-output.zip", "a") as zipped:
        zipped.writestr("training/", b"")
    result = resolve_training(settings, tmp_path / "input", tmp_path / "result")
    assert result["updates"] == 500 and result["verified_exported_files"] == 5


@pytest.mark.parametrize("name,payload", [("unexpected.bin", b"extra bytes"), ("empty/", b"hidden bytes")])
def test_directory_support_does_not_allow_unverified_bytes(tmp_path, name, payload):
    settings = archive(tmp_path / "input")
    with zipfile.ZipFile(tmp_path / "input/fog-experiment-output.zip", "a") as zipped:
        zipped.writestr(name, payload)
    with pytest.raises(ValueError, match="unverified"):
        resolve_training(settings, tmp_path / "input", tmp_path / "result")
    assert not (tmp_path / "result").exists()


def test_real_cpu_export_hashes_nested_source_inventory(tmp_path):
    from fog.scripts.teacher_pilot_kaggle_worker import export_output

    output = tmp_path / "output"
    output.mkdir()
    (output / "pipeline_state.json").write_text('{"status":"complete"}')
    (output / "source-training").mkdir()
    nested = output / "source-training/output_hashes.json"
    nested.write_text('{"training/latest.pt":"hash from the original GPU"}')
    export_output(output, tmp_path / "export")
    with zipfile.ZipFile(tmp_path / "export.zip") as zipped:
        inventory = verified_inventory(zipped)
        assert inventory["source-training/output_hashes.json"] == hashlib.sha256(nested.read_bytes()).hexdigest()
        assert set(inventory) == {"source-training/output_hashes.json", "pipeline_state.json"}


def test_make_archive_output_can_be_resolved_by_the_cpu_queue(tmp_path):
    from fog.scripts.teacher_pilot_kaggle_worker import export_output

    settings = archive(tmp_path / "original")
    output = tmp_path / "gpu-output"
    with zipfile.ZipFile(tmp_path / "original/fog-experiment-output.zip") as zipped:
        zipped.extractall(output)
    mounted = tmp_path / "mounted"
    mounted.mkdir()
    export_output(output, mounted / "fog-experiment-output")
    result = resolve_training(settings, mounted, tmp_path / "verified")
    assert result["updates"] == 500


def test_superseded_teacher_package_cannot_be_submitted(tmp_path, monkeypatch):
    workflow, cpu = dispatch_fixture(tmp_path, monkeypatch)
    state = json.loads((cpu / "state.json").read_text())
    state["phase"] = "superseded"
    workflow.write_json(cpu / "state.json", state)
    monkeypatch.setattr(workflow, "cli", lambda *args: pytest.fail("superseded package must not contact Kaggle"))
    with pytest.raises(ValueError, match="superseded"):
        workflow.submit(cpu)
