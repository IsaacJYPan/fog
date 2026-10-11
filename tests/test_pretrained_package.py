import pytest
import json
from pathlib import Path

from fog.pretrained.kakuna import sha256
from fog.scripts import pretrained_experiment as workflow
from fog.scripts.pretrained_experiment import verify_published_records, write_notebook_runner
from fog.scripts.pretrained_kaggle_worker import export_outputs


def test_changed_replay_invalidates_previously_audited_manifest(tmp_path):
    replay = tmp_path / "game.json.lz4"
    replay.write_bytes(b"frozen replay")
    manifest = {"records": [{"path": replay.name, "sha256": sha256(replay)}]}
    verify_published_records(tmp_path, manifest)
    replay.write_bytes(b"changed replay")
    with pytest.raises(ValueError, match="Audited published replay changed"):
        verify_published_records(tmp_path, manifest)


def test_freezing_warm_start_preserves_unique_dataset_and_notebook_identity(tmp_path, monkeypatch):
    from fog.pretrained import initialization, loading

    root = tmp_path / "workspace"
    def write(name, value):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) if isinstance(value, dict) else value)
        return path

    original = write("original.pt", "original weights")
    original_sha = sha256(original)
    write("fog/example.py", "pass\n")
    for name in ("LICENSE", "integrations/THIRD_PARTY.md", "integrations/LICENSE.kakuna"):
        write(name, "license")
    write("configs/kaggle_results.json", {"kernel": "owner/existing"})
    write("published/manifest.json", {
        "matches": 1, "bytes": 10, "records": [], "pool_manifest_sha256": "pool",
        "current_rules_screen": {"native_sample": {"passed": True}},
    })
    collection = write("fresh/collection.json", {"pool_sha256": "train", "split": "train"})
    write("fresh/manifest.json", {"passed": True, "native_load_perspectives": 2000,
                                 "collection_sha256": sha256(collection), "files": {}})
    isolation = write("isolation.json", {
        "passed": True, "pool_base_rosters_disjoint": True,
        "published_manifest_sha256": sha256(root / "published/manifest.json"),
        "pool_manifest_sha256": "pool", "training_pool_sha256": "train",
    })
    write("contract/cpu/contract.json", {"passed": True, "checkpoint_sha256": original_sha})
    write("contract/inputs.pt", "samples")
    write("contract/cpu/probabilities.pt", "probabilities")
    sources = {}
    for condition in ("ordinary", "joint"):
        sources[condition] = {field: str(write(f"parents/{condition}/{field}.pt", field))
                              for field in ("native", "auxiliary", "reference")}
    config = write("config.json", {"condition": "joint", "updates": 1,
                                   "warm_start": {"sources": sources}})
    monkeypatch.setattr(workflow, "ROOT", root)
    monkeypatch.setattr(workflow, "UPSTREAM", {})
    monkeypatch.setattr(workflow, "KAKUNA_34_SHA256", original_sha)
    monkeypatch.setattr(workflow, "directory", lambda name: root / "runs" / name)
    monkeypatch.setattr(loading, "fresh_files", lambda *a, **k: ([], {"eligible": 1000}))
    monkeypatch.setattr(initialization, "validate_warm_start", lambda *a, **k: {})
    monkeypatch.setattr(workflow.subprocess, "check_output",
                        lambda args, **kwargs: "revision\n" if kwargs.get("text") else b"numpy==2.5.3\n")
    monkeypatch.setattr(workflow, "write_notebook_runner", lambda path, timeout: path.write_text("pass\n"))
    monkeypatch.setattr(workflow, "handoff", lambda *args: None)
    folder = workflow.freeze(
        "train", checkpoint=original, native_python=root / "python.exe", config=config,
        published=root / "published", fresh=root / "fresh", contract=root / "contract",
        isolation=isolation, cpu_preflight=True, timeout=1800,
    )
    experiment = json.loads((folder / "experiment.json").read_text())
    state = json.loads((folder / "state.json").read_text())
    workflow.validate_submission_identity(folder, experiment, state, gpu_requested=False)
    assert state["dataset"] == "owner/" + folder.name
    assert state["kernel"] == "owner/" + folder.name + "-run"
    assert folder.name.startswith("fog-native-train-") and folder.name != "reference"
    assert len([name for name in experiment["files"] if name.startswith("inputs/warm-start/")]) == 6


def test_notebook_watchdog_is_bound_before_reading_inputs(tmp_path):
    script = tmp_path / "run.py"
    write_notebook_runner(script, 1800)
    code = script.read_text()
    compile(code, str(script), "exec")
    assert "FROZEN_TIMEOUT_SECONDS = 1800" in code
    assert "timeout=FROZEN_TIMEOUT_SECONDS" in code


def test_scheduled_runner_is_real_notebook_json_with_clean_child_argv(tmp_path):
    path = tmp_path / "run.ipynb"
    write_notebook_runner(path, 1800)
    notebook = json.loads(path.read_text())
    assert notebook["nbformat"] == 4 and len(notebook["cells"]) == 1
    cell = "".join(notebook["cells"][0]["source"])
    compile(cell, "<notebook>", "exec")
    assert "FROZEN_TIMEOUT_SECONDS = 1800" in cell
    assert "subprocess.run([sys.executable, str(runner)], check=True)" in cell


def test_notebook_executes_with_missing_temp_directory_and_kernel_arguments(tmp_path, monkeypatch):
    import sys

    source_root = tmp_path / "source"
    source = source_root / "fog/scripts/pretrained_kaggle_worker.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "from pathlib import Path\nimport json\nimport sys\n"
        "FROZEN_TIMEOUT_SECONDS = 43200\n"
        "assert len(sys.argv) == 1\n"
        "Path(__file__).with_suffix('.json').write_text(json.dumps("
        "{'timeout': FROZEN_TIMEOUT_SECONDS, 'executed': True}))\n"
    )
    monkeypatch.setattr(workflow, "ROOT", source_root)
    monkeypatch.setattr(sys, "argv", ["ipykernel_launcher.py", "-f", "connection.json"])
    notebook_path = tmp_path / "run.ipynb"
    write_notebook_runner(notebook_path, 1800)
    cell = "".join(json.loads(notebook_path.read_text())["cells"][0]["source"])
    entry = tmp_path / "absent/kaggle/temp/fog-evaluation-entry.py"
    assert not entry.parent.exists()
    cell = cell.replace("'/kaggle/temp/fog-evaluation-entry.py'", repr(str(entry)))
    exec(compile(cell, "<notebook>", "exec"), {})
    assert json.loads(entry.with_suffix(".json").read_text()) == {
        "timeout": 1800, "executed": True,
    }


@pytest.mark.parametrize("stage", ["train", "evaluate"])
def test_cpu_jobs_never_reserve_or_request_gpu(tmp_path, monkeypatch, stage):
    folder = tmp_path / "cpu-smoke"
    folder.mkdir()
    (folder / "experiment.json").write_text(
        json.dumps(
            {
                "id": folder.name,
                "cpu_preflight": stage == "train",
                "stage": stage,
                "timeout_seconds": 1800,
            }
        )
    )
    (folder / "state.json").write_text(
        json.dumps(
            {
                "stage": stage,
                "phase": "uploaded",
                "kernel": "owner/cpu-smoke-run",
                "dataset": "owner/cpu-smoke",
            }
        )
    )
    (folder / "dataset").mkdir()
    (folder / "notebook").mkdir()
    (folder / "dataset/dataset-metadata.json").write_text(json.dumps({"id": "owner/cpu-smoke"}))
    (folder / "notebook/kernel-metadata.json").write_text(
        json.dumps({"id": "owner/cpu-smoke-run", "is_private": True, "enable_gpu": False})
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


@pytest.mark.parametrize(
    "override",
    [
        {"id": "owner/another-run"},
        {"is_private": False},
        {"enable_gpu": True},
    ],
)
def test_invalid_notebook_identity_or_resources_fail_before_network(
    tmp_path, monkeypatch, override
):
    folder = tmp_path / "safe-run"
    (folder / "dataset").mkdir(parents=True)
    (folder / "notebook").mkdir()
    (folder / "experiment.json").write_text(json.dumps({"id": "safe-run", "stage": "evaluate"}))
    (folder / "state.json").write_text(
        json.dumps(
            {
                "stage": "evaluate",
                "phase": "prepared",
                "dataset": "owner/safe-run",
                "kernel": "owner/safe-run-run",
            }
        )
    )
    (folder / "dataset/dataset-metadata.json").write_text(json.dumps({"id": "owner/safe-run"}))
    notebook = {"id": "owner/safe-run-run", "is_private": True, "enable_gpu": False}
    notebook.update(override)
    (folder / "notebook/kernel-metadata.json").write_text(json.dumps(notebook))
    monkeypatch.setattr(workflow, "verify", lambda _: None)
    monkeypatch.setattr(
        workflow, "cli", lambda *_: pytest.fail("Invalid package reached network submission")
    )
    with pytest.raises(ValueError, match="identities must match"):
        workflow.submit(folder)


def test_registering_gpu_notebook_does_not_execute_or_reserve_quota(tmp_path, monkeypatch):
    folder = tmp_path / "registered-train"
    (folder / "dataset").mkdir(parents=True)
    (folder / "notebook").mkdir()
    (folder / "experiment.json").write_text(
        json.dumps(
            {
                "id": folder.name,
                "stage": "train",
                "timeout_seconds": 3600,
            }
        )
    )
    (folder / "state.json").write_text(
        json.dumps(
            {
                "stage": "train",
                "phase": "uploaded",
                "dataset": "owner/registered-train",
                "kernel": "owner/registered-train-run",
            }
        )
    )
    (folder / "dataset/dataset-metadata.json").write_text(
        json.dumps({"id": "owner/registered-train"})
    )
    (folder / "notebook/kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": "owner/registered-train-run",
                "is_private": True,
                "enable_gpu": True,
            }
        )
    )
    monkeypatch.setattr(workflow, "verify", lambda _: None)
    monkeypatch.setattr(workflow, "handoff", lambda *_: None)
    monkeypatch.setattr(workflow, "reserve_gpu", lambda **_: pytest.fail("Save reserved GPU quota"))
    monkeypatch.setattr(workflow, "client", lambda: tmp_path / "kaggle.exe")

    def cli(*args):
        assert args[:2] == ("datasets", "status")
        return json.dumps({"status": "ready"})

    def check_output(command, **_):
        assert command[2] == "fog.scripts.kaggle_quick_save"
        return json.dumps({"kernel": "owner/registered-train-run", "version": 1, "executed": False})

    monkeypatch.setattr(workflow, "cli", cli)
    monkeypatch.setattr(workflow.subprocess, "check_output", check_output)
    workflow.submit(folder, save_only=True)
    assert json.loads((folder / "state.json").read_text())["phase"] == "saved"
    with pytest.raises(ValueError, match="Already submitted"):
        workflow.submit(folder, save_only=True)


def test_gpu_launch_requires_an_armed_cpu_queue_for_the_exact_pair(tmp_path, monkeypatch):
    experiments = [{"id": "train-" + arm, "condition": arm} for arm in ("ordinary", "joint")]
    for row in experiments:
        folder = tmp_path / row["id"]
        folder.mkdir()
        (folder / "experiment.json").write_text(json.dumps(row))
    monkeypatch.setattr(workflow, "directory", lambda name: tmp_path / name)
    monkeypatch.setattr(workflow, "verify", lambda _: None)
    training = tmp_path / experiments[0]["id"]
    with pytest.raises(ValueError, match="Arm exactly one"):
        workflow.require_cpu_queue(training, *experiments)
    evaluation = tmp_path / "fog-native-evaluate-queue"
    (evaluation / "notebook").mkdir(parents=True)
    sources = {
        row["condition"]: {"experiment_sha256": sha256(tmp_path / row["id"] / "experiment.json")}
        for row in experiments
    }
    (evaluation / "experiment.json").write_text(
        json.dumps(
            {
                "id": evaluation.name,
                "stage": "evaluate",
                "device": "cpu",
                "evaluation": {},
                "training_sources": sources,
            }
        )
    )
    (evaluation / "state.json").write_text(json.dumps({"phase": "submitted", "kernel": "owner/queue"}))
    metadata = {
        "is_private": True,
        "enable_gpu": False,
        "kernel_type": "notebook",
        "kernel_sources": ["owner/train-ordinary-run", "owner/train-joint-run"],
    }
    (evaluation / "notebook/kernel-metadata.json").write_text(json.dumps(metadata))
    (evaluation / "SCHEDULE.json").write_text(
        json.dumps(
            {
                "enabled": True,
                "trigger": "data_updates",
                "kernel": "owner/queue",
                "sources": metadata["kernel_sources"],
            }
        )
    )
    workflow.require_cpu_queue(training, *experiments)
    metadata["enable_gpu"] = True
    (evaluation / "notebook/kernel-metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Arm exactly one"):
        workflow.require_cpu_queue(training, *experiments)


def test_collection_preserves_waiting_and_later_scheduled_results(tmp_path, monkeypatch):
    folder = tmp_path / "queued"
    folder.mkdir()
    experiment = {"training_sources": {"ordinary": {}}}
    (folder / "experiment.json").write_text(json.dumps(experiment))
    (folder / "state.json").write_text(json.dumps({"phase": "saved", "kernel": "owner/queued"}))
    monkeypatch.setattr(workflow, "verify", lambda _: None)
    monkeypatch.setattr(workflow, "handoff", lambda *_: None)
    monkeypatch.setattr(workflow, "client", lambda: tmp_path / "kaggle.exe")
    monkeypatch.setattr(workflow, "cli", lambda *_: "COMPLETE")
    versions = iter([(100, "waiting"), (101, "complete"), (101, "complete")])

    def fetch(command, **_):
        destination = tmp_path / command[-1]
        run_id, status = next(versions)
        result = destination / "stub"
        result.mkdir()
        (result / "experiment.json").write_text(json.dumps(experiment))
        (result / "pipeline_state.json").write_text(json.dumps({"status": status}))
        export_outputs(result, destination / "fog-experiment-output.zip")
        (destination / "download.json").write_text(json.dumps({"run_id": run_id}))

    monkeypatch.setattr(workflow.subprocess, "run", fetch)
    workflow.collect(folder)
    first = json.loads((folder / "state.json").read_text())
    workflow.collect(folder)
    second = json.loads((folder / "state.json").read_text())
    assert first["result"] != second["result"]
    assert (
        json.loads((Path(first["result"]) / "pipeline_state.json").read_text())["status"]
        == "waiting"
    )
    assert len(second["collected_runs"]) == 2
    assert len(list((folder / "outputs").glob("*.zip"))) == 2
