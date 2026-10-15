"""Freeze, submit and collect private teacher-pilot training and queued CPU evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile

from fog.scripts.kaggle_experiment import cli, directory, verify
from fog.scripts.kaggle_results import ROOT, client, validate_members, write_json
from fog.training.teacher_replay import digest


def prepare(stage, *, config, recipe, data, initial, validation, initialization, pool, source=None, candidate_checkpoint="latest.pt"):
    import yaml
    from fog.config import load_config
    from fog.training.expert_pilot import validate_recipe

    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if status.strip():
        raise ValueError("Commit and validate the fixed source before freezing the pilot")
    proof = json.loads(validation.read_text())
    if not proof["passed"] or not proof["exact_resume_model_and_optimizer"] or not proof["exact_resume_sampler_and_torch_rng"]:
        raise ValueError("Require real-data training and exact-resume validation")
    cfg = load_config(config)
    settings = json.loads(recipe.read_text())
    validate_recipe(settings, cfg)
    initial_proof = json.loads(initialization.read_text())
    if not initial_proof["independent_neural_weights"] or initial_proof["source_checkpoint"] is not None or initial_proof["checkpoint_sha256"] != digest(initial):
        raise ValueError("Require hash-bound independent initialization")
    if stage not in ("train", "evaluate") or (stage == "evaluate") != (source is not None):
        raise ValueError("CPU evaluation needs its fixed GPU source")
    if re.fullmatch(r"(?:latest|anchor-[0-9]{6})\.pt", candidate_checkpoint) is None or (stage == "train" and candidate_checkpoint != "latest.pt"):
        raise ValueError("Select latest.pt or an immutable six-digit training anchor for CPU evaluation")
    name = f"fog-teacher-{stage}-{datetime.now(timezone.utc):%y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    folder = directory(name)
    folder.mkdir(parents=True)
    upload, notebook = folder / "dataset", folder / "notebook"
    upload.mkdir()
    notebook.mkdir()
    files = {}

    def add(path, target):
        path = Path(path).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or target in files:
            raise ValueError("Unsafe or duplicate private input: " + target)
        files[target] = path

    for path in (ROOT / "fog").rglob("*"):
        if path.is_file() and path.suffix in (".py", ".js", ".json") and "__pycache__" not in path.parts:
            add(path, path.relative_to(ROOT).as_posix())
    for name in ("pyproject.toml", "uv.lock", "README.md", "LICENSE", "integrations/THIRD_PARTY.md", "integrations/LICENSE.kakuna"):
        add(ROOT / name, name)
    for name in ("dist", "lib", "sim", "data"):
        for path in (ROOT / "server/pokemon-showdown" / name).rglob("*"):
            if path.is_file():
                add(path, path.relative_to(ROOT).as_posix())
    for name in ("package.json", "package-lock.json", "pokemon-showdown"):
        add(ROOT / "server/pokemon-showdown" / name, "server/pokemon-showdown/" + name)
    resolved = folder / "config.yaml"
    resolved.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    add(resolved, "config.yaml")
    add(recipe, "recipe.json")
    add(initial, "inputs/random_init.pt")
    add(initialization, "inputs/INITIALIZATION.json")
    add(validation, "inputs/LOCAL_TRAINING_VALIDATION.json")
    add(pool, "inputs/development.json")
    for path in data.rglob("*"):
        if path.is_file():
            add(path, "inputs/teacher-data/" + path.relative_to(data).as_posix())
    descriptor = {
        "format": "fog.teacher-pilot-experiment", "version": 1, "id": folder.name,
        "stage": stage, "device": "cuda" if stage == "train" else "cpu",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "timeout_seconds": 16200 if stage == "train" else 21600,
        "independent_neural_weights": True, "automatic_promotion": False,
        "data_sha256": digest(data / "manifest.json"), "initial_sha256": digest(initial),
        "source_validation_sha256": digest(validation), "games": 256, "native_games": 128,
    }
    if stage == "evaluate":
        from fog.scripts.pretrained_experiment import UPSTREAM, tracked_sources

        expected = json.loads((source / "experiment.json").read_text())
        verify(source)
        if expected["stage"] != "train" or expected["device"] != "cuda" or expected["data_sha256"] != descriptor["data_sha256"] or expected["initial_sha256"] != descriptor["initial_sha256"]:
            raise ValueError("CPU source does not match the frozen pilot")
        kernel = json.loads((source / "state.json").read_text())["kernel"]
        descriptor["training_source"] = {"experiment": expected, "kernel": kernel, "minimum_updates": 500}
        descriptor["candidate_checkpoint"] = candidate_checkpoint
        from fog.pretrained.kakuna import KAKUNA_34_SHA256

        reference = ROOT / "runs/pretrained/assets/kakuna--ckpts--policy_weights--policy_epoch_34.pt"
        if digest(reference) != KAKUNA_34_SHA256:
            raise ValueError("Original Kakuna reference changed")
        add(reference, "inputs/kakuna34.pt")
        for upstream in ("metamon", "amago", "poke-env"):
            checkout = ROOT / ".tools/upstream" / upstream
            revision = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
            if revision != UPSTREAM[upstream]:
                raise ValueError("CPU native reference revision differs")
            subprocess.run(["git", "-C", str(checkout), "diff", "--exit-code", "--quiet", "HEAD"], check=True)
            subtree = "src" if upstream == "poke-env" else upstream
            for path in tracked_sources(checkout, [subtree, "pyproject.toml", "requirements.txt", "README.md", "LICENSE"]):
                add(path, "upstream/" + upstream + "/" + path.relative_to(checkout).as_posix())
        prior = ROOT / "runs/experiments/fog-native-evaluate-261012-074715-b0e399/requirements.native.txt"
        add(prior, "requirements.native.txt")
    descriptor["files"] = {name: digest(path) for name, path in sorted(files.items())}
    write_json(folder / "experiment.json", descriptor)
    with zipfile.ZipFile(upload / "bundle.bin", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
        archive.write(folder / "experiment.json", "experiment.json")
    owner = json.loads((ROOT / "configs/kaggle_results.json").read_text())["kernel"].split("/")[0]
    identity, kernel = owner + "/" + folder.name, owner + "/" + folder.name + "-run"
    write_json(upload / "dataset-metadata.json", {"id": identity, "title": folder.name, "licenses": [{"name": "other"}]})
    shutil.copy2(ROOT / "fog/scripts/teacher_pilot_kaggle_worker.py", notebook / "run.py")
    write_json(notebook / "kernel-metadata.json", {
        "id": kernel, "title": folder.name + "-run", "code_file": "run.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": stage == "train", "enable_internet": True,
        "dataset_sources": [identity], "kernel_sources": [descriptor["training_source"]["kernel"]] if source else [],
        "competition_sources": [], "model_sources": [],
    })
    paths = [upload / "bundle.bin", upload / "dataset-metadata.json", notebook / "run.py", notebook / "kernel-metadata.json", folder / "experiment.json", resolved]
    write_json(folder / "seal.json", {p.relative_to(folder).as_posix(): digest(p) for p in paths})
    write_json(folder / "state.json", {"phase": "prepared", "stage": stage, "kernel": kernel, "dataset": identity, "gpu_requested": stage == "train"})
    (folder / "HANDOFF.md").write_text(f"# {folder.name}\n\nPrivate fixed {stage} package. Read experiment.json, seal.json and state.json.\n\nNo automatic model promotion. Previous models are preserved.\n", encoding="utf-8")
    print(folder.name, flush=True)
    return folder


def submit(folder):
    verify(folder)
    state = json.loads((folder / "state.json").read_text())
    if state["phase"] not in ("prepared", "uploaded"):
        raise ValueError("Submission is complete, superseded or uncertain; inspect remote state before retrying")
    quota = json.loads(cli("quota", "--format", "json"))
    if state["gpu_requested"]:
        gpu = next(row for row in quota if row["resource"] == "GPU")
        if float(gpu["remaining"].removesuffix("h")) < 4.5:
            raise ValueError("Fresh quota does not cover the bounded pilot")
    if state["phase"] == "prepared":
        state.update(phase="uploading", quota_before=quota)
        write_json(folder / "state.json", state)
        print(cli("datasets", "create", "-p", str(folder / "dataset")), flush=True)
        state["phase"] = "uploaded"
        write_json(folder / "state.json", state)
    # A newly uploaded private dataset can briefly return 403 while Kaggle
    # finishes processing it. Keep its uploaded identity; never upload twice.
    readiness = None
    for attempt in range(4):
        try:
            readiness = json.loads(cli("datasets", "status", state["dataset"], "--format", "json"))
            break
        except RuntimeError as error:
            if "403" not in str(error):
                raise
            if attempt == 3:
                print("Private dataset is not yet readable; submission remains uploaded and can be resumed.", flush=True)
                return
            time.sleep(3 * (attempt + 1))
    if "ready" not in readiness.get("status", "").lower():
        print("Private dataset is processing; no notebook was submitted.", flush=True)
        return
    state["phase"] = "submitting"
    write_json(folder / "state.json", state)
    command = ["kernels", "push", "-p", str(folder / "notebook"), "--timeout", str(json.loads((folder / "experiment.json").read_text())["timeout_seconds"])]
    if state["gpu_requested"]:
        command.extend(["--accelerator", "NvidiaTeslaT4"])
    response = cli(*command)
    if "Kernel version" not in response or "successfully pushed" not in response:
        raise RuntimeError("Submission outcome is uncertain; inspect before retrying: " + response)
    state.update(phase="submitted", submitted_at=datetime.now(timezone.utc).isoformat(), submission=response,
                 url="https://www.kaggle.com/code/" + state["kernel"],
                 source_attachment_pending="not valid kernel sources" in response)
    write_json(folder / "state.json", state)
    print(response, flush=True)


def collect(folder):
    verify(folder)
    state = json.loads((folder / "state.json").read_text())
    status = cli("kernels", "status", state["kernel"])
    if "RUNNING" in status or "QUEUED" in status:
        print(status, flush=True)
        return
    with tempfile.TemporaryDirectory(dir=folder, prefix="fetch-") as temporary:
        python = client().with_name("python.exe" if sys.platform == "win32" else "python")
        subprocess.run([str(python), "-m", "fog.scripts.kaggle_experiment", "_fetch", folder.name, "--destination", temporary], cwd=ROOT, check=True)
        archive_path = Path(temporary) / "fog-experiment-output.zip"
        archive_digest = digest(archive_path)
        result = folder / "results" / archive_digest[:16]
        run_id = json.loads((Path(temporary) / "download.json").read_text())["run_id"]
        with zipfile.ZipFile(archive_path) as archive:
            validate_members(archive)
            if json.loads(archive.read("experiment.json")) != json.loads((folder / "experiment.json").read_text()):
                raise ValueError("Returned pilot belongs to different frozen inputs")
            from fog.scripts.teacher_pilot_queue import verified_inventory

            verified_inventory(archive)
            if not result.exists():
                result.mkdir(parents=True)
                archive.extractall(result)
        saved = folder / "outputs" / f"{run_id}-{archive_digest[:16]}.zip"
        saved.parent.mkdir(exist_ok=True)
        if not saved.exists():
            shutil.copy2(archive_path, saved)
    state.update(phase="collected", result=str(result), run_id=run_id, output_archive=str(saved))
    write_json(folder / "state.json", state)
    print(result, flush=True)


def dispatch_evaluation(folder):
    """Resume a waiting CPU export with the same sealed inputs, once GPU work is done."""
    from fog.scripts.teacher_pilot_queue import resolve_training

    verify(folder)
    experiment = json.loads((folder / "experiment.json").read_text())
    state = json.loads((folder / "state.json").read_text())
    metadata = json.loads((folder / "notebook/kernel-metadata.json").read_text())
    if experiment["stage"] != "evaluate" or experiment["device"] != "cpu" or state["gpu_requested"]:
        raise ValueError("Only the separately allocated CPU screen can be dispatched")
    source = experiment["training_source"]
    if metadata["is_private"] is not True or metadata["enable_gpu"] is not False or metadata["kernel_sources"] != [source["kernel"]]:
        raise ValueError("CPU metadata differs from its private, fixed GPU source")
    if state["phase"] not in ("submitted", "collected"):
        raise ValueError("CPU submission is absent or uncertain; inspect before dispatching")
    training = directory(source["experiment"]["id"])
    verify(training)
    if json.loads((training / "experiment.json").read_text()) != source["experiment"]:
        raise ValueError("Local GPU source differs from the sealed CPU descriptor")
    training_state = json.loads((training / "state.json").read_text())
    if training_state["kernel"] != source["kernel"]:
        raise ValueError("Local GPU kernel differs from the sealed source")
    training_status = cli("kernels", "status", source["kernel"])
    if "RUNNING" in training_status or "QUEUED" in training_status:
        print("GPU pilot is still running; no CPU execution was submitted.", flush=True)
        return
    if "COMPLETE" not in training_status:
        raise ValueError("GPU pilot failed; collect and diagnose it before CPU dispatch")
    cpu_status = cli("kernels", "status", state["kernel"])
    if "RUNNING" in cpu_status or "QUEUED" in cpu_status:
        print("CPU notebook is active; no duplicate execution was submitted.", flush=True)
        return
    if "COMPLETE" not in cpu_status:
        raise ValueError("CPU notebook failed; inspect it before dispatching")
    # Refresh terminal outputs even when a previous version was already collected.
    collect(folder)
    state = json.loads((folder / "state.json").read_text())
    pipeline = json.loads((Path(state["result"]) / "pipeline_state.json").read_text())
    if pipeline["status"] != "waiting" or pipeline["error"] is not None or pipeline["returncode"] != 0:
        raise ValueError("CPU screen already ran or failed; do not repeat it automatically")
    collect(training)
    training_state = json.loads((training / "state.json").read_text())
    archive = Path(training_state["output_archive"]).resolve()
    if not archive.is_relative_to((training / "outputs").resolve()) or not archive.is_file():
        raise ValueError("GPU archive is outside its collected source folder")
    with tempfile.TemporaryDirectory(dir=folder, prefix="verify-source-") as temporary:
        stage = Path(temporary)
        inputs = stage / "input"
        inputs.mkdir()
        shutil.copy2(archive, inputs / "fog-experiment-output.zip")
        proof = resolve_training(source, inputs, stage / "verified")
        if proof is None:
            raise ValueError("Collected GPU export is missing its fixed source")
        proof.pop("directory")
    begun = datetime.now(timezone.utc)
    record = {"status": "submitting", "started_at": begun.isoformat(), "gpu_requested": False,
              "automatic_promotion": False, "previous_waiting_run_id": state["run_id"],
              "previous_waiting_result": state["result"], "gpu_archive_sha256": digest(archive),
              "verified_source": proof, "experiment_sha256": digest(folder / "experiment.json"),
              "seal_sha256": digest(folder / "seal.json")}
    receipt = folder / "cpu-dispatches" / (begun.strftime("%Y%m%d-%H%M%S") + ".json")
    write_json(receipt, record)
    state["phase"] = "submitting"
    write_json(folder / "state.json", state)
    try:
        response = cli("kernels", "push", "-p", str(folder / "notebook"), "--timeout", str(experiment["timeout_seconds"]))
        if "Kernel version" not in response or "successfully pushed" not in response or "not valid kernel sources" in response:
            raise RuntimeError("CPU dispatch outcome or required source is uncertain; inspect before retrying: " + response)
        record.update(status="submitted", completed_at=datetime.now(timezone.utc).isoformat(), submission=response)
        state.update(phase="submitted", submitted_at=record["completed_at"], latest_submission=response,
                     source_attachment_pending=False)
        state.setdefault("cpu_dispatches", []).append(str(receipt))
        write_json(receipt, record)
        write_json(folder / "state.json", state)
        print(response, flush=True)
    except BaseException as error:
        record.update(status="inspect_before_retry", error=repr(error))
        write_json(receipt, record)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("prepare")
    freeze.add_argument("--stage", choices=("train", "evaluate"), required=True)
    for name in ("config", "recipe", "data", "initial", "validation", "initialization", "pool"):
        freeze.add_argument("--" + name, type=Path, required=True)
    freeze.add_argument("--source", type=Path)
    freeze.add_argument("--candidate-checkpoint", default="latest.pt")
    for name in ("submit", "collect", "status", "dispatch-evaluation"):
        commands.add_parser(name).add_argument("experiment")
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "prepare":
        prepare(**args)
    else:
        folder = directory(args["experiment"])
        if command == "submit":
            submit(folder)
        elif command == "collect":
            collect(folder)
        elif command == "dispatch-evaluation":
            dispatch_evaluation(folder)
        else:
            print(cli("kernels", "status", json.loads((folder / "state.json").read_text())["kernel"]))


if __name__ == "__main__":
    main()
