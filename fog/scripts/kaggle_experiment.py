"""Freeze, submit, collect and evaluate private Kaggle training experiments."""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
import zipfile

from fog.scripts.kaggle_results import ROOT, client, digest, import_archive, remote, write_json

HOME = ROOT / "runs/experiments"


def directory(name: str) -> Path:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,70}", name):
        raise ValueError("Use an experiment ID from prepare (lowercase letters, digits, hyphens)")
    return HOME / name


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def runtime_seconds(job: dict) -> int:
    value = job.get("kaggle", {}).get("timeout_seconds", 7200)
    if isinstance(value, bool) or not isinstance(value, int) or not 600 <= value <= 43200:
        raise ValueError("kaggle.timeout_seconds must be an integer between 600 and 43200")
    return value


def cli(*args: str) -> str:
    result = subprocess.run([str(client()), *args], cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(f"Kaggle {' '.join(args[:2])} failed: {result.stderr[-1500:]}")
    return result.stdout.strip()


def prepare(job_path: Path, label: str, trial: bool = False, parent: Path | None = None) -> Path:
    import yaml
    from fog.scripts.kaggle_bundle import bundle_for_kaggle
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,22}", label):
        raise ValueError("Label must have 2–23 lowercase letters, digits or hyphens")
    name = f"fog-{label}-{datetime.now(UTC):%y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    folder = directory(name)
    folder.mkdir(parents=True)
    job = yaml.safe_load(job_path.read_text())
    timeout = runtime_seconds(job)
    if parent is not None:
        if not parent.is_file():
            raise ValueError("Parent checkpoint must exist locally")
        job["initial_checkpoint"] = parent.resolve().relative_to(ROOT).as_posix()
    if not job.get("safety", {}).get("auto_refresh"):
        raise ValueError("Automatic refresh of both audits is required")
    job.update(name=name, output_dir="runs/experiment")
    job["evaluation"] = {"enabled": False}
    job["training"]["blocks"] = 1
    if trial:
        job["training"].update(decisions_per_block=64, batch_decisions=64,
                               minibatch_size=32, checkpoint_every_decisions=64)
    job["safety"]["device"] = "cuda"
    job.setdefault("overrides", {}).update({
        "device": "cuda", "env.node_binary": "node", "env.actor_threads": 1,
        "env.rollout_workers": 2, "env.showdown_root": "server/pokemon-showdown",
        "preflight.report": "runs/experiment/audits/direct/summary.json",
        "preflight.replay_report": "runs/experiment/audits/replay/summary.json",
    })
    recipe = folder / "job.yaml"
    recipe.write_text(yaml.safe_dump(job, sort_keys=False), encoding="utf-8")
    upload = folder / "dataset"
    upload.mkdir()
    bundle = upload / "bundle.zip"
    bundle_for_kaggle(recipe, bundle)
    with zipfile.ZipFile(bundle, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        inventory = {n.removeprefix("fog/"): hashlib.sha256(archive.read(n)).hexdigest()
                     for n in archive.namelist() if not n.endswith("/")}
        experiment = {"id": name, "job": recipe.relative_to(ROOT).as_posix(),
                      "created_at": datetime.now(UTC).isoformat(), "trial": trial,
                      "timeout_seconds": timeout,
                      "git_revision": subprocess.check_output(
                          ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                      "files": inventory, "parent": job["initial_checkpoint"],
                      "training_pool": job["overrides"].get("env.team_pool")}
        archive.writestr("fog/experiment.json", json.dumps(experiment, indent=2))
    write_json(folder / "experiment.json", experiment)
    owner = read(ROOT / "configs/kaggle_results.json")["kernel"].split("/")[0]
    identifier = f"{owner}/{name}"
    kernel = f"{identifier}-train"
    write_json(upload / "dataset-metadata.json", {"id": identifier, "title": name,
                                                 "licenses": [{"name": "other"}]})
    notebook = folder / "notebook"
    notebook.mkdir()
    shutil.copy2(ROOT / "fog/scripts/kaggle_worker.py", notebook / "train.py")
    write_json(notebook / "kernel-metadata.json", {
        "id": kernel, "title": name + "-train", "code_file": "train.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True,
        "enable_internet": True, "dataset_sources": [identifier],
        "competition_sources": [], "kernel_sources": [], "model_sources": [],
    })
    sealed = [bundle, upload / "dataset-metadata.json", notebook / "train.py",
              notebook / "kernel-metadata.json", folder / "experiment.json", recipe]
    write_json(folder / "seal.json", {p.relative_to(folder).as_posix(): digest(p) for p in sealed})
    write_json(folder / "state.json", {"phase": "prepared", "kernel": kernel, "dataset": identifier,
                                       "trial": trial})
    handoff(folder, "Prepared; next: submit. No resources have been uploaded yet.")
    print(name)
    return folder


def verify(folder: Path) -> None:
    for name, expected in read(folder / "seal.json").items():
        if digest(folder / name) != expected:
            raise ValueError(f"Frozen experiment changed: {name}; prepare a new experiment")


def handoff(folder: Path, message: str) -> None:
    (folder / "HANDOFF.md").write_text(
        f"# {folder.name}\n\n{message}\n\n"
        "Read experiment.json for frozen input hashes and state.json for remote/result links.\n"
        "Evaluate on a held-out pool before promotion. No automatic champion replacement.\n",
        encoding="utf-8")
    write_json(HOME / "latest.json", {"experiment": folder.name, "directory": str(folder)})


def submit(folder: Path) -> None:
    verify(folder)
    state = read(folder / "state.json")
    if state["phase"] == "superseded":
        raise RuntimeError("Experiment was superseded; submit its replacement instead")
    if state["phase"] in {"submitting", "submitted", "collected"}:
        raise RuntimeError("Already submitted or submission uncertain; inspect status, do not resubmit")
    if state["phase"] == "prepared":
        print(cli("quota", "--format", "json"), flush=True)
        state["phase"] = "uploading"
        write_json(folder / "state.json", state)
        print(cli("datasets", "create", "-p", str(folder / "dataset")), flush=True)
        state["phase"] = "uploaded"
        write_json(folder / "state.json", state)
    try:
        readiness = cli("datasets", "status", state.get("dataset", state["kernel"]),
                        "--format", "json")
    except RuntimeError as error:
        handoff(folder, "Dataset upload attempted; inspect dataset status and run submit again once ready.")
        if "403" in str(error) or "404" in str(error):
            print("Dataset is not accessible yet. Processing can take a minute; run submit again. "
                  "If this persists, check account permissions. No GPU job was submitted.")
            return
        raise
    print(readiness, flush=True)
    if "ready" not in str(json.loads(readiness).get("status", "")).lower():
        print("Dataset is processing. Run submit again after it becomes ready.")
        return
    state["phase"] = "submitting"
    write_json(folder / "state.json", state)
    descriptor = folder / "experiment.json"
    timeout = read(descriptor).get("timeout_seconds", 7200) if descriptor.exists() else 7200
    result = cli("kernels", "push", "-p", str(folder / "notebook"),
                 "--accelerator", "NvidiaTeslaT4", "--timeout", str(timeout))
    # Kaggle can print an API rejection while returning process exit code zero.
    if re.search(r"^Kernel push error:", result, flags=re.MULTILINE | re.IGNORECASE):
        capacity_rejection = "maximum batch gpu session count" in result.lower()
        state.update(phase="uploaded" if capacity_rejection else "submitting", submission=result)
        write_json(folder / "state.json", state)
        handoff(folder, (
            "GPU session limit rejected submission. Dataset is ready; submit once a slot is free."
            if capacity_rejection else
            "Kaggle reported a push error. Inspect remote status before retrying; submission is uncertain."
        ))
        raise RuntimeError(result)
    state.update(phase="submitted", submission=result,
                 url=f"https://www.kaggle.com/code/{state['kernel']}")
    write_json(folder / "state.json", state)
    handoff(folder, "Submitted as a private T4 job; next: status, then collect.")
    print(result)


def collect(folder: Path) -> None:
    verify(folder)
    state = read(folder / "state.json")
    if state.get("result"):
        print(f"Already collected: {state['result']}")
        return
    with tempfile.TemporaryDirectory(prefix="fetch-", dir=folder) as temporary:
        subprocess.run([str(client().with_name("python.exe" if sys.platform == "win32" else "python")),
                        "-m", "fog.scripts.kaggle_experiment", "_fetch", folder.name,
                        "--destination", temporary], cwd=ROOT, check=True)
        archive = Path(temporary) / "fog-experiment-output.zip"
        with zipfile.ZipFile(archive) as z:
            if json.loads(z.read("experiment.json")) != read(folder / "experiment.json"):
                raise ValueError("Downloaded output belongs to different experiment inputs")
        saved = folder / "output.zip"
        shutil.copy2(archive, saved)
        run = read(Path(temporary) / "download.json")
    result = import_archive(saved, folder / "results", state["kernel"], 1)
    state.update(phase="collected", result=str(result), run_id=run["run_id"])
    write_json(folder / "state.json", state)
    handoff(folder, f"Collected. Read {result / 'HANDOFF.md'}. Next: evaluate with a held-out pool.")


def evaluate(folder: Path, pool: Path, games: int) -> None:
    import yaml
    from fog.scripts.automated_training import _evaluation_command, summarize_evaluation
    if games < 4 or games % 4:
        raise ValueError("Evaluation games must be a positive multiple of four")
    state = read(folder / "state.json")
    result = Path(state["result"])
    report = read(result / "RESULTS.json")
    if report["pipeline_status"] != "complete" or not report["checkpoints"]:
        raise ValueError("Cannot evaluate an incomplete training result")
    experiment = read(folder / "experiment.json")
    verify(folder)
    evaluation = folder / f"evaluation-{games}-{digest(pool)[:12]}"
    evaluation.mkdir(exist_ok=True)
    with zipfile.ZipFile(folder / "dataset/bundle.zip") as z:
        train_pool = experiment["training_pool"]
        if not train_pool:
            raise ValueError("Frozen experiment must identify its training pool")
        from fog.env.teams import load_pool
        training = json.loads(z.read("fog/" + train_pool.replace("\\", "/")))
        ensure_disjoint(training["teams"] if isinstance(training, dict) else training,
                        load_pool(pool))
        parent = evaluation / "parent.pt"
        parent.write_bytes(z.read("fog/" + experiment["parent"].replace("\\", "/")))
    cfg = yaml.safe_load((result / "block_001/config.resolved.yaml").read_text())
    cfg["device"] = "cpu"
    cfg["env"].update(showdown_root=str(ROOT / "server/pokemon-showdown"), node_binary="node")
    config = evaluation / "config.yaml"
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    candidate = Path(report["checkpoints"][-1]["path"])
    settings = {"pool": str(pool.resolve()), "games": games, "device": "cpu",
                "head_to_head": True, "opponents": ["simpleheuristics", "maxdamage"]}
    out = evaluation / "report.json"
    command = _evaluation_command({}, config, parent, candidate, out, 20260922, settings=settings)
    write_json(evaluation / "protocol.json", {"pool_sha256": digest(pool),
               "candidate_sha256": digest(candidate), "parent_sha256": digest(parent),
               "local_source_hashes": {p.relative_to(ROOT).as_posix(): digest(p)
                                       for p in sorted((ROOT / "fog").rglob("*.py"))},
               "command": command, "promotion": "manual review only"})
    subprocess.run(command, cwd=ROOT, check=True)
    write_json(evaluation / "summary.json", summarize_evaluation(read(out), settings))
    state["evaluation"] = str(evaluation)
    write_json(folder / "state.json", state)
    handoff(folder, f"Evaluation ready for review: {evaluation}. No champion was changed.")


def ensure_disjoint(training: list[str], evaluation: list[str]) -> None:
    from fog.team_identity import team_group_key
    if not training or not evaluation:
        raise ValueError("Training and evaluation pools must be nonempty")
    known = {team_group_key(team) for team in training}
    if known.intersection(team_group_key(team) for team in evaluation):
        raise ValueError("Evaluation pool overlaps the frozen training team compositions")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "submit", "status", "collect", "evaluate", "_fetch"])
    p.add_argument("experiment", nargs="?")
    p.add_argument("--job", type=Path, default=ROOT / "configs/automated_gen9ou_candidates_kaggle.yaml")
    p.add_argument("--label", default="candidates")
    p.add_argument("--trial", action="store_true")
    p.add_argument("--parent", type=Path, help="Downloaded checkpoint inside this repository")
    p.add_argument("--pool", type=Path)
    p.add_argument("--games", type=int, default=64)
    p.add_argument("--destination", type=Path)
    args = p.parse_args()
    if args.command == "prepare":
        prepare(args.job, args.label, args.trial, args.parent)
        return
    folder = directory(args.experiment or read(HOME / "latest.json")["experiment"])
    if args.command == "submit":
        submit(folder)
    elif args.command == "status":
        print(cli("kernels", "status", read(folder / "state.json")["kernel"]))
    elif args.command == "collect":
        collect(folder)
    elif args.command == "_fetch":
        remote(read(folder / "state.json")["kernel"], 1, args.destination,
               "fog-experiment-output.zip", None)
    else:
        if not args.pool:
            p.error("evaluate requires an explicitly chosen held-out --pool")
        evaluate(folder, args.pool, args.games)


if __name__ == "__main__":
    main()
