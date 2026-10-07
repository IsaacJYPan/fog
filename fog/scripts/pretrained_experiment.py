"""Freeze, submit and collect private native experiments without changing previous runs."""

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

from fog.pretrained.kakuna import KAKUNA_34_SHA256, sha256
from fog.pretrained.protocol import reserve_gpu
from fog.scripts.kaggle_experiment import cli, verify, directory, handoff
from fog.scripts.kaggle_results import ROOT, client, validate_members, write_json

UPSTREAM = {
    "metamon": "0a00a759c9a4382a2877088d828302ec294a05a5",
    "amago": "0974781a9096ff43df1b708312256f96fc2ab127",
    "poke-env": "e1268d270c3f2bd32c7ff5713e01062302020579",
    "foul-play": "8f541f80f91f960603673796559f796b10ccf0fb",
}


def tracked_sources(folder, paths):
    result = subprocess.check_output(["git", "-C", str(folder), "ls-files", "-z", "--", *paths])
    return [folder / path.decode("utf-8") for path in result.split(b"\0") if path]


def verify_published_records(root, manifest):
    """A manifest hash alone cannot detect a replay changed after its isolation audit."""
    root = root.resolve()
    for row in manifest["records"]:
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root) or sha256(path) != row["sha256"]:
            raise ValueError("Audited published replay changed: " + row["path"])


def freeze(
    stage,
    *,
    checkpoint,
    native_python,
    pools=None,
    learners=(),
    games=1000,
    shard=0,
    shards=1,
    config=None,
    published=None,
    fresh=None,
    contract=None,
    isolation=None,
    timeout=43200,
    phase="pilot",
):
    if sha256(checkpoint) != KAKUNA_34_SHA256:
        raise ValueError("Freeze requires original Kakuna checkpoint 34")
    if not 600 <= timeout <= 43200:
        raise ValueError("Kaggle background budget must be 10 minutes to 12 hours")
    if stage == "train":
        limits = {"pilot": 3600, "main": 14400, "refinement": 14400}
        if timeout > limits[phase]:
            raise ValueError("Per-arm pretrained GPU phase budget exceeded")
        from fog.pretrained.loading import fresh_files

        _, fresh_audit = fresh_files(fresh, minimum_games=1000)
        data = json.loads((published / "manifest.json").read_text())
        if not 1 <= data.get("matches", 0) <= 100_000 or data["bytes"] > 15_000_000_000:
            raise ValueError("Require a bounded, complete-pair human replay freeze")
        if not data.get("current_rules_screen", {}).get("native_sample", {}).get("passed"):
            raise ValueError(
                "Published human data require current-rule screening and a native input audit"
            )
        proof = json.loads(isolation.read_text()) if isolation else {}
        if (
            not proof.get("passed")
            or not proof.get("pool_base_rosters_disjoint")
            or proof.get("published_manifest_sha256") != sha256(published / "manifest.json")
            or proof.get("pool_manifest_sha256") != data["pool_manifest_sha256"]
        ):
            raise ValueError("A passed, hash-matched original-roster isolation audit is required")
        verify_published_records(published, data)
        collection = json.loads((fresh / "collection.json").read_text())
        fresh_proof = json.loads((fresh / "manifest.json").read_text())
        if (
            not fresh_proof.get("passed")
            or fresh_proof.get("native_load_perspectives") != 2 * fresh_audit["eligible"]
            or fresh_proof.get("collection_sha256") != sha256(fresh / "collection.json")
            or collection["pool_sha256"] != proof["training_pool_sha256"]
            or collection["split"] != "train"
        ):
            raise ValueError(
                "Fresh games require both-POV native validation and the same training pool"
            )
        for name, digest in fresh_proof["files"].items():
            path = (fresh / name).resolve()
            if not path.is_relative_to(fresh.resolve()) or sha256(path) != digest:
                raise ValueError("Audited fresh game changed: " + name)
        reference = json.loads((contract / "cpu/contract.json").read_text())
        if not reference["passed"] or reference["checkpoint_sha256"] != KAKUNA_34_SHA256:
            raise ValueError("CPU probability contract required before GPU packaging")
    name = f"fog-native-{stage}-{datetime.now(UTC):%y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    folder = directory(name)
    folder.mkdir(parents=True)
    upload = folder / "dataset"
    upload.mkdir()
    additions = {}

    def add(path, target):
        path = Path(path).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError("Bundle input must be an existing workspace file: " + str(path))
        if target in additions:
            raise ValueError("Duplicate bundle destination: " + target)
        additions[target] = path

    for path in (ROOT / "fog").rglob("*"):
        if (
            path.is_file()
            and path.suffix in (".py", ".js", ".json")
            and "__pycache__" not in path.parts
        ):
            add(path, path.relative_to(ROOT).as_posix())
    add(ROOT / "LICENSE", "LICENSE")
    add(ROOT / "integrations/THIRD_PARTY.md", "integrations/THIRD_PARTY.md")
    add(ROOT / "integrations/LICENSE.kakuna", "integrations/LICENSE.kakuna")
    add(checkpoint, "inputs/kakuna34.pt")
    provenance = checkpoint.with_suffix(checkpoint.suffix + ".json")
    if provenance.exists():
        add(provenance, "inputs/kakuna34.provenance.json")
    revisions = {}
    for upstream, expected in UPSTREAM.items():
        if upstream == "foul-play" and stage != "collect":
            continue
        source = ROOT / ".tools/upstream" / upstream
        revision = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != expected:
            raise ValueError("Pinned upstream revision mismatch: " + upstream)
        subprocess.run(
            ["git", "-C", str(source), "diff", "--exit-code", "--quiet", "HEAD"], check=True
        )
        revisions[upstream] = revision
        subtree = "src" if upstream == "poke-env" else "fp" if upstream == "foul-play" else upstream
        for path in tracked_sources(
            source, [subtree, "pyproject.toml", "requirements.txt", "README.md", "LICENSE"]
        ):
            add(path, "upstream/" + upstream + "/" + path.relative_to(source).as_posix())
    frozen = {
        "id": name,
        "stage": stage,
        "phase": phase if stage == "train" else "CPU data collection",
        "created_at": datetime.now(UTC).isoformat(),
        "timeout_seconds": timeout,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "upstream_revisions": revisions,
        "trial": stage == "collect" and games < 1000,
        "automatic_promotion": False,
        "torch": "Kaggle 2.11.0; actual native probability gate required",
    }
    if stage == "collect":
        if games < 20 or games % 20 or not 0 <= shard < shards:
            raise ValueError("Collection uses complete 20-game opponent/seat-balanced schedules")
        add(pools / "train.json", "inputs/pools/train.json")
        add(pools / "manifest.json", "inputs/pools/manifest.json")
        add(ROOT / "integrations/foul_play_worker.py", "integrations/foul_play_worker.py")
        add(ROOT / "integrations/foul_play_sampling.py", "integrations/foul_play_sampling.py")
        add(ROOT / "integrations/LICENSE.foul-play", "integrations/LICENSE.foul-play")
        source = ROOT / ".tools/upstream/foul-play"
        for cache in ("pkmn_sets_cache", "smogon_stats_cache"):
            for path in (source / "fp/data" / cache).rglob("*.json"):
                add(path, "upstream/foul-play/" + path.relative_to(source).as_posix())
        for path in (ROOT / "server/pokemon-showdown/dist").rglob("*.js"):
            add(path, path.relative_to(ROOT).as_posix())
        for filename in ("package.json", "package-lock.json", "LICENSE"):
            add(ROOT / "server/pokemon-showdown" / filename, "server/pokemon-showdown/" + filename)
        source_learners = []
        import yaml

        for index, (cfg_path, weights) in enumerate(learners):
            target_cfg, target_weights = (
                f"inputs/learners/{index}.yaml",
                f"inputs/learners/{index}.pt",
            )
            cfg = yaml.safe_load(Path(cfg_path).read_text())
            portable = {
                "encoding": cfg.get("encoding", {}),
                "model": cfg.get("model", {}),
                "device": "cpu",
            }
            local_cfg = folder / f"learner-{index}.yaml"
            local_cfg.write_text(yaml.safe_dump(portable))
            add(local_cfg, target_cfg)
            add(weights, target_weights)
            source_learners.append(target_cfg + "=" + target_weights)
        if not source_learners:
            raise ValueError("A historical learner is required in the 30% opponent component")
        frozen["collection"] = {
            "games": games,
            "seed": 20261006,
            "shard": shard,
            "shards": shards,
            "learners": source_learners,
            "mix": [0.4, 0.3, 0.3],
        }
    else:
        recipe = json.loads(config.read_text())
        recipe.update(
            checkpoint="inputs/kakuna34.pt", published="inputs/published", fresh="inputs/fresh"
        )
        recipe_path = folder / "recipe.json"
        recipe_path.write_text(json.dumps(recipe, indent=2) + "\n")
        add(recipe_path, "recipe.json")
        for path in published.rglob("*"):
            if path.is_file() and path.suffix in (".lz4", ".json"):
                add(path, "inputs/published/" + path.relative_to(published).as_posix())
        for path in fresh.glob("*.json.gz"):
            add(path, "inputs/fresh/" + path.name)
        add(fresh / "collection.json", "inputs/fresh/collection.json")
        add(fresh / "manifest.json", "inputs/fresh/manifest.json")
        add(isolation, "inputs/published-isolation.json")
        for path in contract.rglob("*"):
            if path.is_file() and path.suffix in (".json", ".pt"):
                add(path, "inputs/contract/" + path.relative_to(contract).as_posix())
        frozen["fresh_audit"] = fresh_audit
        paired = {key: value for key, value in recipe.items() if key != "condition"}
        frozen["matched_recipe_sha256"] = hashlib.sha256(
            json.dumps(paired, sort_keys=True).encode()
        ).hexdigest()
    packages = subprocess.check_output(
        ["uv", "--no-config", "pip", "freeze", "--python", str(native_python)], text=True
    )
    requirements = [
        line
        for line in packages.splitlines()
        if "==" in line and not line.lower().startswith(("torch==", "pytest=="))
    ]
    requirements_path = folder / "requirements.native.txt"
    requirements_path.write_text("\n".join(requirements) + "\n")
    add(requirements_path, "requirements.native.txt")
    frozen["files"] = {target: sha256(path) for target, path in sorted(additions.items())}
    if stage == "train":
        matched_inputs = {
            key: value for key, value in frozen["files"].items() if key != "recipe.json"
        }
        frozen["matched_inputs_sha256"] = hashlib.sha256(
            json.dumps(matched_inputs, sort_keys=True).encode()
        ).hexdigest()
        frozen["condition"] = recipe["condition"]
        frozen["updates"] = recipe["updates"]
    write_json(folder / "experiment.json", frozen)
    with zipfile.ZipFile(
        upload / "bundle.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3
    ) as archive:
        for target, path in sorted(additions.items()):
            digest = hashlib.sha256()
            with path.open("rb") as original, archive.open("fog/" + target, "w") as output:
                while block := original.read(1024 * 1024):
                    digest.update(block)
                    output.write(block)
            if digest.hexdigest() != frozen["files"][target]:
                raise ValueError("Bundle input changed during preparation: " + target)
        archive.writestr("fog/experiment.json", json.dumps(frozen, indent=2))
    owner = json.loads((ROOT / "configs/kaggle_results.json").read_text())["kernel"].split("/")[0]
    identifier, kernel = f"{owner}/{name}", f"{owner}/{name}-run"
    write_json(
        upload / "dataset-metadata.json",
        {"id": identifier, "title": name, "licenses": [{"name": "other"}]},
    )
    notebook = folder / "notebook"
    notebook.mkdir()
    shutil.copy2(ROOT / "fog/scripts/pretrained_kaggle_worker.py", notebook / "run.py")
    write_json(
        notebook / "kernel-metadata.json",
        {
            "id": kernel,
            "title": name + "-run",
            "code_file": "run.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": stage == "train",
            "enable_internet": True,
            "dataset_sources": [identifier],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        },
    )
    sealed = [
        upload / "bundle.zip",
        upload / "dataset-metadata.json",
        notebook / "run.py",
        notebook / "kernel-metadata.json",
        folder / "experiment.json",
    ]
    write_json(
        folder / "seal.json", {path.relative_to(folder).as_posix(): sha256(path) for path in sealed}
    )
    write_json(
        folder / "state.json",
        {
            "phase": "prepared",
            "kernel": kernel,
            "dataset": identifier,
            "trial": frozen["trial"],
            "stage": stage,
        },
    )
    handoff(
        folder,
        f"Prepared immutable {stage} package. Next: submit; then collect and locally audit results.",
    )
    print(name, flush=True)
    return folder


def submit(folder):
    verify(folder)
    state = json.loads((folder / "state.json").read_text())
    if state["phase"] in ("submitting", "submitted", "collected"):
        raise ValueError("Already submitted or uncertain; inspect status before any retry")
    experiment = json.loads((folder / "experiment.json").read_text())
    if state["stage"] == "train":
        # Both sealed arms must exist before either is launched. Compare the actual
        # source/data inventories as well as recipes; paths alone do not establish parity.
        partners = []
        for descriptor in folder.parent.glob("fog-native-train-*/experiment.json"):
            other = json.loads(descriptor.read_text())
            keys = (
                "phase",
                "matched_recipe_sha256",
                "matched_inputs_sha256",
                "timeout_seconds",
                "updates",
            )
            if other.get("condition") != experiment["condition"] and all(
                other.get(key) == experiment[key] for key in keys
            ):
                verify(descriptor.parent)
                partners.append(other)
        if len(partners) != 1:
            raise ValueError(
                "Prepare exactly one sealed opposite-condition partner with identical inputs and budget"
            )
        quota = json.loads(cli("quota", "--format", "json"))
        gpu = next((row for row in quota if row.get("resource") == "GPU"), None)
        partner_state = json.loads((directory(partners[0]["id"]) / "state.json").read_text())
        needed = experiment["timeout_seconds"]
        if partner_state["phase"] not in ("submitted", "collected"):
            needed += partners[0]["timeout_seconds"]
        if gpu is None or float(gpu["remaining"].removesuffix("h")) * 3600 < needed:
            raise ValueError("A fresh quota check must cover both requested GPU arms")
    if state["phase"] == "prepared":
        print(cli("quota", "--format", "json"), flush=True)
        state["phase"] = "uploading"
        write_json(folder / "state.json", state)
        print(cli("datasets", "create", "-p", str(folder / "dataset")), flush=True)
        state["phase"] = "uploaded"
        write_json(folder / "state.json", state)
    try:
        status = json.loads(cli("datasets", "status", state["dataset"], "--format", "json"))
    except RuntimeError as error:
        if "403" in str(error) or "404" in str(error):
            handoff(
                folder,
                "Dataset is still becoming accessible. Retry submit once ready; no notebook was started.",
            )
            print("Dataset is still processing. No notebook started.", flush=True)
            return
        raise
    if "ready" not in str(status.get("status", "")).lower():
        handoff(
            folder, "Dataset processing; run submit again once it is ready. No notebook started."
        )
        print(json.dumps(status), flush=True)
        return
    if state["stage"] == "train":
        family = experiment["matched_recipe_sha256"] + ":" + experiment["matched_inputs_sha256"]
        reserve_gpu(
            ROOT / "runs/pretrained/gpu_budget.json",
            run_id=folder.name,
            phase=experiment["phase"],
            condition=experiment["condition"],
            seconds=experiment["timeout_seconds"],
            family=family,
        )
    state["phase"] = "submitting"
    write_json(folder / "state.json", state)
    arguments = [
        "kernels",
        "push",
        "-p",
        str(folder / "notebook"),
        "--timeout",
        str(json.loads((folder / "experiment.json").read_text())["timeout_seconds"]),
    ]
    if state["stage"] == "train":
        arguments += ["--accelerator", "NvidiaTeslaT4"]
    result = cli(*arguments)
    if re.search(r"^Kernel push error:", result, re.MULTILINE | re.IGNORECASE):
        raise RuntimeError(result)
    state.update(
        phase="submitted", submission=result, url="https://www.kaggle.com/code/" + state["kernel"]
    )
    write_json(folder / "state.json", state)
    handoff(
        folder,
        "Submitted privately. Collect, verify output hashes, and evaluate locally before continuing.",
    )
    print(result, flush=True)


def collect(folder):
    verify(folder)
    state = json.loads((folder / "state.json").read_text())
    if state.get("result"):
        print(state["result"])
        return
    status = cli("kernels", "status", state["kernel"])
    if any(word in status.lower() for word in ("running", "queued")):
        print("Notebook is still running or queued; results are not available yet.", flush=True)
        return
    with tempfile.TemporaryDirectory(prefix="fetch-native-", dir=folder) as temporary:
        python = client().with_name("python.exe" if sys.platform == "win32" else "python")
        subprocess.run(
            [
                str(python),
                "-m",
                "fog.scripts.kaggle_experiment",
                "_fetch",
                folder.name,
                "--destination",
                temporary,
            ],
            cwd=ROOT,
            check=True,
        )
        archive = Path(temporary) / "fog-experiment-output.zip"
        digest = sha256(archive)
        result = folder / "results" / digest[:16]
        with zipfile.ZipFile(archive) as zipped:
            validate_members(zipped)
            if json.loads(zipped.read("experiment.json")) != json.loads(
                (folder / "experiment.json").read_text()
            ):
                raise ValueError("Returned result does not match the frozen experiment")
            hashes = json.loads(zipped.read("output_hashes.json"))
            for name, expected in hashes.items():
                if hashlib.sha256(zipped.read(name)).hexdigest() != expected:
                    raise ValueError("Returned output hash mismatch: " + name)
            result.mkdir(parents=True)
            zipped.extractall(result)
        shutil.copy2(archive, folder / "output.zip")
        run = json.loads((Path(temporary) / "download.json").read_text())
    state.update(phase="collected", result=str(result), run_id=run["run_id"])
    write_json(folder / "state.json", state)
    handoff(
        folder,
        f"Collected into {result}. Inspect pipeline_state.json and hash-verified data/state. No model promoted.",
    )
    print(result, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--stage", choices=["collect", "train"], required=True)
    prepare.add_argument("--checkpoint", type=Path, required=True)
    prepare.add_argument("--native-python", type=Path, required=True)
    prepare.add_argument("--pools", type=Path)
    prepare.add_argument("--learner", action="append", default=[])
    prepare.add_argument("--games", type=int, default=1000)
    prepare.add_argument("--shard", type=int, default=0)
    prepare.add_argument("--shards", type=int, default=1)
    prepare.add_argument("--config", type=Path)
    prepare.add_argument("--published", type=Path)
    prepare.add_argument("--fresh", type=Path)
    prepare.add_argument("--contract", type=Path)
    prepare.add_argument("--isolation", type=Path)
    prepare.add_argument("--timeout", type=int, default=43200)
    prepare.add_argument("--phase", choices=["pilot", "main", "refinement"], default="pilot")
    for name in ("submit", "collect", "status"):
        command = commands.add_parser(name)
        command.add_argument("experiment")
    args = parser.parse_args()
    if args.command == "prepare":
        kwargs = vars(args).copy()
        kwargs.pop("command")
        kwargs["learners"] = [
            tuple(Path(part) for part in specification.split("=", 1))
            for specification in kwargs.pop("learner")
        ]
        stage = kwargs.pop("stage")
        freeze(stage, **kwargs)
    else:
        folder = directory(args.experiment)
        if args.command == "submit":
            submit(folder)
        elif args.command == "collect":
            collect(folder)
        else:
            print(
                cli("kernels", "status", json.loads((folder / "state.json").read_text())["kernel"])
            )


if __name__ == "__main__":
    main()
