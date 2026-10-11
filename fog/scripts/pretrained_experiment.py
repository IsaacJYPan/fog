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
from fog.pretrained.protocol import PHASE_SECONDS, reserve_gpu
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
    slots=None,
    attempt_offset=0,
    cpu_preflight=False,
):
    if stage not in ("collect", "train", "evaluate"):
        raise ValueError("Unknown native experiment stage")
    if cpu_preflight and stage != "train":
        raise ValueError("CPU pipeline preflight uses the training path")
    if attempt_offset < 0 or (stage != "collect" and (slots is not None or attempt_offset)):
        raise ValueError("Selected slots and retry offsets apply only to collection")
    if sha256(checkpoint) != KAKUNA_34_SHA256:
        raise ValueError("Freeze requires original Kakuna checkpoint 34")
    if not 600 <= timeout <= 43200:
        raise ValueError("Kaggle background budget must be 10 minutes to 12 hours")
    if stage == "train":
        if timeout > PHASE_SECONDS[phase] // 2:
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
        "phase": "CPU development evaluation"
        if stage == "evaluate"
        else "CPU pipeline preflight"
        if cpu_preflight
        else phase
        if stage == "train"
        else "CPU data collection",
        "created_at": datetime.now(UTC).isoformat(),
        "timeout_seconds": timeout,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "upstream_revisions": revisions,
        "trial": cpu_preflight or (stage == "collect" and (games < 1000 or slots is not None)),
        "automatic_promotion": False,
        "device": "cpu" if cpu_preflight or stage == "evaluate" else "cuda",
        "cpu_preflight": cpu_preflight,
        "requires_cpu_evaluation_queue": stage == "train" and not cpu_preflight,
        "torch": "Kaggle 2.11.0; actual native probability gate required",
    }
    if stage in ("collect", "evaluate"):
        for path in (ROOT / "server/pokemon-showdown/dist").rglob("*.js"):
            add(path, path.relative_to(ROOT).as_posix())
        for filename in ("package.json", "package-lock.json", "LICENSE"):
            add(ROOT / "server/pokemon-showdown" / filename, "server/pokemon-showdown/" + filename)
    if stage == "evaluate":
        from fog.scripts.pretrained_evaluation_job import validate_recipe
        from fog.scripts.pretrained_evaluation_queue import validate_queue
        from fog.pretrained.data import assert_disjoint

        recipe = json.loads(config.read_text())
        queued = "training_sources" in recipe
        if queued:
            validate_queue(recipe, ROOT)
        else:
            validate_recipe(recipe)
        pool_data = {
            split: json.loads((pools / (split + ".json")).read_text())
            for split in ("train", "development", "final")
        }
        assert_disjoint(pool_data)
        for filename in ("train.json", "development.json", "final.json", "manifest.json"):
            add(pools / filename, "inputs/pools/" + filename)
        # Include the same integration sources in the evaluation fingerprint.
        for path in (ROOT / "integrations").rglob("*"):
            if (
                path.is_file()
                and path.suffix in (".py", ".js", ".json")
                and "__pycache__" not in path.parts
            ):
                add(path, path.relative_to(ROOT).as_posix())
        for condition, row in recipe.get("training_sources", {}).items():
            descriptor = Path(row["experiment"])
            verify(descriptor.parent)
            source_state = json.loads((descriptor.parent / "state.json").read_text())
            if source_state["kernel"] != row["kernel"]:
                raise ValueError("Training notebook differs from its frozen identity")
            target = f"inputs/training/{condition}/experiment.json"
            add(descriptor, target)
            row["experiment"] = target
        for condition, row in recipe.get("conditions", {}).items():
            native = Path(row["native"])
            if sha256(native) != row["native_sha256"]:
                raise ValueError("Evaluation candidate changed: " + condition)
            for field, filename in (
                ("native", "native.pt"),
                ("reference", "probabilities.pt"),
                ("auxiliary", "auxiliary.pt"),
            ):
                if field in row:
                    target = f"inputs/candidates/{condition}/{filename}"
                    add(row[field], target)
                    row[field] = target
        if recipe.get("belief"):
            cohort = Path(recipe["belief"]["fresh"])
            collection = json.loads((cohort / "collection.json").read_text())
            if collection["split"] != "development" or collection["pool_sha256"] != sha256(
                pools / "development.json"
            ):
                raise ValueError("Belief evaluation requires the held-out development pool")
            add(cohort / "collection.json", "inputs/belief/collection.json")
            records = sorted(cohort.glob("*.json.gz"))
            if len(records) != recipe["belief"]["games"]:
                raise ValueError("Belief evaluation cohort must be complete")
            from fog.pretrained.loading import read_record

            for path in records:
                record = read_record(path)
                if (
                    not record["training_eligible"]
                    or record.get("failures")
                    or record.get("fallbacks")
                ):
                    raise ValueError("Failed belief evaluation game requires diagnosis")
                add(path, "inputs/belief/" + path.name)
            recipe["belief"]["fresh"] = "inputs/belief"
        add(contract / "inputs.pt", "inputs/contract/inputs.pt")
        add(contract / "cpu/probabilities.pt", "inputs/contract/cpu/probabilities.pt")
        recipe_path = folder / "evaluation.json"
        write_json(recipe_path, recipe)
        add(recipe_path, "evaluation.json")
        frozen["evaluation"] = {key: value for key, value in recipe.items() if key != "conditions"}
        if queued:
            frozen["training_sources"] = recipe["training_sources"]
            frozen["trigger_sources"] = recipe.get("trigger_sources", [])
        else:
            frozen["conditions"] = {
                name: row["native_sha256"] for name, row in recipe["conditions"].items()
            }
    elif stage == "collect":
        if games < 20 or games % 20 or not 0 <= shard < shards:
            raise ValueError("Collection uses complete 20-game opponent/seat-balanced schedules")
        add(pools / "train.json", "inputs/pools/train.json")
        add(pools / "manifest.json", "inputs/pools/manifest.json")
        add(ROOT / "integrations/foul_play_worker.py", "integrations/foul_play_worker.py")
        add(ROOT / "integrations/foul_play_sampling.py", "integrations/foul_play_sampling.py")
        add(ROOT / "integrations/foul_play_processes.py", "integrations/foul_play_processes.py")
        add(ROOT / "integrations/LICENSE.foul-play", "integrations/LICENSE.foul-play")
        source = ROOT / ".tools/upstream/foul-play"
        for cache in ("pkmn_sets_cache", "smogon_stats_cache"):
            for path in (source / "fp/data" / cache).rglob("*.json"):
                add(path, "upstream/foul-play/" + path.relative_to(source).as_posix())
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
            "attempt_offset": attempt_offset,
        }
        if slots is not None:
            selected = json.loads(slots.read_text())
            if (
                not selected
                or len(selected) != len(set(selected))
                or any(type(index) is not int or not 0 <= index < games for index in selected)
            ):
                raise ValueError("Recovery slots must identify unique existing schedule positions")
            add(slots, "inputs/slots.json")
            frozen["collection"]["slots"] = selected
            if attempt_offset:
                provenance = slots.with_suffix(".provenance.json")
                if not provenance.is_file():
                    raise ValueError(
                        "Recovery requires parent result provenance beside selected slots"
                    )
                add(provenance, "inputs/recovery_provenance.json")
    elif stage == "train":
        recipe = json.loads(config.read_text())
        if cpu_preflight and (recipe["condition"] != "joint" or not 1 <= recipe["updates"] <= 2):
            raise ValueError("CPU preflight requires one or two full joint optimizer updates")
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
        upload / "bundle.bin", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3
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
    entrypoint = "run.ipynb" if frozen.get("training_sources") else "run.py"
    write_notebook_runner(notebook / entrypoint, timeout)
    write_json(
        notebook / "kernel-metadata.json",
        {
            "id": kernel,
            "title": name + "-run",
            "code_file": entrypoint,
            "language": "python",
            "kernel_type": "notebook" if entrypoint.endswith(".ipynb") else "script",
            "is_private": True,
            "enable_gpu": stage == "train" and not cpu_preflight,
            "enable_internet": True,
            "dataset_sources": [identifier],
            "competition_sources": [],
            "kernel_sources": [row["kernel"] for row in frozen.get("training_sources", {}).values()]
            + frozen.get("trigger_sources", []),
            "model_sources": [],
        },
    )
    sealed = [
        upload / "bundle.bin",
        upload / "dataset-metadata.json",
        notebook / entrypoint,
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
            "gpu_requested": stage == "train" and not cpu_preflight,
        },
    )
    handoff(
        folder,
        f"Prepared immutable {stage} package. Next: submit; then collect and locally audit results.",
    )
    print(name, flush=True)
    return folder


def write_notebook_runner(destination, timeout):
    source = (ROOT / "fog/scripts/pretrained_kaggle_worker.py").read_text(encoding="utf-8")
    marker = "FROZEN_TIMEOUT_SECONDS = 43200"
    if source.count(marker) != 1:
        raise ValueError("Expected exactly one frozen notebook deadline")
    source = source.replace(marker, "FROZEN_TIMEOUT_SECONDS = " + str(timeout))
    if destination.suffix == ".ipynb":
        # Scheduled script versions can be treated as notebook JSON by Kaggle.
        # A real notebook runs the same sealed watchdog with a clean argv.
        cell = (
            "from pathlib import Path\nimport subprocess\nimport sys\n"
            "runner = Path('/kaggle/temp/fog-evaluation-entry.py')\n"
            f"runner.write_text({source!r}, encoding='utf-8')\n"
            "subprocess.run([sys.executable, str(runner)], check=True)\n"
        )
        write_json(
            destination,
            {
                "nbformat": 4,
                "nbformat_minor": 5,
                "metadata": {
                    "kernelspec": {
                        "display_name": "Python 3",
                        "language": "python",
                        "name": "python3",
                    }
                },
                "cells": [
                    {
                        "cell_type": "code",
                        "id": "evaluation",
                        "metadata": {},
                        "execution_count": None,
                        "outputs": [],
                        "source": cell.splitlines(keepends=True),
                    }
                ],
            },
        )
    else:
        destination.write_text(source, encoding="utf-8")


def submit(folder, *, save_only=False, bootstrap=False):
    verify(folder)
    state = json.loads((folder / "state.json").read_text())
    if state["phase"] in ("saving", "submitting", "submitted", "collected") or (
        save_only and state["phase"] in ("saved", "registered")
    ):
        raise ValueError("Already submitted or uncertain; inspect status before any retry")
    experiment = json.loads((folder / "experiment.json").read_text())
    gpu_requested = state["stage"] == "train" and not experiment.get("cpu_preflight", False)
    if save_only and experiment.get("training_sources"):
        raise ValueError(
            "Queued evaluation requires an initial CPU submission; Quick Save can skip scheduled execution"
        )
    if bootstrap and (not save_only or not gpu_requested):
        raise ValueError("Registration is a CPU placeholder for a frozen GPU training notebook")
    validate_submission_identity(folder, experiment, state, gpu_requested=gpu_requested)
    if gpu_requested and not save_only:
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
        if experiment.get("requires_cpu_evaluation_queue"):
            require_cpu_queue(folder, experiment, partners[0])
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
    if gpu_requested and not save_only:
        family = experiment["matched_recipe_sha256"] + ":" + experiment["matched_inputs_sha256"]
        reserve_gpu(
            ROOT / "runs/pretrained/gpu_budget.json",
            run_id=folder.name,
            phase=experiment["phase"],
            condition=experiment["condition"],
            seconds=experiment["timeout_seconds"],
            family=family,
        )
    state["phase"] = "saving" if save_only else "submitting"
    write_json(folder / "state.json", state)
    if save_only:
        python = client().with_name("python.exe" if sys.platform == "win32" else "python")
        command = [
            str(python),
            "-m",
            "fog.scripts.kaggle_quick_save",
            "--folder",
            str(folder / "notebook"),
            "--timeout",
            str(experiment["timeout_seconds"]),
        ]
        if bootstrap:
            command += ["--bootstrap", "--experiment-sha256", sha256(folder / "experiment.json")]
        result = subprocess.check_output(command, cwd=ROOT, text=True)
        registration = json.loads(result)
        if registration["executed"] is not bootstrap or registration["kernel"] != state["kernel"]:
            raise ValueError("Non-executing registration returned an unexpected identity")
        state.update(
            phase="registered" if bootstrap else "saved",
            registration=registration,
            url="https://www.kaggle.com/code/" + state["kernel"],
        )
        write_json(folder / "state.json", state)
        handoff(
            folder,
            "Registered a CPU placeholder; training has not started. Wait for its output, "
            "save and arm the queued CPU evaluation, then submit the real GPU package."
            if bootstrap
            else "Saved privately without execution. Configure the data-update schedule "
            "for queued CPU evaluation, or submit this registered training package to launch.",
        )
        print(result, flush=True)
        return
    arguments = [
        "kernels",
        "push",
        "-p",
        str(folder / "notebook"),
        "--timeout",
        str(json.loads((folder / "experiment.json").read_text())["timeout_seconds"]),
    ]
    if gpu_requested:
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


def validate_submission_identity(folder, experiment, state, *, gpu_requested):
    dataset = json.loads((folder / "dataset/dataset-metadata.json").read_text())
    notebook = json.loads((folder / "notebook/kernel-metadata.json").read_text())
    if (
        experiment["id"] != folder.name
        or dataset["id"] != state["dataset"]
        or dataset["id"].split("/")[-1] != folder.name
        or notebook["id"] != state["kernel"]
        or notebook["id"] != dataset["id"] + "-run"
        or state["stage"] != experiment["stage"]
        or notebook.get("is_private") is not True
        or notebook.get("enable_gpu") is not gpu_requested
        or notebook.get("kernel_sources", [])
        != (
            [row["kernel"] for row in experiment.get("training_sources", {}).values()]
            + experiment.get("trigger_sources", [])
        )
    ):
        raise ValueError(
            "Frozen experiment, dataset, private notebook and resource identities must match"
        )
    if state["phase"] not in ("prepared", "uploading", "uploaded", "saved", "registered"):
        raise ValueError("This experiment is not in a valid submission phase")


def require_cpu_queue(folder, experiment, partner):
    """Do not allocate new GPU time until its private CPU handoff has been armed."""
    expected = {
        row["condition"]: sha256(directory(row["id"]) / "experiment.json")
        for row in (experiment, partner)
    }
    ready = []
    for path in folder.parent.glob("fog-native-evaluate-*/experiment.json"):
        candidate = json.loads(path.read_text())
        sources = candidate.get("training_sources", {})
        if {arm: row["experiment_sha256"] for arm, row in sources.items()} != expected:
            continue
        verify(path.parent)
        state = json.loads((path.parent / "state.json").read_text())
        receipt_path = path.parent / "SCHEDULE.json"
        if not receipt_path.is_file():
            continue
        receipt = json.loads(receipt_path.read_text())
        metadata = json.loads((path.parent / "notebook/kernel-metadata.json").read_text())
        if (
            candidate["stage"] == "evaluate"
            and candidate["device"] == "cpu"
            and not candidate["evaluation"].get("pilot_only")
            and metadata["is_private"] is True
            and metadata["enable_gpu"] is False
            and metadata["kernel_type"] == "notebook"
            and receipt.get("enabled") is True
            and receipt.get("trigger") == "data_updates"
            and receipt.get("kernel") == state["kernel"]
            and receipt.get("sources") == metadata["kernel_sources"]
            and state["phase"] in ("submitted", "collected")
        ):
            ready.append(candidate["id"])
    if len(ready) != 1:
        raise ValueError(
            "Arm exactly one hash-matched private CPU evaluation schedule before GPU launch"
        )


def collect(folder):
    verify(folder)
    state = json.loads((folder / "state.json").read_text())
    experiment = json.loads((folder / "experiment.json").read_text())
    scheduled = bool(experiment.get("training_sources"))
    if state.get("result") and not scheduled:
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
        run = json.loads((Path(temporary) / "download.json").read_text())
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
            if not result.exists():
                result.mkdir(parents=True)
                zipped.extractall(result)
        saved_archive = (
            folder / "outputs" / f"{run['run_id']}-{digest[:16]}.zip"
            if scheduled
            else folder / "output.zip"
        )
        if not saved_archive.exists():
            saved_archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(archive, saved_archive)
    state.update(
        phase="collected",
        result=str(result),
        run_id=run["run_id"],
        output_archive=str(saved_archive),
    )
    if scheduled:
        entry = {
            "run_id": run["run_id"],
            "result": str(result),
            "archive": str(saved_archive),
            "pipeline_status": json.loads((result / "pipeline_state.json").read_text())["status"],
        }
        if entry not in state.setdefault("collected_runs", []):
            state["collected_runs"].append(entry)
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
    prepare.add_argument("--stage", choices=["collect", "train", "evaluate"], required=True)
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
    prepare.add_argument("--phase", choices=list(PHASE_SECONDS), default="pilot")
    prepare.add_argument("--slots", type=Path)
    prepare.add_argument("--attempt-offset", type=int, default=0)
    prepare.add_argument("--cpu-preflight", action="store_true")
    for name in ("submit", "save", "register", "collect", "status"):
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
        if args.command in ("submit", "save", "register"):
            submit(folder, save_only=args.command != "submit", bootstrap=args.command == "register")
        elif args.command == "collect":
            collect(folder)
        else:
            print(
                cli("kernels", "status", json.loads((folder / "state.json").read_text())["kernel"])
            )


if __name__ == "__main__":
    main()
