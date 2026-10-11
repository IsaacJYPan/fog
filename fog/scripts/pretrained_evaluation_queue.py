"""Resolve sealed GPU training outputs for a separately scheduled CPU evaluation."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re
import zipfile

from fog.pretrained.kakuna import KAKUNA_34_SHA256, sha256
from fog.scripts.kaggle_results import validate_members, write_json
from fog.scripts.pretrained_compare import matched_reports
from fog.scripts.pretrained_evaluation_job import validate_recipe

ARMS = ("ordinary", "joint")
MATCHED_KEYS = (
    "phase",
    "matched_recipe_sha256",
    "matched_inputs_sha256",
    "timeout_seconds",
    "updates",
)


def validate_queue(recipe, root):
    """Bind the evaluation to two immutable recipes before their weights exist."""
    sources = recipe.get("training_sources", {})
    if set(sources) != set(ARMS) or "conditions" in recipe or "matched" in recipe:
        raise ValueError("Queued evaluation requires two training sources, without candidates")
    experiments = []
    for arm in ARMS:
        source = sources[arm]
        path = (root / source["experiment"]).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != source["experiment_sha256"]:
            raise ValueError("Frozen training descriptor changed: " + arm)
        experiment = json.loads(path.read_text())
        if (
            experiment.get("stage") != "train"
            or experiment.get("condition") != arm
            or experiment.get("device") != "cuda"
            or experiment.get("cpu_preflight")
            or experiment.get("trial")
            or type(experiment.get("updates")) is not int
            or experiment["updates"] < 1
            or not re.fullmatch(
                r"[\w-]+/" + re.escape(experiment["id"]) + r"-run", source["kernel"]
            )
        ):
            raise ValueError("Queue source must be the corresponding frozen GPU training job")
        experiments.append(experiment)
    if experiments[0]["id"] == experiments[1]["id"] or any(
        experiments[0][key] != experiments[1][key] for key in MATCHED_KEYS
    ):
        raise ValueError("Queued training recipes, inputs and update budgets must match")
    triggers = recipe.get("trigger_sources", [])
    if (
        not isinstance(triggers, list)
        or len(triggers) != len(set(triggers))
        or any(
            not isinstance(value, str) or not re.fullmatch(r"[\w-]+/[\w-]+", value)
            for value in triggers
        )
        or (triggers and recipe.get("pilot_only") is not True)
        or set(triggers) & {source["kernel"] for source in sources.values()}
    ):
        raise ValueError(
            "Extra cloud trigger sources are limited to separately frozen queue pilots"
        )
    marker = recipe.get("activation_marker")
    if marker and (
        not triggers
        or recipe.get("pilot_only") is not True
        or not re.fullmatch(r"[\w-]+\.json", marker.get("filename", ""))
        or not re.fullmatch(r"[a-f0-9]{32}", marker.get("value", ""))
    ):
        raise ValueError("A queue-pilot activation marker requires a fixed CPU trigger")
    # Apply the same development-only, crossed-game and belief-cohort constraints.
    placeholder = deepcopy(recipe)
    placeholder.pop("training_sources")
    placeholder["conditions"] = {
        arm: {"native": arm, "reference": arm, "auxiliary": arm, "native_sha256": arm}
        for arm in ARMS
    }
    placeholder["matched"] = {"models": list(ARMS)}
    validate_recipe(placeholder)
    return experiments


def verify_source(archive_path, expected, destination, *, archive_sha256=None):
    """Stream all output hashes, retaining only evaluation inputs, not optimizer copies."""
    if archive_sha256 and sha256(archive_path) != archive_sha256:
        raise ValueError("Training output archive changed")
    update = expected["updates"]
    retained = {
        "experiment.json",
        "pipeline_state.json",
        "training/RESULTS.json",
        "training/recipe.json",
        "training/metrics.jsonl",
        "training/native_final.pt",
        f"training/native_{update:06d}.pt",
        f"training/auxiliary_{update:06d}.pt",
        "contract/contract.json",
        "contract-trained/cpu/contract.json",
        "contract-trained/cpu/probabilities.pt",
        "contract-trained/cuda/contract.json",
    }
    with zipfile.ZipFile(archive_path) as archive:
        validate_members(archive)
        if json.loads(archive.read("experiment.json")) != expected:
            raise ValueError("Training output belongs to a different frozen experiment")
        hashes = json.loads(archive.read("output_hashes.json"))
        members = {row.filename for row in archive.infolist() if not row.is_dir()}
        if members != set(hashes) | {"output_hashes.json"} or not retained <= set(hashes):
            raise ValueError("Training export lacks complete hashed evaluation inputs")
        for name, digest in hashes.items():
            target = destination / name
            if name in retained:
                target.parent.mkdir(parents=True, exist_ok=True)
            sink = target.open("wb") if name in retained else None
            calculated = hashlib.sha256()
            try:
                with archive.open(name) as stream:
                    while block := stream.read(1024 * 1024):
                        calculated.update(block)
                        if sink:
                            sink.write(block)
            finally:
                if sink:
                    sink.close()
            if calculated.hexdigest() != digest:
                raise ValueError("Training output hash mismatch: " + name)
        write_json(destination / "output_hashes.json", hashes)
    pipeline = json.loads((destination / "pipeline_state.json").read_text())
    report = json.loads((destination / "training/RESULTS.json").read_text())
    if (
        pipeline["status"] != "complete"
        or report.get("complete") is not True
        or report["completed_updates"] != update
        or report["planned_updates"] != update
        or report["condition"] != expected["condition"]
        or report.get("promotion")
    ):
        raise ValueError("Incomplete or failed GPU training cannot enter evaluation")
    config = json.loads((destination / "training/recipe.json").read_text())
    identity = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    paired = {key: value for key, value in config.items() if key != "condition"}
    if (
        identity != report["identity"]
        or hashlib.sha256(json.dumps(paired, sort_keys=True).encode()).hexdigest()
        != expected["matched_recipe_sha256"]
        or config["condition"] != expected["condition"]
        or config["updates"] != update
    ):
        raise ValueError("Returned training recipe differs from the frozen queue source")
    native_sha = hashes["training/native_final.pt"]
    if hashes[f"training/native_{update:06d}.pt"] != native_sha:
        raise ValueError("Final and selected common-update exports differ")
    samples = expected["files"]["inputs/contract/inputs.pt"]
    for name, checkpoint_sha in (
        ("contract/contract.json", KAKUNA_34_SHA256),
        ("contract-trained/cpu/contract.json", native_sha),
        ("contract-trained/cuda/contract.json", native_sha),
    ):
        contract = json.loads((destination / name).read_text())
        if (
            not contract.get("passed")
            or not contract.get("weights_unchanged")
            or contract["checkpoint_sha256"] != checkpoint_sha
            or contract["samples_sha256"] != samples
            or not 0 < contract["tolerance"] <= 1e-3
            or not contract.get("records")
        ):
            raise ValueError("GPU source probability contract failed")
        if name == "contract-trained/cuda/contract.json" and not contract.get("cpu_cuda_verified"):
            raise ValueError("Trained CPU/CUDA probability agreement is required")
        for row in contract["records"]:
            errors = [row["cache_max_absolute_error"]]
            if name == "contract-trained/cuda/contract.json":
                errors.append(row["cpu_cuda_max_absolute_error"])
            if any(not math.isfinite(value) or not 0 <= value <= 1e-3 for value in errors):
                raise ValueError("GPU source numerical error exceeds the contract")
    rows = [
        json.loads(line)
        for line in (destination / "training/metrics.jsonl").read_text().splitlines()
    ]
    if any(
        not math.isfinite(row["unclipped_gradient_norm"])
        or not all(math.isfinite(value) for value in row["losses"].values())
        for row in rows
    ):
        raise ValueError("Training source contains nonfinite optimization metrics")
    return {
        "experiment_id": expected["id"],
        "archive_sha256": sha256(archive_path),
        "output_hashes": hashes,
        "completed_updates": update,
        "native_sha256": native_sha,
        "all_export_hashes_verified": True,
        "full_training_state_tensor_audit": "Deferred to result review; state bytes were hash checked",
    }


def resolve_queue(recipe, input_root, work, output):
    """Return None while a source is absent; refuse wrong or failed completed sources."""
    experiments = validate_queue(recipe, work)
    marker = recipe.get("activation_marker")
    if marker:
        files = list(input_root.rglob(marker["filename"]))
        if not files:
            write_json(
                output / "queue/STATE.json",
                {
                    "status": "waiting",
                    "awaiting_activation": marker["filename"],
                    "gpu_requested": False,
                    "automatic_promotion": False,
                },
            )
            return None
        if len(files) != 1 or json.loads(files[0].read_text()) != {"activation": marker["value"]}:
            raise ValueError("Unexpected queue-pilot activation source")
    archives = {}
    expected_ids = {row["id"] for row in experiments}
    for path in input_root.rglob("fog-experiment-output.zip"):
        with zipfile.ZipFile(path) as archive:
            identifier = json.loads(archive.read("experiment.json"))["id"]
        if identifier in expected_ids:
            if identifier in archives:
                raise ValueError("Ambiguous duplicate training output")
            archives[identifier] = path
    missing = sorted(expected_ids - archives.keys())
    if missing:
        write_json(
            output / "queue/STATE.json",
            {
                "status": "waiting",
                "missing_experiments": missing,
                "gpu_requested": False,
                "automatic_promotion": False,
            },
        )
        return None
    roots, audits = [], {}
    for arm, experiment in zip(ARMS, experiments):
        if experiment["files"]["inputs/contract/inputs.pt"] != sha256(
            work / "inputs/contract/inputs.pt"
        ):
            raise ValueError("Queue and training probability fixtures differ")
        destination = work / "inputs/resolved" / arm
        audits[arm] = verify_source(
            archives[experiment["id"]],
            experiment,
            destination,
            archive_sha256=recipe["training_sources"][arm].get("archive_sha256"),
        )
        roots.append(destination)
    pair = matched_reports(
        roots[0] / "experiment.json", roots[1] / "experiment.json", roots[0], roots[1]
    )
    resolved = deepcopy(recipe)
    resolved.pop("training_sources")
    resolved["matched"] = pair
    resolved["conditions"] = {
        arm: {
            "native": (root / f"training/native_{pair['updates']:06d}.pt")
            .relative_to(work)
            .as_posix(),
            "native_sha256": pair["models"][index],
            "reference": (root / "contract-trained/cpu/probabilities.pt")
            .relative_to(work)
            .as_posix(),
            "auxiliary": (root / f"training/auxiliary_{pair['updates']:06d}.pt")
            .relative_to(work)
            .as_posix(),
        }
        for index, (arm, root) in enumerate(zip(ARMS, roots))
    }
    validate_recipe(resolved)
    write_json(work / "evaluation.resolved.json", resolved)
    write_json(
        output / "queue/RESOLVED.json",
        {
            "training_sources": audits,
            "evaluation": resolved,
            "automatic_promotion": False,
            "gpu_requested": False,
        },
    )
    write_json(output / "queue/STATE.json", {"status": "ready", "gpu_requested": False})
    return resolved
