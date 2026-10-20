"""Thin sealed GPU-head-training / separate CPU-evaluation notebook entrypoint."""

from pathlib import Path
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile


def execute_pair(commands, environment, output, timeout):
    """One independent condition per device; stop its sibling on failure."""
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from fog.scripts.pretrained_kaggle_worker import execute

    cancelled = threading.Event()

    def one(label, command, device):
        try:
            return execute(
                command, dict(environment, CUDA_VISIBLE_DEVICES=str(device)),
                Path(output) / f"{label}.log", timeout, isolated=False,
                cancel_event=cancelled,
                progress=lambda: {"condition": label, "status": "running"},
            )
        except BaseException:
            cancelled.set()
            raise

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(one, label, command, device) for label, command, device in commands]
        return [future.result() for future in futures]


def run():
    started = time.monotonic()
    work = Path("/kaggle/temp/fog")
    output = Path("/kaggle/working/prior-heads")
    bundles = list(Path("/kaggle/input").rglob("bundle.bin"))
    if len(bundles) != 1:
        raise ValueError("Require one sealed prior-head experiment")
    with zipfile.ZipFile(bundles[0]) as archive:
        descriptor = json.loads(archive.read("experiment.json"))
        names = archive.namelist()
        if (
            descriptor["format"] != "fog.prior-heads-experiment"
            or descriptor["version"] != 1
            or len(set(names)) != len(names)
            or set(names) != set(descriptor["files"]) | {"experiment.json"}
            or any(not (work / name).resolve().is_relative_to(work) for name in names)
        ):
            raise ValueError("Unsafe or inconsistent sealed input inventory")
        archive.extractall(work)
    for name, expected in descriptor["files"].items():
        with (work / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError("Changed sealed input: " + name)
    schema = json.loads((work / "inputs/LABEL_SCHEMA_AUDIT.json").read_text())
    if (
        not schema["passed"]
        or schema["data_manifest_sha256"] != descriptor["files"]["inputs/data/manifest.json"]
    ):
        raise ValueError("Cached label-schema audit does not bind this data manifest")
    if any(
        descriptor["files"][name] != expected for name, expected in schema["source_files"].items()
    ):
        raise ValueError("Cached labels and runtime categorical schemas differ")
    os.chdir(work)
    sys.path.insert(0, str(work))
    from fog.scripts.pretrained_kaggle_worker import execute
    from fog.scripts.teacher_pilot_kaggle_worker import export_output

    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(work / "experiment.json", output / "experiment.json")
    environment = dict(
        os.environ,
        PYTHONPATH=str(work),
        PYTHONUTF8="1",
        PYTHONIOENCODING="utf-8",
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        TORCHDYNAMO_DISABLE="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        HF_HUB_OFFLINE="1",
    )
    status, error, phase = "failed", None, "setup"

    def child(command):
        remaining = descriptor["timeout_seconds"] - (time.monotonic() - started) - 120
        return execute(
            command,
            environment,
            output / "runner.log",
            remaining,
            isolated=False,
            progress=lambda: {"phase": phase, "elapsed_seconds": time.monotonic() - started},
        )

    try:
        import torch
        import numpy

        platform = {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": numpy.__version__,
            "cuda": torch.version.cuda,
            "gpu_available": torch.cuda.is_available(),
            "requested_device": descriptor["device"],
            "cuda_devices": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ],
        }
        (output / "PLATFORM.json").write_text(json.dumps(platform, indent=2) + "\n")
        if descriptor["stage"] == "train":
            if (
                descriptor["device"] != "cuda"
                or not torch.cuda.is_available()
                or torch.cuda.device_count() != 2
                or torch.__version__.split("+")[0] != "2.11.0"
            ):
                raise ValueError(
                    "This GPU recipe requires two CUDA devices and tested PyTorch 2.11"
                )
            config = json.loads((work / "inputs/recipe.json").read_text())
            if config.get("execution") != "parallel_conditions":
                raise ValueError("An explicit independent two-device recipe is required")
            phase = "actual CUDA full-state resume and initial probability proof"
            execute_pair(
                [
                    (mode, [sys.executable, "-m", "fog.scripts.prior_heads_check",
                            "--config", "inputs/recipe.json", "--out",
                            str(output / f"startup-proof-{mode}"), "--device", "cuda",
                            "--condition", mode], device)
                    for device, mode in enumerate(("facts", "native"))
                ],
                environment, output / "startup-logs",
                descriptor["timeout_seconds"] - (time.monotonic() - started) - 120,
            )
            phase = "concurrent production-shape CUDA throughput"
            execute_pair(
                [
                    (mode, [sys.executable, "-m", "fog.scripts.prior_heads_check",
                            "--config", "inputs/recipe.json", "--out",
                            str(output / f"startup-proof-{mode}-throughput"), "--device", "cuda",
                            "--condition", mode, "--benchmark-only"], device)
                    for device, mode in enumerate(("facts", "native"))
                ],
                environment, output / "throughput-logs",
                descriptor["timeout_seconds"] - (time.monotonic() - started) - 120,
            )
            measurements = []
            for mode in ("facts", "native"):
                proof = json.loads((output / f"startup-proof-{mode}/RESULTS.json").read_text())
                throughput = json.loads(
                    (output / f"startup-proof-{mode}-throughput/RESULTS.json").read_text()
                )
                measurements.append((throughput["measurement_started_unix"],
                                     throughput["measurement_finished_unix"]))
                if (
                    not proof["passed"] or not proof["CUDA_rng_saved"]
                    or proof["condition"] != mode or throughput["condition"] != mode
                    or not proof["probability_agreement"]["actual_cuda"]
                    or not throughput["passed"] or not throughput["actual_cuda"]
                    or throughput["main_updates_on_device"] != config["updates"]
                    or not throughput["main_initialization_unchanged"]
                ):
                    raise ValueError("Actual per-device startup/throughput proof failed: " + mode)
            overlap = min(end for _, end in measurements) - max(start for start, _ in measurements)
            if overlap < 0.5 * min(end - start for start, end in measurements):
                raise ValueError("Throughput measurements did not sufficiently overlap")
            phase = "matched frozen-policy residual head training"
            execute_pair(
                [
                    (mode, [sys.executable, "-m", "fog.scripts.train_prior_heads",
                            "--config", "inputs/recipe.json", "--out",
                            str(output / "training" / mode), "--device", "cuda",
                            "--condition", mode], device)
                    for device, mode in enumerate(("facts", "native"))
                ],
                environment, output / "training-logs",
                descriptor["timeout_seconds"] - (time.monotonic() - started) - 120,
            )
            from fog.pretrained.paired_belief import merge_conditions

            result = merge_conditions(config, output / "training")
            if (
                not result["complete"]
                or result["steps"] != 2 * config["updates"]
                or not result["matched_query_routes"]
                or not result["outcome_heads_identical"]
                or result["native_policy_changed"]
                or result["native_optimizer_present"]
            ):
                raise ValueError("Matched GPU result failed its exposure/isolation audit")
            phase = "final CPU/CUDA probability proof"
            from fog.scripts.prior_heads_check import probabilities
            from fog.pretrained.frozen_belief import FrozenBeliefData
            import random

            data = FrozenBeliefData(config["data"], config["data_sha256"])
            specs = data.draw(random.Random(config["seed"]), 8, split="development")
            contracts = {}
            for label in ["prior_only", "facts_heads_final", "native_heads_final"]:
                state = torch.load(
                    output / f"training/{label}.pt", map_location="cpu", weights_only=True
                )
                selected = "cuda:1" if label == "native_heads_final" else "cuda:0"
                contracts[label] = probabilities(state["heads"], config, specs, device=selected)
            (output / "FINAL_PROBABILITIES.json").write_text(json.dumps(contracts, indent=2) + "\n")
        elif descriptor["stage"] == "evaluate":
            if descriptor["device"] != "cpu":
                raise ValueError("The declared downstream evaluation is CPU-only")
            phase = "pinned native/evaluator runtime and simulator setup"
            child(
                [
                    sys.executable,
                    "-m",
                    "fog.scripts.pretrained_setup",
                    "--stage",
                    "evaluate",
                    "--device",
                    "cpu",
                    "--root",
                    str(work),
                    "--out",
                    str(output / "environment"),
                ]
            )
            runtimes = json.loads((output / "environment/runtimes.json").read_text())
            phase = "fresh disjoint-roster development collection"
            child(
                [
                    runtimes["evaluator"],
                    "-m",
                    "fog.scripts.prior_belief_contexts",
                    "collect",
                    "--config",
                    "inputs/CPU_RECIPE.json",
                    "--out",
                    str(output / "fresh"),
                    "--native-python",
                    runtimes["native"],
                ]
            )
            phase = "whole-history frozen native context extraction"
            from fog.pretrained.runtime import native_environment

            native_env = native_environment(work)
            for key in ("PYTHONPATH", "METAMON_CACHE_DIR"):
                environment[key] = native_env[key]
            child(
                [
                    runtimes["native"],
                    "-m",
                    "fog.scripts.prior_belief_contexts",
                    "encode",
                    "--config",
                    "inputs/CPU_RECIPE.json",
                    "--source",
                    str(output / "fresh"),
                    "--out",
                    str(output / "fresh-data"),
                ]
            )
            # Keep upstream native source paths out of the evaluator interpreter.
            environment["PYTHONPATH"] = str(work)
            config = json.loads((work / "inputs/CPU_RECIPE.json").read_text())
            config.update(
                training="inputs/training",
                data="inputs/data",
                fresh_data=str(output / "fresh-data"),
                initial_heads="inputs/initial-heads.pt",
                prior="inputs/prior.npz",
            )
            local = output / "evaluation_config.json"
            local.write_text(json.dumps(config, indent=2) + "\n")
            phase = "matched AR reconstruction, calibration and context ablation"
            child(
                [
                    runtimes["evaluator"],
                    "-m",
                    "fog.scripts.evaluate_prior_heads",
                    "--config",
                    str(local),
                    "--out",
                    str(output / "evaluation"),
                ]
            )
            if not json.loads((output / "evaluation/RESULTS.json").read_text())["complete"]:
                raise ValueError("CPU reconstruction evaluation did not complete")
        else:
            raise ValueError("Unknown sealed prior-head experiment stage")
        status = "complete"
    except Exception as failure:
        error = type(failure).__name__ + ": " + str(failure)
        raise
    finally:
        state = {
            "status": status,
            "error": error,
            "phase": phase,
            "returncode": 0 if status == "complete" else 1,
            "elapsed_seconds": time.monotonic() - started,
            "gpu_requested": descriptor["device"] == "cuda",
            "native_policy_changed": False,
            "automatic_promotion": False,
        }
        (output / "pipeline_state.json").write_text(json.dumps(state, indent=2) + "\n")
        export_output(output, "/kaggle/working/fog-experiment-output")
        print(json.dumps(state), flush=True)


if __name__ == "__main__":
    run()
