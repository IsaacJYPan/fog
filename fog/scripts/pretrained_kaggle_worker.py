"""Thin entrypoint for a private, immutable native experiment."""

from pathlib import Path
import hashlib
import gzip
import json
import os
import shutil
import subprocess
import sys
import time
import threading
import zipfile


def execute(command, environment, log, timeout):
    """Keep console and downloadable logs while bounding the whole process tree."""
    import signal

    process = subprocess.Popen(
        command,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,
        bufsize=1,
    )

    def copy_output():
        with log.open("a", encoding="utf-8") as stream:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end="", flush=True)

    reader = threading.Thread(target=copy_output, daemon=True)
    reader.start()
    try:
        code = process.wait(timeout=max(1.0, timeout))
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        code = 124
    reader.join(timeout=5)
    process.stdout.close()
    if code:
        raise RuntimeError("Experiment subprocess exit code " + str(code))
    return code


def run():
    started = time.monotonic()
    work = Path("/kaggle/temp/fog")
    output = Path("/kaggle/working/results")
    archives = list(Path("/kaggle/input").rglob("bundle.zip"))
    if archives:
        if len(archives) != 1:
            raise ValueError("Expected one immutable private experiment bundle")
        with zipfile.ZipFile(archives[0]) as archive:
            for entry in archive.infolist():
                target = (work.parent / entry.filename).resolve()
                if not target.is_relative_to(work) or entry.file_size > 16_000_000_000:
                    raise ValueError("Unsafe bundle member")
            archive.extractall(work.parent)
    else:
        # Kaggle automatically expands uploaded ZIP datasets on some runtimes.
        manifests = list(Path("/kaggle/input").rglob("experiment.json"))
        if len(manifests) != 1:
            raise ValueError("Expected one immutable private experiment inventory")
        shutil.copytree(manifests[0].parent, work)
    os.chdir(work)
    output.mkdir(parents=True, exist_ok=True)
    experiment = json.loads((work / "experiment.json").read_text())
    for name, expected in experiment["files"].items():
        path = (work / name).resolve()
        if not path.is_relative_to(work):
            raise ValueError("Unsafe frozen inventory path")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError("Frozen file changed: " + name)
    shutil.copy2(work / "experiment.json", output / "experiment.json")
    environment = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        TORCHDYNAMO_DISABLE="1",
        METAMON_CACHE_DIR=str(work / ".native-cache"),
        WANDB_MODE="disabled",
        HF_HUB_DISABLE_TELEMETRY="1",
    )
    status, error, code = "failed", None, 1
    try:

        def remaining():
            return experiment["timeout_seconds"] - (time.monotonic() - started) - 300

        execute(
            [
                sys.executable,
                "-m",
                "fog.scripts.pretrained_setup",
                "--stage",
                experiment["stage"],
                "--out",
                str(output / "environment"),
            ],
            environment,
            output / "runner.log",
            remaining(),
        )
        runtimes = json.loads((output / "environment/runtimes.json").read_text())
        if experiment["stage"] == "collect":
            recipe = experiment["collection"]
            command = [
                runtimes["collector"],
                "-m",
                "fog.scripts.pretrained_collect",
                "--pools",
                "inputs/pools",
                "--out",
                str(output / "fresh"),
                "--games",
                str(recipe["games"]),
                "--seed",
                str(recipe["seed"]),
                "--split",
                "train",
                "--native-python",
                runtimes["native"],
                "--checkpoint",
                "inputs/kakuna34.pt",
                "--foul-play-python",
                runtimes["foulplay"],
                "--foul-play-source",
                "upstream/foul-play",
                "--shard",
                str(recipe["shard"]),
                "--shards",
                str(recipe["shards"]),
            ]
            for learner in recipe["learners"]:
                command += ["--learner", learner]
            command += ["--attempt-offset", str(recipe.get("attempt_offset", 0))]
            if "slots" in recipe:
                command += ["--slots", "inputs/slots.json"]
        else:
            from fog.pretrained.runtime import native_environment

            environment["PYTHONPATH"] = native_environment(work)["PYTHONPATH"]
            execute(
                [
                    runtimes["native"],
                    "-m",
                    "fog.scripts.pretrained_contract",
                    "--checkpoint",
                    "inputs/kakuna34.pt",
                    "--samples",
                    "inputs/contract/inputs.pt",
                    "--reference",
                    "inputs/contract/cpu/probabilities.pt",
                    "--device",
                    "cuda",
                    "--out",
                    str(output / "contract"),
                ],
                environment,
                output / "runner.log",
                remaining(),
            )
            command = [
                runtimes["native"],
                "-m",
                "fog.scripts.pretrained_train",
                "--config",
                "recipe.json",
                "--out",
                str(output / "training"),
                "--device",
                "cuda",
            ]
            command += ["--max-seconds", str(max(1.0, remaining() - 180))]
        if remaining() < 1:
            raise RuntimeError("Setup exhausted the frozen job budget")
        code = execute(command, environment, output / "runner.log", remaining())
        if experiment["stage"] == "train":
            report = json.loads((output / "training/RESULTS.json").read_text())
            checkpoint = str(output / "training/native_final.pt")
            for device in ("cpu", "cuda"):
                contract_command = [
                    runtimes["native"],
                    "-m",
                    "fog.scripts.pretrained_contract",
                    "--checkpoint",
                    checkpoint,
                    "--trained",
                    "--samples",
                    "inputs/contract/inputs.pt",
                    "--device",
                    device,
                    "--out",
                    str(output / "contract-trained" / device),
                ]
                if device == "cuda":
                    contract_command += [
                        "--reference",
                        str(output / "contract-trained/cpu/probabilities.pt"),
                    ]
                execute(contract_command, environment, output / "runner.log", remaining())
            status = "complete" if report["complete"] else "interrupted"
        else:
            recipe = experiment["collection"]
            selected = recipe.get("slots", range(recipe["games"]))
            expected = sum(index % recipe["shards"] == recipe["shard"] for index in selected)
            saved = list((output / "fresh").glob("*.json.gz"))
            expected_indices = {
                index for index in selected if index % recipe["shards"] == recipe["shard"]
            }
            actual_indices = {int(path.name.split(".")[0]) for path in saved}
            if not actual_indices <= expected_indices:
                raise ValueError("Returned collection contains an unplanned schedule slot")
            rows = [json.loads(gzip.decompress(path.read_bytes())) for path in saved]
            eligible = sum(bool(row.get("training_eligible")) for row in rows)
            audit = {
                "expected_slots": expected,
                "saved_slots": len(rows),
                "eligible_games": eligible,
                "failed_slots": len(rows) - eligible,
                "failed_attempts": len(list((output / "fresh/failed_attempts").glob("*.json.gz"))),
            }
            (output / "collection_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
            if len(rows) == expected and eligible != expected:
                raise RuntimeError(
                    "Collection completed with permanently failed slots; inspect recorded failures"
                )
            status = "complete" if eligible == expected else "interrupted"
    except Exception as caught:
        error = type(caught).__name__ + ": " + str(caught)
        print(error, flush=True)
    finally:
        (output / "pipeline_state.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "stage": experiment["stage"],
                    "error": error,
                    "elapsed_seconds": time.monotonic() - started,
                    "returncode": code,
                },
                indent=2,
            )
            + "\n"
        )
        files = {}
        for path in sorted(output.rglob("*")):
            if path.is_file():
                with path.open("rb") as stream:
                    files[path.relative_to(output).as_posix()] = hashlib.file_digest(
                        stream, "sha256"
                    ).hexdigest()
        (output / "output_hashes.json").write_text(json.dumps(files, indent=2) + "\n")
        shutil.make_archive("/kaggle/working/fog-experiment-output", "zip", output)
    if status == "failed":
        raise RuntimeError(error)


if __name__ == "__main__":
    run()
