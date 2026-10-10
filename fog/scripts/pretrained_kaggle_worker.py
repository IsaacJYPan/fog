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

# Replaced when sealing a notebook, before any input preparation can begin.
FROZEN_TIMEOUT_SECONDS = 43200
EXPORT_RESERVE_SECONDS = 60
CLEANUP_RESERVE_SECONDS = 5


class DeadlineExceeded(RuntimeError):
    pass


def stop_process_tree(process, *, isolated):
    import signal

    if os.name == "nt" and isolated:
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW,
            check=False,
        )
    elif isolated:
        # The group can still contain descendants after its leader has exited.
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                break
    elif process.poll() is None:
        process.kill()
    if process.poll() is None:
        process.wait(timeout=5)


def execute(command, environment, log, timeout, *, isolated=True, progress=None):
    """Keep console and downloadable logs while bounding the whole process tree."""
    if timeout <= 0:
        raise DeadlineExceeded("Frozen job deadline reached before starting subprocess")
    log.parent.mkdir(parents=True, exist_ok=True)
    flags = {}
    if os.name == "nt":
        flags["creationflags"] = subprocess.CREATE_NO_WINDOW
        if isolated:
            flags["creationflags"] |= subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        flags["start_new_session"] = isolated
    process = subprocess.Popen(
        command,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        **flags,
    )

    def copy_output():
        with log.open("a", encoding="utf-8") as stream:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end="", flush=True)

    reader = threading.Thread(target=copy_output, daemon=True)
    reader.start()
    deadline = time.monotonic() + timeout
    timed_out = False
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            try:
                code = process.wait(timeout=min(30, remaining))
                break
            except subprocess.TimeoutExpired:
                if progress and time.monotonic() < deadline:
                    print(json.dumps(progress()), flush=True)
    finally:
        # Nested commands remain in the worker's group so the supervisor also
        # stops installers, battle engines, and any abandoned grandchildren.
        if isolated or timed_out:
            stop_process_tree(process, isolated=isolated)
        reader.join(timeout=1)
    process.stdout.close()
    if timed_out:
        raise DeadlineExceeded("Frozen subprocess deadline exceeded")
    if code:
        raise RuntimeError("Experiment subprocess exit code " + str(code))
    return code


def prepare_inputs(input_root, work, output):
    write_progress(output, "input preparation")
    # Kaggle recursively expands ZIP and gzip dataset uploads. An opaque filename
    # preserves the exact compressed records and their frozen SHA-256 hashes.
    archives = list(input_root.rglob("bundle.bin"))
    if not archives:
        archives = list(input_root.rglob("bundle.zip"))
    if archives:
        if len(archives) != 1:
            raise ValueError("Expected one immutable private experiment bundle")
        with zipfile.ZipFile(archives[0]) as archive:
            experiment = json.loads(archive.read("fog/experiment.json"))
            write_json(output / "experiment.json", experiment)
            for entry in archive.infolist():
                target = (work.parent / entry.filename).resolve()
                if not target.is_relative_to(work) or entry.file_size > 16_000_000_000:
                    raise ValueError("Unsafe bundle member")
            archive.extractall(work.parent)
    else:
        # Kaggle automatically expands uploaded ZIP datasets on some runtimes.
        manifests = list(input_root.rglob("experiment.json"))
        if len(manifests) != 1:
            raise ValueError("Expected one immutable private experiment inventory")
        write_json(output / "experiment.json", json.loads(manifests[0].read_text()))
        shutil.copytree(manifests[0].parent, work)
    experiment = json.loads((work / "experiment.json").read_text())
    write_progress(output, "frozen input verification")
    for name, expected in experiment["files"].items():
        path = (work / name).resolve()
        if not path.is_relative_to(work):
            raise ValueError("Unsafe frozen inventory path")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError("Frozen file changed: " + name)
    shutil.copy2(work / "experiment.json", output / "experiment.json")
    return experiment


def write_json(path, value):
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_progress(output, stage):
    write_json(output / "pipeline_progress.json", {"stage": stage})
    print("Pipeline:", stage, flush=True)


def activate_frozen_code(work, environment):
    """A script retains its original sys.path[0] after chdir; activate verified code."""
    sys.path.insert(0, str(work))
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(work) + (os.pathsep + existing if existing else "")
    return environment


def run_pipeline(input_root, work, output, timeout):
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    experiment = prepare_inputs(input_root, work, output)
    if experiment["timeout_seconds"] != timeout:
        raise ValueError("Notebook deadline must match the frozen experiment")
    device = experiment.get("device", "cuda")
    if device not in ("cpu", "cuda") or (device == "cpu" and not experiment.get("cpu_preflight")):
        raise ValueError("CPU execution requires a separately frozen preflight recipe")
    os.chdir(work)
    environment = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        TORCHDYNAMO_DISABLE="1",
        METAMON_CACHE_DIR=str(work / ".native-cache"),
        WANDB_MODE="disabled",
        HF_HUB_DISABLE_TELEMETRY="1",
    )
    activate_frozen_code(work, environment)
    status, error, code = "failed", None, 1
    try:

        def remaining():
            return timeout - (time.monotonic() - started) - EXPORT_RESERVE_SECONDS - 10

        def child(command):
            return execute(command, environment, output / "runner.log", remaining(), isolated=False)

        write_progress(output, "environment setup")
        child(
            [
                sys.executable,
                "-m",
                "fog.scripts.pretrained_setup",
                "--stage",
                experiment["stage"],
                "--out",
                str(output / "environment"),
                "--device",
                device,
            ],
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
            write_progress(output, "original " + device + " probability gate")
            child(
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
                    device,
                    "--out",
                    str(output / "contract"),
                ],
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
                device,
            ]
            command += ["--max-seconds", str(max(1.0, remaining() - 180))]
        if remaining() < 1:
            raise RuntimeError("Setup exhausted the frozen job budget")
        write_progress(output, "training" if experiment["stage"] == "train" else "collection")
        code = child(command)
        if experiment["stage"] == "train":
            write_progress(output, "trained " + device + " probability gate")
            report = json.loads((output / "training/RESULTS.json").read_text())
            checkpoint = str(output / "training/native_final.pt")
            for contract_device in ("cpu", "cuda") if device == "cuda" else ("cpu",):
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
                    contract_device,
                    "--out",
                    str(output / "contract-trained" / contract_device),
                ]
                if contract_device == "cuda":
                    contract_command += [
                        "--reference",
                        str(output / "contract-trained/cpu/probabilities.pt"),
                    ]
                child(contract_command)
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
                    "operation": json.loads((output / "pipeline_progress.json").read_text())[
                        "stage"
                    ],
                    "error": error,
                    "elapsed_seconds": time.monotonic() - started,
                    "returncode": code,
                },
                indent=2,
            )
            + "\n"
        )
    if status == "failed":
        raise RuntimeError(error)


def export_outputs(output, archive_path):
    """Hash while streaming once; a deadline never publishes a partial ZIP."""
    temporary = archive_path.with_suffix(".part")
    files = {}
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(output.rglob("*")):
            if not path.is_file() or path.name == "output_hashes.json":
                continue
            if path.is_symlink():
                raise ValueError("Output symlinks are not exportable")
            name = path.relative_to(output).as_posix()
            digest = hashlib.sha256()
            with path.open("rb") as original, archive.open(name, "w", force_zip64=True) as target:
                while block := original.read(1024 * 1024):
                    digest.update(block)
                    target.write(block)
            files[name] = digest.hexdigest()
        contents = json.dumps(files, indent=2) + "\n"
        archive.writestr("output_hashes.json", contents)
    (output / "output_hashes.json").write_text(contents, encoding="utf-8")
    temporary.replace(archive_path)


def run(
    *,
    input_root=Path("/kaggle/input"),
    work=Path("/kaggle/temp/fog"),
    output=Path("/kaggle/working/results"),
    timeout=FROZEN_TIMEOUT_SECONDS,
    export_reserve=EXPORT_RESERVE_SECONDS,
    worker_command=None,
):
    started = time.monotonic()
    deadline = started + timeout
    output.mkdir(parents=True, exist_ok=True)
    log = output.parent / "supervisor.log"
    script = str(Path(__file__).resolve())
    environment = dict(os.environ, PYTHONUNBUFFERED="1")
    command = worker_command or [
        sys.executable,
        script,
        "--worker",
        "--input-root",
        str(input_root),
        "--work",
        str(work),
        "--output",
        str(output),
        "--timeout",
        str(timeout),
    ]
    error = None

    def progress():
        path = output / "pipeline_progress.json"
        stage = json.loads(path.read_text())["stage"] if path.exists() else "worker startup"
        return {
            "stage": stage,
            "elapsed_seconds": time.monotonic() - started,
            "remaining_seconds": max(0, deadline - time.monotonic()),
        }

    write_json(
        output / "supervisor_state.json",
        {
            "status": "running",
            "timeout_seconds": timeout,
            "export_reserve_seconds": export_reserve,
        },
    )
    print("Supervisor: whole job budget", timeout, "seconds", flush=True)
    try:
        execute(
            command,
            environment,
            log,
            deadline - time.monotonic() - export_reserve - CLEANUP_RESERVE_SECONDS,
            progress=progress,
        )
        if not (output / "pipeline_state.json").exists():
            raise RuntimeError("Worker exited without a pipeline result")
    except Exception as caught:
        error = type(caught).__name__ + ": " + str(caught)
        state_path = output / "pipeline_state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        state.update(
            status="failed", supervisor_error=error, elapsed_seconds=time.monotonic() - started
        )
        state.setdefault("stage", progress()["stage"])
        state.setdefault("error", error)
        write_json(state_path, state)
        print(error, flush=True)
    finally:
        if log.exists():
            shutil.copy2(log, output / "supervisor.log")
        write_json(
            output / "supervisor_state.json",
            {
                "status": "failed" if error else "worker returned",
                "error": error,
                "timeout_seconds": timeout,
                "export_reserve_seconds": export_reserve,
                "worker_elapsed_seconds": time.monotonic() - started,
            },
        )
        archive_path = output.parent / "fog-experiment-output.zip"
        print("Supervisor: exporting results", flush=True)
        try:
            execute(
                [sys.executable, script, "--export", str(output), str(archive_path)],
                environment,
                output.parent / "export.log",
                deadline - time.monotonic() - CLEANUP_RESERVE_SECONDS,
            )
        except Exception as caught:
            write_json(
                output.parent / "export_failure.json",
                {
                    "error": type(caught).__name__ + ": " + str(caught),
                    "elapsed_seconds": time.monotonic() - started,
                    "partial_results": str(output),
                },
            )
            raise
    if error:
        raise RuntimeError(error)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--export", nargs=2, type=Path)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--work", type=Path, default=Path("/kaggle/temp/fog"))
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/results"))
    parser.add_argument("--timeout", type=float, default=FROZEN_TIMEOUT_SECONDS)
    args = parser.parse_args()
    if args.export:
        export_outputs(*args.export)
    elif args.worker:
        run_pipeline(args.input_root, args.work, args.output, args.timeout)
    else:
        run(input_root=args.input_root, work=args.work, output=args.output, timeout=args.timeout)
