"""Run fog commands (collection, evaluation) on a hosted CPU machine.

A script kernel with internet and no GPU, and one private dataset attached that holds the
network weights the commands need. It builds the pinned Showdown and the Gen 9 engine, then
runs each of `STEPS` from the repository root with one math thread a process. In a command,
`{out}` is the output folder, `{input:NAME}` the path of the file `NAME` in the dataset and
`{round:NAME}` the folder of the uploaded collection `NAME` (the host unpacks `NAME.tar`).
Every folder a step leaves in the output folder is stored as `<folder>.tar`; `job.json`
records times and exit statuses.

A job that overruns its timeout loses its output: give collection `--max-seconds`, size an
evaluation to finish, and submit with a timeout above the expected run time
(`kaggle kernels push -p <folder> --timeout <seconds>`).

Collection on several machines: give every machine its own `--seed` and its own `--out`
folder; `python -m fog.training.search_distill train --data` takes all the folders.

The hosted machine reports four CPUs that are two cores with two threads each. The search
budget is wall-clock, so use one process a core (`--workers 2`, `--shards 2`): with four
collection processes a label got 43,000 simulations where a four-core machine gives
117,000 (ledger, 2026-11-16).
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

COMMIT = "887bf60"
STEPS = [
    {"name": "round3b", "command": [
        "-m", "fog.training.search_distill", "collect", "--pool", "data/teams/replay_v1/train",
        "--out", "{out}/round3b", "--checkpoint", "{input:net2.pt}", "--games", "600",
        "--beta", "0.25", "--seed", "16003", "--agreement-every", "40", "--workers", "2",
        "--max-seconds", "7200"]},
]
NODE = "v22.11.0"

started = time.time()
temp, out = Path("/kaggle/temp"), Path("/kaggle/working")
root = temp / "fog"
temp.mkdir(parents=True, exist_ok=True)


def run(command, **kwargs):
    print("+", command if isinstance(command, str) else " ".join(map(str, command)), flush=True)
    subprocess.check_call(command, shell=isinstance(command, str), **kwargs)


print(subprocess.run("lscpu | grep -E 'Model name|Thread|Core|^CPU\\(s\\)'", shell=True,
                     capture_output=True, text=True).stdout, flush=True)
run(["git", "clone", "https://github.com/IsaacJYPan/fog.git", str(root)])
run(["git", "checkout", "--detach", COMMIT], cwd=root)
run(["git", "submodule", "update", "--init", "--depth", "1", "server/pokemon-showdown"], cwd=root)

# Node for the Showdown worker.
node_home = temp / f"node-{NODE}-linux-x64"
run(f"curl -sSfL https://nodejs.org/dist/{NODE}/node-{NODE}-linux-x64.tar.xz | tar -xJ -C {temp}")
os.environ["PATH"] = f"{node_home / 'bin'}:{Path.home() / '.cargo' / 'bin'}:{os.environ['PATH']}"
showdown = root / "server" / "pokemon-showdown"
run("npm ci --ignore-scripts --no-audit --no-fund", cwd=showdown)
run("npm run build", cwd=showdown)

# The engine is built from source with the Gen 9 feature, as pyproject.toml says.
if shutil.which("cargo") is None:
    run("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal")
run([sys.executable, "-m", "pip", "install", "-q", "--no-binary", "poke-engine",
     "poke-engine==0.0.48", "--config-settings",
     "build-args=--features poke-engine/terastallization --no-default-features"])
run([sys.executable, "-m", "pip", "install", "-q", "poke-env==0.15.0"])
setup_seconds = time.time() - started


def resolve(argument: str) -> str:
    def dataset_file(match):
        found = sorted(Path("/kaggle/input").rglob(match.group(1)))
        if not found:
            raise SystemExit(f"{match.group(1)} is not in the attached dataset")
        return str(found[0])

    def collection(match):
        found = sorted(path.parent for path in Path("/kaggle/input").rglob("game_0000000.npz")
                       if path.parent.name == match.group(1))
        if not found:
            raise SystemExit(f"collection {match.group(1)} is not in the attached dataset")
        return str(found[0])

    argument = re.sub(r"\{round:([^}]+)\}", collection, argument)
    return re.sub(r"\{input:([^}]+)\}", dataset_file, argument).replace("{out}", str(out))


env = {**os.environ, "PYTHONPATH": str(root), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
       "OPENBLAS_NUM_THREADS": "1"}
record = []
for step in STEPS:
    command = [sys.executable, *[resolve(argument) for argument in step["command"]]]
    print("+", " ".join(command), flush=True)
    step_started = time.time()
    status = subprocess.call(command, cwd=root, env=env)
    record.append({"name": step["name"], "exit_status": status,
                   "seconds": time.time() - step_started})
for folder in sorted(path for path in out.iterdir() if path.is_dir()):
    for log in sorted(folder.glob("*.log")):
        print(log.name, log.read_text(encoding="utf-8", errors="replace")[-1200:], flush=True)
    for leftover in folder.glob("game_*.lock"):
        leftover.unlink()
    with tarfile.open(out / f"{folder.name}.tar", "w") as archive:
        archive.add(folder, arcname=folder.name)
    shutil.rmtree(folder)
(out / "job.json").write_text(json.dumps({
    "commit": COMMIT, "cpus": os.cpu_count(), "setup_seconds": setup_seconds,
    "seconds": time.time() - started, "steps": record}, indent=2) + "\n")
if any(step["exit_status"] != 0 for step in record):
    raise SystemExit(1)
