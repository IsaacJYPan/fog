"""Train search-distillation networks on a hosted GPU from uploaded rounds.

A script kernel with a GPU and internet, and one private dataset attached that holds a
`roundN.tar` for each collection directory (`tar -cf round0.tar -C runs/distill round0`).
Set COMMIT to the `fog` commit to train with and RUNS to the trainings wanted: each has a
name, the rounds it reads (all uploaded rounds when the list is empty) and extra trainer
arguments. Outputs: `<name>/policy_best.pt`, `<name>/value_best.pt`, `<name>/metrics.json`
for each run, and `job.json`. Submit with
`kaggle kernels push -p <folder> --accelerator NvidiaTeslaT4 --timeout 3600`.
"""
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

COMMIT = "79283c9d780448edfc1938aa90642fb87a211bbe"
RUNS = [{"name": "net", "rounds": [], "args": []}]

started = time.time()
root, data, out = Path("/kaggle/temp/fog"), Path("/kaggle/temp/data"), Path("/kaggle/working")
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout, flush=True)
root.parent.mkdir(parents=True, exist_ok=True)
subprocess.check_call(["git", "clone", "https://github.com/IsaacJYPan/fog.git", str(root)])
subprocess.check_call(["git", "checkout", "--detach", COMMIT], cwd=root)
data.mkdir(parents=True, exist_ok=True)
inputs = Path("/kaggle/input")
for archive in sorted(inputs.rglob("*.tar")):
    with tarfile.open(archive) as handle:
        handle.extractall(data)
games = list(data.rglob("game_*.npz")) + list(inputs.rglob("game_*.npz"))
available = {Path(p).name: p for p in sorted({str(path.parent) for path in games})}
print("rounds", sorted(available), "games", len(games), flush=True)
import torch  # noqa: E402

print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "cpus", os.cpu_count(),
      flush=True)
record = []
for run in RUNS:
    rounds = run["rounds"] or sorted(available)
    run_started = time.time()
    command = [sys.executable, "-m", "fog.training.search_distill", "train",
               "--config", "configs/search_distill_cpu.yaml",
               "--data", *[available[name] for name in rounds],
               "--out", str(out / run["name"]), "--device", "cuda", "--threads", "4",
               *run["args"]]
    print("run", run["name"], rounds, run["args"], flush=True)
    subprocess.check_call(command, cwd=root, env={**os.environ, "PYTHONPATH": str(root)})
    (out / run["name"] / "state.pt").unlink(missing_ok=True)
    (out / run["name"] / "policy_final.pt").unlink(missing_ok=True)
    record.append({**run, "rounds": rounds, "seconds": time.time() - run_started})
(out / "job.json").write_text(json.dumps({
    "commit": COMMIT, "seconds": time.time() - started, "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(0), "games": len(games), "runs": record}, indent=2) + "\n")
