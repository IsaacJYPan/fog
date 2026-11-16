"""Train the search-distillation network on a hosted GPU from uploaded rounds.

A script kernel with a GPU and internet, and one private dataset attached that holds a
`roundN.tar` for each collection directory (`tar -cf round0.tar -C runs/distill round0`).
Set COMMIT to the `fog` commit to train with. Outputs: `net/policy_best.pt`,
`net/metrics.json`, `job.json`. Submitted on 2026-11-16 with
`kaggle kernels push -p <folder> --accelerator NvidiaTeslaT4 --timeout 2400`.
"""
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

COMMIT = "f2f11c6cedb77e37884079f30ecbd49108267fd7"
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
directories = sorted({str(path.parent) for path in games}, key=lambda p: Path(p).name)
print("rounds", directories, "games", len(games), flush=True)
import torch

print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "cpus", os.cpu_count(),
      flush=True)
command = [sys.executable, "-m", "fog.training.search_distill", "train",
           "--config", "configs/search_distill_cpu.yaml", "--data", *directories,
           "--out", str(out / "net"), "--device", "cuda", "--threads", "4"]
subprocess.check_call(command, cwd=root, env={**os.environ, "PYTHONPATH": str(root)})
(out / "net" / "state.pt").unlink(missing_ok=True)
(out / "job.json").write_text(json.dumps({
    "commit": COMMIT, "seconds": time.time() - started, "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(0), "rounds": [Path(d).name for d in directories],
    "games": len(games)}, indent=2) + "\n")
