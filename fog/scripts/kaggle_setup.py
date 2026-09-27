"""Install the direct-Showdown runtime without replacing Kaggle's GPU PyTorch."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path


def runtime_requirements(root: Path) -> list[str]:
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    versions = {package["name"]: package["version"] for package in lock["package"]}
    requirements = []
    for dependency in project["project"]["dependencies"]:
        name = re.split(r"[<>=!~\[; ]", dependency, maxsplit=1)[0]
        # The direct simulator needs neither the Rust engine nor a new PyTorch wheel.
        if name not in {"torch", "poke-engine"}:
            # Kaggle's preloaded google namespace exposes Protobuf 5.29.5 even
            # with a newer wheel in the venv. TensorBoard 2.20 supports it.
            version = "2.20.0" if name == "tensorboard" else versions[name]
            requirements.append(f"{name}=={version}")
    return requirements


def setup(root: Path) -> Path:
    if sys.platform != "linux":
        raise RuntimeError("Run this installer inside a Kaggle Linux notebook; use uv sync locally.")
    python = root / ".venv" / "bin" / "python"
    # Kaggle supplies pip globally; its ensurepip bootstrap can fail even though
    # that pip is usable. Inherit it alongside the preinstalled CUDA PyTorch.
    subprocess.run([sys.executable, "-m", "pip", "--version"], check=True)
    subprocess.run([
        sys.executable, "-m", "venv", "--system-site-packages", "--without-pip",
        str(root / ".venv"),
    ], check=True)
    subprocess.run([str(python), "-m", "pip", "install", *runtime_requirements(root)], check=True)
    subprocess.run([str(python), "-m", "pip", "install", "--no-deps", "-e", str(root)], check=True)
    if not shutil.which("node") or not shutil.which("npm"):
        raise RuntimeError("Node.js and npm are required. Install them, then rerun this cell.")
    subprocess.run([
        "npm", "--prefix", str(root / "server/pokemon-showdown"), "ci",
        "--omit=dev", "--omit=optional", "--ignore-scripts", "--no-audit", "--no-fund",
    ], check=True)
    # Use the new process so any packages installed above are imported afresh.
    probe = '''
import json, platform, subprocess, torch
from fog.env.showdown_direct import ShowdownDirectClient
from torch.utils.tensorboard import SummaryWriter
assert torch.cuda.is_available(), "Enable the notebook GPU accelerator before training."
with SummaryWriter("runs/kaggle_environment/tensorboard_smoke") as writer:
    writer.add_scalar("setup/ok", 1, 0)
with ShowdownDirectClient() as client:
    assert client.ping()["pong"]
print(json.dumps({"python": platform.python_version(), "torch": torch.__version__,
                  "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                  "node": subprocess.check_output(["node", "--version"], text=True).strip()}))
'''
    result = subprocess.run([str(python), "-c", probe], cwd=root, capture_output=True, text=True)
    print(result.stdout, end="")
    if result.returncode:
        print(result.stderr, file=sys.stderr, end="")
        result.check_returncode()
    output = root / "runs" / "kaggle_environment"
    output.mkdir(parents=True, exist_ok=True)
    (output / "runtime.json").write_text(json.dumps(json.loads(result.stdout), indent=2) + "\n")
    packages = subprocess.check_output([str(python), "-m", "pip", "freeze"], text=True)
    (output / "requirements.txt").write_text(packages, encoding="utf-8")
    return python


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--print-requirements", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.print_requirements:
        print("\n".join(runtime_requirements(root)))
    else:
        print(f"Training interpreter: {setup(root)}")


if __name__ == "__main__":
    main()
