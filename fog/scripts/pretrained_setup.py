"""Install isolated native/opponent runtimes from the frozen package on Kaggle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def command(args, **kwargs):
    print("Setup:", " ".join(str(arg) for arg in args[:5]), flush=True)
    subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def environment(root: Path, name: str):
    destination = root / ".runtime" / name
    command([sys.executable, "-m", "venv", "--system-site-packages", "--without-pip", destination])
    return destination / "bin/python"


def setup(root: Path, output: Path, *, stage):
    if sys.platform != "linux":
        raise RuntimeError(
            "Use the tested local environments on Windows; this installer targets Kaggle Linux"
        )
    output.mkdir(parents=True, exist_ok=True)
    # Kaggle's CUDA build is preserved. Its exact version and numerical behavior
    # are gates, rather than allowing an unplanned multi-GB wheel replacement.
    probe_code = "import json,torch,platform; print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.is_available()}))"
    probe = subprocess.check_output([sys.executable, "-c", probe_code], text=True)
    platform = json.loads(probe)
    if stage == "collect" and platform["torch"].split("+")[0] != "2.11.0":
        command(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "torch==2.11.0",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
            ]
        )
        platform = json.loads(
            subprocess.check_output([sys.executable, "-c", probe_code], text=True)
        )
    if platform["torch"].split("+")[0] != "2.11.0":
        raise RuntimeError(
            "Expected the tested Kaggle PyTorch 2.11.0; freeze and validate a new environment recipe"
        )
    if stage == "train" and not platform["gpu"]:
        raise RuntimeError("GPU training package requires a real CUDA device")
    (output / "platform.json").write_text(json.dumps(platform, indent=2) + "\n")
    native = environment(root, "native")
    # All transitive native dependencies are frozen explicitly. Resolving them
    # again can replace the system CUDA Torch with a newer multi-GB build.
    command([native, "-m", "pip", "install", "--no-deps", "-r", root / "requirements.native.txt"])
    for name in ("amago", "poke-env", "metamon"):
        command([native, "-m", "pip", "install", "--no-deps", "-e", root / "upstream" / name])
    from fog.pretrained.runtime import native_environment

    native_probe = """import contextlib,json,sys,amago,metamon,poke_env,gymnasium,numpy,torch
from fog.pretrained.kakuna import load_kakuna
torch.set_num_threads(1)
with contextlib.redirect_stdout(sys.stderr):
    model=load_kakuna('inputs/kakuna34.pt')
print(json.dumps({'prefix':sys.prefix,'amago':amago.__file__,'metamon':metamon.__file__,
    'poke_env':poke_env.__file__,'gymnasium':gymnasium.__version__,'numpy':numpy.__version__,
    'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.is_available(),
    'checkpoint_sha256':model.source_sha256,'state_entries':len(model.model.state_dict())}))
"""
    result = subprocess.check_output(
        [native, "-c", native_probe], env=native_environment(root), text=True
    )
    loaded = json.loads(result)
    if loaded["torch"] != platform["torch"] or loaded["prefix"] != str(root / ".runtime/native"):
        raise RuntimeError("Native model must use the frozen virtual environment and base Torch")
    if stage == "train" and not loaded["gpu"]:
        raise RuntimeError("Native model lost access to the allocated CUDA device")
    (output / "native-imports.json").write_text(result)
    # Native code resolves FOG by cwd. Editable upstream installs preserve its
    # tokenizer, gin, and Dex resources that current wheels omit.
    if stage == "collect":
        parent = environment(root, "collector")
        command(
            [parent, "-m", "pip", "install", "poke-env==0.15.0", "pyyaml==6.0.3", "numpy==2.5.3"]
        )
        foul = environment(root, "foulplay")
        if not shutil.which("cargo"):
            installer = root / ".runtime/rustup-init.sh"
            import urllib.request

            urllib.request.urlretrieve("https://sh.rustup.rs", installer)
            command(
                ["sh", installer, "-y", "--profile", "minimal", "--default-toolchain", "1.89.0"]
            )
            os.environ["PATH"] = str(Path.home() / ".cargo/bin") + os.pathsep + os.environ["PATH"]
        command(
            [
                foul,
                "-m",
                "pip",
                "install",
                "requests==2.33.0",
                "websockets==14.1",
                "python-dateutil==2.8.0",
                "poke-engine==0.0.48",
                "--no-binary=poke-engine",
                "--config-settings=build-args=--features poke-engine/terastallization --no-default-features",
            ]
        )
        if not shutil.which("node"):
            command(["apt-get", "update"])
            command(["apt-get", "install", "-y", "nodejs", "npm"])
        command(
            [
                "npm",
                "--prefix",
                root / "server/pokemon-showdown",
                "ci",
                "--omit=dev",
                "--omit=optional",
                "--ignore-scripts",
                "--no-audit",
                "--no-fund",
            ]
        )
        runtimes = {"native": str(native), "collector": str(parent), "foulplay": str(foul)}
    else:
        runtimes = {"native": str(native)}
    for name, python in runtimes.items():
        installed = subprocess.check_output([python, "-m", "pip", "freeze"], text=True)
        (output / (name + "-packages.txt")).write_text(installed)
    (output / "runtimes.json").write_text(json.dumps(runtimes, indent=2) + "\n")
    return runtimes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=["collect", "train"], required=True)
    args = parser.parse_args()
    setup(args.root, args.out, stage=args.stage)


if __name__ == "__main__":
    main()
