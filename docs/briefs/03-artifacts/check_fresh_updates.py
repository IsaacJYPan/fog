"""Compare 20 real updates with each BC module initialized in a fresh process.

Run serially after the active training run has finished::

    uv run --no-sync python docs/briefs/03-artifacts/check_fresh_updates.py

The controller and child entry point import only the standard library until the
target trainer is initialized. Each child then reuses check_real_updates.run,
including its normal full holdout baseline and six-epoch learning-rate schedule.
This diagnoses startup order; it is not a full historical reproduction run.
All generated outputs, including tensor snapshots, remain in ignored runs/.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = "f0dc3d16eeecab8fbad545aed93af2b2de20a2c4"
WORK = Path("runs/blueprint/original-current-real20/fresh-process")
SHARED_RESULT = WORK.parent / "comparison.json"
HELPER = ROOT / "docs/briefs/03-artifacts/check_real_updates.py"
INVOCATION = "uv run --no-sync python docs/briefs/03-artifacts/check_fresh_updates.py"
RELEVANT_MODULES = {
    "numpy", "numpy._core", "numpy._core._multiarray_umath", "numpy.linalg", "numpy.random",
    "torch", "torch._C", "torch.storage", "fog.training.bc", "fresh_original_bc",
}


def _load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _child(name: str) -> None:
    started = time.perf_counter()
    before = {module: module in sys.modules for module in ("numpy", "torch")}
    current_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    with tempfile.TemporaryDirectory() as temporary:
        if name == "original":
            source = Path(temporary) / "original_bc.py"
            source_command = ["git", "show", f"{BASE}:fog/training/bc.py"]
            source.write_bytes(subprocess.check_output(source_command))
            target = _load_file("fresh_original_bc", source)
            initialization = "git show + importlib execution of original bc.py"
        else:
            source_command = None
            target = importlib.import_module("fog.training.bc")
            initialization = "import fog.training.bc"

        # sys.modules is insertion-ordered. This records insertion, not module
        # completion times, before importing the shared helper/current trainer.
        insertion_order = [
            {"index": index, "module": module}
            for index, module in enumerate(sys.modules)
            if module in RELEVANT_MODULES
        ]
        after_target = {module: module in sys.modules for module in ("numpy", "torch")}
        helper = _load_file("real_updates_helper", HELPER)
        losses, state, run_metadata = helper.run(target, f"fresh_{name}")

    # Torch/NumPy were initialized by the target or helper, never preimported here.
    torch = sys.modules["torch"]
    numpy = sys.modules["numpy"]
    tensor_path = WORK / f"{name}-state.pt"
    torch.save(state, tensor_path)
    metadata = {
        "status": "completed_20_update_diagnostic_not_a_full_run",
        "target": name,
        "target_bc_commit": BASE if name == "original" else current_commit,
        "current_shared_code_commit": current_commit,
        "command": shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]),
        "target_initialization": initialization,
        "original_source_command": shlex.join(source_command) if source_command else None,
        "modules_present_before_target_loading": before,
        "modules_present_after_target_loading_before_helper": after_target,
        "relevant_module_insertion_order_after_target_before_helper": insertion_order,
        "insertion_order_limit": "sys.modules insertion order, not module completion times",
        "torch_version": str(torch.__version__),
        "numpy_version": str(numpy.__version__),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "losses": losses,
        "state_dict_tensor_count": len(state),
        "state_dict_path": str(tensor_path),
        "run": run_metadata,
        "wall_seconds": time.perf_counter() - started,
    }
    (WORK / f"{name}.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


def _loss_comparison(before: list, after: list) -> dict:
    return {
        "identical": before == after,
        "lengths": [len(before), len(after)],
        "maximum_absolute_difference": max(
            (abs(a - b) for a, b in zip(before, after)), default=0.0
        ),
    }


def _write_comparison(result: dict) -> None:
    output = WORK / "comparison.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"Fresh-process diagnostic JSON: {output}", flush=True)


def _controller() -> None:
    started = time.perf_counter()
    result = {
        "status": "running",
        "full_training_run": False,
        "scope": "original/current startup order followed by 20 real CPU updates in clean children",
        "command": INVOCATION,
        "controller_command": shlex.join([sys.executable, str(Path(__file__).resolve())]),
        "base_commit": BASE,
        "current_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "children": {},
    }
    for name in ("original", "current"):
        command = [sys.executable, str(Path(__file__).resolve()), "--child", name]
        for suffix in (".json", "-state.pt"):
            (WORK / f"{name}{suffix}").unlink(missing_ok=True)
        print(f"Starting {name}: {shlex.join(command)}", flush=True)
        child_started = time.perf_counter()
        output_path = WORK / f"{name}-output.txt"
        with output_path.open("w", encoding="utf-8") as stream:
            stream.write(f"$ {shlex.join(command)}\n")
            stream.flush()
            child = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
        details = {
            "command": shlex.join(command),
            "exit_code": child.returncode,
            "wall_seconds": time.perf_counter() - child_started,
            "output": str(output_path),
        }
        if child.returncode == 0:
            details["metadata"] = json.loads((WORK / f"{name}.json").read_text())
        result["children"][name] = details
        print(f"Finished {name}: exit={child.returncode}", flush=True)

    if any(child["exit_code"] for child in result["children"].values()):
        result.update(status="failed_child_execution", wall_seconds=time.perf_counter() - started)
        _write_comparison(result)
        raise SystemExit("fresh-process child failed; see its output and comparison.json")

    # Both clean children have finished. Only now may the controller import Torch.
    import torch

    original = result["children"]["original"]["metadata"]
    current = result["children"]["current"]["metadata"]
    old_state = torch.load(WORK / "original-state.pt", map_location="cpu", weights_only=True)
    new_state = torch.load(WORK / "current-state.pt", map_location="cpu", weights_only=True)
    differing_names = sorted(
        name for name in old_state.keys() | new_state.keys()
        if name not in old_state or name not in new_state
        or not torch.equal(old_state[name], new_state[name])
    )
    loss_comparison = _loss_comparison(original["losses"], current["losses"])
    startup_valid = not any(
        any(metadata["modules_present_before_target_loading"].values())
        for metadata in (original, current)
    )
    identical = loss_comparison["identical"] and not differing_names
    result.update(
        status=("passed_20_update_startup_control" if identical and startup_valid
                else "failed_20_update_startup_control"),
        startup_modules_initially_absent=startup_valid,
        first_20_losses_before=original["losses"],
        first_20_losses_after=current["losses"],
        loss_comparison=loss_comparison,
        all_parameters_and_persistent_buffers_identical=not differing_names,
        differing_state_dict_tensor_names=differing_names,
        state_dict_tensor_counts=[len(old_state), len(new_state)],
        controller_torch_version=str(torch.__version__),
    )
    supplemental = {"source": str(SHARED_RESULT), "available": SHARED_RESULT.is_file()}
    if SHARED_RESULT.is_file():
        try:
            shared = json.loads(SHARED_RESULT.read_text())
            supplemental.update(
                source_current_commit=shared.get("current_commit"),
                fresh_original_vs_shared_original=_loss_comparison(
                    shared["first_20_losses_before"], original["losses"]
                ),
                fresh_current_vs_shared_current=_loss_comparison(
                    shared["first_20_losses_after"], current["losses"]
                ),
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            # An optional old result must not suppress the new control's JSON.
            supplemental["read_error"] = str(error)
    result["supplemental_shared_process_loss_comparison"] = supplemental
    result["wall_seconds"] = time.perf_counter() - started
    _write_comparison(result)
    if not identical or not startup_valid:
        raise SystemExit("fresh-process control differs or was preloaded; see comparison.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", choices=("original", "current"))
    args = parser.parse_args()
    os.chdir(ROOT)
    WORK.mkdir(parents=True, exist_ok=True)
    if args.child:
        _child(args.child)
    else:
        _controller()


if __name__ == "__main__":
    main()
