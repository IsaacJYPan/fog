"""Discarded short training paths must reproduce a complete native training state."""

import argparse
import gc
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np
import torch


def identical(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(identical(v, b[k]) for k, v in a.items())
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(identical(x, y) for x, y in zip(a, b))
    return a == b


def check(config, out, device="cuda"):
    from fog.scripts.kakuna_latent_train import run

    out = Path(out)
    if os.name == "nt" and not out.resolve().is_relative_to(Path.cwd().resolve()):
        raise ValueError("Windows validation cleanup must stay inside the workspace")
    out.mkdir(parents=True, exist_ok=True)
    trial = dict(config, warm_updates=2, condition_updates=2, effective_batch=4, microbatch=2,
                 save_every=1, evaluation_contexts=2, training_seconds=600, export_native=False)
    if config.get("retention_replay"):
        trial["retention_replay"] = dict(config["retention_replay"], contexts=16, batch=4, audit_every=1,
                                         extra=config["retention_replay"].get("extra", [])[:4])
    with tempfile.TemporaryDirectory(dir=out, prefix="discarded-") as temporary:
        root = Path(temporary)
        if not root.resolve().is_relative_to(out.resolve()):
            raise ValueError("Unexpected validation temporary path")
        direct = run(trial, root / "direct", device=device)
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        partial = run(trial, root / "split", device=device, stop_after_steps=5)
        assert not partial["complete"] and partial["steps"] == 5
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        resumed = run(trial, root / "split", device=device, resume=root / "split/latest.state.pt")
        a = torch.load(root / "direct/latest.state.pt", weights_only=False, map_location="cpu")
        b = torch.load(root / "split/latest.state.pt", weights_only=False, map_location="cpu")
        keys = ("upper", "heads", "head_optimizer", "native_optimizer", "sampler", "route_rng", "rng",
                "warm_heads", "control_heads", "rejected", "native_accepted", "steps", "phase", "phase_update",
                "retention_guard", "retention_bank_sha256")
        checks = {key: identical(a[key], b[key]) for key in keys}
        rng_checks = {key: identical(value, b["rng"][key]) for key, value in a["rng"].items()}
        checks["query_schedules"] = [r["query_schedule_sha256"] for r in a["history"]] == [r["query_schedule_sha256"] for r in b["history"]]
        checks["frozen_native"] = direct["frozen_native_unchanged"] and resumed["frozen_native_unchanged"]
        checks["completed"] = direct["complete"] and resumed["complete"]
        report = {"passed": all(checks.values()), "device": device, "checks": checks, "rng_components": rng_checks,
                  "discarded_head_updates": direct["steps"] + resumed["steps"],
                  "discarded_accepted_native_updates": direct["native_accepted_updates"] + resumed["native_accepted_updates"],
                  "discarded_optimized_contexts": (direct["steps"] + resumed["steps"]) * trial["effective_batch"],
                  "initialized_main_run": False}
        if not report["passed"]:
            from fog.pretrained.state import atomic_save

            atomic_save({"direct": a, "resumed": b}, out / "FAILED_STATES.pt")
    (out / "RESULTS.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise ValueError("Complete native training-state resume proof failed")
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    check(json.loads(args.config.read_text()), args.out, args.device)


if __name__ == "__main__":
    main()
