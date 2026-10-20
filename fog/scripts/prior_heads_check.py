"""Discarded complete-state resume and CPU/CUDA checks for frozen prior heads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fog.pretrained.frozen_belief import FrozenBeliefData, public_payload
from fog.models.donor_prior import DonorPrior
from fog.pretrained.joint import BeliefHeads
from fog.scripts.kakuna_latent_resume_check import identical
from fog.scripts.train_prior_heads import run


def probabilities(heads, config, specifications, *, device="cuda", tolerance=1e-3):
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    data = FrozenBeliefData(
        config["data"], config["data_sha256"], allow_trial=config.get("allow_local_data", False)
    )
    prior = DonorPrior(
        Path(config["prior"]), max_donors=config["max_donors"], smoothing=config["smoothing"]
    )
    distributions = []
    devices = ["cpu", device] if device.startswith("cuda") else ["cpu", "cpu"]
    for selected in devices:
        model = BeliefHeads().to(selected).eval()
        model.load_state_dict(heads, strict=True)
        batch = data.batch(specifications, device=selected)
        payload = public_payload(model, batch["labels"]["reveals"], prior)
        with torch.no_grad():
            output = model.belief(model.context_projection(batch["latent"]), payload)
            distributions.append(
                {key: value.softmax(-1).cpu() for key, value in output.logits.items()}
            )
    errors = {
        key: float((distributions[0][key] - distributions[1][key]).abs().max())
        for key in distributions[0]
    }
    if not all(value <= tolerance for value in errors.values()):
        raise ValueError("Frozen-head CPU/CUDA probability agreement failed")
    return {
        "passed": True,
        "devices": devices,
        "tolerance": tolerance,
        "maximum_errors": errors,
        "actual_cuda": device.startswith("cuda"),
        "contexts": len(specifications),
    }


def check(config, out, *, device="cpu"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    trial = dict(
        config,
        updates=2,
        effective_batch=4,
        microbatch=2,
        chain_queries=2,
        save_every=1,
        warmup_updates=1,
        evaluation_games=1,
        training_seconds=600,
    )
    direct = run(trial, out / "direct", device=device)
    partial = run(trial, out / "split", device=device, stop_after_steps=3)
    if partial["complete"] or partial["steps"] != 3:
        raise ValueError("Resume proof did not interrupt after a complete update")
    resumed = run(trial, out / "split", device=device, resume=out / "split/latest.state.pt")
    a = torch.load(out / "direct/latest.state.pt", weights_only=False, map_location="cpu")
    b = torch.load(out / "split/latest.state.pt", weights_only=False, map_location="cpu")
    keys = [
        "model",
        "initial",
        "optimizer",
        "sampler",
        "route_rng",
        "rng",
        "phase",
        "completed",
        "steps",
        "histories",
        "evaluations",
    ]
    checks = {key: identical(a[key], b[key]) for key in keys}
    checks["complete"] = direct["complete"] and resumed["complete"]
    checks["no_native_parameters"] = not any(key.startswith("native.") for key in a["model"])
    checks["matched_exposure"] = direct["matched_query_routes"] and resumed["matched_query_routes"]
    checks["outcome_head_control"] = (
        direct["outcome_heads_identical"] and resumed["outcome_heads_identical"]
    )
    checks["CUDA_rng_present"] = not device.startswith("cuda") or bool(a["rng"]["cuda"])
    for phase, mode in enumerate(["facts", "native"]):
        left = torch.load(out / f"direct/{mode}_full.state.pt", weights_only=False, map_location="cpu")
        right = torch.load(out / f"split/{mode}_full.state.pt", weights_only=False, map_location="cpu")
        checks[f"{mode}_full_state"] = (
            left["phase"] == phase
            and left["completed"] == trial["updates"]
            and all(identical(left[key], right[key]) for key in keys)
        )
    data = FrozenBeliefData(
        config["data"], config["data_sha256"], allow_trial=config.get("allow_local_data", False)
    )
    import random

    specs = data.draw(random.Random(config["seed"]), 4, split="development")
    agreement = probabilities(a["model"], config, specs, device=device)
    result = {
        "passed": all(checks.values()) and agreement["passed"],
        "checks": checks,
        "device": device,
        "probability_agreement": agreement,
        "discarded_optimizer_updates": 8,
        "main_initialization_unchanged": True,
        "native_updates": 0,
        "CUDA_rng_saved": bool(a["rng"]["cuda"]),
    }
    (out / "RESULTS.json").write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        raise ValueError("Full-state frozen-head resume proof failed")
    print(json.dumps(result), flush=True)
    return result


def benchmark(config, out, *, device="cuda"):
    """Discarded production-shape updates; reject an infeasible fixed budget."""
    import random
    import time
    import numpy as np
    from fog.pretrained.frozen_belief import initialize_residual, update

    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("Throughput proof requires an actual CUDA device")
    data = FrozenBeliefData(config["data"], config["data_sha256"])
    prior = DonorPrior(
        Path(config["prior"]), max_donors=config["max_donors"], smoothing=config["smoothing"]
    )
    model = initialize_residual(
        config["initial_heads"],
        expected_sha256=config["initial_heads_sha256"],
        native_sha256=config["native_sha256"],
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0.01)
    torch.manual_seed(config["seed"] + 40)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(config["seed"] + 40)
    sampler = random.Random(config["seed"] + 41)
    generator = torch.Generator(device=device).manual_seed(config["seed"] + 42)
    seconds = []
    for _ in range(8):
        started = time.monotonic()
        batch = data.batch(data.draw(sampler, config["effective_batch"]), device=device)
        payload = public_payload(model, batch["labels"]["reveals"], prior)
        update(
            model,
            optimizer,
            batch,
            payload,
            generator,
            mode="native",
            chain_queries=config["chain_queries"],
            microbatch=config["microbatch"],
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        seconds.append(time.monotonic() - started)
    estimate = float(np.quantile(seconds[2:], 0.9)) * config["updates"] * 2 * 1.2
    result = {
        "passed": estimate < config["training_seconds"],
        "actual_cuda": device.startswith("cuda"),
        "production_batch": config["effective_batch"],
        "microbatch": config["microbatch"],
        "AR_queries_per_context": config["chain_queries"],
        "update_seconds": seconds,
        "estimated_main_seconds_with_20pct_margin": estimate,
        "main_seconds": config["training_seconds"],
        "discarded_optimizer_updates": 8,
        "main_initialization_unchanged": True,
        "retrieval_cache": prior.retrieve.cache_info()._asdict(),
    }
    (out / "RESULTS.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if not result["passed"]:
        raise ValueError(
            "Measured CUDA throughput cannot safely fit the fixed matched update budget"
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    check(json.loads(args.config.read_text()), args.out, device=args.device)
    if args.benchmark:
        benchmark(
            json.loads(args.config.read_text()),
            args.out.with_name(args.out.name + "-throughput"),
            device=args.device,
        )


if __name__ == "__main__":
    main()
