"""Matched fact-only/native-context AR residual training with immutable Kakuna weights."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from fog.models.donor_prior import DonorPrior
from fog.pretrained.frozen_belief import (
    FrozenBeliefData,
    digest,
    export_heads,
    initialize_residual,
    public_payload,
    update,
)
from fog.pretrained.prior_head_metrics import evaluation_queries, score_distributions
from fog.pretrained.state import atomic_save, restore_rng, rng_state


def run(config, out, *, device="cpu", resume=None, stop_after_steps=None, condition=None):
    if condition not in (None, "facts", "native"):
        raise ValueError("Unknown independent belief condition")
    for key in [
        "updates",
        "effective_batch",
        "microbatch",
        "chain_queries",
        "save_every",
        "warmup_updates",
    ]:
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError("Frozen positive integer required: " + key)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("Actual CUDA device required")
    torch.set_num_threads(config.get("cpu_threads", 1))
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
    if device.startswith("cuda") and config.get("allow_local_data"):
        raise ValueError("Local trial data are prohibited on the production CUDA path")
    data = FrozenBeliefData(
        config["data"], config["data_sha256"], allow_trial=config.get("allow_local_data", False)
    )
    if (
        data.manifest["native_sha256"] != config["native_sha256"]
        or digest(config["prior"]) != config["prior_sha256"]
    ):
        raise ValueError("Native/prior identities differ from the recipe")
    prior = DonorPrior(
        Path(config["prior"]), max_donors=config["max_donors"], smoothing=config["smoothing"]
    )
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    identity = hashlib.sha256(canonical.encode()).hexdigest()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "recipe.json").exists() and (out / "recipe.json").read_text() != canonical + "\n":
        raise ValueError("Output belongs to another immutable experiment")
    if not resume and (out / "latest.state.pt").exists():
        raise ValueError("Existing training requires explicit full-state resume")
    (out / "recipe.json").write_text(canonical + "\n")
    model = initialize_residual(
        config["initial_heads"],
        expected_sha256=config["initial_heads_sha256"],
        native_sha256=config["native_sha256"],
        device=device,
    )
    initial = deepcopy(model.state_dict())
    seed = config["seed"]
    phase, completed, steps, elapsed_prior = 0, 0, 0, 0.0
    histories, evaluations = [], {}
    modes = [condition] if condition else ["facts", "native"]

    def streams():
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return random.Random(seed + 1), torch.Generator(device=device).manual_seed(seed + 2)

    def optimizer_for():
        return torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0.01)

    sampler, route = streams()
    optimizer = optimizer_for()
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        if (
            saved["format"] != "fog.prior-head-training"
            or saved["version"] != 2
            or saved["condition"] != condition
            or saved["identity"] != identity
            or saved["config"] != config
        ):
            raise ValueError("Full training-state identity/schema/recipe mismatch")
        phase, completed, steps = saved["phase"], saved["completed"], saved["steps"]
        if (
            phase not in (0, 1)
            or steps != phase * config["updates"] + completed
            or not 0 <= completed <= config["updates"]
        ):
            raise ValueError("Training exposure counters disagree")
        model.load_state_dict(saved["model"], strict=True)
        initial = saved["initial"]
        optimizer.load_state_dict(saved["optimizer"])
        sampler.setstate(saved["sampler"])
        route.set_state(saved["route_rng"].cpu())
        histories, evaluations, elapsed_prior = (
            saved["histories"],
            saved["evaluations"],
            saved["elapsed_seconds"],
        )
        restore_rng(saved["rng"])
        log = out / "metrics.jsonl"
        expected = "".join(json.dumps(v) + "\n" for v in histories)
        if not log.exists():
            log.write_text(expected)
        elif log.read_text() != expected:
            archive = out / ("metrics.interrupted-" + digest(log)[:16] + ".jsonl")
            if not archive.exists():
                archive.write_bytes(log.read_bytes())
            log.write_text(expected)
    queries = evaluation_queries(data, maximum_games=config.get("evaluation_games", 32))
    started = time.monotonic()

    def save(filename="latest.state.pt"):
        atomic_save(
            {
                "format": "fog.prior-head-training",
                "version": 2,
                "condition": condition,
                "identity": identity,
                "config": config,
                "phase": phase,
                "completed": completed,
                "steps": steps,
                "model": model.state_dict(),
                "initial": initial,
                "optimizer": optimizer.state_dict(),
                "sampler": sampler.getstate(),
                "route_rng": route.get_state(),
                "rng": rng_state(),
                "histories": histories,
                "evaluations": evaluations,
                "elapsed_seconds": elapsed_prior + time.monotonic() - started,
                "save_boundary": "complete optimizer update; no pending gradients",
            },
            out / filename,
        )

    if not resume:
        evaluations["prior_only"] = score_distributions(
            model, data, prior, queries, mode="native", device=device, seed=seed + 3
        )
        export_heads(
            model, out / "prior_only.pt", config=config, mode="native", update=0, identity=identity
        )
    stop_reason = "planned_updates"
    while phase < len(modes):
        mode = modes[phase]
        if completed == 0:
            model.load_state_dict(initial, strict=True)
            optimizer = optimizer_for()
            sampler, route = streams()
        while completed < config["updates"]:
            if elapsed_prior + time.monotonic() - started >= config["training_seconds"]:
                stop_reason = "time_budget"
                save()
                break
            specifications = data.draw(sampler, config["effective_batch"])
            batch = data.batch(specifications, device=device)
            payload = public_payload(model, batch["labels"]["reveals"], prior)
            progress = completed / max(1, config["updates"] - 1)
            schedule = min(1.0, (completed + 1) / config["warmup_updates"]) * (
                0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2
            )
            for group in optimizer.param_groups:
                group["lr"] = config["learning_rate"] * schedule
            route_before = hashlib.sha256(route.get_state().cpu().numpy().tobytes()).hexdigest()
            result = update(
                model,
                optimizer,
                batch,
                payload,
                route,
                mode=mode,
                chain_queries=config["chain_queries"],
                microbatch=config["microbatch"],
            )
            completed += 1
            steps += 1
            row = {
                "mode": mode,
                "update": completed,
                "steps": steps,
                **result,
                "query_sha256": hashlib.sha256(json.dumps(specifications).encode()).hexdigest(),
                "route_before_sha256": route_before,
                "route_after_sha256": hashlib.sha256(
                    route.get_state().cpu().numpy().tobytes()
                ).hexdigest(),
                "lr": optimizer.param_groups[0]["lr"],
            }
            histories.append(row)
            with (out / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            if completed % config["save_every"] == 0 or completed == config["updates"]:
                if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
                    raise FloatingPointError("Nonfinite decoder weights")
                export_heads(
                    model,
                    out / f"{mode}_heads_{completed:06d}.pt",
                    config=config,
                    mode=mode,
                    update=completed,
                    identity=identity,
                )
                save()
                print(
                    json.dumps(
                        {
                            "mode": mode,
                            "update": completed,
                            "losses": row["losses"],
                            "elapsed_seconds": elapsed_prior + time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
            if stop_after_steps is not None and steps >= stop_after_steps:
                save()
                return {"complete": False, "stop_reason": "explicit_resume_probe", "steps": steps}
        if stop_reason != "planned_updates":
            break
        evaluations[mode] = score_distributions(
            model, data, prior, queries, mode=mode, device=device, seed=seed + 3
        )
        export_heads(
            model,
            out / f"{mode}_heads_final.pt",
            config=config,
            mode=mode,
            update=completed,
            identity=identity,
        )
        save()
        save(f"{mode}_full.state.pt")
        if phase == len(modes) - 1:
            break
        phase, completed = phase + 1, 0
        save()
    complete = phase == len(modes) - 1 and completed == config["updates"]
    matched = complete and condition is None
    if matched:
        left = [v for v in histories if v["mode"] == "facts"]
        right = [v for v in histories if v["mode"] == "native"]
        if any(
            a[key] != b[key]
            for a, b in zip(left, right)
            for key in ["query_sha256", "route_before_sha256", "route_after_sha256"]
        ):
            raise ValueError("Matched conditions consumed different data/teacher queries")
        a = torch.load(out / "facts_heads_final.pt", map_location="cpu", weights_only=True)
        b = torch.load(out / "native_heads_final.pt", map_location="cpu", weights_only=True)
        if not all(
            torch.equal(v, b["heads"][k]) for k, v in a["heads"].items() if k.startswith("outcome.")
        ):
            raise ValueError("Outcome heads unexpectedly differ across belief-context ablations")
    result = {
        "complete": complete,
        "condition": condition,
        "stop_reason": stop_reason,
        "identity": identity,
        "steps": steps,
        "updates_per_condition": config["updates"],
        "AR_queries_per_context": config["chain_queries"],
        "matched_query_routes": matched,
        "outcome_heads_identical": matched,
        "native_policy_changed": False,
        "native_optimizer_present": False,
        "native_sha256": config["native_sha256"],
        "prior_sha256": config["prior_sha256"],
        "evaluations": evaluations,
        "elapsed_seconds": elapsed_prior + time.monotonic() - started,
        "automatic_promotion": False,
        "search_connected": False,
    }
    (out / "RESULTS.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--condition", choices=["facts", "native"])
    args = parser.parse_args()
    result = run(
        json.loads(args.config.read_text()), args.out, device=args.device, resume=args.resume,
        condition=args.condition,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "evaluations"}))
    if not result["complete"]:
        raise RuntimeError("Incomplete matched experiment; preserve outputs and diagnose")


if __name__ == "__main__":
    main()
