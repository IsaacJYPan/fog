"""Matched frozen/last-block Kakuna belief training with full-state continuation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
import numpy as np

from fog.pretrained.checkpoints import AUXILIARY_VERSION
from fog.pretrained.joint import JointKakuna
from fog.pretrained.kakuna import INTERFACE_VERSION, KAKUNA_34_SHA256, load_kakuna, sha256
from fog.pretrained.latent_cache import tensor_digest
from fog.pretrained.latent_training import (
    FeatureData, fixed_queries, head_parameters, head_state, initialize_heads, load_heads,
    query_schedule, score_beliefs, train_update,
)
from fog.pretrained.retention import RetentionLimits, configure_upper_training, upper_modules
from fog.pretrained.state import atomic_save, restore_rng, rng_state


def upper_state(native):
    block, norm = upper_modules(native)
    return {"block": copy.deepcopy(block.state_dict()), "norm": copy.deepcopy(norm.state_dict())}


def load_upper(native, state):
    block, norm = upper_modules(native)
    block.load_state_dict(state["block"], strict=True)
    norm.load_state_dict(state["norm"], strict=True)


def frozen_fingerprint(native):
    return tensor_digest({key: value for key, value in native.state_dict().items()
                          if not key.startswith(("traj_encoder.tformer.layers.9.", "traj_encoder.tformer.norm."))})


def export_heads(model, path, *, native_sha256, identity, config, update):
    atomic_save({"version": AUXILIARY_VERSION, "interface": INTERFACE_VERSION,
                 "identity": identity, "update": update, "config": config,
                 "native_sha256": native_sha256, "heads": head_state(model)}, path)


def run(config, out, *, device="cpu", resume=None, stop_after_steps=None):
    for key in ("warm_updates", "condition_updates", "effective_batch", "microbatch", "save_every", "evaluation_contexts"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError("Require a positive frozen integer: " + key)
    if config["effective_batch"] % config["microbatch"]:
        raise ValueError("Microbatches must divide the effective context batch")
    if device.startswith("cuda") and (not torch.cuda.is_available() or config.get("allow_local_cache")):
        raise ValueError("CUDA training requires real CUDA and the complete production cache")
    torch.set_num_threads(config.get("cpu_threads", 1))
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    data = FeatureData(config["cache"], config["cache_sha256"], allow_trial=config.get("allow_local_cache", False))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    identity = hashlib.sha256(canonical.encode()).hexdigest()
    recipe_path = out / "recipe.json"
    if recipe_path.exists() and recipe_path.read_text() != canonical + "\n":
        raise ValueError("Output belongs to a different immutable recipe")
    if not resume and (out / "latest.state.pt").exists():
        raise FileExistsError("An existing run requires explicit complete-state resume")
    recipe_path.write_text(canonical + "\n")
    runtime = load_kakuna(config["checkpoint"], device=device)
    # First-time upstream imports may consume random numbers. Head initialization
    # and training streams must not depend on import caches in the host process.
    torch.manual_seed(seed + 1)
    model = JointKakuna(runtime.model).to(device)
    initialization = None
    if config.get("head_initialization"):
        initialization = initialize_heads(model, **config["head_initialization"])
    random.seed(seed + 2)
    np.random.seed(seed + 2)
    torch.manual_seed(seed + 2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + 2)
    fixed = frozen_fingerprint(model.native)
    upper_parameters = configure_upper_training(model.native, adapt=True)
    limits = RetentionLimits(**config["limits"])
    phases = [("warm", config["warm_updates"]), ("control", config["condition_updates"]), ("adapted", config["condition_updates"])]
    phase, update, steps, rejected, native_accepted = 0, 0, 0, 0, 0
    warm_heads, control_heads, histories, evaluations = None, None, [], {}
    elapsed_prior = 0.0
    warm_source = None

    def optimizers():
        # Explicit resets between phases; same head optimizer initialization in
        # both paired conditions. Native Adam is active only in the adapted arm.
        return (torch.optim.AdamW(head_parameters(model), lr=config["head_learning_rate"], weight_decay=0.01),
                torch.optim.AdamW(upper_parameters, lr=config["native_learning_rate"], weight_decay=0.0))

    def generators(index):
        offset = 100 if index == 0 else 200
        return random.Random(seed + offset), torch.Generator(device=device).manual_seed(seed + offset + 1)

    head_optimizer, native_optimizer = optimizers()
    sampler, route = generators(phase)
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        if saved["format"] != "fog.kakuna-belief-retention" or saved["version"] != 1 or saved["identity"] != identity or saved["config"] != config:
            raise ValueError("Training-state identity, schema or recipe mismatch")
        phase, update, steps = saved["phase"], saved["phase_update"], saved["steps"]
        if not 0 <= phase < len(phases) or not 0 <= update <= phases[phase][1]:
            raise ValueError("Invalid training progress")
        load_upper(model.native, saved["upper"])
        load_heads(model, saved["heads"])
        head_optimizer.load_state_dict(saved["head_optimizer"])
        native_optimizer.load_state_dict(saved["native_optimizer"])
        sampler, route = generators(phase)
        sampler.setstate(saved["sampler"])
        route.set_state(saved["route_rng"].cpu())
        restore_rng(saved["rng"])
        warm_heads, control_heads = saved["warm_heads"], saved["control_heads"]
        rejected, native_accepted = saved["rejected"], saved["native_accepted"]
        histories, evaluations = saved["history"], saved["evaluations"]
        elapsed_prior = saved["elapsed_seconds"]
        warm_source = saved.get("warm_source")
        if len(histories) != steps or fixed != saved["frozen_fingerprint"]:
            raise ValueError("Saved exposure or frozen-native fingerprint differs")
    sentinel_specs = fixed_queries(data, seed=seed + 709, count=config["effective_batch"])
    sentinel = data.batch(sentinel_specs, device)
    development = fixed_queries(data, seed=seed + 991, count=config["evaluation_contexts"], split="development")
    started, last_save = time.monotonic(), time.monotonic()

    def save():
        if frozen_fingerprint(model.native) != fixed:
            raise ValueError("Protected lower/actor/critic native parameters or buffers changed")
        atomic_save({"format": "fog.kakuna-belief-retention", "version": 1, "identity": identity,
                     "config": config, "phase": phase, "phase_update": update, "steps": steps,
                     "upper": upper_state(model.native), "heads": head_state(model),
                     "head_optimizer": head_optimizer.state_dict(), "native_optimizer": native_optimizer.state_dict(),
                     "sampler": sampler.getstate(), "route_rng": route.get_state(), "rng": rng_state(),
                     "warm_heads": warm_heads, "control_heads": control_heads, "rejected": rejected,
                     "native_accepted": native_accepted, "history": histories, "evaluations": evaluations,
                     "frozen_fingerprint": fixed, "elapsed_seconds": elapsed_prior + time.monotonic() - started,
                     "initialization": initialization, "warm_source": warm_source,
                     "boundary": "completed head update and accepted/restored native update"}, out / "latest.state.pt")
        (out / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in histories))

    stopped, reason = False, "planned_updates"
    while phase < len(phases):
        label, planned = phases[phase]
        adapt = label == "adapted"
        configure_upper_training(model.native, adapt=adapt)
        if update == 0 and label not in evaluations:
            evaluations[label + "-before"] = score_beliefs(model, data, development, adapt=adapt, device=device, seed=seed + 881)
        while update < planned:
            if stop_after_steps is not None and steps >= stop_after_steps:
                stopped, reason = True, "explicit_local_interruption"
                break
            if elapsed_prior + time.monotonic() - started >= config["training_seconds"]:
                stopped, reason = True, "training_time_budget"
                break
            begin = time.monotonic()
            specs = data.draw(sampler, config["effective_batch"])
            batch = data.batch(specs, device)
            fraction = update / max(1, planned - 1)
            schedule = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * fraction))
            if label == "warm":
                schedule *= min(1.0, (update + 1) / min(100, planned))
            for group in head_optimizer.param_groups:
                group["lr"] = config["head_learning_rate"] * schedule
            for group in native_optimizer.param_groups:
                group["lr"] = config["native_learning_rate"] * schedule * 0.5 ** min(rejected, 10)
            devices = [torch.device(device).index or 0] if device.startswith("cuda") else []
            route_before = hashlib.sha256(route.get_state().cpu().numpy().tobytes()).hexdigest()
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(seed + update + (10000 if label == "warm" else 20000))
                metrics = train_update(model, batch, head_optimizer=head_optimizer,
                                       native_optimizer=native_optimizer if adapt else None,
                                       route_generator=route, weights=config["weights"],
                                       policy_penalty=config["policy_penalty"], value_penalty=config["value_penalty"],
                                       limits=limits, sentinel=sentinel, microbatch=config["microbatch"])
            update += 1
            steps += 1
            native_accepted += metrics["native_optimizer_steps"]
            rejected += int(adapt and not metrics["native_update_accepted"])
            row = {"stage": label, "update": update, "steps": steps, "contexts": len(specs),
                   "query_schedule_sha256": query_schedule(specs), "head_lr": head_optimizer.param_groups[0]["lr"],
                   "route_rng_before_sha256": route_before,
                   "route_rng_after_sha256": hashlib.sha256(route.get_state().cpu().numpy().tobytes()).hexdigest(),
                   "native_lr": native_optimizer.param_groups[0]["lr"], "native_accepted_total": native_accepted,
                   "native_rejected_total": rejected, "seconds": time.monotonic() - begin, **metrics}
            histories.append(row)
            print(json.dumps(row), flush=True)
            if not metrics["retained_batch_within_limits"]:
                stopped, reason = True, "uncovered_behavior_drift"
            if adapt and update >= 100 and rejected / update > 0.3:
                stopped, reason = True, "excessive_native_rejections"
            if update % config["save_every"] == 0 or time.monotonic() - last_save >= 300 or stopped:
                save()
                last_save = time.monotonic()
            if stopped:
                break
        if stopped:
            break
        evaluations[label] = score_beliefs(model, data, development, adapt=adapt, device=device, seed=seed + 881)
        if label == "warm":
            warm_heads = head_state(model)
            warm_source = hashlib.sha256(b"".join(v.contiguous().numpy().tobytes() for _, v in sorted(warm_heads.items()))).hexdigest()
            export_heads(model, out / "warm_auxiliary.pt", native_sha256=KAKUNA_34_SHA256,
                         identity=identity, config=config, update=update)
        elif label == "control":
            control_heads = head_state(model)
            export_heads(model, out / "control_auxiliary.pt", native_sha256=KAKUNA_34_SHA256,
                         identity=identity, config=config, update=update)
        else:
            break
        phase += 1
        update = 0
        load_heads(model, warm_heads)
        head_optimizer, native_optimizer = optimizers()
        sampler, route = generators(phase)
        save()
    save()
    complete = not stopped and phase == 2 and update == config["condition_updates"]
    if complete:
        control_rows = [row for row in histories if row["stage"] == "control"]
        adapted_rows = [row for row in histories if row["stage"] == "adapted"]
        if [r["query_schedule_sha256"] for r in control_rows] != [r["query_schedule_sha256"] for r in adapted_rows]:
            raise ValueError("Paired conditions consumed different query schedules")
        for key in ("route_rng_before_sha256", "route_rng_after_sha256"):
            if [row[key] for row in control_rows] != [row[key] for row in adapted_rows]:
                raise ValueError("Paired belief queries consumed different route randomness")
        if config.get("export_native", True):
            atomic_save(model.native.state_dict(), out / "native_final.pt")
            export_heads(model, out / "adapted_auxiliary.pt", native_sha256=sha256(out / "native_final.pt"),
                         identity=identity, config=config, update=update)
    report = {"complete": complete, "stop_reason": reason, "identity": identity,
              "steps": steps, "head_updates_per_condition": config["condition_updates"] if complete else None,
              "native_accepted_updates": native_accepted, "native_rejected_updates": rejected,
              "evaluations": evaluations, "frozen_native_unchanged": frozen_fingerprint(model.native) == fixed,
              "elapsed_seconds": elapsed_prior + time.monotonic() - started,
              "initialization": initialization, "warm_head_fingerprint": warm_source,
              "automatic_promotion": False, "cache_sha256": data.signature}
    (out / "RESULTS.json").write_text(json.dumps(report, indent=2) + "\n")
    if sha256(Path(config["checkpoint"])) != KAKUNA_34_SHA256:
        raise ValueError("Protected starting checkpoint changed")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-steps", type=int)
    args = parser.parse_args()
    report = run(json.loads(args.config.read_text()), args.out, device=args.device,
                 resume=args.resume, stop_after_steps=args.stop_after_steps)
    print(json.dumps({key: value for key, value in report.items() if key != "evaluations"}))


if __name__ == "__main__":
    main()
