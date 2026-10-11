"""Matched native Kakuna fine-tuning with optional shared belief/outcome supervision."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("METAMON_CACHE_DIR", str(Path("runs/pretrained/cache").resolve()))

import numpy as np
import torch

from fog.pretrained.data import MatchedMixture
from fog.pretrained.checkpoints import anchor_updates, export_candidates
from fog.pretrained.joint import JointKakuna
from fog.pretrained.initialization import (
    initialize_auxiliary, set_learning_rates, starting_checkpoint,
    training_optimizer, validate_warm_start,
)
from fog.pretrained.kakuna import load_kakuna
from fog.pretrained.loading import SequenceLoader, fresh_files
from fog.pretrained.optimization import AuxiliaryGradientCap
from fog.pretrained.state import save_training_state, load_training_state


def auxiliary_forward(model, latent, targets, generator):
    from fog.constants.tables import TABLES
    from fog.encoding.reveals import masks_for_tokens

    length = targets["outcomes"].shape[1]
    latent = latent[:, :length].reshape(-1, latent.shape[-1])
    flat = {key: value.reshape(-1, *value.shape[2:]) for key, value in targets.items()}
    active = flat["outcome_valid"].bool()
    latent = latent[active]
    flat = {key: value[active] for key, value in flat.items()}
    fresh = flat["verified"].flatten(1).any(-1)
    # Belief prediction need not run on published trajectories with no verified sets.
    context_loss = model.outcome(latent).squeeze(-1).tanh()
    outcome = (context_loss - flat["outcomes"]).square().mean()
    if not bool(fresh.any()):
        return {"marginal": outcome * 0, "autoregressive": outcome * 0, "outcome": outcome}
    reveals = flat["reveals"][fresh]
    rows = [masks_for_tokens(row, TABLES) for row in reveals.detach().cpu().numpy()]
    masks = {
        key: torch.as_tensor(np.stack([row[key] for row in rows]), device=latent.device)
        for key in rows[0]
    }
    losses = model.auxiliary_losses(
        latent[fresh],
        reveals,
        flat["truth"][fresh],
        flat["verified"][fresh],
        flat["outcomes"][fresh],
        flat["outcome_valid"][fresh],
        masks=masks,
        generator=generator,
    )
    losses["outcome"] = outcome
    return losses


def train(config, *, out: Path, device="cuda", resume=None, max_seconds=None):
    torch.set_num_threads(config.get("cpu_threads", 2))
    # AMAGO feeds probability-smoothed buffer actions to OneHotCategorical.
    # Its log_prob scores argmax, but Torch's optional one-hot support assertion
    # rejects the smoothing first. Loader checks enforce the discrete action
    # contract explicitly; disabling assertions preserves the native objective.
    torch.distributions.Distribution.set_default_validate_args(False)
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Frozen GPU experiment cannot silently fall back to CPU")
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    out.mkdir(parents=True, exist_ok=True)
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    identity = hashlib.sha256(canonical.encode()).hexdigest()
    descriptor = out / "recipe.json"
    if descriptor.exists() and descriptor.read_text() != canonical + "\n":
        raise ValueError("Output folder already belongs to a different immutable recipe")
    descriptor.write_text(canonical + "\n")
    published = [
        str(path) for path in sorted(Path(config["published"]).glob("train/gen9ou/*.json.lz4"))
    ]
    fresh, audit = fresh_files(
        Path(config["fresh"]), minimum_games=config.get("minimum_fresh_games", 1000)
    )
    (out / "data_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    sampler = MatchedMixture(published, fresh, seed)
    initialization = validate_warm_start(config)
    runtime = load_kakuna(
        starting_checkpoint(config), device=device, training=True,
        verify_original=not bool(initialization),
    )
    # Both arms construct the same heads before restoring the primary RNG stream,
    # keeping subsequent native stochastic operations identically seeded.
    primary_rng = torch.get_rng_state()
    model = JointKakuna(runtime.model).to(device)
    initialize_auxiliary(model, config)
    torch.set_rng_state(primary_rng)
    runtime.model.train()
    model.train()
    joint = config["condition"] == "joint"
    if config["condition"] not in ("ordinary", "joint"):
        raise ValueError("Unsupported matched condition")
    optimizer, parameters = training_optimizer(model, config)
    (out / "initialization.json").write_text(json.dumps({
        "mode": "weight_warm_start" if initialization else "original_kakuna34",
        "sources": initialization, "selected_condition": config["condition"],
        "optimizer": "same_recipe_resume" if resume else "reset",
        "sampler": "same_recipe_resume" if resume else "reset",
        "seed": seed,
    }, indent=2) + "\n")
    cap = AuxiliaryGradientCap(model.shared_parameters(), config["auxiliary_gradient_fraction"])
    route_generator = torch.Generator(device=device).manual_seed(seed + 8675309)
    loader = SequenceLoader(runtime, sequence_length=config["sequence_length"])
    update, previous_elapsed = 0, 0.0
    if resume:
        state = load_training_state(
            resume,
            model=model,
            optimizer=optimizer,
            sampler=sampler,
            route_generator=route_generator,
            identity=identity,
            config=config,
        )
        update, previous_elapsed = state["update"], state["elapsed_seconds"]
    started, last_save = time.monotonic(), time.monotonic()
    micro = config["micro_batch_size"]
    effective = config["effective_batch_size"]
    if effective % micro:
        raise ValueError("Microbatches must divide the effective batch exactly")
    accumulation = effective // micro
    history = out / "metrics.jsonl"
    if resume and history.exists():
        retained = [
            line
            for line in history.read_text().splitlines()
            if json.loads(line)["update"] <= update
        ]
        history.write_text("".join(line + "\n" for line in retained))
    stopped = False
    while update < config["updates"]:
        if max_seconds and time.monotonic() - started >= max_seconds:
            stopped = True
            break
        begin = time.monotonic()
        samples = sampler.next_batch(effective)
        schedule_hash = hashlib.sha256(
            json.dumps(samples, separators=(",", ":")).encode()
        ).hexdigest()
        optimizer.zero_grad(set_to_none=True)
        totals = {"primary": 0.0, "marginal": 0.0, "autoregressive": 0.0, "outcome": 0.0}
        counts = {
            "positions": 0,
            "verified_fields": 0,
            "published_trajectories": sum(s[0] == "published" for s in samples),
        }
        for index in range(0, effective, micro):
            batch, targets = loader.batch(samples[index : index + micro], device)
            primary, latent = model.primary_with_latent(batch)
            if not bool(torch.isfinite(primary)):
                raise FloatingPointError("Nonfinite native primary loss")
            totals["primary"] += float(primary.detach()) / accumulation
            counts["positions"] += int(targets["outcome_valid"].sum())
            counts["verified_fields"] += int(targets["verified"].sum())
            if joint:
                # Belief Transformer dropout must not advance the native policy's
                # RNG stream. AR routes use their own serializable generator.
                devices = [torch.device(device).index or 0] if device.startswith("cuda") else []
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(seed + update * accumulation + index // micro + 104729)
                    losses = auxiliary_forward(model, latent, targets, route_generator)
                auxiliary = sum(
                    config["auxiliary_weights"][key] * loss for key, loss in losses.items()
                )
                if not bool(torch.isfinite(auxiliary)):
                    raise FloatingPointError("Nonfinite normalized auxiliary loss")
                cap.backward(primary, auxiliary, scale=1 / accumulation)
                for key, loss in losses.items():
                    totals[key] += float(loss.detach()) / accumulation
            else:
                (primary / accumulation).backward()
        gradient_metrics = cap.finish() if joint else {}
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                parameters, config["gradient_clip"], error_if_nonfinite=True
            )
        )
        learning_rates = set_learning_rates(optimizer, config, update)
        optimizer.step()
        runtime.model.soft_sync_targets()
        update += 1
        elapsed = previous_elapsed + time.monotonic() - started
        row = {
            "update": update,
            "losses": totals,
            "counts": counts,
            "learning_rate": learning_rates["native"],
            "learning_rates": learning_rates,
            "unclipped_gradient_norm": gradient_norm,
            "seconds": time.monotonic() - begin,
            "elapsed_seconds": elapsed,
            "schedule_sha256": schedule_hash,
            **gradient_metrics,
        }
        with history.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if (
            update % config["save_every_updates"] == 0
            or time.monotonic() - last_save >= config["save_every_seconds"]
            or update in anchor_updates(config["updates"])
        ):
            previous = out / "previous.state.pt"
            latest = out / "latest.state.pt"
            if latest.exists():
                latest.replace(previous)
            save_training_state(
                out / "latest.state.pt",
                model=model,
                optimizer=optimizer,
                sampler=sampler,
                route_generator=route_generator,
                update=update,
                identity=identity,
                config=config,
                elapsed_seconds=elapsed,
            )
            export_candidates(model, out, update=update, identity=identity, config=config)
            last_save = time.monotonic()
    save_training_state(
        out / "latest.state.pt",
        model=model,
        optimizer=optimizer,
        sampler=sampler,
        route_generator=route_generator,
        update=update,
        identity=identity,
        config=config,
        elapsed_seconds=previous_elapsed + time.monotonic() - started,
    )
    export_candidates(model, out, update=update, identity=identity, config=config, final=True)
    result = {
        "identity": identity,
        "completed_updates": update,
        "planned_updates": config["updates"],
        "complete": update == config["updates"],
        "time_budget_reached": stopped,
        "elapsed_seconds": previous_elapsed + time.monotonic() - started,
        "condition": config["condition"],
        "promotion": False,
    }
    (out / "RESULTS.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-seconds", type=float)
    args = parser.parse_args()
    print(
        json.dumps(
            train(
                json.loads(args.config.read_text()),
                out=args.out,
                device=args.device,
                resume=args.resume,
                max_seconds=args.max_seconds,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
