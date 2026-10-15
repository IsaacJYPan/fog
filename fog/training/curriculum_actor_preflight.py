"""Check fixed CUDA rollout numerics, speed and exact-team collection before training."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import tempfile

import numpy as np
import torch

from fog.config import load_config
from fog.models.agent import FogAgent
from fog.training.curriculum_pilot import ReverseSchedule, VerifiedBank, collect, train
from fog.training.ppo import _index_nested


def check(config, recipe, bank, checkpoint, out, *, parent=None, anchor_checkpoint=None):
    if out.exists() or recipe.get("actor_device") != "cuda" or not torch.cuda.is_available():
        raise ValueError("Require a new report and an available, declared CUDA actor")
    torch.set_num_threads(1)
    batch = bank.contract_batch()
    visible_keys = {"entities", "entity_ids", "field", "history", "history_ids", "history_state", "history_len",
                    "active_idx", "legality_mask", "action_features", "opp_revealed"}
    visible = {key: value for key, value in batch.items() if key in visible_keys}
    actor = FogAgent.from_tables(config).eval()
    actor.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
    probabilities, durations = {}, {}
    with torch.inference_mode():
        for device in ("cpu", "cuda"):
            actor.to(device)

            def moved(value):
                return {key: moved(item) for key, item in value.items()} if isinstance(value, dict) else value.to(device)

            inputs = [_index_nested(moved(visible), torch.tensor([i], device=device)) for i in range(len(batch["action"]))]
            for sample in inputs:
                actor(sample)
            measured, probs = [], []
            for _ in range(2):
                for sample in inputs:
                    if device == "cuda":
                        torch.cuda.synchronize()
                    start = time.monotonic()
                    result = actor(sample).log_probs.exp()
                    if device == "cuda":
                        torch.cuda.synchronize()
                    measured.append(time.monotonic()-start)
                    probs.append(result.cpu())
            probabilities[device] = torch.cat(probs)
            durations[device] = {"mean_seconds": float(np.mean(measured)), "maximum_seconds": max(measured),
                                 "forwards": len(measured)}
    error = float((probabilities["cpu"]-probabilities["cuda"]).abs().max())
    speed_ratio = durations["cuda"]["mean_seconds"]/durations["cpu"]["mean_seconds"]
    numeric_speed_passed = error <= 1e-3 and speed_ratio <= 1.5
    result = {"passed": False, "numeric_speed_passed": numeric_speed_passed, "cpu_cuda_max_error": error,
              "tolerance": 1e-3, "cuda_cpu_mean_time_ratio": speed_ratio, "maximum_allowed_ratio": 1.5,
              "durations": durations, "timing_scope": "Forward calls only; excludes transfers and simulator work",
              "development_contexts_used_only_for_numeric_and_speed_checks": True}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2)+"\n")
    if not numeric_speed_passed:
        raise ValueError("CUDA actor failed fixed probability or throughput preflight")
    anchor = deepcopy(actor).cpu()
    from fog.env.showdown_direct import ShowdownDirectClient

    probe = dict(recipe, condition="ordinary", positions_per_batch=32)
    start = time.monotonic()
    with ShowdownDirectClient() as client, torch.inference_mode():
        sample, episodes, collected = collect(client, actor, anchor, bank, np.random.default_rng(20261016),
                                              ReverseSchedule(), probe)
    if not episodes or any(row["prefix"] for row in episodes) or not sample.get("belief_targets"):
        raise ValueError("CUDA collection lost full games or exact-team supervision")
    if not torch.isfinite(sample["old_log_probs"]).all():
        raise ValueError("CUDA collection lost finite behavior probabilities")
    result.update(cuda_real_collection_passed=True, full_games=len(episodes), collected_positions=collected,
                  real_collection_seconds=time.monotonic()-start,
                  optimized_positions=0, training_data_only=True)
    out.write_text(json.dumps(result, indent=2)+"\n")
    if parent is not None:
        trial = dict(recipe, max_batches=2, positions_per_batch=32, minibatch_size=16,
                     epochs=1, warmup_batches=1, save_every=1, training_seconds=300)

        def equal(a, b):
            if isinstance(a, torch.Tensor):
                return torch.equal(a, b)
            if isinstance(a, np.ndarray):
                return np.array_equal(a, b)
            if isinstance(a, dict):
                return a.keys() == b.keys() and all(equal(a[key], b[key]) for key in a)
            if isinstance(a, (list, tuple)):
                return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
            return a == b

        # Discarded trial branches never become the training initializer.
        with tempfile.TemporaryDirectory(prefix="fog-cuda-resume-") as temporary:
            direct, resumed = Path(temporary)/"direct", Path(temporary)/"resumed"
            options = {"device": "cuda", "parent": parent, "anchor_checkpoint": anchor_checkpoint}
            train(config, trial, bank, checkpoint, direct, **options)
            train(config, trial, bank, checkpoint, resumed, stop_after_batches=1, **options)
            train(config, trial, bank, checkpoint, resumed, resume=resumed/"latest.state.pt", **options)
            a = torch.load(direct/"latest.state.pt", weights_only=False, map_location="cpu")
            b = torch.load(resumed/"latest.state.pt", weights_only=False, map_location="cpu")
            checks = {key: equal(a[key], b[key]) for key in ("agent", "optimizer", "random", "schedule", "lineage")}
            result.update(cuda_resume_checks=checks, discarded_trial_optimizer_steps=12)
            if not all(checks.values()):
                out.write_text(json.dumps(result, indent=2)+"\n")
                raise ValueError("Actual CUDA full-state continuation/resume differs")
        result["cuda_full_state_resume_verified"] = True
    result["passed"] = True
    out.write_text(json.dumps(result, indent=2)+"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "recipe", "bank", "pool", "checkpoint", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--anchor-checkpoint", type=Path)
    args = parser.parse_args()
    report = check(load_config(args.config), json.loads(args.recipe.read_text()),
                   VerifiedBank(args.bank, args.pool), args.checkpoint, args.out,
                   parent=args.parent, anchor_checkpoint=args.anchor_checkpoint)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
