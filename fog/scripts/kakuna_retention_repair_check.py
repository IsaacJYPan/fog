"""Discarded real-checkpoint regression for the previously uncovered policy drift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from types import SimpleNamespace

import torch

from fog.pretrained.kakuna import KAKUNA_34_SHA256, load_kakuna, sha256
from fog.pretrained.latent_training import FeatureData, assess_behavior
from fog.pretrained.retention import RetentionLimits, configure_upper_training, upper_modules
from fog.pretrained.retention_replay import OriginalReferenceRepair


def check(config, fixture, out, *, device="cpu"):
    torch.set_num_threads(config.get("cpu_threads", 1))
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    source = torch.load(fixture, map_location="cpu", weights_only=True)
    if source["checkpoint_sha256"] != KAKUNA_34_SHA256:
        raise ValueError("Repair regression does not belong to protected Kakuna34")
    data = FeatureData(
        config["cache"], config["cache_sha256"], allow_trial=config.get("allow_local_cache", False)
    )
    model = SimpleNamespace(native=load_kakuna(config["checkpoint"], device=device).model)
    parameters = configure_upper_training(model.native, adapt=True)
    buffers = {key: value.detach().clone() for key, value in model.native.named_buffers()}
    limits = RetentionLimits(**config["limits"])
    repair = OriginalReferenceRepair(parameters, limits=limits)
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0.0)
    for parameter in parameters:
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    assert optimizer.state
    for module, group in zip(upper_modules(model.native), ("block", "norm")):
        module.load_state_dict(source["upper"][group], strict=True)
    batches = {
        "stopped_training_batch": data.batch(source["specifications"], device),
        "sentinel": data.batch(data.draw(random.Random(config["seed"] + 709), 8), device),
        "independent_training": data.batch(data.draw(random.Random(20261020), 16), device),
    }

    def assess():
        return {
            name: assess_behavior(model, batch, adapt=True, microbatch=config["microbatch"])
            for name, batch in batches.items()
        }

    before = assess()
    assert not repair.accepts(before)
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if device.startswith("cuda") else []
    result = repair.ensure(assess, optimizer)
    assert (
        result["repaired"]
        and result["original_reference_scale"] < 1
        and repair.accepts(result["after"])
    )
    assert not optimizer.state and torch.equal(rng, torch.get_rng_state())
    assert all(
        torch.equal(a, b)
        for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all() if cuda_rng else [])
    )
    assert all(torch.equal(value, buffers[key]) for key, value in model.native.named_buffers())
    assert sha256(Path(config["checkpoint"])) == KAKUNA_34_SHA256
    report = {
        "passed": True,
        "device": device,
        "fixture_sha256": sha256(Path(fixture)),
        "source_state_sha256": source["source_state_sha256"],
        "before": before,
        "repair": result,
        "native_Adam_reset_verified": True,
        "native_buffers_unchanged": True,
        "query_rng_unchanged": True,
        "protected_checkpoint_unchanged": True,
        "discarded_optimized_contexts": 0,
        "initialized_main_run": False,
        "scope": "Exact failed public training batch and two independent training probes. No global retention or playing-strength guarantee.",
    }
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "RESULTS.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    check(json.loads(args.config.read_text()), args.fixture, args.out, device=args.device)


if __name__ == "__main__":
    main()
