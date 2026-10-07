"""Freeze native CPU probabilities and enforce cached/CUDA agreement before training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("METAMON_CACHE_DIR", str(Path("runs/pretrained/cache").resolve()))

import torch

from fog.pretrained.kakuna import load_kakuna, sha256
from fog.pretrained.loading import SequenceLoader
from fog.pretrained.state import atomic_save


def probabilities(runtime, example, device, *, cached=False):
    obs = {key: value.to(device) for key, value in example["obs"].items()}
    rl2, times = example["rl2s"].to(device), example["time_idxs"].to(device)
    started = time.perf_counter()
    with torch.inference_mode():
        if cached:
            hidden = runtime.model.traj_encoder.init_hidden_state(1, torch.device(device))
            pieces = []
            for index in range(rl2.shape[1]):
                piece, _, hidden = runtime.distribution(
                    {key: value[:, index : index + 1] for key, value in obs.items()},
                    rl2[:, index : index + 1],
                    times[:, index : index + 1],
                    hidden,
                )
                pieces.append(piece.cpu())
            result = torch.cat(pieces, 1)
        else:
            result, _, _ = runtime.distribution(obs, rl2, times)
            result = result.cpu()
    return result, time.perf_counter() - started


def run(checkpoint, samples, out, *, device="cpu", reference=None, trained=False):
    torch.set_num_threads(1)
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA contract requires an actual CUDA device")
    before = sha256(checkpoint)
    runtime = load_kakuna(checkpoint, device=device, verify_original=not trained)
    examples = torch.load(samples, map_location="cpu", weights_only=True)
    expected = torch.load(reference, map_location="cpu", weights_only=True) if reference else None
    if expected and (
        expected["checkpoint_sha256"] != before or expected["samples_sha256"] != sha256(samples)
    ):
        raise ValueError("CPU reference inputs or checkpoint mismatch")
    results, records = [], []
    for index, example in enumerate(examples):
        dense, seconds = probabilities(runtime, example, device)
        cached, cached_seconds = probabilities(runtime, example, device, cached=True)
        cache_error = float((dense - cached).abs().max())
        cross_error = (
            float((dense - expected["probabilities"][index]).abs().max()) if expected else None
        )
        if not torch.isfinite(dense).all() or not torch.isfinite(cached).all():
            raise ValueError("Nonfinite native probability contract")
        if cache_error > 1e-3 or (cross_error is not None and cross_error > 1e-3):
            raise ValueError(
                f"Native probability contract failed: cache={cache_error}, CPU/CUDA={cross_error}"
            )
        illegal = example["obs"]["illegal_actions"].bool()
        if bool((dense[illegal] != 0).any()) or not torch.allclose(
            dense.sum(-1), torch.ones_like(dense.sum(-1)), atol=1e-5
        ):
            raise ValueError("Policy probability or action-mask contract failed")
        results.append(dense)
        records.append(
            {
                "positions": dense.shape[1],
                "cache_max_absolute_error": cache_error,
                "cpu_cuda_max_absolute_error": cross_error,
                "dense_seconds": seconds,
                "mean_cached_decision_seconds": cached_seconds / dense.shape[1],
            }
        )
    unchanged = before == sha256(checkpoint)
    if not unchanged:
        raise ValueError("Probability validation modified its starting checkpoint")
    out.mkdir(parents=True, exist_ok=True)
    atomic_save(
        {"checkpoint_sha256": before, "samples_sha256": sha256(samples), "probabilities": results},
        out / "probabilities.pt",
    )
    report = {
        "passed": True,
        "device": device,
        "torch_version": torch.__version__,
        "interface": runtime.interface_version,
        "checkpoint_sha256": before,
        "samples_sha256": sha256(samples),
        "weights_unchanged": unchanged,
        "tolerance": 1e-3,
        "cpu_cuda_verified": expected is not None and device.startswith("cuda"),
        "records": records,
    }
    (out / "contract.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--trained", action="store_true")
    parser.add_argument("--prepare-fresh", type=Path)
    args = parser.parse_args()
    if args.prepare_fresh:
        torch.set_num_threads(1)
        runtime = load_kakuna(args.checkpoint)
        loader = SequenceLoader(runtime)
        examples = []
        files = sorted(args.prepare_fresh.glob("*.json.gz"))[:4]
        if len(files) < 2:
            raise ValueError(
                "At least two exact-team games are required for the probability contract"
            )
        for index, path in enumerate(files):
            sample, _ = loader.load(("fresh", str(path), index + 1))
            length = len(sample.actions)
            examples.append(
                {
                    "obs": {
                        key: value[None, :length]
                        for key, value in sample.obs.items()
                        if key != "missing_action_mask"
                    },
                    "rl2s": sample.rl2s[None, :length],
                    "time_idxs": sample.time_idxs[None, :length],
                }
            )
        # Exercise AMAGO cache rollover past 128 in addition to real battle sequences.
        first = examples[0]
        repetitions = 160 // first["rl2s"].shape[1] + 1
        examples.append(
            {
                "obs": {
                    key: value.repeat(1, repetitions, 1)[:, :160]
                    for key, value in first["obs"].items()
                },
                "rl2s": first["rl2s"].repeat(1, repetitions, 1)[:, :160],
                "time_idxs": torch.arange(160)[None, :, None],
            }
        )
        args.samples.parent.mkdir(parents=True, exist_ok=True)
        atomic_save(examples, args.samples)
        del loader, runtime
    run(
        args.checkpoint,
        args.samples,
        args.out,
        device=args.device,
        reference=args.reference,
        trained=args.trained,
    )


if __name__ == "__main__":
    main()
