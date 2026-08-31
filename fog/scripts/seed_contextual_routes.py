"""Add zero-initialized contextual preservation routes without changing model outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from fog.config import load_config
from fog.eval.encoder_information_probe_live import TensorObservationDataset
from fog.models.agent import FogAgent

_ALLOWED_NEW_PREFIXES = (
    "shared_latent.dynamic_entity_encoder.",
    "shared_latent.global_state_encoder.",
)


def _digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _max_difference(left: torch.Tensor | None, right: torch.Tensor | None) -> float:
    if left is None or right is None:
        if left is right:
            return 0.0
        raise RuntimeError("output presence changed during contextual-route seeding")
    if torch.equal(left, right):
        return 0.0
    if not torch.equal(torch.isfinite(left), torch.isfinite(right)):
        return float("inf")
    finite = torch.isfinite(left)
    return float((left[finite] - right[finite]).abs().max())


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    base = FogAgent.from_tables(load_config(args.base_config)).to(device).eval()
    base.load_state_dict(source)
    enhanced = FogAgent.from_tables(load_config(args.enhanced_config)).to(device).eval()
    migration = enhanced.load_compatible_state_dict(source)
    initialized = list(migration["initialized"])
    illegal_initialized = [
        name for name in initialized if not name.startswith(_ALLOWED_NEW_PREFIXES)
    ]
    if illegal_initialized:
        raise RuntimeError(f"unexpected newly initialized tensors: {illegal_initialized[:12]}")
    if migration["mismatched"] or migration["obsolete"]:
        raise RuntimeError(
            "contextual-route seed must not replace existing tensors: "
            f"mismatched={list(migration['mismatched'])[:8]} obsolete={migration['obsolete'][:8]}"
        )

    dataset = TensorObservationDataset(args.data)
    batch = next(dataset.batches(args.batch_size, shuffle=False, seed=0, device=device))
    with torch.no_grad():
        base_contexts = base.encode_contexts(batch)
        enhanced_contexts = enhanced.encode_contexts(batch)
        base_latent = base.shared_latent(batch, base_contexts, base.backbone.embeds)
        enhanced_latent = enhanced.shared_latent(batch, enhanced_contexts, enhanced.backbone.embeds)
        base_output = base(batch)
        enhanced_output = enhanced(batch)
    differences = {
        "shared_latent": _max_difference(base_latent, enhanced_latent),
        "policy_log_probs": _max_difference(base_output.log_probs, enhanced_output.log_probs),
        "value": _max_difference(base_output.value, enhanced_output.value),
        "belief": _max_difference(base_output.belief, enhanced_output.belief),
        "belief_chain": _max_difference(base_output.belief_chain, enhanced_output.belief_chain),
    }
    if any(value != 0.0 for value in differences.values()):
        raise RuntimeError(f"zero-initialized contextual routes changed outputs: {differences}")

    seeded = enhanced.state_dict()
    preserved = {
        name: _digest(value) == _digest(seeded[name])
        for name, value in source.items()
        if name in seeded and tuple(value.shape) == tuple(seeded[name].shape)
    }
    if not all(preserved.values()):
        changed = [name for name, same in preserved.items() if not same]
        raise RuntimeError(f"existing checkpoint tensors changed while seeding: {changed[:12]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(seeded, args.out)
    report = {
        "status": "complete",
        "source": str(args.checkpoint),
        "output": str(args.out),
        "base_config": str(args.base_config),
        "enhanced_config": str(args.enhanced_config),
        "rows_checked": int(batch["entities"].shape[0]),
        "outputs_byte_identical": True,
        "existing_tensors_byte_identical": True,
        "initialized": initialized,
        "initialized_only_contextual_routes": not illegal_initialized,
        "differences": differences,
        "migration": migration,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--enhanced-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
