"""Create an auditable compatible seed for a wider typed-token trunk."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from fog.config import load_config
from fog.eval.encoder_information_probe_live import TensorObservationDataset
from fog.models.agent import FogAgent

_ALLOWED_REINITIALIZED_PREFIXES = (
    "shared_latent.",
    "shared_to_belief.",
    "shared_to_policy.",
    "shared_to_value.",
    "policy_value.",
)


def _digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _allowed(name: str) -> bool:
    return name.startswith(_ALLOWED_REINITIALIZED_PREFIXES)


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    source_agent = FogAgent.from_tables(load_config(args.source_config)).eval()
    source_agent.load_state_dict(source)
    target = FogAgent.from_tables(load_config(args.target_config)).to(device).eval()
    migration = target.load_compatible_state_dict(source)

    illegal_initialized = [name for name in migration["initialized"] if not _allowed(name)]
    illegal_mismatched = [name for name in migration["mismatched"] if not _allowed(name)]
    if illegal_initialized or illegal_mismatched or migration["obsolete"]:
        raise RuntimeError(
            "wide-token migration changed tensors outside typed-trunk/head interfaces: "
            f"initialized={illegal_initialized[:12]} mismatched={illegal_mismatched[:12]} "
            f"obsolete={migration['obsolete'][:12]}"
        )

    seeded = target.state_dict()
    preserved = {
        name: _digest(value) == _digest(seeded[name])
        for name, value in source.items()
        if name in seeded and tuple(value.shape) == tuple(seeded[name].shape)
    }
    changed_preserved = [name for name, same in preserved.items() if not same]
    if changed_preserved:
        raise RuntimeError(f"compatible tensors changed while widening: {changed_preserved[:12]}")

    dataset = TensorObservationDataset(args.data)
    batch = next(dataset.batches(args.batch_size, shuffle=False, seed=0, device=device))
    with torch.no_grad():
        contexts = target.encode_contexts(batch)
        latent = target.shared_latent(batch, contexts, target.backbone.embeds)
        output = target(batch)
    expected_shape = (
        int(batch["entities"].shape[0]),
        int(target.shared_latent.token_count * target.shared_latent.token_dim),
    )
    if tuple(latent.shape) != expected_shape or not torch.isfinite(latent).all():
        raise RuntimeError(f"invalid widened latent: got {tuple(latent.shape)}, expected {expected_shape}")
    if not torch.isfinite(output.value).all():
        raise RuntimeError("widened value output is nonfinite")
    finite_policy = output.log_probs[torch.isfinite(output.log_probs)]
    if finite_policy.numel() == 0 or not torch.isfinite(finite_policy).all():
        raise RuntimeError("widened policy output has no finite legal logits")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(seeded, args.out)
    report = {
        "status": "complete",
        "source": str(args.checkpoint),
        "output": str(args.out),
        "source_config": str(args.source_config),
        "target_config": str(args.target_config),
        "source_parameter_count": source_agent.n_params(),
        "target_parameter_count": target.n_params(),
        "parameter_increase": target.n_params() - source_agent.n_params(),
        "token_count": target.shared_latent.token_count,
        "source_token_dim": source_agent.shared_latent.token_dim,
        "target_token_dim": target.shared_latent.token_dim,
        "target_latent_dim": target.shared_latent.latent_dim,
        "rows_checked": int(batch["entities"].shape[0]),
        "compatible_tensors_byte_identical": True,
        "illegal_initialized": illegal_initialized,
        "illegal_mismatched": illegal_mismatched,
        "migration": migration,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
