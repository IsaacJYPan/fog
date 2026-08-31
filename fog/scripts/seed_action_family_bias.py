"""Create a traceable checkpoint with a one-time action-family logit correction.

This is intentionally different from a rollout-only exploration bonus: the correction is part of
the saved policy, so rollout, learning, evaluation, and the viewer all see the same logits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from fog.constants.schema import MOVE_TERA_SLOTS, N_ACTIONS

POLICY_BIAS_KEY = "policy_value.policy.2.bias"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_tera_bias(state: dict[str, torch.Tensor], amount: float) -> dict[str, torch.Tensor]:
    if not torch.isfinite(torch.tensor(amount)) or amount <= 0:
        raise ValueError("amount must be finite and positive")
    if POLICY_BIAS_KEY not in state:
        raise KeyError(f"checkpoint is missing {POLICY_BIAS_KEY}")
    bias = state[POLICY_BIAS_KEY]
    if tuple(bias.shape) != (N_ACTIONS,):
        raise ValueError(f"expected {POLICY_BIAS_KEY} shape ({N_ACTIONS},), got {tuple(bias.shape)}")

    seeded = type(state)((key, value.detach().clone()) for key, value in state.items())
    slots = list(MOVE_TERA_SLOTS)
    seeded[POLICY_BIAS_KEY][slots] += float(amount)
    return seeded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--amount", type=float, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    state = torch.load(args.input, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or not all(torch.is_tensor(value) for value in state.values()):
        raise TypeError("expected a plain model state-dict checkpoint")
    seeded = seed_tera_bias(state, args.amount)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(seeded, args.output)
    report = {
        "input": str(args.input),
        "input_sha256": _sha256(args.input),
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
        "changed_tensor": POLICY_BIAS_KEY,
        "changed_slots": list(MOVE_TERA_SLOTS),
        "added_logit": args.amount,
        "purpose": "persistent powered-up-action exploration seed; no rollout-only bonus",
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
