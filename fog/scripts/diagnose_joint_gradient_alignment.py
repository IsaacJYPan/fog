"""Read-only gradient alignment diagnostic for replay/simulator joint curricula."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from fog.config import load_config
from fog.models.agent import FogAgent
from fog.training.bc import bc_loss
from fog.training.dataset import StreamingReplayDataset

COMPONENTS = {
    "belief": {"action_coef": 0.0, "belief_coef": 1.0, "belief_chain_coef": 0.0, "value_coef": 0.0},
    "chain": {"action_coef": 0.0, "belief_coef": 0.0, "belief_chain_coef": 1.0, "value_coef": 0.0},
    "value": {"action_coef": 0.0, "belief_coef": 0.0, "belief_chain_coef": 0.0, "value_coef": 1.0},
}


def _group(name: str) -> str | None:
    if name.startswith("backbone.embeds."):
        return "backbone_embeddings"
    if name.startswith("backbone.turn_history_encoder."):
        return "history_encoder"
    if name.startswith("shared_latent.contextual_encoder."):
        return "shared_transformer"
    if name.startswith("shared_latent."):
        return "shared_input_routes"
    if name.startswith("shared_to_belief."):
        return "belief_reader"
    if name.startswith("shared_to_value."):
        return "value_reader"
    return None


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = float(left.norm() * right.norm())
    if denominator == 0.0:
        return 0.0
    return float(torch.dot(left, right) / denominator)


def _vectors(agent: FogAgent, batch: dict, component: str) -> tuple[dict[str, torch.Tensor], float]:
    selected = [
        (name, parameter)
        for name, parameter in agent.named_parameters()
        if _group(name) is not None
    ]
    total, _ = bc_loss(agent, batch, COMPONENTS[component])
    gradients = torch.autograd.grad(
        total,
        [parameter for _, parameter in selected],
        allow_unused=True,
    )
    grouped: dict[str, list[torch.Tensor]] = defaultdict(list)
    for (name, parameter), gradient in zip(selected, gradients, strict=True):
        value = torch.zeros_like(parameter) if gradient is None else gradient
        grouped[_group(name)].append(value.detach().float().cpu().flatten())
    return {name: torch.cat(parts) for name, parts in grouped.items()}, float(total.detach())


def diagnose(args: argparse.Namespace) -> dict:
    cfg = load_config(args.config)
    device = torch.device(args.device)
    agent = FogAgent.from_tables(cfg).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False))
    agent.train()  # The belief chain is intentionally produced in training mode.

    replay = StreamingReplayDataset(args.replay_manifest, shard_buffer=args.shard_buffer)
    simulator = StreamingReplayDataset(args.simulator_manifest, shard_buffer=args.shard_buffer)
    replay_iter = replay.batches(args.batch_size, shuffle=True, seed=args.seed, device=device)
    simulator_iter = simulator.batches(
        args.batch_size, shuffle=True, seed=args.seed + 1, device=device
    )
    sums: dict[tuple[str, str, str], torch.Tensor] = {}
    losses: dict[str, dict[str, list[float]]] = {
        source: {component: [] for component in COMPONENTS}
        for source in ("replay", "simulator")
    }
    batch_cosines: dict[str, dict[str, list[float]]] = {
        component: defaultdict(list) for component in COMPONENTS
    }
    for _ in range(args.batches):
        batches = {"replay": next(replay_iter), "simulator": next(simulator_iter)}
        for component in COMPONENTS:
            pair: dict[str, dict[str, torch.Tensor]] = {}
            for source, batch in batches.items():
                vectors, loss = _vectors(agent, batch, component)
                pair[source] = vectors
                losses[source][component].append(loss)
                for group, vector in vectors.items():
                    key = (source, component, group)
                    sums[key] = sums.get(key, torch.zeros_like(vector)) + vector
            for group in pair["replay"]:
                batch_cosines[component][group].append(
                    _cosine(pair["replay"][group], pair["simulator"][group])
                )

    alignment: dict[str, dict] = {}
    for component in COMPONENTS:
        alignment[component] = {}
        groups = sorted(
            group for source, family, group in sums if source == "replay" and family == component
        )
        for group in groups:
            replay_grad = sums[("replay", component, group)] / args.batches
            simulator_grad = sums[("simulator", component, group)] / args.batches
            cosines = batch_cosines[component][group]
            alignment[component][group] = {
                "aggregate_cosine": _cosine(replay_grad, simulator_grad),
                "mean_batch_cosine": sum(cosines) / len(cosines),
                "min_batch_cosine": min(cosines),
                "replay_norm": float(replay_grad.norm()),
                "simulator_norm": float(simulator_grad.norm()),
                "simulator_to_replay_norm_ratio": float(
                    simulator_grad.norm() / replay_grad.norm().clamp_min(1e-12)
                ),
            }
    report = {
        "status": "complete",
        "read_only": True,
        "checkpoint": str(args.checkpoint),
        "replay_manifest": str(args.replay_manifest),
        "simulator_manifest": str(args.simulator_manifest),
        "batch_size": args.batch_size,
        "batches": args.batches,
        "losses": {
            source: {
                component: sum(values) / len(values)
                for component, values in components.items()
            }
            for source, components in losses.items()
        },
        "alignment": alignment,
    }
    if not all(
        math.isfinite(value)
        for source in report["losses"].values()
        for value in source.values()
    ):
        raise RuntimeError("non-finite diagnostic loss")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--replay-manifest", required=True)
    parser.add_argument("--simulator-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--shard-buffer", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261181)
    args = parser.parse_args()
    print(json.dumps(diagnose(args), indent=2))


if __name__ == "__main__":
    main()
