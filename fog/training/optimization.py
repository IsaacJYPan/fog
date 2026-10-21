"""Parameter groups, learning-rate schedules, and multi-task gradient updates."""
from __future__ import annotations

import math
from collections.abc import Iterable

import torch


def trainable_parameters(modules: Iterable[torch.nn.Module | None]) -> list[torch.nn.Parameter]:
    return [
        parameter
        for module in modules
        if module is not None
        for parameter in module.parameters()
        if parameter.requires_grad
    ]


def cosine_schedule(optimizer, final_ratio: float, total_steps: int):
    """Decay every parameter group's initial rate by the same cosine factor."""
    def ratio(step_index: int) -> float:
        progress = min(1.0, max(0.0, step_index / total_steps))
        return final_ratio + (1.0 - final_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=ratio)


def restricted_optimizer_groups(
    agent,
    trainable_prefixes: tuple[str, ...],
    lr_scales: dict[str, float],
    base_lr: float,
) -> tuple[list[dict], list[torch.nn.Parameter], dict[float, list[str]]]:
    """Freeze unselected tensors and assign per-prefix learning-rate scales."""
    if not trainable_prefixes:
        if lr_scales:
            raise ValueError("ppo.trainable_lr_scales requires trainable_prefixes")
        trainable = list(agent.parameters())
        return [{"params": trainable, "lr": base_lr}], trainable, {1.0: ["*"]}
    for prefix, scale in lr_scales.items():
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("ppo.trainable_lr_scales keys must be non-empty prefixes")
        if not math.isfinite(float(scale)) or float(scale) <= 0:
            raise ValueError(
                f"ppo.trainable_lr_scales[{prefix!r}] must be finite and positive"
            )
    grouped: dict[float, list[torch.nn.Parameter]] = {}
    group_names: dict[float, list[str]] = {}
    matched_scale_prefixes: set[str] = set()
    trainable: list[torch.nn.Parameter] = []
    for name, parameter in agent.named_parameters():
        selected = name.startswith(trainable_prefixes)
        parameter.requires_grad_(selected)
        if not selected:
            continue
        trainable.append(parameter)
        matching = [prefix for prefix in lr_scales if name.startswith(prefix)]
        scale_prefix = max(matching, key=len) if matching else None
        scale = float(lr_scales[scale_prefix]) if scale_prefix is not None else 1.0
        if scale_prefix is not None:
            matched_scale_prefixes.add(scale_prefix)
        grouped.setdefault(scale, []).append(parameter)
        group_names.setdefault(scale, []).append(name)
    if not trainable:
        raise ValueError(
            f"ppo.trainable_prefixes matched no parameters: {trainable_prefixes}"
        )
    unmatched = sorted(set(lr_scales) - matched_scale_prefixes)
    if unmatched:
        raise ValueError(
            f"ppo.trainable_lr_scales prefixes matched no trainable parameters: {unmatched}"
        )
    groups = [
        {"params": grouped[scale], "lr": base_lr * scale}
        for scale in sorted(grouped, reverse=True)
    ]
    return groups, trainable, group_names


def pcgrad_pair(
    primary: tuple[torch.Tensor | None, ...],
    auxiliary: tuple[torch.Tensor | None, ...],
    parameters: list[torch.nn.Parameter],
) -> tuple[list[torch.Tensor], dict[str, float]]:
    """Symmetric two-task PCGrad over the complete trainable parameter vector."""
    left = [torch.zeros_like(p) if g is None else g for p, g in zip(parameters, primary, strict=True)]
    right = [torch.zeros_like(p) if g is None else g for p, g in zip(parameters, auxiliary, strict=True)]
    dot = sum((a * b).sum() for a, b in zip(left, right, strict=True))
    left_sq = sum(a.square().sum() for a in left)
    right_sq = sum(b.square().sum() for b in right)
    cosine = dot / (left_sq.sqrt() * right_sq.sqrt()).clamp_min(1e-12)
    if float(dot) < 0.0:
        left_scale = dot / right_sq.clamp_min(1e-12)
        right_scale = dot / left_sq.clamp_min(1e-12)
        combined = [
            0.5 * ((a - left_scale * b) + (b - right_scale * a))
            for a, b in zip(left, right, strict=True)
        ]
    else:
        combined = [0.5 * (a + b) for a, b in zip(left, right, strict=True)]
    return combined, {
        "joint_gradient_cosine": float(cosine.detach()),
        "joint_gradient_conflict": float(dot.detach() < 0),
        "primary_gradient_norm": float(left_sq.sqrt().detach()),
        "auxiliary_gradient_norm": float(right_sq.sqrt().detach()),
    }


def pcgrad_many(
    task_gradients: list[tuple[torch.Tensor | None, ...]],
    parameters: list[torch.nn.Parameter],
) -> tuple[list[torch.Tensor], dict[str, float]]:
    """Deterministic symmetric PCGrad for separately normalized tasks."""
    if len(task_gradients) < 2:
        raise ValueError("multi-task PCGrad requires at least two tasks")
    original = [
        [
            torch.zeros_like(parameter) if gradient is None else gradient
            for parameter, gradient in zip(parameters, gradients, strict=True)
        ]
        for gradients in task_gradients
    ]
    projected = [[gradient.clone() for gradient in task] for task in original]
    metrics: dict[str, float] = {}
    conflicts = 0
    comparisons = 0
    for index, task in enumerate(projected):
        norm = sum(gradient.square().sum() for gradient in original[index]).sqrt()
        metrics[f"task_{index}_gradient_norm"] = float(norm.detach())
        for other_index, other in enumerate(original):
            if index == other_index:
                continue
            dot = sum(
                (left * right).sum() for left, right in zip(task, other, strict=True)
            )
            other_sq = sum(gradient.square().sum() for gradient in other)
            conflict = float(dot.detach() < 0)
            conflicts += int(conflict)
            comparisons += 1
            metrics[f"task_{index}_vs_{other_index}_conflict"] = conflict
            if conflict:
                scale = dot / other_sq.clamp_min(1e-12)
                task[:] = [
                    left - scale * right
                    for left, right in zip(task, other, strict=True)
                ]
    combined = [
        sum(task[parameter_index] for task in projected) / len(projected)
        for parameter_index in range(len(parameters))
    ]
    metrics["joint_gradient_conflict"] = conflicts / max(1, comparisons)
    return combined, metrics


def equal_norm_pcgrad_pair(
    primary: tuple[torch.Tensor | None, ...],
    auxiliary: tuple[torch.Tensor | None, ...],
    parameters: list[torch.nn.Parameter],
) -> tuple[list[torch.Tensor], dict[str, float]]:
    """Balance source norms before PCGrad, preserving their geometric mean scale."""
    left = [torch.zeros_like(p) if g is None else g for p, g in zip(parameters, primary, strict=True)]
    right = [torch.zeros_like(p) if g is None else g for p, g in zip(parameters, auxiliary, strict=True)]
    left_norm = sum(g.square().sum() for g in left).sqrt()
    right_norm = sum(g.square().sum() for g in right).sqrt()
    target = (left_norm * right_norm).clamp_min(1e-24).sqrt()
    if float(left_norm) > 0.0 and float(right_norm) > 0.0:
        left = [g * (target / left_norm) for g in left]
        right = [g * (target / right_norm) for g in right]
    combined, metrics = pcgrad_pair(tuple(left), tuple(right), parameters)
    metrics["primary_gradient_norm_before_balance"] = float(left_norm.detach())
    metrics["auxiliary_gradient_norm_before_balance"] = float(right_norm.detach())
    return combined, metrics
