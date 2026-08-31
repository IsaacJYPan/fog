"""Training stage 1 — behavior cloning (spec §5).

Objective:
    CE(policy, human action)
  + marginal belief loss (censored diagnostic)  [opponent-team belief head]
  + set-AR next-field loss (censored auxiliary) [opponent-team generator]
  + value MSE against final outcome (γ=1)
  + self-belief losses (censored, own team)      [off by default in BC — see note]

Self-belief note: its correct input is the OPPONENT's view of the acting player (a mirror encoding),
which the replay shards don't carry. So in BC `self_belief_coef` defaults to 0; the self-belief /
leakage predictor is trained in self-play (§7 / training/leakage) where both perspectives are natural.
The self-belief LABELS are still parsed and stored for that path.

Optimizer AdamW, lr 3e-4 cosine to 3e-5, batch 512, ~10 epochs, mixed precision on CUDA.
Go/no-go (§5): action top-1 vs strong players ≥45%; belief item top-1 before reveal ≥55%; win-rate vs
SimpleHeuristics ≥65% (the last two are measured by the eval suite once a checkpoint exists).

    python -m fog.training.bc --config configs/bc_ou.yaml
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import torch

from fog.config import config_hash
from fog.training import losses


def _tensor_digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def _policy_supervision_mask(batch: dict, loss_cfg: dict) -> torch.Tensor:
    """Return replay rows allowed to supervise the policy.

    The submitted action is a public replay fact, but its reconstructed candidate features can
    still be inferred or unknown.  Strict replay-policy stages may therefore require the chosen
    candidate itself to have exact provenance without discarding the deliberately sampled private
    state variants used as uncertainty augmentation.
    """
    valid = batch.get(
        "policy_valid", torch.ones_like(batch["action"], dtype=torch.bool)
    ).bool()
    if loss_cfg.get("require_exact_chosen_action_features", False):
        provenance = batch.get("action_feature_provenance")
        if provenance is None:
            return torch.zeros_like(valid)
        chosen = provenance.gather(-1, batch["action"].unsqueeze(-1)).squeeze(-1)
        valid = valid & (chosen == 2)
    if loss_cfg.get("require_exact_chosen_legality", False):
        provenance = batch.get("legal_provenance")
        if provenance is None:
            return torch.zeros_like(valid)
        chosen = provenance.gather(-1, batch["action"].unsqueeze(-1)).squeeze(-1)
        valid = valid & (chosen == 2)
    return valid


def bc_loss(
    agent,
    batch,
    loss_cfg: dict,
    *,
    retention_decoder=None,
    retention_coef: float = 0.0,
    return_components: bool = False,
):
    out = agent(batch)
    policy_valid = _policy_supervision_mask(batch, loss_cfg)
    chosen_nll = -out.log_probs.gather(-1, batch["action"].unsqueeze(-1)).squeeze(-1)
    finite_policy = policy_valid & torch.isfinite(chosen_nll)
    action_ce = (
        chosen_nll[finite_policy].mean()
        if bool(finite_policy.any())
        else out.log_probs.new_zeros(())
    )
    value = losses.value_loss(out.value, batch["outcome"])
    belief = out.log_probs.new_zeros(())
    belief_stats: dict[str, int] = {}
    if out.belief is not None and loss_cfg.get("belief_coef", 1.0) > 0:
        belief = losses.belief_loss(
            out.belief.logits,
            batch["belief_targets"],
            batch["belief_valid"],
            stats=belief_stats,
        )
    belief_chain = out.log_probs.new_zeros(())
    chain_stats: dict[str, int] = {}
    if out.belief_chain is not None and loss_cfg.get("belief_chain_coef", 0.0) > 0:
        belief_chain = losses.belief_output_loss(
            out.belief_chain,
            batch["belief_targets"],
            batch["belief_valid"],
            stats=chain_stats,
        )
    self_belief = out.log_probs.new_zeros(())
    if out.self_belief is not None and loss_cfg.get("self_belief_coef", 0.0) > 0:
        self_belief = losses.belief_loss(
            out.self_belief.logits, batch["self_belief_targets"], batch["self_belief_valid"]
        )
    retention = out.log_probs.new_zeros(())
    if retention_decoder is not None and retention_coef > 0:
        if out.shared_state is None:
            raise ValueError("reconstruction retention requires a shared state")
        from fog.training.shared_latent_reconstruction import _loss as reconstruction_loss

        retention = reconstruction_loss(
            retention_decoder(out.shared_state),
            batch,
            id_weight=4.0,
            id_regression_weight=2.0,
            raw_weight=8.0,
            field_weight=8.0,
            worst_field_weight=3.0,
            rare_value_weight=10.0,
            identity_embedding_weight=1.0,
            tail_fraction=0.25,
            ordinal_ev_decoder=True,
            linear_id_values=True,
            linear_exact_values=True,
            categorical_cross_entropy=True,
        )
    components = {
        "action": loss_cfg.get("action_coef", 1.0) * action_ce,
        "belief": loss_cfg.get("belief_coef", 1.0) * belief,
        "belief_chain": loss_cfg.get("belief_chain_coef", 0.0) * belief_chain,
        "value": loss_cfg.get("value_coef", 0.5) * value,
        "self_belief": loss_cfg.get("self_belief_coef", 0.0) * self_belief,
        "retention": retention_coef * retention,
    }
    total = sum(components.values())
    with torch.no_grad():
        correct = out.log_probs.argmax(-1) == batch["action"]
        top1 = (
            correct[finite_policy].float().mean().item()
            if bool(finite_policy.any())
            else 0.0
        )
    metrics = {
        "loss": float(total.detach()), "action_ce": float(action_ce.detach()),
        "belief": float(belief.detach()), "belief_chain": float(belief_chain.detach()),
        "value": float(value.detach()), "action_top1": top1,
        "reconstruction_retention": float(retention.detach()),
        "belief_valid_count": float(belief_stats.get("valid", 0)),
        "belief_dropped_count": float(belief_stats.get("dropped_nonfinite", 0)),
        "belief_chain_valid_count": float(chain_stats.get("valid", 0)),
        "belief_chain_dropped_count": float(chain_stats.get("dropped_nonfinite", 0)),
        "policy_valid_count": float(finite_policy.sum().item()),
        "policy_total_count": float(policy_valid.numel()),
    }
    if "imputation_id" in batch:
        for imputation in batch["imputation_id"].unique().tolist():
            imputation = int(imputation)
            if imputation < 0:
                continue
            selected = finite_policy & (batch["imputation_id"] == imputation)
            metrics[f"imputation_{imputation}_valid_count"] = float(
                selected.sum().detach()
            )
            metrics[f"imputation_{imputation}_correct_count"] = float(
                correct[selected].sum().detach()
            )
            metrics[f"imputation_{imputation}_nll_sum_count"] = float(
                chosen_nll[selected].sum().detach()
            )
    from fog.training.ppo import _chain_probability_metrics

    if out.belief_chain is not None:
        metrics.update(_chain_probability_metrics(out.belief_chain, batch["belief_valid"]))
    if "legality_mask" in batch:
        # Replay legality masks are RECONSTRUCTED from public state (~3.5 legal actions/row vs ~9
        # from real |request| JSON), which inflates top-1: random guessing on 3.5 options is ~29%.
        # Always read action_top1 against these two — action_top1_lift is the chance-normalized
        # skill signal (1.0 = random over the reconstructed mask).
        avg_legal = batch["legality_mask"].float().sum(-1).clamp(min=1.0)
        metrics["avg_legal_actions"] = float(avg_legal.mean())
        metrics["action_top1_lift"] = top1 * float(avg_legal.mean())
    if return_components:
        return total, metrics, components
    return total, metrics


def _average_metrics(weighted: list[tuple[int, dict]]) -> dict[str, float]:
    """Row-weighted epoch metrics; count fields remain exact sums."""
    if not weighted:
        return {}
    keys = set().union(*(metrics for _, metrics in weighted))
    result: dict[str, float] = {}
    for key in keys:
        values = [(rows, metrics[key]) for rows, metrics in weighted if key in metrics]
        if key.endswith("_count"):
            result[key] = float(sum(value for _, value in values))
        else:
            denominator = sum(rows for rows, _ in values)
            result[key] = float(
                sum(rows * value for rows, value in values) / max(1, denominator)
            )
    valid = result.get("belief_valid_count", 0.0)
    dropped = result.get("belief_dropped_count", 0.0)
    result["belief_dropped_rate"] = dropped / max(1.0, valid + dropped)
    chain_valid = result.get("belief_chain_valid_count", 0.0)
    chain_dropped = result.get("belief_chain_dropped_count", 0.0)
    result["belief_chain_dropped_rate"] = chain_dropped / max(
        1.0, chain_valid + chain_dropped
    )
    imputation_ids = sorted(
        {
            int(match.group(1))
            for key in result
            if (match := re.fullmatch(r"imputation_(\d+)_valid_count", key))
        }
    )
    top1_values = []
    nll_values = []
    for imputation in imputation_ids:
        valid = result[f"imputation_{imputation}_valid_count"]
        top1 = result[f"imputation_{imputation}_correct_count"] / max(1.0, valid)
        nll = result[f"imputation_{imputation}_nll_sum_count"] / max(1.0, valid)
        result[f"imputation_{imputation}_top1"] = top1
        result[f"imputation_{imputation}_nll"] = nll
        if valid:
            top1_values.append(top1)
            nll_values.append(nll)
    if top1_values:
        result["imputation_top1_range"] = max(top1_values) - min(top1_values)
        result["imputation_nll_range"] = max(nll_values) - min(nll_values)
    return result


def _concatenate_batches(primary: dict, auxiliary: dict) -> dict:
    """Combine replay and simulator rows into one optimizer update."""

    if primary.keys() != auxiliary.keys():
        missing_primary = sorted(auxiliary.keys() - primary.keys())
        missing_auxiliary = sorted(primary.keys() - auxiliary.keys())
        raise ValueError(
            "cannot jointly batch mismatched sources: "
            f"missing_primary={missing_primary}, missing_auxiliary={missing_auxiliary}"
        )
    combined: dict = {}
    for key, left in primary.items():
        right = auxiliary[key]
        if isinstance(left, dict) and isinstance(right, dict):
            combined[key] = _concatenate_batches(left, right)
        elif torch.is_tensor(left) and torch.is_tensor(right):
            if left.ndim == 0 or right.ndim == 0:
                raise ValueError(f"joint batch leaf {key!r} must have a batch dimension")
            if (
                key in {"history", "history_ids", "history_state"}
                and left.ndim >= 2
                and left.shape[1] != right.shape[1]
                and left.shape[2:] == right.shape[2:]
            ):
                # Histories are front-padded throughout Fog: the newest turn is always last.
                turns = max(left.shape[1], right.shape[1])

                def front_pad(value: torch.Tensor, target_turns: int = turns) -> torch.Tensor:
                    if value.shape[1] == target_turns:
                        return value
                    shape = (value.shape[0], target_turns, *value.shape[2:])
                    padded = value.new_zeros(shape)
                    padded[:, -value.shape[1] :] = value
                    return padded

                left, right = front_pad(left), front_pad(right)
            if left.shape[1:] != right.shape[1:]:
                raise ValueError(
                    f"joint batch leaf {key!r} shape mismatch: "
                    f"{tuple(left.shape)} vs {tuple(right.shape)}"
                )
            combined[key] = torch.cat((left, right), dim=0)
        else:
            raise TypeError(
                f"joint batch leaf {key!r} must be tensors or nested dictionaries, got "
                f"{type(left).__name__} and {type(right).__name__}"
            )
    return combined


def _pcgrad_pair(
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


def _pcgrad_many(
    task_gradients: list[tuple[torch.Tensor | None, ...]],
    parameters: list[torch.nn.Parameter],
) -> tuple[list[torch.Tensor], dict[str, float]]:
    """Deterministic symmetric PCGrad for three or more separately normalized tasks."""

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


def _equal_norm_pcgrad_pair(
    primary: tuple[torch.Tensor | None, ...],
    auxiliary: tuple[torch.Tensor | None, ...],
    parameters: list[torch.nn.Parameter],
) -> tuple[list[torch.Tensor], dict[str, float]]:
    """PCGrad after equalizing source norms without changing a family's geometric scale.

    Replay and exact-simulator labels have different density and can produce very different
    gradient magnitudes even after each loss is normalized.  Equalizing within one loss family
    prevents the denser source from dominating while the geometric target preserves that
    family's natural scale relative to belief, chain, and value siblings.
    """

    left = [torch.zeros_like(p) if g is None else g for p, g in zip(parameters, primary, strict=True)]
    right = [torch.zeros_like(p) if g is None else g for p, g in zip(parameters, auxiliary, strict=True)]
    left_norm = sum(g.square().sum() for g in left).sqrt()
    right_norm = sum(g.square().sum() for g in right).sqrt()
    target = (left_norm * right_norm).clamp_min(1e-24).sqrt()
    if float(left_norm) > 0.0 and float(right_norm) > 0.0:
        left = [g * (target / left_norm) for g in left]
        right = [g * (target / right_norm) for g in right]
    combined, metrics = _pcgrad_pair(tuple(left), tuple(right), parameters)
    metrics["primary_gradient_norm_before_balance"] = float(left_norm.detach())
    metrics["auxiliary_gradient_norm_before_balance"] = float(right_norm.detach())
    return combined, metrics


def _fixed_eval(
    agent,
    ds,
    batch_size: int,
    loss_cfg: dict,
    device,
    seed: int,
    max_batches: int,
    *,
    retention_decoder=None,
    retention_coef: float = 0.0,
):
    """Deterministic fixed-subset evaluation without disturbing training RNG streams."""
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    was_training = agent.training
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    agent.eval()
    weighted: list[tuple[int, dict]] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(
            ds.batches(batch_size, shuffle=False, seed=seed, device=device)
        ):
            if max_batches > 0 and batch_index >= max_batches:
                break
            # FogAgent normally builds the sampled set-AR training route only in train mode.
            # Fixed evaluation deliberately uses eval mode, so request that auxiliary output
            # explicitly; otherwise the reported chain loss is a misleading constant zero.
            batch["compute_belief_chain"] = True
            if retention_decoder is None or retention_coef <= 0:
                _, metrics = bc_loss(agent, batch, loss_cfg)
            else:
                _, metrics = bc_loss(
                    agent,
                    batch,
                    loss_cfg,
                    retention_decoder=retention_decoder,
                    retention_coef=retention_coef,
                )
            weighted.append((int(batch["action"].shape[0]), metrics))
    if was_training:
        agent.train()
    torch.random.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    return _average_metrics(weighted)


def _write_metrics(path: Path, rows: list[dict]) -> None:
    fields = ["epoch"]
    fields.extend(sorted(set().union(*(row for row in rows)) - {"epoch"}))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _policy_modules(agent) -> list[torch.nn.Module]:
    if agent.policy_value.action_query_policy:
        modules = [
            agent.policy_value.action_query_input,
            agent.policy_value.action_query_attention,
            agent.policy_value.action_query_scorer,
        ]
    else:
        modules = [agent.policy_value.policy]
    if agent.shared_to_policy is not None:
        modules.append(agent.shared_to_policy)
    if agent.policy_value.action_scorer is not None:
        modules.append(agent.policy_value.action_scorer)
    if (
        agent.policy_value.conditional_action_scorer is not None
        and not agent.policy_value.action_query_policy
    ):
        modules.append(agent.policy_value.conditional_action_scorer)
    return modules


def _late_encoder_modules(agent) -> list[torch.nn.Module]:
    """Small representation adapters nearest the policy, excluding embeddings/sequence layers."""
    return [
        agent.backbone.current_state_encoder.head,
        agent.backbone.turn_history_encoder.head,
        agent.backbone.private_state_encoder.head,
        agent.backbone.public_fusion,
    ]


def _shared_latent_adaptive_modules(agent) -> list[torch.nn.Module]:
    """Trainable semantic parts of the shared state vector.

    The exact entity and field projections are deliberately excluded.  They are the
    full-rank paths that provide the audited information-preservation guarantee.
    """
    if agent.shared_latent is None:
        raise ValueError("shared-latent scope requires model.shared_latent_trunk=true")
    if agent.shared_latent.contextual or agent.shared_latent.flat_autoencoder:
        modules = [agent.shared_latent]
        if (
            agent.shared_latent.contextual
            and agent.shared_latent.typed_world_tokens
        ):
            # The revised trunk owns the Pokemon embeddings and temporal encoder conceptually,
            # even though checkpoint compatibility keeps those modules under ``backbone``.
            modules.extend(
                [agent.backbone.embeds, agent.backbone.turn_history_encoder]
            )
        return modules
    return [
        agent.shared_latent.semantic_entity,
        agent.shared_latent.global_context,
    ]


def _select_trainable_parameters(agent, scope: str) -> list[torch.nn.Parameter]:
    """Apply a narrow BC training scope and return exactly the parameters an optimizer may see."""
    if scope == "full":
        return [parameter for parameter in agent.parameters() if parameter.requires_grad]
    for parameter in agent.parameters():
        parameter.requires_grad_(False)
    if scope == "belief":
        module = agent.belief
    elif scope == "belief_shared":
        if agent.shared_to_belief is None:
            raise ValueError("belief_shared requires model.shared_latent_trunk=true")
        modules = [
            agent.belief,
            agent.shared_to_belief,
            *_shared_latent_adaptive_modules(agent),
        ]
        for selected in modules:
            for parameter in selected.parameters():
                parameter.requires_grad_(True)
        return [
            parameter
            for selected in modules
            for parameter in selected.parameters()
            if parameter.requires_grad
        ]
    elif scope in {"belief_value_shared", "belief_value_prior_shared"}:
        if agent.shared_latent is None:
            raise ValueError(f"{scope} requires model.shared_latent_trunk=true")
        modules = [agent.belief, agent.policy_value.value]
        if agent.shared_to_belief is not None:
            modules.append(agent.shared_to_belief)
        if agent.shared_to_value is not None:
            modules.append(agent.shared_to_value)
        if scope == "belief_value_prior_shared":
            if agent.matchup_value_prior is None:
                raise ValueError(
                    "belief_value_prior_shared requires model.matchup_value_prior=true"
                )
            modules.append(agent.matchup_value_prior)
        modules.extend(_shared_latent_adaptive_modules(agent))
        for selected in modules:
            for parameter in selected.parameters():
                parameter.requires_grad_(True)
        return [
            parameter
            for selected in modules
            for parameter in selected.parameters()
            if parameter.requires_grad
        ]
    elif scope == "policy_belief_value_prior_shared":
        if agent.shared_latent is None or agent.matchup_value_prior is None:
            raise ValueError(
                "policy_belief_value_prior_shared requires shared latent and matchup prior"
            )
        modules = [
            *_policy_modules(agent),
            agent.belief,
            agent.policy_value.value,
            agent.matchup_value_prior,
        ]
        if agent.shared_to_belief is not None:
            modules.append(agent.shared_to_belief)
        if agent.shared_to_value is not None:
            modules.append(agent.shared_to_value)
        modules.extend(_shared_latent_adaptive_modules(agent))
        for selected in modules:
            for parameter in selected.parameters():
                parameter.requires_grad_(True)
        return list({id(p): p for m in modules for p in m.parameters() if p.requires_grad}.values())
    elif scope == "belief_structured":
        modules = [
            agent.belief,
            agent.belief_structured_memory,
            agent.structured_to_belief,
        ]
        if any(selected is None for selected in modules):
            raise ValueError("belief_structured requires model.structured_memory=true")
        for selected in modules:
            for parameter in selected.parameters():
                parameter.requires_grad_(True)
        return [parameter for selected in modules for parameter in selected.parameters()]
    elif scope == "value":
        modules = [agent.policy_value.value]
        if agent.shared_to_value is not None:
            modules.append(agent.shared_to_value)
        for selected in modules:
            for parameter in selected.parameters():
                parameter.requires_grad_(True)
        return [parameter for selected in modules for parameter in selected.parameters()]
    elif scope == "value_prior":
        if agent.matchup_value_prior is None:
            raise ValueError("value_prior requires model.matchup_value_prior=true")
        module = agent.matchup_value_prior
    elif scope == "value_shared":
        if agent.shared_latent is None:
            raise ValueError("value_shared requires model.shared_latent_trunk=true")
        modules = [agent.policy_value.value]
        if agent.shared_to_value is not None:
            modules.append(agent.shared_to_value)
        modules.extend(_shared_latent_adaptive_modules(agent))
        for selected in modules:
            for parameter in selected.parameters():
                parameter.requires_grad_(True)
        return [
            parameter
            for selected in modules
            for parameter in selected.parameters()
            if parameter.requires_grad
        ]
    elif scope == "value_structured":
        modules = [
            agent.policy_value.value,
            agent.policy_value.structured_value,
            agent.value_structured_memory,
        ]
        if any(selected is None for selected in modules):
            raise ValueError("value_structured requires model.structured_memory=true")
        for selected in modules:
            for parameter in selected.parameters():
                parameter.requires_grad_(True)
        return [parameter for selected in modules for parameter in selected.parameters()]
    elif scope in {
        "policy",
        "policy_late_encoders",
        "policy_shared",
        "policy_structured",
    }:
        modules = _policy_modules(agent)
        if scope == "policy_late_encoders":
            modules.extend(_late_encoder_modules(agent))
        if scope == "policy_structured":
            if agent.policy_structured_memory is None:
                raise ValueError("policy_structured requires model.structured_memory=true")
            modules.append(agent.policy_structured_memory)
        if scope == "policy_shared":
            modules.extend(_shared_latent_adaptive_modules(agent))
        for selected in modules:
            for parameter in selected.parameters():
                parameter.requires_grad_(True)
        return [
            parameter
            for selected in modules
            for parameter in selected.parameters()
            if parameter.requires_grad
        ]
    else:
        raise ValueError(
            "bc.scope must be 'full', 'belief', 'belief_shared', 'belief_value_shared', "
            "'belief_structured', 'belief_value_prior_shared', "
            "'policy_belief_value_prior_shared', 'value', 'value_prior', "
            "'value_shared', 'value_structured', 'policy', 'policy_shared', "
            "'policy_structured', or "
            "'policy_late_encoders'"
        )
    for parameter in module.parameters():
        parameter.requires_grad_(True)
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def _split_replay_manifest(
    manifest_path: str | Path,
    out_dir: str | Path,
    holdout_fraction: float,
    seed: int,
) -> tuple[Path, Path]:
    """Write a deterministic replay-level train/holdout split and return both manifests.

    Every parsed replay is one shard, so splitting manifest rows prevents adjacent decisions from
    the same battle leaking into evaluation. Hash ordering makes the assignment stable if the
    source manifest is regenerated in a different row order.
    """
    import polars as pl

    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("bc.holdout_fraction must be strictly between 0 and 1")
    manifest = pl.read_parquet(manifest_path)
    if len(manifest) < 2:
        raise ValueError("replay holdout requires at least two parsed games")
    key_column = "id" if "id" in manifest.columns else "shard"
    keys = [str(value) for value in manifest[key_column].to_list()]
    order = sorted(
        range(len(keys)),
        key=lambda index: hashlib.sha256(
            f"{seed}\0{keys[index]}".encode()
        ).digest(),
    )
    n_holdout = min(len(order) - 1, max(1, round(len(order) * holdout_fraction)))
    holdout_keys = [keys[index] for index in order[:n_holdout]]
    holdout = manifest.filter(pl.col(key_column).cast(pl.String).is_in(holdout_keys))
    training = manifest.filter(~pl.col(key_column).cast(pl.String).is_in(holdout_keys))

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    train_path = destination / "train_manifest.parquet"
    holdout_path = destination / "holdout_manifest.parquet"
    training.write_parquet(train_path)
    holdout.write_parquet(holdout_path)
    return train_path, holdout_path


def train(cfg: dict) -> dict:
    from fog.models.agent import FogAgent
    from fog.training.dataset import ReplayDataset, StreamingReplayDataset
    from fog.training.preflight import require_preflight

    require_preflight(cfg)

    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    device = torch.device("cuda" if (cfg.get("device") == "cuda" and torch.cuda.is_available()) else "cpu")

    bc_cfg = cfg.get("bc", {})
    loss_cfg = cfg.get("loss", {})
    manifest = bc_cfg.get("manifest", "data/parsed/manifest.parquet")
    batch_size = int(bc_cfg.get("batch_size", 512))
    epochs = int(bc_cfg.get("epochs", 10))
    early_stopping_patience = int(bc_cfg.get("early_stopping_patience", 0))
    early_stopping_min_delta = float(bc_cfg.get("early_stopping_min_delta", 0.0))
    lr0 = float(bc_cfg.get("lr", 3e-4))
    lr1 = float(bc_cfg.get("lr_final", 3e-5))
    eval_batches = int(bc_cfg.get("eval_batches", 16))
    limit = bc_cfg.get("limit_rows")
    ckpt_dir = Path(cfg.get("log", {}).get("dir", "runs/bc")) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    holdout_fraction = float(bc_cfg.get("holdout_fraction", 0.0))
    eval_manifest = bc_cfg.get("holdout_manifest")
    if eval_manifest is not None and holdout_fraction > 0:
        raise ValueError(
            "bc.holdout_manifest and bc.holdout_fraction are mutually exclusive"
        )
    if holdout_fraction > 0:
        manifest, eval_manifest = _split_replay_manifest(
            manifest, ckpt_dir.parent, holdout_fraction, seed
        )

    streaming = bool(bc_cfg.get("streaming", False))
    shard_buffer = int(bc_cfg.get("shard_buffer", 64))

    def make_dataset(path, *, training: bool, force_policy_valid: bool = False):
        if streaming:  # 150k-game corpora: shard-streaming, no full concat
            return StreamingReplayDataset(
                path,
                shard_buffer=shard_buffer,
                force_policy_valid=force_policy_valid,
            )
        if force_policy_valid:
            raise ValueError("force_policy_valid currently requires bc.streaming=true")
        return ReplayDataset(path, limit_rows=limit if training else None)

    force_primary_policy_valid = bool(bc_cfg.get("force_primary_policy_valid", False))
    force_auxiliary_policy_valid = bool(bc_cfg.get("force_auxiliary_policy_valid", False))
    ds = make_dataset(
        manifest, training=True, force_policy_valid=force_primary_policy_valid
    )
    eval_ds = (
        make_dataset(
            eval_manifest,
            training=False,
            force_policy_valid=force_primary_policy_valid,
        )
        if eval_manifest else ds
    )
    auxiliary_manifest = bc_cfg.get("auxiliary_manifest")
    auxiliary_holdout_manifest = bc_cfg.get("auxiliary_holdout_manifest")
    if (auxiliary_manifest is None) != (auxiliary_holdout_manifest is None):
        raise ValueError(
            "bc.auxiliary_manifest and bc.auxiliary_holdout_manifest must be supplied together"
        )
    auxiliary_ds = (
        make_dataset(
            auxiliary_manifest,
            training=True,
            force_policy_valid=force_auxiliary_policy_valid,
        )
        if auxiliary_manifest is not None else None
    )
    auxiliary_eval_ds = (
        make_dataset(
            auxiliary_holdout_manifest,
            training=False,
            force_policy_valid=force_auxiliary_policy_valid,
        )
        if auxiliary_holdout_manifest is not None else None
    )
    guard_holdout_manifest = bc_cfg.get("guard_holdout_manifest")
    guard_train_manifest = bc_cfg.get("guard_train_manifest")
    guard_train_ds = (
        make_dataset(guard_train_manifest, training=True)
        if guard_train_manifest is not None else None
    )
    guard_eval_ds = (
        make_dataset(guard_holdout_manifest, training=False)
        if guard_holdout_manifest is not None else None
    )
    guard_metric = str(bc_cfg.get("guard_metric", "loss"))
    auxiliary_every = int(bc_cfg.get("auxiliary_every_primary_batches", 1))
    if auxiliary_every < 1:
        raise ValueError("bc.auxiliary_every_primary_batches must be positive")
    auxiliary_eval_weight = float(bc_cfg.get("auxiliary_eval_weight", 1.0))
    if auxiliary_eval_weight <= 0:
        raise ValueError("bc.auxiliary_eval_weight must be positive")
    auxiliary_joint_batch = bool(bc_cfg.get("auxiliary_joint_batch", False))
    if auxiliary_joint_batch and auxiliary_ds is None:
        raise ValueError("bc.auxiliary_joint_batch requires an auxiliary dataset")
    if auxiliary_joint_batch and auxiliary_every != 1:
        raise ValueError(
            "bc.auxiliary_joint_batch requires auxiliary_every_primary_batches=1"
        )
    auxiliary_gradient_surgery = str(
        bc_cfg.get("auxiliary_gradient_surgery", "none")
    ).lower()
    if auxiliary_gradient_surgery not in {"none", "pcgrad", "component_pcgrad"}:
        raise ValueError(
            "bc.auxiliary_gradient_surgery must be 'none', 'pcgrad', or "
            "'component_pcgrad'"
        )
    if auxiliary_gradient_surgery != "none" and not auxiliary_joint_batch:
        raise ValueError(
            "bc.auxiliary_gradient_surgery requires bc.auxiliary_joint_batch=true"
        )
    if guard_train_ds is not None and (
        not auxiliary_joint_batch or auxiliary_gradient_surgery != "pcgrad"
    ):
        raise ValueError(
            "bc.guard_train_manifest requires joint batches with pcgrad surgery"
        )
    guard_loss_cfg = dict(loss_cfg)
    auxiliary_loss_cfg = dict(loss_cfg)
    auxiliary_loss_cfg.update(bc_cfg.get("auxiliary_loss_overrides", {}) or {})
    if bool(bc_cfg.get("guard_train_value_only", True)):
        for key in (
            "action_coef",
            "belief_coef",
            "belief_chain_coef",
            "self_belief_coef",
            "self_belief_chain_coef",
            "preview_coef",
        ):
            guard_loss_cfg[key] = 0.0
        guard_loss_cfg["value_coef"] = 1.0
    split_note = (
        f"; holdout={len(eval_ds)} rows" if eval_manifest is not None else "; eval=train subset"
    )
    print(
        f"BC dataset: train={len(ds)} decision rows{split_note}  "
        f"(config_hash={config_hash(cfg)})"
    )
    agent = FogAgent.from_tables(cfg).to(device)
    if bc_cfg.get("init_ckpt"):
        initial = torch.load(bc_cfg["init_ckpt"], map_location=device)
        if bc_cfg.get("init_compatible", False):
            manifest = agent.load_compatible_state_dict(initial)
            (ckpt_dir.parent / "init_compatibility.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            print(
                f"[init] compatible warm start: {len(manifest['loaded'])} tensors loaded, "
                f"{len(manifest['initialized'])} initialized",
                flush=True,
            )
        else:
            agent.load_state_dict(initial)

    retention_decoder = None
    retention_coef = float(bc_cfg.get("reconstruction_coef", 0.0))
    retention_path = bc_cfg.get("reconstruction_decoder")
    if retention_coef > 0:
        if not retention_path:
            raise ValueError(
                "bc.reconstruction_coef requires bc.reconstruction_decoder"
            )
        if agent.shared_latent is None:
            raise ValueError("reconstruction retention requires a shared trunk")
        from fog.training.shared_latent_reconstruction import (
            SharedLatentDecoder,
            _class_sizes,
        )

        embeds = agent.backbone.embeds
        identity_weights = [
            embeds.species.weight,
            embeds.item.weight,
            embeds.ability.weight,
            *([embeds.move.weight] * 5),
            embeds.nature.weight,
            *([embeds.ev.weight] * 6),
            embeds.item.weight,
            embeds.ability.weight,
        ]
        retention_decoder = SharedLatentDecoder(
            agent.shared_latent.latent_dim,
            _class_sizes(agent),
            slot_dim=int(bc_cfg.get("reconstruction_slot_dim", 256)),
            latent_entity_dim=agent.shared_latent.entity_dim,
            identity_weights=identity_weights,
            contextual=agent.shared_latent.contextual,
            flat=agent.shared_latent.flat_autoencoder,
            linear_exact_values=True,
        ).to(device)
        retention_decoder.load_state_dict(
            torch.load(retention_path, map_location=device, weights_only=False)
        )
        retention_decoder.requires_grad_(False).eval()

    scope = str(bc_cfg.get("scope", "full"))
    trainable = _select_trainable_parameters(agent, scope)
    if not trainable:
        raise ValueError(f"bc.scope={scope!r} selected no trainable parameters")
    trainable_storage = {
        parameter.untyped_storage().data_ptr() for parameter in trainable
    }
    allowed_state_names = {
        name
        for name, value in agent.state_dict().items()
        if value.untyped_storage().data_ptr() in trainable_storage
    }
    initial_digests = {
        name: _tensor_digest(value) for name, value in agent.state_dict().items()
    }
    primary_steps = max(1, math.ceil(len(ds) / batch_size))
    max_primary_batches = int(bc_cfg.get("max_primary_batches_per_epoch", 0))
    if max_primary_batches < 0:
        raise ValueError("bc.max_primary_batches_per_epoch must be non-negative")
    if max_primary_batches:
        primary_steps = min(primary_steps, max_primary_batches)
    auxiliary_steps = (
        math.ceil(primary_steps / auxiliary_every) if auxiliary_ds is not None else 0
    )
    steps_per_epoch = primary_steps if auxiliary_joint_batch else primary_steps + auxiliary_steps
    total_steps = steps_per_epoch * epochs
    if scope == "policy_late_encoders":
        encoder_lr_scale = float(bc_cfg.get("encoder_lr_scale", 0.1))
        if not 0.0 < encoder_lr_scale <= 1.0:
            raise ValueError("bc.encoder_lr_scale must be in (0, 1]")
        policy_parameters = [
            parameter
            for module in _policy_modules(agent)
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        late_encoder_parameters = [
            parameter
            for module in _late_encoder_modules(agent)
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        opt = torch.optim.AdamW(
            [
                {"params": policy_parameters, "lr": lr0},
                {"params": late_encoder_parameters, "lr": lr0 * encoder_lr_scale},
            ]
        )
        final_ratio = lr1 / lr0

        def cosine_ratio(step_index: int) -> float:
            progress = min(1.0, max(0.0, step_index / total_steps))
            return final_ratio + (1.0 - final_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=cosine_ratio)
    elif scope in {"belief_shared", "belief_value_shared", "belief_value_prior_shared"}:
        shared_lr_scale = float(bc_cfg.get("shared_lr_scale", 0.1))
        if not 0.0 < shared_lr_scale <= 1.0:
            raise ValueError("bc.shared_lr_scale must be in (0, 1]")
        primary_modules = [agent.belief, agent.shared_to_belief]
        if scope in {"belief_value_shared", "belief_value_prior_shared"}:
            primary_modules.extend([agent.policy_value.value, agent.shared_to_value])
        if scope == "belief_value_prior_shared":
            primary_modules.append(agent.matchup_value_prior)
        primary_parameters = [
            parameter
            for module in primary_modules
            if module is not None
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        shared_parameters = [
            parameter
            for module in _shared_latent_adaptive_modules(agent)
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        opt = torch.optim.AdamW(
            [
                {"params": primary_parameters, "lr": lr0},
                {"params": shared_parameters, "lr": lr0 * shared_lr_scale},
            ]
        )
        final_ratio = lr1 / lr0

        def cosine_ratio(step_index: int) -> float:
            progress = min(1.0, max(0.0, step_index / total_steps))
            return final_ratio + (1.0 - final_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=cosine_ratio)
    elif scope == "policy_belief_value_prior_shared":
        shared_lr_scale = float(bc_cfg.get("shared_lr_scale", 0.05))
        auxiliary_head_lr_scale = float(bc_cfg.get("auxiliary_head_lr_scale", 0.5))
        if not 0.0 < shared_lr_scale <= 1.0:
            raise ValueError("bc.shared_lr_scale must be in (0, 1]")
        if not 0.0 < auxiliary_head_lr_scale <= 1.0:
            raise ValueError("bc.auxiliary_head_lr_scale must be in (0, 1]")
        policy_parameters = list({
            id(parameter): parameter
            for module in _policy_modules(agent)
            for parameter in module.parameters()
            if parameter.requires_grad
        }.values())
        auxiliary_modules = [agent.belief, agent.policy_value.value, agent.matchup_value_prior]
        auxiliary_modules.extend(
            module for module in (agent.shared_to_belief, agent.shared_to_value) if module is not None
        )
        auxiliary_parameters = list({
            id(parameter): parameter
            for module in auxiliary_modules
            for parameter in module.parameters()
            if parameter.requires_grad
        }.values())
        occupied = {id(parameter) for parameter in policy_parameters + auxiliary_parameters}
        shared_parameters = list({
            id(parameter): parameter
            for module in _shared_latent_adaptive_modules(agent)
            for parameter in module.parameters()
            if parameter.requires_grad and id(parameter) not in occupied
        }.values())
        opt = torch.optim.AdamW(
            [
                {"params": policy_parameters, "lr": lr0},
                {"params": auxiliary_parameters, "lr": lr0 * auxiliary_head_lr_scale},
                {"params": shared_parameters, "lr": lr0 * shared_lr_scale},
            ]
        )
        final_ratio = lr1 / lr0

        def cosine_ratio(step_index: int) -> float:
            progress = min(1.0, max(0.0, step_index / total_steps))
            return final_ratio + (1.0 - final_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=cosine_ratio)
    elif scope in {"value_shared", "policy_shared"}:
        shared_lr_scale = float(bc_cfg.get("shared_lr_scale", 0.1))
        if not 0.0 < shared_lr_scale <= 1.0:
            raise ValueError("bc.shared_lr_scale must be in (0, 1]")
        primary_modules = (
            [
                module
                for module in [agent.policy_value.value, agent.shared_to_value]
                if module is not None
            ]
            if scope == "value_shared"
            else _policy_modules(agent)
        )
        primary_parameters = [
            parameter
            for module in primary_modules
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        shared_parameters = [
            parameter
            for module in _shared_latent_adaptive_modules(agent)
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        opt = torch.optim.AdamW(
            [
                {"params": primary_parameters, "lr": lr0},
                {"params": shared_parameters, "lr": lr0 * shared_lr_scale},
            ]
        )
        final_ratio = lr1 / lr0

        def cosine_ratio(step_index: int) -> float:
            progress = min(1.0, max(0.0, step_index / total_steps))
            return final_ratio + (1.0 - final_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=cosine_ratio)
    elif scope in {"belief_structured", "value_structured", "policy_structured"}:
        reader_lr_scale = float(bc_cfg.get("reader_lr_scale", 0.1))
        if not 0.0 < reader_lr_scale <= 1.0:
            raise ValueError("bc.reader_lr_scale must be in (0, 1]")
        if scope == "belief_structured":
            primary_modules = [agent.belief, agent.structured_to_belief]
            reader_module = agent.belief_structured_memory
        elif scope == "value_structured":
            primary_modules = [agent.policy_value.value, agent.policy_value.structured_value]
            reader_module = agent.value_structured_memory
        else:
            primary_modules = _policy_modules(agent)
            reader_module = agent.policy_structured_memory
        primary_parameters = [
            parameter
            for module in primary_modules
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        reader_parameters = [
            parameter for parameter in reader_module.parameters() if parameter.requires_grad
        ]
        opt = torch.optim.AdamW(
            [
                {"params": primary_parameters, "lr": lr0},
                {"params": reader_parameters, "lr": lr0 * reader_lr_scale},
            ]
        )
        final_ratio = lr1 / lr0

        def cosine_ratio(step_index: int) -> float:
            progress = min(1.0, max(0.0, step_index / total_steps))
            return final_ratio + (1.0 - final_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=cosine_ratio)
    else:
        opt = torch.optim.AdamW(trainable, lr=lr0)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=total_steps, eta_min=lr1
        )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    metrics_path = ckpt_dir.parent / "metrics.csv"
    history: list[dict] = []
    baseline = _fixed_eval(
        agent,
        eval_ds,
        batch_size,
        loss_cfg,
        device,
        seed + 90_001,
        eval_batches,
        retention_decoder=retention_decoder,
        retention_coef=retention_coef,
    )
    auxiliary_baseline = (
        _fixed_eval(
            agent,
            auxiliary_eval_ds,
            batch_size,
            auxiliary_loss_cfg,
            device,
            seed + 190_001,
            int(bc_cfg.get("auxiliary_eval_batches", eval_batches)),
            retention_decoder=retention_decoder,
            retention_coef=retention_coef,
        )
        if auxiliary_eval_ds is not None else None
    )
    guard_baseline = (
        _fixed_eval(
            agent,
            guard_eval_ds,
            batch_size,
            guard_loss_cfg,
            device,
            seed + 290_001,
            int(bc_cfg.get("guard_eval_batches", eval_batches)),
            retention_decoder=retention_decoder,
            retention_coef=retention_coef,
        )
        if guard_eval_ds is not None else None
    )
    if guard_baseline is not None and guard_metric not in guard_baseline:
        raise ValueError(
            f"bc.guard_metric={guard_metric!r} is not an available evaluation metric"
        )
    primary_selection_metrics = tuple(
        bc_cfg.get("primary_selection_metrics", ("loss",))
    )
    auxiliary_selection_metrics = tuple(
        bc_cfg.get("auxiliary_selection_metrics", ("loss",))
    )
    if not primary_selection_metrics:
        raise ValueError("bc.primary_selection_metrics must not be empty")
    if auxiliary_baseline is not None and not auxiliary_selection_metrics:
        raise ValueError("bc.auxiliary_selection_metrics must not be empty")
    for label, selected_metrics, source_baseline in (
        ("primary", primary_selection_metrics, baseline),
        ("auxiliary", auxiliary_selection_metrics, auxiliary_baseline),
    ):
        if source_baseline is None:
            continue
        for metric_name in selected_metrics:
            if metric_name not in source_baseline:
                raise ValueError(
                    f"bc.{label}_selection_metrics contains unavailable metric "
                    f"{metric_name!r}"
                )
            if not math.isfinite(float(source_baseline[metric_name])) or float(
                source_baseline[metric_name]
            ) <= 0.0:
                raise ValueError(
                    f"bc.{label}_selection_metrics baseline {metric_name!r} must be "
                    "finite and positive"
                )
    baseline_row = {"epoch": 0, **{f"eval_{key}": value for key, value in baseline.items()}}
    if auxiliary_baseline is not None:
        baseline_row.update(
            {f"aux_eval_{key}": value for key, value in auxiliary_baseline.items()}
        )
        baseline_row["weighted_eval_loss"] = (
            baseline["loss"] + auxiliary_eval_weight * auxiliary_baseline["loss"]
        ) / (1.0 + auxiliary_eval_weight)
        baseline_row["worst_relative_eval_loss"] = 1.0
    for metric_name in primary_selection_metrics:
        baseline_row[f"eval_{metric_name}_relative"] = 1.0
    if auxiliary_baseline is not None:
        for metric_name in auxiliary_selection_metrics:
            baseline_row[f"aux_eval_{metric_name}_relative"] = 1.0
    if guard_baseline is not None:
        baseline_row.update(
            {f"guard_eval_{key}": value for key, value in guard_baseline.items()}
        )
        baseline_row["worst_relative_eval_loss"] = 1.0
    history.append(baseline_row)
    _write_metrics(metrics_path, history)
    print(
        f"[bc] baseline  loss={baseline['loss']:.3f}  "
        f"belief={baseline['belief']:.3f}  chain={baseline['belief_chain']:.3f}",
        flush=True,
    )
    if auxiliary_baseline is not None:
        print(
            f"[bc] auxiliary baseline loss={auxiliary_baseline['loss']:.3f}  "
            f"belief={auxiliary_baseline['belief']:.3f}  "
            f"chain={auxiliary_baseline['belief_chain']:.3f}  "
            f"value={auxiliary_baseline['value']:.3f}",
            flush=True,
        )
    if guard_baseline is not None:
        print(
            f"[bc] guard baseline {guard_metric}={guard_baseline[guard_metric]:.3f}  "
            f"loss={guard_baseline['loss']:.3f} value={guard_baseline['value']:.3f}",
            flush=True,
        )

    metrics: dict = {}
    # The unchanged initialization is a real evaluated checkpoint for both one- and two-source
    # runs. Preserve it unless training actually improves the selection metric. Previously only
    # auxiliary runs received this protection, so a single-source run could save an epoch that was
    # strictly worse than its baseline merely because ``best_eval_loss`` started at infinity.
    custom_relative_selection = (
        primary_selection_metrics != ("loss",)
        or auxiliary_baseline is not None
        or guard_baseline is not None
    )
    initial_selection_loss = 1.0 if custom_relative_selection else baseline["loss"]
    best_eval_loss = initial_selection_loss
    best_meaningful_eval_loss = initial_selection_loss
    best_epoch = 0
    torch.save(agent.state_dict(), ckpt_dir / "bc_best.pt")
    epochs_without_improvement = 0
    step = 0
    for epoch in range(epochs):
        agent.train()
        train_weighted: list[tuple[int, dict]] = []
        primary_weighted: list[tuple[int, dict]] = []
        auxiliary_weighted: list[tuple[int, dict]] = []
        guard_train_weighted: list[tuple[int, dict]] = []

        def auxiliary_batches_forever(epoch_index: int):
            cycle = 0
            while True:
                yielded = False
                for auxiliary_batch in auxiliary_ds.batches(
                    batch_size,
                    shuffle=True,
                    seed=seed + 50_000 + epoch_index * 1_000 + cycle,
                    device=device,
                ):
                    yielded = True
                    yield auxiliary_batch
                if not yielded:
                    raise RuntimeError("auxiliary dataset yielded no batches")
                cycle += 1

        auxiliary_iterator = (
            auxiliary_batches_forever(epoch) if auxiliary_ds is not None else None
        )

        def guard_batches_forever(epoch_index: int):
            cycle = 0
            while True:
                yielded = False
                for guard_batch in guard_train_ds.batches(
                    batch_size,
                    shuffle=True,
                    seed=seed + 70_000 + epoch_index * 1_000 + cycle,
                    device=device,
                ):
                    yielded = True
                    yield guard_batch
                if not yielded:
                    raise RuntimeError("guard training dataset yielded no batches")
                cycle += 1

        guard_iterator = guard_batches_forever(epoch) if guard_train_ds is not None else None

        def update(
            batch,
            source: str,
            train_rows: list[tuple[int, dict]],
            primary_rows: list[tuple[int, dict]],
            auxiliary_rows: list[tuple[int, dict]],
        ) -> None:
            nonlocal step
            opt.zero_grad(set_to_none=True)
            optimizer_stepped = True
            source_loss_cfg = auxiliary_loss_cfg if source == "auxiliary" else loss_cfg
            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    total, metrics = bc_loss(
                        agent,
                        batch,
                        source_loss_cfg,
                        retention_decoder=retention_decoder,
                        retention_coef=retention_coef,
                    )
                scaler.scale(total).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                scale_before = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                optimizer_stepped = scaler.get_scale() >= scale_before
            else:
                total, metrics = bc_loss(
                    agent,
                    batch,
                    source_loss_cfg,
                    retention_decoder=retention_decoder,
                    retention_coef=retention_coef,
                )
                total.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                opt.step()
            # GradScaler deliberately skips an update when it detects non-finite scaled
            # gradients.  Do not advance the learning-rate schedule past a step that did not
            # happen.
            if optimizer_stepped:
                sched.step()
            step += 1
            weighted = (int(batch["action"].shape[0]), metrics)
            train_rows.append(weighted)
            if source == "auxiliary":
                auxiliary_rows.append(weighted)
            elif source == "primary":
                primary_rows.append(weighted)

        def update_joint(
            primary_batch,
            auxiliary_batch,
            guard_batch,
            train_rows: list[tuple[int, dict]],
            primary_rows: list[tuple[int, dict]],
            auxiliary_rows: list[tuple[int, dict]],
            guard_rows: list[tuple[int, dict]],
        ) -> None:
            """One optimizer step with separately normalized losses from both sources."""

            nonlocal step
            opt.zero_grad(set_to_none=True)
            optimizer_stepped = True
            if auxiliary_gradient_surgery in {"pcgrad", "component_pcgrad"}:
                autocast_device = "cuda" if device.type == "cuda" else "cpu"
                with torch.amp.autocast(autocast_device, enabled=device.type == "cuda"):
                    primary_result = bc_loss(
                        agent,
                        primary_batch,
                        loss_cfg,
                        retention_decoder=retention_decoder,
                        retention_coef=retention_coef,
                        return_components=auxiliary_gradient_surgery == "component_pcgrad",
                    )
                    auxiliary_result = bc_loss(
                        agent,
                        auxiliary_batch,
                        auxiliary_loss_cfg,
                        retention_decoder=retention_decoder,
                        retention_coef=retention_coef,
                        return_components=auxiliary_gradient_surgery == "component_pcgrad",
                    )
                    guard_result = (
                        bc_loss(agent, guard_batch, guard_loss_cfg)
                        if guard_batch is not None else None
                    )
                if auxiliary_gradient_surgery == "pcgrad":
                    primary_total, primary_metrics = primary_result
                    auxiliary_total, auxiliary_metrics = auxiliary_result
                    primary_gradients = torch.autograd.grad(
                        primary_total, trainable, allow_unused=True
                    )
                    auxiliary_gradients = torch.autograd.grad(
                        auxiliary_total,
                        trainable,
                        allow_unused=True,
                        retain_graph=guard_result is not None,
                    )
                    if guard_result is None:
                        gradients, surgery_metrics = _pcgrad_pair(
                            primary_gradients, auxiliary_gradients, trainable
                        )
                        guard_metrics = None
                    else:
                        guard_total, guard_metrics = guard_result
                        guard_gradients = torch.autograd.grad(
                            guard_total, trainable, allow_unused=True
                        )
                        gradients, surgery_metrics = _pcgrad_many(
                            [primary_gradients, auxiliary_gradients, guard_gradients],
                            trainable,
                        )
                else:
                    primary_total, primary_metrics, primary_components = primary_result
                    auxiliary_total, auxiliary_metrics, auxiliary_components = auxiliary_result
                    active_names = [
                        name
                        for name in primary_components
                        if primary_components[name].requires_grad
                        and auxiliary_components[name].requires_grad
                        and (
                            float(primary_components[name].detach().abs()) > 0.0
                            or float(auxiliary_components[name].detach().abs()) > 0.0
                        )
                    ]
                    gradients = [torch.zeros_like(parameter) for parameter in trainable]
                    surgery_metrics = {}
                    for name in active_names:
                        primary_gradients = torch.autograd.grad(
                            primary_components[name],
                            trainable,
                            allow_unused=True,
                            retain_graph=True,
                        )
                        auxiliary_gradients = torch.autograd.grad(
                            auxiliary_components[name],
                            trainable,
                            allow_unused=True,
                            retain_graph=True,
                        )
                        family_gradients, family_metrics = _equal_norm_pcgrad_pair(
                            primary_gradients, auxiliary_gradients, trainable
                        )
                        gradients = [
                            total_gradient + family_gradient
                            for total_gradient, family_gradient in zip(
                                gradients, family_gradients, strict=True
                            )
                        ]
                        surgery_metrics.update(
                            {
                                f"{name}_{key}": value
                                for key, value in family_metrics.items()
                            }
                        )
                    if not active_names:
                        raise RuntimeError("component PCGrad found no active loss families")
                if not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
                    raise RuntimeError("non-finite PCGrad gradient")
                for parameter, gradient in zip(trainable, gradients, strict=True):
                    parameter.grad = gradient
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                opt.step()
                primary_metrics.update(surgery_metrics)
                auxiliary_metrics.update(surgery_metrics)
                if guard_result is not None:
                    guard_metrics.update(surgery_metrics)
            elif device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    primary_total, primary_metrics = bc_loss(
                        agent,
                        primary_batch,
                        loss_cfg,
                        retention_decoder=retention_decoder,
                        retention_coef=retention_coef,
                    )
                    auxiliary_total, auxiliary_metrics = bc_loss(
                        agent,
                        auxiliary_batch,
                        auxiliary_loss_cfg,
                        retention_decoder=retention_decoder,
                        retention_coef=retention_coef,
                    )
                    # Normalize each source independently before averaging. Exact simulator rows
                    # have far denser belief labels than censored replays, so concatenating rows
                    # and normalizing once would silently let the simulator dominate the trunk.
                    total = 0.5 * (primary_total + auxiliary_total)
                scaler.scale(total).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                scale_before = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                optimizer_stepped = scaler.get_scale() >= scale_before
            else:
                primary_total, primary_metrics = bc_loss(
                    agent,
                    primary_batch,
                    loss_cfg,
                    retention_decoder=retention_decoder,
                    retention_coef=retention_coef,
                )
                auxiliary_total, auxiliary_metrics = bc_loss(
                    agent,
                    auxiliary_batch,
                    auxiliary_loss_cfg,
                    retention_decoder=retention_decoder,
                    retention_coef=retention_coef,
                )
                (0.5 * (primary_total + auxiliary_total)).backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                opt.step()
            if optimizer_stepped:
                sched.step()
            step += 1
            primary_weighted_row = (
                int(primary_batch["action"].shape[0]),
                primary_metrics,
            )
            auxiliary_weighted_row = (
                int(auxiliary_batch["action"].shape[0]),
                auxiliary_metrics,
            )
            primary_rows.append(primary_weighted_row)
            auxiliary_rows.append(auxiliary_weighted_row)
            train_rows.extend((primary_weighted_row, auxiliary_weighted_row))
            if guard_batch is not None:
                guard_weighted_row = (
                    int(guard_batch["action"].shape[0]),
                    guard_metrics,
                )
                guard_rows.append(guard_weighted_row)
                train_rows.append(guard_weighted_row)

        for batch_index, batch in enumerate(
            ds.batches(batch_size, shuffle=True, seed=seed + epoch, device=device),
            start=1,
        ):
            if auxiliary_joint_batch:
                update_joint(
                    batch,
                    next(auxiliary_iterator),
                    next(guard_iterator) if guard_iterator is not None else None,
                    train_weighted,
                    primary_weighted,
                    auxiliary_weighted,
                    guard_train_weighted,
                )
            else:
                update(
                    batch,
                    "primary",
                    train_weighted,
                    primary_weighted,
                    auxiliary_weighted,
                )
                if auxiliary_iterator is not None and batch_index % auxiliary_every == 0:
                    update(
                        next(auxiliary_iterator),
                        "auxiliary",
                        train_weighted,
                        primary_weighted,
                        auxiliary_weighted,
                    )
            if max_primary_batches and batch_index >= max_primary_batches:
                break
        train_metrics = _average_metrics(train_weighted)
        primary_train_metrics = _average_metrics(primary_weighted)
        auxiliary_train_metrics = _average_metrics(auxiliary_weighted)
        guard_train_metrics = _average_metrics(guard_train_weighted)
        eval_metrics = _fixed_eval(
            agent,
            eval_ds,
            batch_size,
            loss_cfg,
            device,
            seed + 90_001,
            eval_batches,
            retention_decoder=retention_decoder,
            retention_coef=retention_coef,
        )
        auxiliary_eval_metrics = (
            _fixed_eval(
                agent,
                auxiliary_eval_ds,
                batch_size,
                auxiliary_loss_cfg,
                device,
                seed + 190_001,
                int(bc_cfg.get("auxiliary_eval_batches", eval_batches)),
                retention_decoder=retention_decoder,
                retention_coef=retention_coef,
            )
            if auxiliary_eval_ds is not None else None
        )
        guard_eval_metrics = (
            _fixed_eval(
                agent,
                guard_eval_ds,
                batch_size,
                guard_loss_cfg,
                device,
                seed + 290_001,
                int(bc_cfg.get("guard_eval_batches", eval_batches)),
                retention_decoder=retention_decoder,
                retention_coef=retention_coef,
            )
            if guard_eval_ds is not None else None
        )
        row = {"epoch": epoch + 1}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"primary_train_{key}": value for key, value in primary_train_metrics.items()})
        row.update({f"aux_train_{key}": value for key, value in auxiliary_train_metrics.items()})
        row.update({f"guard_train_{key}": value for key, value in guard_train_metrics.items()})
        row.update({f"eval_{key}": value for key, value in eval_metrics.items()})
        selection_loss = eval_metrics["loss"]
        relative_losses = []
        for metric_name in primary_selection_metrics:
            relative = eval_metrics[metric_name] / baseline[metric_name]
            relative_losses.append(relative)
            row[f"eval_{metric_name}_relative"] = relative
        if auxiliary_eval_metrics is not None:
            row.update(
                {f"aux_eval_{key}": value for key, value in auxiliary_eval_metrics.items()}
            )
            weighted_eval_loss = (
                eval_metrics["loss"]
                + auxiliary_eval_weight * auxiliary_eval_metrics["loss"]
            ) / (1.0 + auxiliary_eval_weight)
            for metric_name in auxiliary_selection_metrics:
                relative = (
                    auxiliary_eval_metrics[metric_name]
                    / auxiliary_baseline[metric_name]
                )
                relative_losses.append(relative)
                row[f"aux_eval_{metric_name}_relative"] = relative
            # Select the checkpoint that improves the weaker source, rather than allowing the
            # much larger replay corpus or the cleaner simulator corpus to hide regression in the
            # other. The weighted absolute loss remains a diagnostic.
            row["weighted_eval_loss"] = weighted_eval_loss
        if guard_eval_metrics is not None:
            row.update(
                {f"guard_eval_{key}": value for key, value in guard_eval_metrics.items()}
            )
            relative_losses.append(
                guard_eval_metrics[guard_metric]
                / max(1e-12, guard_baseline[guard_metric])
            )
        if custom_relative_selection:
            selection_loss = max(relative_losses)
            row["worst_relative_eval_loss"] = selection_loss
        history.append(row)
        _write_metrics(metrics_path, history)
        if selection_loss < best_eval_loss:
            best_eval_loss = selection_loss
            best_epoch = epoch + 1
            torch.save(agent.state_dict(), ckpt_dir / "bc_best.pt")

        if selection_loss < best_meaningful_eval_loss - early_stopping_min_delta:
            best_meaningful_eval_loss = selection_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        print(
            f"[bc] epoch {epoch+1}/{epochs}  train_loss={train_metrics['loss']:.3f}  "
            f"eval_loss={eval_metrics['loss']:.3f}  "
            f"belief={eval_metrics['belief']:.3f}  chain={eval_metrics['belief_chain']:.3f}  "
            f"value={eval_metrics['value']:.3f}  "
            f"query_p={eval_metrics.get('belief_chain_true_prob', 0.0):.3f}  "
            f"dropped={eval_metrics['belief_chain_dropped_rate']:.3%}",
            flush=True,
        )
        if auxiliary_eval_metrics is not None:
            print(
                f"[bc] auxiliary eval loss={auxiliary_eval_metrics['loss']:.3f}  "
                f"belief={auxiliary_eval_metrics['belief']:.3f}  "
                f"chain={auxiliary_eval_metrics['belief_chain']:.3f}  "
                f"value={auxiliary_eval_metrics['value']:.3f}  "
                f"worst_relative={selection_loss:.4f}",
                flush=True,
            )
        if guard_eval_metrics is not None:
            print(
                f"[bc] guard eval {guard_metric}={guard_eval_metrics[guard_metric]:.3f}  "
                f"loss={guard_eval_metrics['loss']:.3f} "
                f"value={guard_eval_metrics['value']:.3f}  "
                f"worst_relative={selection_loss:.4f}",
                flush=True,
            )
        metrics = eval_metrics
        torch.save(agent.state_dict(), ckpt_dir / f"bc_epoch{epoch+1}.pt")
        if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
            print(
                "[bc] early stop: no held-out improvement of at least "
                f"{early_stopping_min_delta:.6g} for {epochs_without_improvement} epochs",
                flush=True,
            )
            break

    final = ckpt_dir / "bc_final.pt"
    torch.save(agent.state_dict(), final)
    best_path = ckpt_dir / "bc_best.pt"
    best_state = torch.load(best_path, map_location="cpu", weights_only=False)
    changed_names = sorted(
        name
        for name, value in best_state.items()
        if _tensor_digest(value) != initial_digests[name]
    )
    illegal_names = sorted(set(changed_names) - allowed_state_names)
    scope_report = {
        "status": "passed" if not illegal_names else "failed",
        "scope": scope,
        "source": str(bc_cfg.get("init_ckpt")) if bc_cfg.get("init_ckpt") else None,
        "checkpoint": str(best_path),
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "allowed_state_tensor_count": len(allowed_state_names),
        "changed_tensor_count": len(changed_names),
        "changed_tensors": changed_names,
        "illegal_changed_tensors": illegal_names,
        "policy_frozen": scope in {"belief_value_shared", "belief_value_prior_shared"},
        "validation_teams_used": False,
    }
    (ckpt_dir.parent / "scope_verification.json").write_text(
        json.dumps(scope_report, indent=2), encoding="utf-8"
    )
    if illegal_names:
        raise RuntimeError(
            f"bc.scope={scope!r} changed tensors outside the selected scope: {illegal_names[:8]}"
        )
    print(
        f"[bc] done -> {final}  (best eval epoch={best_epoch}, loss={best_eval_loss:.3f})",
        flush=True,
    )
    return metrics


def main() -> None:
    from fog.config import load_config_with_args

    ap = argparse.ArgumentParser(description="FOG behavior cloning (§5).")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", dest="overrides", help="dotted overrides, e.g. bc.epochs=3")
    args = ap.parse_args()
    train(load_config_with_args(args.config, args.overrides))


if __name__ == "__main__":
    main()
