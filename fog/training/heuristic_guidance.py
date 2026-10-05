"""Public-information teacher labels on learner-visited training states.

The teacher never selects the acting policy's action. Its independent random stream supplies
supervised targets only; accepted/rejected choices retain or discard those targets with the
ordinary rollout fragment. Labels can supply a PPO auxiliary loss or the primary cloning
objective. Neither mode has a DAgger regret guarantee.
"""
from __future__ import annotations

import math
import random

import torch


def guidance_weight(settings: dict, decisions: int, total: int) -> float:
    initial = float(settings.get("heuristic_coef", 0.0))
    final = float(settings.get("heuristic_coef_final", 0.0))
    if not all(math.isfinite(v) and v >= 0 for v in (initial, final)):
        raise ValueError("heuristic guidance coefficients must be finite and non-negative")
    horizon = settings.get("heuristic_decay_decisions", max(1, total))
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("ppo.heuristic_decay_decisions must be a positive integer")
    fraction = min(1.0, max(0, decisions) / horizon)
    return initial + (final - initial) * fraction


def validate_update_objective(settings: dict) -> tuple[str, str]:
    objective = settings.get("policy_objective", "ppo")
    value_target = settings.get("value_target", "gae")
    if objective not in ("ppo", "heuristic_bc"):
        raise ValueError("ppo.policy_objective must be ppo or heuristic_bc")
    if value_target not in ("gae", "outcome"):
        raise ValueError("ppo.value_target must be gae or outcome")
    weight = float(settings.get("heuristic_voluntary_switch_weight", 1.0))
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("heuristic voluntary-switch weight must be finite and positive")
    if objective == "heuristic_bc":
        if settings.get("target_kl") is not None:
            raise ValueError("heuristic_bc requires target_kl: null; PPO's stop gate is inapplicable")
        if any(float(settings.get(key, 0.0)) != 0 for key in ("heuristic_coef", "heuristic_coef_final")):
            raise ValueError("heuristic_bc uses teacher CE directly; heuristic coefficients must be zero")
    return objective, value_target


def validate_guidance(config: dict) -> None:
    settings = config.get("ppo", {})
    objective, _ = validate_update_objective(settings)
    teacher = settings.get("heuristic_teacher")
    if teacher not in (None, "simpleheuristics", "maxdamage"):
        raise ValueError("ppo.heuristic_teacher must be simpleheuristics or maxdamage")
    initial = guidance_weight(settings, 0, int(settings.get("total_decisions", 100000)))
    final = float(settings.get("heuristic_coef_final", 0.0))
    if teacher is None and (initial > 0 or final > 0):
        raise ValueError("positive heuristic guidance requires ppo.heuristic_teacher")
    if teacher is None and objective == "heuristic_bc":
        raise ValueError("heuristic_bc requires ppo.heuristic_teacher")
    if teacher is not None and config.get("env", {}).get("backend") != "showdown_direct":
        raise ValueError("heuristic guidance requires the showdown_direct backend")


class PublicHeuristicTeacher:
    """Seeded teacher with access only to the acting player's live battle view."""

    def __init__(self, name: str):
        self.name = name
        self.rng = random.Random(0)

    def start_game(self, seed: int) -> None:
        self.rng.seed(int(seed) ^ 0xD1B54A32D192ED03)

    def __call__(self, battle, observation, _output) -> int:
        from fog.env.actions import map_order_to_action
        from fog.env.showdown_local_runner import _scripted_order

        action = int(map_order_to_action(battle, _scripted_order(self.name, battle, self.rng)))
        if not 0 <= action < len(observation.legality_mask) or not observation.legality_mask[action]:
            raise ValueError(f"public heuristic teacher returned illegal action {action}")
        return action


def attach_guidance(adapter, config: dict) -> PublicHeuristicTeacher | None:
    validate_guidance(config)
    name = config.get("ppo", {}).get("heuristic_teacher")
    if name is None:
        return None
    teacher = PublicHeuristicTeacher(name)
    adapter.supervised_label_provider = teacher
    return teacher


def guidance_loss(log_probs: torch.Tensor, labels: torch.Tensor,
                  legality: torch.Tensor, *, voluntary_switch_weight: float = 1.0
                  ) -> tuple[torch.Tensor, dict]:
    if not math.isfinite(voluntary_switch_weight) or voluntary_switch_weight <= 0:
        raise ValueError("heuristic voluntary-switch weight must be finite and positive")
    if labels.shape != log_probs.shape[:1] or legality.shape != log_probs.shape:
        raise ValueError("heuristic labels and legality must align with policy rows")
    if labels.dtype != torch.long or bool(((labels < 0) | (labels >= log_probs.shape[1])).any()):
        raise ValueError("heuristic labels must be valid integer action indices")
    if not bool(legality.gather(1, labels[:, None]).all()):
        raise ValueError("heuristic target is illegal in its recorded observation")
    # One-option requests contain no policy choice and must not dilute the imitation objective.
    choices = legality.sum(-1) > 1
    count = int(choices.sum())
    voluntary_switch = (labels >= 8) & (labels < 13) & legality[:, :8].any(-1)
    metrics = {}
    if count:
        chosen = log_probs.gather(1, labels[:, None]).squeeze(1)[choices]
        if not torch.isfinite(chosen).all():
            raise FloatingPointError("nonfinite teacher action probability")
        weights = torch.where(voluntary_switch[choices], voluntary_switch_weight, 1.0)
        loss = -(chosen * weights).sum() / weights.sum()
        # Keep unweighted NLL comparable between recipes with different class weights.
        nll = -chosen.mean()
        accuracy = (log_probs[choices].argmax(-1) == labels[choices]).float().mean()
    else:
        loss = torch.where(legality, log_probs, 0.0).sum() * 0.0
        nll = loss.detach()
        accuracy = loss.detach()
    for name, selected in {
        "voluntary_switch": voluntary_switch,
        "forced_switch": (labels >= 8) & (labels < 13) & ~legality[:, :8].any(-1),
        "move": labels < 8,
    }.items():
        selected = selected & choices
        metrics[f"heuristic_{name}_rows"] = int(selected.sum())
        if bool(selected.any()):
            metrics[f"heuristic_{name}_nll"] = -log_probs.gather(1, labels[:, None])[selected].mean().detach()
            metrics[f"heuristic_{name}_top1"] = (log_probs[selected].argmax(-1) == labels[selected]).float().mean()
    teacher_move = choices & (labels < 8) & legality[:, :8].any(-1)
    if bool(teacher_move.any()):
        prediction = log_probs[teacher_move].argmax(-1)
        metrics["heuristic_false_voluntary_switch_rate"] = ((prediction >= 8) & (prediction < 13)).float().mean()
    return loss, {**metrics, "heuristic_loss": loss.detach(), "heuristic_nll": nll.detach(),
                  "heuristic_top1": accuracy.detach(), "heuristic_choice_rows": count,
                  "heuristic_voluntary_switch_weight": voluntary_switch_weight,
                  "heuristic_target_tera_rate": ((labels >= 4) & (labels < 8)).float().mean(),
                  "heuristic_target_switch_rate": ((labels >= 8) & (labels < 13)).float().mean()}
