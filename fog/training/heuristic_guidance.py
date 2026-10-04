"""Optional public-information teacher labels on learner-visited PPO states.

The teacher never selects the acting policy's action. Its independent random stream supplies
supervised targets only; accepted/rejected choices retain or discard those targets with the
ordinary rollout fragment. This is an online imitation aid, without a DAgger regret guarantee.
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


def validate_guidance(config: dict) -> None:
    settings = config.get("ppo", {})
    teacher = settings.get("heuristic_teacher")
    if teacher not in (None, "simpleheuristics", "maxdamage"):
        raise ValueError("ppo.heuristic_teacher must be simpleheuristics or maxdamage")
    initial = guidance_weight(settings, 0, int(settings.get("total_decisions", 100000)))
    final = float(settings.get("heuristic_coef_final", 0.0))
    if teacher is None and (initial > 0 or final > 0):
        raise ValueError("positive heuristic guidance requires ppo.heuristic_teacher")
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
                  legality: torch.Tensor) -> tuple[torch.Tensor, dict]:
    if labels.shape != log_probs.shape[:1] or legality.shape != log_probs.shape:
        raise ValueError("heuristic labels and legality must align with policy rows")
    if labels.dtype != torch.long or bool(((labels < 0) | (labels >= log_probs.shape[1])).any()):
        raise ValueError("heuristic labels must be valid integer action indices")
    if not bool(legality.gather(1, labels[:, None]).all()):
        raise ValueError("heuristic target is illegal in its recorded observation")
    # One-option requests contain no policy choice and must not dilute the imitation objective.
    choices = legality.sum(-1) > 1
    count = int(choices.sum())
    if count:
        chosen = log_probs.gather(1, labels[:, None]).squeeze(1)[choices]
        if not torch.isfinite(chosen).all():
            raise FloatingPointError("nonfinite teacher action probability")
        loss = -chosen.mean()
        accuracy = (log_probs[choices].argmax(-1) == labels[choices]).float().mean()
    else:
        loss = torch.where(legality, log_probs, 0.0).sum() * 0.0
        accuracy = loss.detach()
    return loss, {"heuristic_nll": loss.detach(), "heuristic_top1": accuracy.detach(),
                  "heuristic_choice_rows": count,
                  "heuristic_target_tera_rate": ((labels >= 4) & (labels < 8)).float().mean(),
                  "heuristic_target_switch_rate": ((labels >= 8) & (labels < 13)).float().mean()}
