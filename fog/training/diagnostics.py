"""Read-only optimization measurements and deterministic rollout schedules."""
from __future__ import annotations

import math
import random

import torch


def direct_rng_streams(setup):
    """Optionally separate game schedules from opponent-choice RNG consumption."""
    if not setup.env_cfg.get("independent_opponent_rng", False):
        return setup.rng, setup.rng
    games = random.Random(setup.seed + 100_000)
    if setup.resume_backend_rng_state is not None:
        games.setstate(setup.resume_backend_rng_state)
    setup.backend_rng = games
    return games, setup.rng


def tera_bonus_at(ppo: dict, decisions: int, total: int) -> float:
    initial = float(ppo.get("rollout_tera_logit_bonus_initial", ppo.get("rollout_tera_logit_bonus", 0)))
    final = float(ppo.get("rollout_tera_logit_bonus_final", initial))
    horizon = int(ppo.get("rollout_tera_logit_bonus_decay_decisions", total))
    if not all(math.isfinite(x) and x >= 0 for x in (initial, final)) or horizon <= 0:
        raise ValueError("Tera bonus endpoints must be finite/non-negative and decay decisions positive")
    return initial + (final - initial) * min(1., max(0., decisions / horizon))


def gradient_comparison(policy_loss, replay_nll, parameters, coefficient: float) -> dict[str, float]:
    """Compare unscaled policy/replay gradients on the same trainable coordinates.

    autograd.grad does not accumulate .grad or step the optimizer. No value/belief/entropy terms
    enter the reference policy-gradient norm; absent gradients are zero in the shared vector.
    """
    params = [p for p in parameters if p.requires_grad]
    policy = torch.autograd.grad(policy_loss, params, allow_unused=True, retain_graph=True)
    replay = torch.autograd.grad(replay_nll, params, allow_unused=True, retain_graph=True)
    p2 = sum(float(g.detach().double().square().sum()) for g in policy if g is not None)
    r2 = sum(float(g.detach().double().square().sum()) for g in replay if g is not None)
    dot = sum(float((p.detach().double() * r.detach().double()).sum())
              for p, r in zip(policy, replay, strict=True) if p is not None and r is not None)
    pn, rn = math.sqrt(p2), math.sqrt(r2)
    return {"policy_gradient_norm": pn, "replay_unweighted_gradient_norm": rn,
            "replay_weighted_gradient_ratio": coefficient * rn / max(pn, 1e-30),
            "replay_policy_gradient_cosine": dot / max(pn * rn, 1e-30)}
