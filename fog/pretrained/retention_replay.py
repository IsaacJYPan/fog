"""Training-only retention replay and repair toward the protected initialization."""

from __future__ import annotations

import hashlib
import json
import math
import random

import torch
from torch.nn import functional as F

from fog.pretrained.latent_cache import tensor_digest
from fog.pretrained.retention import RetentionLimits


def budget_retention_terms(
    probabilities, reference_probabilities, values, reference_values, legal, *, limits, fraction
):
    """A separable dead zone; hard acceptance still uses the original limits.

    Penalizing every tiny KL change made retention dominate useful auxiliary
    gradients. The margins apply per context/action, so effective-batch and
    microbatch reductions agree. They are an objective, not a global guarantee.
    Inputs are validated by retention_terms at the calling boundary.
    """
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("Retention margin fraction must be in (0,1)")
    reference = reference_probabilities.detach()
    kl = (
        reference * (reference.clamp_min(1e-12).log() - probabilities.clamp_min(1e-12).log())
    ).sum(-1)
    relative = (values - reference_values.detach()).abs() / (1 + reference_values.detach().abs())
    policy = F.relu(kl - fraction * limits.policy_kl_mean).mean()
    # Average within a context first, so microbatches with different numbers of
    # legal actions keep the same effective-context weighting.
    value = F.relu(relative - fraction * limits.value_relative_mean).square()
    per_context = (value * legal).sum(-1) / legal.sum(-1)
    return policy, per_context.mean()


def replay_specifications(data, *, count, seed, extra=()):
    """A deterministic, deduplicated training-only bank with optional hard cases."""
    if type(count) is not int or count < 1:
        raise ValueError("Require a positive retention-bank size")
    training = {row["path"]: row["positions"] for row in data.rows["train"]}
    specs, seen = [], set()
    for name, index in extra:
        if name not in training or type(index) is not int or not 0 <= index < training[name]:
            raise ValueError("Retention replay must use verified training contexts")
        pair = (name, index)
        if pair not in seen:
            specs.append(pair)
            seen.add(pair)
    if len(specs) > count or sum(training.values()) < count:
        raise ValueError("Retention-bank budget is inconsistent with available contexts")
    rng = random.Random(seed)
    while len(specs) < count:
        for pair in data.draw(rng, min(count - len(specs), 64)):
            pair = tuple(pair)
            if pair not in seen:
                specs.append(pair)
                seen.add(pair)
    signature = hashlib.sha256(json.dumps(specs, separators=(",", ":")).encode()).hexdigest()
    return specs, signature


class OriginalReferenceRepair:
    """Restore sampled bounds by contracting toward immutable original weights.

    Rejecting a new step cannot fix an already unsafe pre-step model. This
    repair tests the current weights, then contracts their total displacement
    from the protected initialization. Zero is tested explicitly and copied
    exactly. Only a successful checked repair resets native Adam moments; head
    weights, head optimization and query/RNG streams are untouched.
    """

    def __init__(self, parameters, *, limits=None, max_backtracks=8):
        self.parameters = list(parameters)
        if not self.parameters or type(max_backtracks) is not int or not 1 <= max_backtracks <= 16:
            raise ValueError("Require native parameters and a bounded repair search")
        self.original = [parameter.detach().clone() for parameter in self.parameters]
        self.signature = tensor_digest({str(i): value for i, value in enumerate(self.original)})
        self.limits = limits or RetentionLimits()
        self.max_backtracks = max_backtracks
        self.repairs = 0

    def state_dict(self):
        return {
            "reference_sha256": self.signature,
            "repairs": self.repairs,
            "max_backtracks": self.max_backtracks,
        }

    def load_state_dict(self, state):
        if (
            state.get("reference_sha256") != self.signature
            or state.get("max_backtracks") != self.max_backtracks
            or type(state.get("repairs")) is not int
            or state["repairs"] < 0
        ):
            raise ValueError("Retention repair reference or progress changed")
        self.repairs = state["repairs"]

    def accepts(self, measurements):
        return bool(measurements) and all(
            self.limits.accepts(value) for value in measurements.values()
        )

    @torch.no_grad()
    def ensure(self, assess, optimizer):
        before = assess()
        if self.accepts(before):
            return {
                "repaired": False,
                "original_reference_scale": 1.0,
                "attempts": 0,
                "before": before,
                "after": before,
            }
        candidate = [p.detach().clone() for p in self.parameters]
        try:
            for attempt, scale in enumerate(
                [0.5**i for i in range(1, self.max_backtracks + 1)] + [0.0], 1
            ):
                for parameter, original, value in zip(self.parameters, self.original, candidate):
                    parameter.copy_(
                        original if scale == 0 else original + scale * (value - original)
                    )
                measured = assess()
                if self.accepts(measured):
                    optimizer.state.clear()
                    for parameter in self.parameters:
                        parameter.grad = None
                    self.repairs += 1
                    return {
                        "repaired": True,
                        "original_reference_scale": scale,
                        "attempts": attempt,
                        "before": before,
                        "after": measured,
                        "native_Adam_reset": True,
                    }
        except BaseException:
            for parameter, value in zip(self.parameters, candidate):
                parameter.copy_(value)
            raise
        for parameter, value in zip(self.parameters, candidate):
            parameter.copy_(value)
        raise FloatingPointError("Protected original cannot satisfy the sampled retention contract")
