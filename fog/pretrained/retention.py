"""Small shared-representation updates with explicit native behavior checks."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch


def upper_modules(native):
    """This cache boundary is specific to the pinned ten-layer Kakuna model."""
    transformer = native.traj_encoder.tformer
    if len(transformer.layers) != 10:
        raise ValueError("The cached adaptation boundary requires ten native layers")
    block = transformer.layers[-1]
    attention = block.attention_layer.self_attention
    if not attention.causal or tuple(attention.window_size) != (96, 0):
        raise ValueError("Cached upper inference requires the native causal 96-step window")
    return block, transformer.norm


def configure_upper_training(native, *, adapt):
    # eval() is intentional: SigmaReparam updates buffers in train(), even for
    # parameters with requires_grad=False. Gradients still work in eval mode.
    native.eval()
    for parameter in native.parameters():
        parameter.requires_grad_(False)
    block, norm = upper_modules(native)
    parameters = list(block.parameters()) + list(norm.parameters())
    if adapt:
        for parameter in parameters:
            parameter.requires_grad_(True)
    return parameters if adapt else []


def upper_forward(native, features, positions, lengths):
    block, norm = upper_modules(native)
    if features.ndim != 3 or positions.shape != features.shape[:2]:
        raise ValueError("Upper features and absolute positions have incompatible shapes")
    if lengths.shape != (len(features),) or bool((lengths < 1).any()) or bool((lengths > features.shape[1]).any()):
        raise ValueError("Invalid right-padded upper sequence lengths")
    positional = positions if native.traj_encoder.tformer.use_rope else None
    sequence = norm(block(features, pos_idxs=positional))
    return sequence[torch.arange(len(features), device=features.device), lengths - 1]


def legal_policy(native, latent, actor_observations):
    probabilities = native.actor(latent, straight_from_obs=actor_observations).probs[..., -1, :]
    legal = ~actor_observations["illegal_actions"].bool()
    if probabilities.shape != legal.shape or bool((~legal.any(-1)).any()):
        raise ValueError("Native behavior check needs at least one authoritative legal action")
    probabilities = torch.where(legal, probabilities, 0.0)
    total = probabilities.sum(-1, keepdim=True)
    if not torch.isfinite(probabilities).all() or bool((total <= 0).any()):
        raise FloatingPointError("Nonfinite or empty native action distribution")
    return probabilities / total


def action_values(native, latent):
    """Differentiable counterpart of the inference-only native critic readout."""
    if latent.ndim != 3:
        raise ValueError("Native critic needs [batch,time,latent] inputs")
    batch, length = latent.shape[:2]
    actions = torch.eye(13, device=latent.device, dtype=latent.dtype)[:, None, None, None]
    actions = actions.expand(13, batch, length, len(native.gammas), 13)
    raw = native.critics.bin_dist_to_raw_vals(native.critics(latent, actions))
    if raw.ndim != 6 or raw.shape[:3] != (13, batch, length) or raw.shape[4:] != (len(native.gammas), 1):
        raise ValueError("Unexpected native critic layout")
    return raw[..., -1, 0].mean(3).permute(1, 2, 0) / float(native.reward_multiplier)


def retention_terms(probabilities, reference_probabilities, values, reference_values, legal):
    if not (probabilities.shape == reference_probabilities.shape == values.shape == reference_values.shape == legal.shape):
        raise ValueError("Retention tensors must share the native action layout")
    if not bool(legal.any(-1).all()) or any(not torch.isfinite(x).all() for x in (probabilities, reference_probabilities, values, reference_values)):
        raise FloatingPointError("Retention inputs must be finite with legal actions")
    if bool((probabilities[~legal] != 0).any()) or bool((reference_probabilities[~legal] != 0).any()):
        raise ValueError("Illegal actions cannot carry retention probability mass")
    if any(bool((x < 0).any()) or not torch.allclose(x.sum(-1), torch.ones_like(x[..., 0]), atol=1e-5) for x in (probabilities, reference_probabilities)):
        raise ValueError("Retention probabilities must be normalized")
    reference_probabilities = reference_probabilities.detach()
    reference_values = reference_values.detach()
    kl = (reference_probabilities * (reference_probabilities.clamp_min(1e-12).log() - probabilities.clamp_min(1e-12).log())).sum(-1)
    # Native Q is shaped-return prediction, not a win probability. This relative
    # scale has a one-return-unit floor and does not alter either critic's units.
    relative = (values - reference_values) / (1.0 + reference_values.abs())
    loss = relative[legal].square().mean()
    metrics = {
        "policy_kl_mean": max(0.0, float(kl.detach().mean())),
        "policy_kl_max": max(0.0, float(kl.detach().max())),
        "argmax_disagreement": float((probabilities.argmax(-1) != reference_probabilities.argmax(-1)).float().mean()),
        "value_relative_mean": float(relative.detach()[legal].abs().mean()),
        "value_relative_max": float(relative.detach()[legal].abs().max()),
    }
    return kl.mean(), loss, metrics


@dataclass(frozen=True)
class RetentionLimits:
    policy_kl_mean: float = 0.001
    policy_kl_max: float = 0.03
    argmax_disagreement: float = 0.05
    value_relative_mean: float = 0.01
    value_relative_max: float = 0.05

    def __post_init__(self):
        import math

        if any(not math.isfinite(v) or not 0 < v < 1 for v in vars(self).values()):
            raise ValueError("Retention limits must be finite fractions in (0,1)")

    def accepts(self, metrics):
        import math

        return all(key in metrics and math.isfinite(metrics[key]) and 0 <= metrics[key] <= bound for key, bound in vars(self).items())


class NativeStep:
    """Restore only the native update and its Adam state when a check rejects it.

    Head optimization is independent and is not discarded. Capture happens after
    gradients are computed, so this object does not modify random/query streams.
    """

    def __init__(self, parameters, optimizer):
        self.parameters = list(parameters)
        self.values = [p.detach().clone() for p in self.parameters]
        self.optimizer = optimizer
        self.state = copy.deepcopy(optimizer.state_dict())

    @torch.no_grad()
    def restore(self):
        for parameter, value in zip(self.parameters, self.values):
            parameter.copy_(value)
        self.optimizer.load_state_dict(self.state)
