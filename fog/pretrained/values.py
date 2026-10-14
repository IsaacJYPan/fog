"""Inference-only native critic readout in its trained shaped-return units."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass
class NativeValue:
    value: torch.Tensor
    action_values: torch.Tensor
    critic_dispersion: torch.Tensor
    gamma: float
    reward_multiplier: float


@torch.no_grad()
def policy_value(model, latent: torch.Tensor, probabilities: torch.Tensor) -> NativeValue:
    """Exact legal-policy expectation over 13 actions at the deployed last gamma.

    MultiTaskAgent's distributional critics accept one-hot actions with a
    separate gamma axis. bin_dist_to_raw_vals yields the scaled training return;
    divide by the loss-only reward multiplier to match environment rewards.
    Do not apply PopArt again or clamp these values to outcome units.
    Ensemble dispersion is descriptive, not a calibrated confidence interval.
    """
    if latent.ndim != 3 or probabilities.shape != (*latent.shape[:2], 13):
        raise ValueError("Native value needs latent [B,T,D] and probabilities [B,T,13]")
    if (not torch.isfinite(probabilities).all() or bool((probabilities < 0).any())
            or not torch.allclose(probabilities.sum(-1), torch.ones_like(probabilities[..., 0]), atol=1e-5)):
        raise ValueError("Native value needs a finite normalized legal policy")
    gammas = model.gammas
    if gammas.ndim != 1 or not len(gammas) or not torch.isfinite(gammas).all():
        raise ValueError("Native critic has an invalid gamma axis")
    gamma = float(gammas[-1])
    if not 0 <= gamma <= 1:
        raise ValueError("Native deployed discount must lie in [0,1]")
    multiplier = float(model.reward_multiplier)
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("Native reward multiplier must be finite and positive")
    batch, length = latent.shape[:2]
    actions = torch.eye(13, device=latent.device, dtype=latent.dtype)[:, None, None, None, :]
    actions = actions.expand(13, batch, length, len(gammas), 13)
    distribution = model.critics(latent, actions)
    raw = model.critics.bin_dist_to_raw_vals(distribution)
    if (raw.ndim != 6 or raw.shape[:3] != (13, batch, length)
            or raw.shape[3] < 1 or raw.shape[4:] != (len(gammas), 1)
            or not torch.isfinite(raw).all()):
        raise ValueError("Native critic returned invalid [A,B,T,C,G,1] raw values")
    deployed = raw[..., -1, 0]
    q = deployed.mean(3).permute(1, 2, 0) / multiplier
    dispersion = deployed.std(3, unbiased=False).permute(1, 2, 0) / multiplier
    return NativeValue((q * probabilities).sum(-1), q, dispersion, gamma, multiplier)
