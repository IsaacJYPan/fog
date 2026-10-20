"""Cap outcome supervision on shared weights without weakening head training."""

from __future__ import annotations

import math

import torch


class OutcomeGradientCap:
    """Accumulate effective-batch task gradients before applying a norm cap.

    The ordinary backward still trains every head and accumulates retention
    gradients. Removing only excess outcome gradients from shared parameters
    leaves those other gradients intact. The cap concerns actual shared-weight
    gradients, rather than their differently scaled latent-space counterparts.
    """

    def __init__(self, parameters, maximum_ratio):
        if not math.isfinite(maximum_ratio) or not 0 <= maximum_ratio <= 1:
            raise ValueError("Outcome/shared-belief gradient ratio must be in [0,1]")
        self.parameters = list(parameters)
        if not self.parameters or any(not p.requires_grad for p in self.parameters):
            raise ValueError("The outcome cap needs trainable shared parameters")
        self.maximum_ratio = float(maximum_ratio)
        self.belief = [torch.zeros_like(p) for p in self.parameters]
        self.outcome = [torch.zeros_like(p) for p in self.parameters]

    def accumulate(self, belief_loss, weighted_outcome_loss):
        for loss, accumulator in (
            (belief_loss, self.belief),
            (weighted_outcome_loss, self.outcome),
        ):
            gradients = torch.autograd.grad(
                loss, self.parameters, retain_graph=True, allow_unused=True
            )
            for total, gradient in zip(accumulator, gradients):
                if gradient is not None:
                    total.add_(gradient.detach())

    @torch.no_grad()
    def apply(self):
        belief_norm = torch.linalg.vector_norm(torch.stack([g.norm() for g in self.belief]))
        outcome_norm = torch.linalg.vector_norm(torch.stack([g.norm() for g in self.outcome]))
        if not torch.isfinite(belief_norm) or not torch.isfinite(outcome_norm):
            raise FloatingPointError("Nonfinite shared task gradient")
        scale = min(1.0, self.maximum_ratio * float(belief_norm) / max(float(outcome_norm), 1e-30))
        retention_norm = torch.linalg.vector_norm(
            torch.stack(
                [
                    (
                        (p.grad if p.grad is not None else torch.zeros_like(p)) - belief - outcome
                    ).norm()
                    for p, belief, outcome in zip(self.parameters, self.belief, self.outcome)
                ]
            )
        )
        for parameter, gradient in zip(self.parameters, self.outcome):
            if parameter.grad is not None:
                parameter.grad.add_(gradient, alpha=scale - 1.0)
        return {
            "native_belief_gradient_norm": float(belief_norm),
            "native_outcome_gradient_norm_before_cap": float(outcome_norm),
            "native_outcome_gradient_norm_after_cap": scale * float(outcome_norm),
            "native_outcome_gradient_scale": scale,
            "native_outcome_maximum_ratio": self.maximum_ratio,
            "native_retention_gradient_norm": float(retention_norm),
        }
