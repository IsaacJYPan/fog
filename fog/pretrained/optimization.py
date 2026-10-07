"""Separate accumulated primary/auxiliary gradients, capped before global clipping."""

from __future__ import annotations

import math
from typing import Iterable

import torch


class AuxiliaryGradientCap:
    """Cap the *accumulated* auxiliary backbone gradient at a primary-norm fraction.

    Capping each microbatch separately would not bound the resulting update when
    primary gradients cancel. Head gradients are left unchanged. A zero primary
    backbone norm suppresses the auxiliary backbone update for that optimizer step.
    """

    def __init__(self, shared: Iterable[torch.nn.Parameter], fraction=0.25):
        self.shared = list(shared)
        self.fraction = fraction
        self.primary = {}
        self.auxiliary = {}

    @staticmethod
    def _accumulate(storage, parameters, gradients):
        for parameter, gradient in zip(parameters, gradients):
            if gradient is not None:
                key = id(parameter)
                value = gradient.detach()
                if key in storage:
                    storage[key].add_(value)
                else:
                    storage[key] = value.clone()

    def backward(self, primary, auxiliary, *, scale=1.0):
        active = [p for p in self.shared if p.requires_grad]
        for parameter in active:
            parameter.grad = None
        (primary * scale).backward(retain_graph=True)
        self._accumulate(self.primary, active, [p.grad for p in active])
        for parameter in active:
            parameter.grad = None
        (auxiliary * scale).backward()
        self._accumulate(self.auxiliary, active, [p.grad for p in active])
        for parameter in active:
            parameter.grad = None

    def finish(self) -> dict:
        primary_norm = math.sqrt(
            sum(float(g.double().square().sum()) for g in self.primary.values())
        )
        auxiliary_norm = math.sqrt(
            sum(float(g.double().square().sum()) for g in self.auxiliary.values())
        )
        dot = sum(
            float((gradient * self.auxiliary[key]).sum(dtype=torch.float64))
            for key, gradient in self.primary.items()
            if key in self.auxiliary
        )
        cosine = dot / (primary_norm * auxiliary_norm) if primary_norm and auxiliary_norm else None
        ratio = min(1.0, self.fraction * primary_norm / auxiliary_norm) if auxiliary_norm else 1.0
        for parameter in self.shared:
            primary = self.primary.get(id(parameter))
            auxiliary = self.auxiliary.get(id(parameter))
            if primary is None and auxiliary is None:
                continue
            parameter.grad = torch.zeros_like(parameter) if primary is None else primary.clone()
            if auxiliary is not None:
                parameter.grad.add_(auxiliary, alpha=ratio)
        self.primary.clear()
        self.auxiliary.clear()
        return {
            "primary_backbone_norm": primary_norm,
            "auxiliary_backbone_norm": auxiliary_norm,
            "auxiliary_scale": ratio,
            "capped_auxiliary_backbone_norm": ratio * auxiliary_norm,
            "auxiliary_to_primary_norm": auxiliary_norm / primary_norm if primary_norm else None,
            "primary_auxiliary_cosine": cosine,
            "auxiliary_cap_active": ratio < 1.0,
        }
