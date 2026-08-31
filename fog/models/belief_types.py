"""Value objects returned by the belief model, separate from its network implementation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BeliefOutput:
    logits: dict[str, torch.Tensor]
    loss_valid: dict[str, torch.Tensor] | None = None
    loss_targets: dict[str, torch.Tensor] | None = None
    order: torch.Tensor | None = None
    context_count: torch.Tensor | None = None
    route_length: torch.Tensor | None = None


@dataclass
class BeliefSamples:
    teams: torch.Tensor
    log_value_prob: torch.Tensor
    log_route_prob: torch.Tensor
    order: torch.Tensor

    @property
    def log_joint_path_prob(self) -> torch.Tensor:
        return self.log_value_prob + self.log_route_prob
