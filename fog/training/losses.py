"""Shared loss terms for BC and self-play (spec §3.2, §5, §6).

The belief / self-belief losses are sums of masked per-attribute cross-entropies. In replay training
the labels are CENSORED (only attributes eventually revealed contribute — the rest are masked out of
the loss); in self-play the simulator provides full ground truth every game.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_cross_entropy(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """CE over entries where `valid` is True; returns 0 if nothing is valid.

    logits [..., C], target [...], valid [...] bool — leading dims are flattened, so this accepts both
    the flat BC action case ([N, C]) and the belief-attribute shapes ([B, 6, C], [B, 6, 4, C]).
    Targets of -1 are treated as invalid regardless of `valid`.
    """
    c = logits.shape[-1]
    logits = logits.reshape(-1, c)
    target = target.reshape(-1)
    valid = valid.reshape(-1) & (target >= 0) & (target < c)
    # Guard: a target sitting on a legality-masked (-inf) class would make CE = +inf and poison the
    # whole update. This only happens when the belief legality mask is not a superset of the truth
    # (usage-derived item/spread priors, or a learnset table gap). Drop those entries from the loss
    # rather than diverging — they are rare and a hard mask genuinely can't represent them.
    if valid.any():
        tgt_logit = logits.gather(1, target.clamp(min=0).unsqueeze(1)).squeeze(1)
        valid = valid & torch.isfinite(tgt_logit)
    if valid.sum() == 0:
        return logits.new_zeros(())
    return F.cross_entropy(logits[valid], target[valid])


def belief_loss(attr_logits: dict, attr_targets: dict, attr_valid: dict) -> torch.Tensor:
    """Sum of masked CEs across all belief attributes (§3.2). `attr_valid` carries the censoring."""
    total = None
    for name, logits in attr_logits.items():
        term = masked_cross_entropy(logits, attr_targets[name], attr_valid[name])
        total = term if total is None else total + term
    return total if total is not None else torch.zeros(())


def value_loss(pred_value: torch.Tensor, outcome: torch.Tensor) -> torch.Tensor:
    """MSE against final outcome in [-1, 1]. Discount γ=1 within a game (episodic)."""
    return F.mse_loss(pred_value, outcome)
