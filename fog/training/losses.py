"""Shared loss terms for BC and self-play (spec §3.2, §5, §6).

The belief / self-belief losses are sums of masked per-attribute cross-entropies. In replay training
the labels are CENSORED (only attributes eventually revealed contribute — the rest are masked out of
the loss); in self-play the simulator provides full ground truth every game.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_cross_entropy(
    logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
    stats: dict | None = None,
) -> torch.Tensor:
    """CE over entries where `valid` is True; returns 0 if nothing is valid.

    logits [..., C], target [...], valid [...] bool — leading dims are flattened, so this accepts both
    the flat BC action case ([N, C]) and the belief-attribute shapes ([B, 6, C], [B, 6, 4, C]).
    Targets of -1 are treated as invalid regardless of `valid`.

    `stats`, when passed, accumulates {"valid": n_scored, "dropped_nonfinite": n_dropped} across
    calls — the dropped-entry-rate counter. A persistently nonzero dropped rate is a CONVENTION
    BUG signal, not noise: exactly this guard silently swallowed every revealed move whose paste
    position differed from its clamp position until the 2026-07-12 alignment fix
    (reveals.align_truth_to_views). Post-alignment the only legitimate drops are a truth label
    outside its legality mask (usage-derived spread priors, learnset table gaps) — genuinely rare.
    """
    c = logits.shape[-1]
    logits = logits.reshape(-1, c)
    target = target.reshape(-1)
    valid = valid.reshape(-1) & (target >= 0) & (target < c)
    # Guard: a target sitting on a legality-masked (-inf) class would make CE = +inf and poison
    # the whole update. Drop those entries from the loss rather than diverging — a hard mask
    # genuinely can't represent them (and see the stats note above: watch the rate).
    if valid.any():
        tgt_logit = logits.gather(1, target.clamp(min=0).unsqueeze(1)).squeeze(1)
        finite = torch.isfinite(tgt_logit)
        if stats is not None:
            stats["dropped_nonfinite"] = (
                stats.get("dropped_nonfinite", 0) + int((valid & ~finite).sum())
            )
        valid = valid & finite
    if stats is not None:
        stats["valid"] = stats.get("valid", 0) + int(valid.sum())
    if valid.sum() == 0:
        return logits.new_zeros(())
    return F.cross_entropy(logits[valid], target[valid])


def belief_loss(
    attr_logits: dict, attr_targets: dict, attr_valid: dict, stats: dict | None = None
) -> torch.Tensor:
    """Sum of masked CEs across all belief attributes (§3.2). `attr_valid` carries the censoring;
    `stats` threads through to masked_cross_entropy's dropped-entry counter."""
    total = None
    for name, logits in attr_logits.items():
        term = masked_cross_entropy(logits, attr_targets[name], attr_valid[name], stats=stats)
        total = term if total is None else total + term
    return total if total is not None else torch.zeros(())


def value_loss(pred_value: torch.Tensor, outcome: torch.Tensor) -> torch.Tensor:
    """MSE against final outcome in [-1, 1]. Discount γ=1 within a game (episodic)."""
    return F.mse_loss(pred_value, outcome)
