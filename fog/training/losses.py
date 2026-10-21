"""Shared loss terms for BC and self-play.

The belief / self-belief losses are sums of masked per-attribute cross-entropies. In replay training
the labels are CENSORED (only attributes eventually revealed contribute — the rest are masked out of
the loss); in self-play the simulator provides full ground truth every game.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _masked_cross_entropy_sum(
    logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
    stats: dict | None = None,
) -> tuple[torch.Tensor, int]:
    """Return the summed finite CE and its entry count for one vocabulary family."""
    c = logits.shape[-1]
    logits = logits.reshape(-1, c)
    target = target.reshape(-1)
    valid = valid.reshape(-1) & (target >= 0) & (target < c)
    if valid.any():
        safe_target = target.clamp(min=0, max=c - 1)
        tgt_logit = logits.gather(1, safe_target.unsqueeze(1)).squeeze(1)
        finite = torch.isfinite(tgt_logit)
        if stats is not None:
            dropped = valid & ~finite
            stats["dropped_nonfinite"] = (
                stats.get("dropped_nonfinite", 0) + int(dropped.sum())
            )
            if bool(dropped.any()):
                indices = torch.nonzero(dropped, as_tuple=False).flatten()[:8]
                stats.setdefault("dropped_examples", []).extend(
                    {
                        "flat_index": int(index),
                        "target": int(target[index]),
                    }
                    for index in indices
                )
        valid = valid & finite
    count = int(valid.sum())
    if stats is not None:
        stats["valid"] = stats.get("valid", 0) + count
    if count == 0:
        # Keep an all-censored batch connected to the model graph. Replay shuffling can rarely
        # produce a batch with no eventually revealed belief fields; a detached scalar zero makes
        # the otherwise valid no-op update crash at backward(). Summing an empty view is finite
        # even when masked logits contain -inf and gives every upstream parameter a zero gradient.
        return logits.reshape(-1)[:0].sum(), 0
    return F.cross_entropy(logits[valid], target[valid], reduction="sum"), count


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
    outside its legality mask (learnset table gaps or an invalid EV total) — genuinely rare.
    """
    total, count = _masked_cross_entropy_sum(logits, target, valid, stats)
    return total / count if count else total


def belief_loss(
    attr_logits: dict, attr_targets: dict, attr_valid: dict, stats: dict | None = None
) -> torch.Tensor:
    """Sum of masked CEs across all belief attributes. `attr_valid` carries the censoring;
    `stats` threads through to masked_cross_entropy's dropped-entry counter."""
    total = None
    for name, logits in attr_logits.items():
        family_stats: dict | None = {} if stats is not None else None
        term = masked_cross_entropy(
            logits, attr_targets[name], attr_valid[name], stats=family_stats
        )
        if stats is not None and family_stats is not None:
            stats["valid"] = stats.get("valid", 0) + family_stats.get("valid", 0)
            stats["dropped_nonfinite"] = (
                stats.get("dropped_nonfinite", 0)
                + family_stats.get("dropped_nonfinite", 0)
            )
            stats.setdefault("by_family", {})[name] = dict(family_stats)
        total = term if total is None else total + term
    return total if total is not None else torch.zeros(())


def belief_output_loss(
    output, attr_targets: dict, attr_valid: dict, stats: dict | None = None
) -> torch.Tensor:
    """Loss for either a marginal ``BeliefOutput`` or a set-AR route-step output.

    Marginal outputs use the caller's ordinary labels and censoring masks.  A set-autoregressive
    training output additionally supplies:

    * ``loss_valid`` — the single next field selected by the sampled generation route; and
    * ``loss_targets`` — the same truth grid after exchangeable hidden moves were randomly
      permuted.

    Intersecting the masks here is load-bearing.  Applying the old all-fields mask to a one-step
    conditional would silently train values for facts already placed in the context and would turn
    the intended matched autoregressive objective back into an inconsistent pseudo-likelihood.
    """
    if output.loss_valid is None:
        return belief_loss(output.logits, attr_targets, attr_valid, stats=stats)

    targets = output.loss_targets if output.loss_targets is not None else attr_targets
    total = None
    count = 0
    for name, logits in output.logits.items():
        term, n = _masked_cross_entropy_sum(
            logits, targets[name], attr_valid[name] & output.loss_valid[name], stats=stats
        )
        total = term if total is None else total + term
        count += n
    if total is None:
        return torch.zeros(())
    # A random route and uniform cut select every hidden field with probability 1/L. Averaging all
    # selected rows therefore estimates per-hidden-field route NLL without changing that
    # distribution through family-dependent reduction weights.
    return total / count if count else total


def value_loss(pred_value: torch.Tensor, outcome: torch.Tensor) -> torch.Tensor:
    """MSE against final outcome in [-1, 1]. Discount γ=1 within a game (episodic)."""
    return F.mse_loss(pred_value, outcome)
