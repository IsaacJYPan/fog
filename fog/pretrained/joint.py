"""FOG fact-set belief and outcome supervision on Kakuna's shared latent."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from fog.models.belief import BeliefModel


def normalized_belief_loss(output, truth, verified) -> torch.Tensor:
    """Mean hidden-field NLL / log(vocabulary), with strict provenance masking.

    Unlike a sum over families, reduction does not grow with the number of fields.
    Impossible verified targets are a data/interface failure, never silently dropped.
    """
    targets = BeliefModel._split_flat(truth)
    masks = BeliefModel._split_flat(verified)
    if output.loss_targets is not None:
        targets = output.loss_targets
    total = next(iter(output.logits.values())).reshape(-1)[:0].sum()
    count = 0
    for family, logits in output.logits.items():
        target = targets[family]
        valid = masks[family] & (target >= 0)
        if output.loss_valid is not None:
            valid = valid & output.loss_valid[family]
        if not bool(valid.any()):
            continue
        scored = logits[valid]
        selected = target[valid]
        if bool((selected >= scored.shape[-1]).any()):
            raise ValueError("Verified belief label outside vocabulary")
        if not bool(torch.isfinite(scored.gather(-1, selected[:, None])).all()):
            raise ValueError("Verified belief target excluded by decoder constraints")
        total = total + F.cross_entropy(scored, selected, reduction="sum") / math.log(
            scored.shape[-1]
        )
        count += int(valid.sum())
    return total / max(count, 1)


class BeliefHeads(nn.Module):
    """The same auxiliary heads, independently trainable on verified frozen latents."""

    def __init__(self, *, native_dim=900, belief_dim=128, tables=None):
        super().__init__()
        if tables is None:
            from fog.constants.tables import TABLES

            tables = TABLES
        self.context_projection = nn.Sequential(
            nn.LayerNorm(native_dim),
            nn.Linear(native_dim, 256),
            nn.GELU(),
        )
        self.belief = BeliefModel(
            state_dim=256,
            d=belief_dim,
            n_species=tables.n_species,
            n_items=tables.n_items,
            n_abilities=tables.n_abilities,
            n_moves=tables.n_moves,
        )
        self.outcome = nn.Sequential(
            nn.LayerNorm(native_dim),
            nn.Linear(native_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def auxiliary_losses(
        self,
        latent,
        reveal_tokens,
        truth,
        verified,
        outcomes,
        outcome_valid,
        *,
        masks=None,
        generator=None,
        prior=None,
        chain_queries=1,
        selected_chain=False,
    ):
        context = self.context_projection(latent)
        # Unverified fields cannot become teacher prefix facts either.
        safe_truth = torch.where(verified, truth, -1)
        reveals = {"tokens": reveal_tokens, "masks": masks or {}}
        if prior is not None:
            reveals["prior"] = prior
        marginal = self.belief(context, reveals)
        hidden = verified & ~self.belief._revealed_mask(reveal_tokens).reshape_as(verified)
        marginal_loss = normalized_belief_loss(marginal, safe_truth, hidden)
        if not isinstance(chain_queries, int) or chain_queries < 1:
            raise ValueError("Positive integer AR query count required")
        chain_context, chain_reveals, chain_truth = context, reveals, safe_truth
        if chain_queries > 1:
            chain_context = context.repeat_interleave(chain_queries, dim=0)
            chain_truth = safe_truth.repeat_interleave(chain_queries, dim=0)
            chain_reveals = {
                "tokens": reveal_tokens.repeat_interleave(chain_queries, dim=0),
                "masks": {key: value.repeat_interleave(chain_queries, dim=0) for key, value in reveals["masks"].items()},
            }
            if prior is not None:
                chain_reveals["prior"] = prior.repeat_interleave(chain_queries)
        chain = self.belief(chain_context, chain_reveals, teacher_values=chain_truth,
                            generator=generator, selected_only=selected_chain)
        # Exchangeable move fields have their validity permuted with the truth. A partially
        # verified moveset uses -1 entries, so loss_targets itself supplies the safe validity.
        chain_valid = verified.repeat_interleave(chain_queries, dim=0)
        if chain.loss_targets is not None:
            permuted = chain.loss_targets["move"]
            chain_valid[:, :, 3:7] = permuted >= 0
        chain_loss = normalized_belief_loss(chain, chain_truth, chain_valid)
        predictions = self.outcome(latent).squeeze(-1)
        valid = outcome_valid.bool() & torch.isfinite(outcomes)
        outcome_loss = (
            F.mse_loss(predictions[valid].tanh(), outcomes[valid])
            if bool(valid.any())
            else predictions[:0].sum()
        )
        return {"marginal": marginal_loss, "autoregressive": chain_loss, "outcome": outcome_loss}

    def sample_team(self, latent, reveals, **kwargs):
        return self.belief.sample_team(self.context_projection(latent), reveals, **kwargs)


class JointKakuna(BeliefHeads):
    """Native actor/critic plus auxiliary heads; predictions never feed ordinary policy."""

    def __init__(self, native: nn.Module, *, belief_dim=128, tables=None):
        super().__init__(native_dim=native.traj_encoder.emb_dim, belief_dim=belief_dim, tables=tables)
        self.native = native

    def shared_parameters(self):
        return list(self.native.tstep_encoder.parameters()) + list(
            self.native.traj_encoder.parameters()
        )

    def trainable_parameters(self):
        return (
            list(self.native.trainable_params)
            + list(self.context_projection.parameters())
            + list(self.belief.parameters())
            + list(self.outcome.parameters())
        )

    def primary_with_latent(self, batch):
        latent = []
        handle = self.native.traj_encoder.register_forward_hook(
            lambda _m, _i, output: latent.append(output[0])
        )
        try:
            loss = self.native(batch, log_step=False)
        finally:
            handle.remove()
        if len(latent) != 1:
            raise RuntimeError("Expected one native shared-latent forward")
        return loss, latent[0]
