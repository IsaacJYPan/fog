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


class JointKakuna(nn.Module):
    """Native actor/critic plus auxiliary heads; predictions never feed ordinary policy."""

    def __init__(self, native: nn.Module, *, belief_dim=128, tables=None):
        super().__init__()
        if tables is None:
            from fog.constants.tables import TABLES

            tables = TABLES
        self.native = native
        self.context_projection = nn.Sequential(
            nn.LayerNorm(native.traj_encoder.emb_dim),
            nn.Linear(native.traj_encoder.emb_dim, 256),
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
            nn.LayerNorm(native.traj_encoder.emb_dim),
            nn.Linear(native.traj_encoder.emb_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

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
    ):
        context = self.context_projection(latent)
        # Unverified fields cannot become teacher prefix facts either.
        safe_truth = torch.where(verified, truth, -1)
        reveals = {"tokens": reveal_tokens, "masks": masks or {}}
        marginal = self.belief(context, reveals)
        hidden = verified & ~self.belief._revealed_mask(reveal_tokens).reshape_as(verified)
        marginal_loss = normalized_belief_loss(marginal, safe_truth, hidden)
        chain = self.belief(context, reveals, teacher_values=safe_truth, generator=generator)
        # Exchangeable move fields have their validity permuted with the truth. A partially
        # verified moveset uses -1 entries, so loss_targets itself supplies the safe validity.
        chain_valid = verified.clone()
        if chain.loss_targets is not None:
            permuted = chain.loss_targets["move"]
            chain_valid[:, :, 3:7] = permuted >= 0
        chain_loss = normalized_belief_loss(chain, safe_truth, chain_valid)
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
