"""Public-only correlated set priors for the fact-set autoregressive decoder.

Donors are evidence for a proposal, not verified labels or a calibrated posterior.
The conditional is smoothed over every currently legal value; an unfamiliar set
can therefore escape the bank. Each insertion conditions on the actual prefix,
including unordered moves. Neural logits learn a residual over these log priors.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from fog.constants.schema import SET_MOVE_COLS, SET_N_ATTRS
from fog.training.replay_candidates import ReplayCandidateBank


@dataclass
class DonorPriorBatch:
    tokens: torch.Tensor  # [batch,6,donors,15], canonical unordered donor moves
    weights: torch.Tensor  # [batch,6,donors], normalized over public-compatible sets
    smoothing: float
    residual_mode: str = "all"

    def __post_init__(self):
        if (
            self.tokens.ndim != 4
            or self.tokens.shape[1] != 6
            or self.tokens.shape[-1] != SET_N_ATTRS
        ):
            raise ValueError("Donor tokens must have shape [batch,6,donors,15]")
        if self.weights.shape != self.tokens.shape[:-1] or not 0 < self.smoothing < 1:
            raise ValueError("Donor weights/smoothing disagree with their token batch")
        if not bool(torch.isfinite(self.weights).all()) or bool((self.weights < 0).any()):
            raise ValueError("Donor weights must be finite and nonnegative")
        if self.residual_mode not in ("all", "fallback_only"):
            raise ValueError("Unknown donor residual mode")

    def index_select(self, indices):
        return DonorPriorBatch(
            self.tokens[indices], self.weights[indices], self.smoothing, self.residual_mode
        )

    def repeat_interleave(self, count):
        return DonorPriorBatch(
            self.tokens.repeat_interleave(count, dim=0),
            self.weights.repeat_interleave(count, dim=0),
            self.smoothing,
            self.residual_mode,
        )

    def condition(self, logits, fed, position):
        """Add a normalized conditional log prior to already legality-masked logits.

        Unknown or contradicted donors back off to the uniform legal distribution.
        No private truth, predicted answer, route selection or training split enters
        this operation. Zero neural logits recover this smoothed AR proposal exactly.
        """
        if fed.shape != (len(logits), 6 * SET_N_ATTRS) or len(fed) != len(self.tokens):
            raise ValueError("Prior rows do not match the decoder's fact contexts")
        if not 0 <= int(position) < 6 * SET_N_ATTRS:
            raise ValueError("Prior query outside the six-slot schema")
        legal = torch.isfinite(logits)
        if not bool(legal.any(-1).all()):
            raise ValueError("A prior conditional needs at least one legal value")
        slot, column = divmod(int(position), SET_N_ATTRS)
        donors, mass = self.tokens[:, slot], self.weights[:, slot]
        facts = fed.reshape(-1, 6, SET_N_ATTRS)[:, slot]
        compatible = mass > 0
        # The current target has not been inserted. Public/other assumed facts
        # constrain the remaining donor sets without distinguishing move columns.
        scalar_columns = [c for c in (0, 1, 2, *range(7, SET_N_ATTRS)) if c != column]
        scalar_facts = facts[:, scalar_columns]
        compatible &= (
            (scalar_facts[:, None] < 0) | (donors[:, :, scalar_columns] == scalar_facts[:, None])
        ).all(-1)
        moves = donors[:, :, list(SET_MOVE_COLS)]
        known_moves = facts[:, list(SET_MOVE_COLS)]
        contains = (moves[:, :, :, None] == known_moves[:, None, None, :]).any(-2)
        compatible &= ((known_moves[:, None] < 0) | contains).all(-1)
        mass = mass * compatible
        counts = logits.new_zeros(logits.shape)
        if column in SET_MOVE_COLS:
            remaining = ~(
                (moves[:, :, :, None] == known_moves[:, None, None, :])
                & (known_moves[:, None, None, :] >= 0)
            ).any(-1)
            valid = remaining & (moves >= 0) & (moves < logits.shape[-1])
            safe = moves.clamp(0, logits.shape[-1] - 1)
            valid &= legal.gather(1, safe.flatten(1)).reshape_as(moves)
            denominator = valid.sum(-1).clamp_min(1)
            additions = mass[:, :, None] * valid / denominator[:, :, None]
            counts.scatter_add_(1, safe.flatten(1), additions.flatten(1).to(counts.dtype))
        else:
            values = donors[:, :, column]
            safe = values.clamp(0, logits.shape[-1] - 1)
            valid = (values >= 0) & (values < logits.shape[-1]) & legal.gather(1, safe)
            counts.scatter_add_(1, safe, (mass * valid).to(counts.dtype))
        total = counts.sum(-1, keepdim=True)
        uniform = legal.to(logits.dtype) / legal.sum(-1, keepdim=True)
        empirical = counts / total.clamp_min(torch.finfo(logits.dtype).tiny)
        probabilities = torch.where(
            total > 0, (1 - self.smoothing) * empirical + self.smoothing * uniform, uniform
        )
        if self.residual_mode == "fallback_only":
            # Preserve the donor conditional exactly while it has support. This
            # conservative mode cannot learn to correct a supported but wrong
            # donor prior; it is an explicit comparison, not a general posterior.
            neutral = torch.where(legal, torch.zeros_like(logits), logits)
            logits = torch.where(total > 0, neutral, logits)
        # Retain -inf on illegal values and a finite path to every legal target.
        return logits + probabilities.clamp_min(torch.finfo(logits.dtype).tiny).log()


class DonorPrior:
    """Retrieve a bounded public-compatible, roster-weighted set bank.

    Excluding the query roster is applied in every split, using only public team
    preview. This prevents a training team from retrieving its own exact sheet.
    The cap is applied after public conditioning; rare revealed facts can retrieve
    a set that would not appear in an unconditional species top-k.
    """

    def __init__(
        self, bank: ReplayCandidateBank | Path, *, max_donors=128, smoothing=0.001,
        residual_mode="all",
    ):
        if not isinstance(bank, ReplayCandidateBank):
            bank = ReplayCandidateBank(Path(bank))
        if max_donors < 1 or not 0 < smoothing < 1:
            raise ValueError("Positive donor cap and nonzero smoothing below one required")
        if residual_mode not in ("all", "fallback_only"):
            raise ValueError("Unknown donor residual mode")
        self.residual_mode = residual_mode
        self.bank, self.max_donors, self.smoothing = bank, int(max_donors), float(smoothing)
        self.rosters = np.sort(bank.rosters, axis=1)
        if (bank.tokens < 0).any():
            raise ValueError("AR donor conditioning requires complete represented set rows")
        # A deterministic public-state cache, not part of the optimization state.
        # It changes neither probabilities nor RNG exposure after a cold resume.
        self.retrieve = lru_cache(maxsize=8192)(self._retrieve)

    def _retrieve(self, roster_values, row_values):
        row, roster = np.asarray(row_values), np.asarray(roster_values)
        indices = self.bank.by_species.get(int(row[0]), np.empty(0, dtype=int))
        empty = (np.empty((0, SET_N_ATTRS), dtype=np.int64), np.empty(0, dtype=np.float32))
        if not len(indices):
            return empty
        cand = self.bank.tokens[indices]
        compatible = ~np.all(self.rosters[indices] == roster, axis=1)
        for col in (1, 2, *range(7, SET_N_ATTRS)):
            if row[col] >= 0:
                compatible &= cand[:, col] == row[col]
        for move in row[list(SET_MOVE_COLS)]:
            if move >= 0:
                compatible &= (cand[:, list(SET_MOVE_COLS)] == move).any(axis=1)
        indices = indices[compatible]
        if not len(indices):
            return empty
        cand = self.bank.tokens[indices]
        overlap = np.isin(self.bank.rosters[indices], roster[roster >= 0]).sum(1)
        mass = 1 + np.maximum(overlap - 1, 0) / 5
        unique, inverse = np.unique(cand, axis=0, return_inverse=True)
        mass = np.bincount(inverse, weights=mass, minlength=len(unique))
        selected = np.argsort(-mass, kind="stable")[: self.max_donors]
        return unique[selected], (mass[selected] / mass[selected].sum()).astype(np.float32)

    def prepare(self, public, *, device="cpu"):
        """Public tokens only; callers must normalize current-item sentinels first."""
        public = np.asarray(public, dtype=np.int64)
        if public.ndim != 3 or public.shape[1:] != (6, SET_N_ATTRS):
            raise ValueError("Prior retrieval requires a public [batch,6,15] token grid")
        tokens = np.zeros((len(public), 6, self.max_donors, SET_N_ATTRS), dtype=np.int64)
        weights = np.zeros(tokens.shape[:-1], dtype=np.float32)
        for batch, team in enumerate(public):
            roster = tuple(sorted(int(v) for v in team[:, 0]))
            for slot, row in enumerate(team):
                selected, mass = self.retrieve(roster, tuple(int(v) for v in row))
                count = len(selected)
                tokens[batch, slot, :count] = selected
                weights[batch, slot, :count] = mass
        return DonorPriorBatch(
            torch.as_tensor(tokens, device=device),
            torch.as_tensor(weights, device=device),
            self.smoothing,
            self.residual_mode,
        )
