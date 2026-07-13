"""Self-belief predictor — the leakage measurer (spec §3.3). Mirror of the belief model (§3.2).

Identical architecture, SEPARATE weights, pointed at MY OWN team. Input: the opponent's view of me (my
entity records with epistemic flags flipped to what THEY can see, plus the same history — history is
symmetric public info). Output: a distribution over MY hidden attributes.

Trained with full ground-truth labels (I always know my own team). Its per-attribute cross-entropy on
my still-hidden attributes is the leakage signal L_t: low CE = I am readable; high CE = I am opaque.

Frozen copies `Φ̂` are used to compute the leakage penalty (§7) so the agent can't cheat by making the
predictor bad instead of the play deceptive.

Tensor conventions (shared with BeliefModel):
  * view dicts ("my_view" = the opponent's view of my team) have the same {"tokens","masks"} contract
    as `opp_revealed` in belief.py — tokens [B, 6, 9], -1 where the OPPONENT can't see the attribute.
  * label dicts are per-family index tensors:
      species/item/ability/spread/tera : [B, 6]
      move                             : [B, 6, 4]
    with -1 for "no label" (never the case for own-team truth, but supported).
  * still-hidden dicts are bools of the same per-family shapes.
"""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from fog.models.belief import SLOT_ATTRS, BeliefModel

# family -> the attr columns it occupies in a [.., 6, 9] token tensor (order = SLOT_ATTRS)
_FAMILY_COLS = {
    "species": (0,),
    "item": (1,),
    "ability": (2,),
    "move": (3, 4, 5, 6),
    "spread": (7,),
    "tera": (8,),
}


def labels_from_tokens(tokens: torch.Tensor) -> dict[str, torch.Tensor]:
    """Split a [B, 6, 9] token tensor into the per-family label dict (see module docstring)."""
    out: dict[str, torch.Tensor] = {}
    for fam, cols in _FAMILY_COLS.items():
        sel = tokens[..., list(cols)]
        out[fam] = sel if fam == "move" else sel.squeeze(-1)
    return out


def tokens_from_labels(labels: dict[str, torch.Tensor]) -> torch.Tensor:
    """Inverse of `labels_from_tokens`: a per-family label dict -> [B, 6, 9] token grid, -1 preserved.

    Used to rebuild the true-value grid teacher-forced into the belief chain (BELIEF_PLAN §3.3 pass B).
    A -1 label (censored/off-table) stays -1, so that tail token feeds <UNK> — exactly the pass-B
    contract ("<UNK> appears in the tail only where the label itself is censored/missing").
    """
    any_fam = next(iter(labels.values()))
    b = any_fam.shape[0]
    grid = torch.full((b, 6, len(SLOT_ATTRS)), -1, dtype=torch.long, device=any_fam.device)
    for fam, cols in _FAMILY_COLS.items():
        vals = labels[fam].to(torch.long)
        if fam == "move":
            grid[..., list(cols)] = vals
        else:
            grid[..., cols[0]] = vals
    return grid


def still_hidden_from_tokens(view_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-family bool masks of what the OPPONENT still can't see: view token == -1."""
    hidden = view_tokens < 0
    return {
        fam: (m if fam == "move" else m.squeeze(-1))
        for fam, m in ((f, hidden[..., list(c)]) for f, c in _FAMILY_COLS.items())
    }


class SelfBeliefPredictor(BeliefModel):
    """Same net as BeliefModel; the difference is only what it's pointed at (my hidden attrs) and how
    its input observation is constructed (opponent's view of me).
    """

    def leakage_ce(
        self,
        s_opp_view: torch.Tensor,
        my_view: dict,
        still_hidden_mask: dict[str, torch.Tensor] | None = None,
        labels: dict[str, torch.Tensor] | None = None,
        family_weights: dict[str, float] | None = None,
        per_family: bool = False,
    ):
        """Summed CE over MY still-hidden attributes = the leakage signal L_t (§3.3). Returns [B],
        or a per-family {name: [B]} dict when `per_family` (family weights are NOT applied there —
        per-family output is for reporting, weights are for the scalar penalty).

        `my_view` is the opponent's view ({"tokens","masks"}, -1 where they can't see). Labels default
        to `my_view["labels"]` (per-family truth dict); still-hidden defaults to view tokens == -1.
        Attributes with label -1 never contribute. `family_weights` rescales each family's summed CE
        in the scalar output (§7 weighted-ΔI ablation: not all bits are strategically equal — a tera
        type is worth more than a fourth filler move). Differentiable when called with grad enabled
        (used as a training loss); wrap in no_grad / use a FrozenProbe for the §7 penalty.
        """
        if labels is None:
            labels = my_view["labels"]
        if still_hidden_mask is None:
            still_hidden_mask = still_hidden_from_tokens(my_view["tokens"])

        out = self.forward(s_opp_view, my_view)
        total = s_opp_view.new_zeros(s_opp_view.shape[0])
        by_family: dict[str, torch.Tensor] = {}
        for fam, logits in out.logits.items():
            lab = labels[fam]
            hidden = still_hidden_mask[fam]
            n_classes = logits.shape[-1]
            valid = hidden & (lab >= 0) & (lab < n_classes)
            logp = F.log_softmax(logits, dim=-1)
            ce = -logp.gather(-1, lab.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            # A truth label on a legality-masked (-inf) class gives ce=+inf -> NaN rewards. Drop it
            # from the sum (same guard as training.losses.masked_cross_entropy).
            valid = valid & torch.isfinite(ce)
            ce = torch.where(valid, ce, torch.zeros_like(ce))
            fam_sum = ce.reshape(ce.shape[0], -1).sum(dim=1)
            by_family[fam] = fam_sum
            w = float(family_weights.get(fam, 1.0)) if family_weights else 1.0
            total = total + w * fam_sum
        return by_family if per_family else total

    def frozen_copy(self) -> "SelfBeliefPredictor":
        """Detached, eval-mode deep copy of the HEAD ONLY. NOT the §7 penalty probe: a frozen head
        fed by the live trunk drifts OOD (the v1 hole) — Φ̂ is `models.probe.FrozenProbe`, which
        freezes the (backbone + self-belief) pair together."""
        clone = copy.deepcopy(self)
        clone.eval()
        for p in clone.parameters():
            p.requires_grad_(False)
        return clone
