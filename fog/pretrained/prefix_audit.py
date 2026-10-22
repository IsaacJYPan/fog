"""Offline observations of donor support; labels never affect returned logits."""

from __future__ import annotations

from collections import defaultdict
import math

import torch

from fog.constants.schema import SET_MOVE_COLS, SET_N_ATTRS

FAMILIES = (
    "species", "item", "ability", "move", "move", "move", "move", "nature",
    "ev", "ev", "ev", "ev", "ev", "ev", "tera",
)


def compatible_sets(donors, facts, verified=None, *, skip_column=None):
    """Compare unordered moves and scalar facts, without consulting an answer."""
    valid = facts >= 0
    if verified is not None:
        valid = valid & verified
    scalar = [c for c in range(SET_N_ATTRS) if c not in SET_MOVE_COLS and c != skip_column]
    result = ((donors[..., scalar] == facts[scalar]) | ~valid[scalar]).all(-1)
    moves = donors[..., list(SET_MOVE_COLS)]
    for column in SET_MOVE_COLS:
        if column != skip_column and bool(valid[column]):
            result = result & (moves == facts[column]).any(-1)
    return result


class PrefixObservations:
    """Scoring-only observer of the actual production conditional calls.

    The wrapped prior computes its result before this observer runs. Private truth
    is used here only to label errors/support. No truth is passed to the decoder,
    retrieval or prior. Counters are diagnostics, not independent observations.
    """

    def __init__(self, truth, verified, public, metadata, *, source):
        self.truth = truth.detach().cpu()
        self.verified = verified.detach().cpu()
        self.public = public.detach().cpu()
        self.metadata = metadata
        self.source = source
        self.rows = []

    def observe(self, prior, ids, logits, result, fed, position):
        slot, column = divmod(int(position), SET_N_ATTRS)
        for row, identity in enumerate(ids.tolist()):
            # Dense forward also visits already inserted fields; those are not queries.
            if int(fed[row, position]) >= 0:
                continue
            truth = self.truth[identity, slot]
            verified = self.verified[identity, slot]
            if not bool(verified[column]):
                continue
            values = prior.tokens[row, slot].detach().cpu()
            weights = prior.weights[row, slot].detach().cpu()
            facts = fed[row].detach().cpu().reshape(6, SET_N_ATTRS)[slot]
            legal = torch.isfinite(logits[row]).detach().cpu()
            donors = (weights > 0) & compatible_sets(values, facts, skip_column=column)
            if column in SET_MOVE_COLS:
                remaining = values[:, list(SET_MOVE_COLS)]
                allowed = (remaining >= 0) & (remaining < len(legal))
                for known in facts[list(SET_MOVE_COLS)]:
                    if int(known) >= 0:
                        allowed &= remaining != known
                allowed &= legal[remaining.clamp(0, len(legal) - 1)]
                viable = donors & allowed.any(-1)
                targets = truth[list(SET_MOVE_COLS)][verified[list(SET_MOVE_COLS)]]
                targets = targets[(targets >= 0) & (targets < len(legal))].unique()
                for known in facts[list(SET_MOVE_COLS)]:
                    if int(known) >= 0:
                        targets = targets[targets != known]
                supported = bool(
                    ((remaining[:, :, None] == targets) & allowed[:, :, None]
                     & viable[:, None, None]).any()
                )
            else:
                query_values = values[:, column]
                valid = (query_values >= 0) & (query_values < len(legal))
                viable = donors & valid & legal[query_values.clamp(0, len(legal) - 1)]
                targets = truth[column:column + 1]
                targets = targets[(targets >= 0) & (targets < len(legal))]
                supported = bool(((query_values[:, None] == targets) & viable[:, None]).any())
            if not len(targets):
                continue
            baseline_logits = torch.where(
                torch.isfinite(logits[row:row + 1]), 0.0, -torch.inf
            )
            baseline = prior.index_select(slice(row, row + 1)).condition(
                baseline_logits, fed[row:row + 1], position
            ).softmax(-1)[0].detach().cpu()
            probabilities = result[row].softmax(-1).detach().cpu()
            consistent = bool(compatible_sets(truth[None], facts, verified)[0])
            complete = bool(verified.all())
            truth_support = complete and bool(
                (compatible_sets(values, truth) & (weights > 0)).any()
            )
            public = self.public[identity, slot]
            assumed = int(((facts >= 0) & (public < 0)).sum())
            positive = probabilities > 0
            kl = float((probabilities[positive] * (
                probabilities[positive].log() - baseline[positive].log()
            )).sum())
            self.rows.append({
                **self.metadata[identity], "source": self.source, "slot": slot,
                "column": column, "family": FAMILIES[column], "assumed_count": assumed,
                "donors": int((weights > 0).sum()), "viable_donors": int(viable.sum()),
                "fallback": not bool(viable.any()), "truth_prefix_compatible": consistent,
                "complete_verified_set": complete, "truth_set_in_retrieved_bank": truth_support,
                "truth_value_supported": supported, "truth_value_legal": bool(legal[targets].any()),
                "prior_true_mass": float(baseline[targets].sum()),
                "model_true_mass": float(probabilities[targets].sum()),
                "kl_model_prior": kl,
            })


class ObservedPriorBatch:
    """Transparent adapter which preserves indexing, sampling RNG and logits."""

    def __init__(self, prior, observations, ids=None):
        self.prior, self.observations = prior, observations
        self.ids = torch.arange(len(prior.tokens)) if ids is None else ids

    def index_select(self, indices):
        selected = indices.detach().cpu() if isinstance(indices, torch.Tensor) else indices
        return ObservedPriorBatch(
            self.prior.index_select(indices), self.observations, self.ids[selected]
        )

    def repeat_interleave(self, count):
        return ObservedPriorBatch(
            self.prior.repeat_interleave(count), self.observations,
            self.ids.repeat_interleave(count),
        )

    def condition(self, logits, fed, position):
        result = self.prior.condition(logits, fed, position)
        with torch.no_grad():
            self.observations.observe(self.prior, self.ids, logits, result, fed, position)
        return result


def summarize(rows):
    """Descriptive stratified counters; no field-level confidence intervals."""
    groups = defaultdict(list)
    for row in rows:
        for key in ["all", "family/" + row["family"], "phase/" + row["phase"],
                    "prefix/" + ("compatible" if row["truth_prefix_compatible"] else "contradictory"),
                    "support/" + ("fallback" if row["fallback"] else "donor"),
                    "assumed/" + str(min(row["assumed_count"], 8))]:
            groups[key].append(row)
    output = {}
    for key, values in groups.items():
        n = len(values)
        output[key] = {"queries": n, **{
            name + "_rate": sum(bool(x[name]) for x in values) / n for name in (
                "fallback", "truth_prefix_compatible", "truth_set_in_retrieved_bank",
                "truth_value_supported", "truth_value_legal",
            )
        }, **{
            name + "_mean": sum(x[name] for x in values) / n
            for name in ["viable_donors", "kl_model_prior"]
        }, **{
            name + "_nll": sum(-math.log(max(x[name], 1e-30)) for x in values) / n
            for name in ["prior_true_mass", "model_true_mass"]
        }}
    return output
