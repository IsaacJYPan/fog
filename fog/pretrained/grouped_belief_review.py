"""Paired uncertainty over connected whole-roster groups, never individual fields."""

from __future__ import annotations

import numpy as np


def connected_groups(keys):
    parent = list(range(len(keys)))
    owner = {}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, values in enumerate(keys):
        for value in values:
            if value in owner:
                parent[find(i)] = find(owner[value])
            else:
                owner[value] = i
    groups = {}
    for i in range(len(keys)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def interval(point, draws):
    values = np.asarray(draws)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite clustered uncertainty")
    constant = bool(np.ptp(values) < 1e-15)
    lo, hi = np.quantile(values, [0.025, 0.975])
    return {
        "point": float(point),
        "low": None if constant else float(lo),
        "high": None if constant else float(hi),
        "degenerate_bootstrap": constant,
        "scope": "Exploratory unadjusted paired whole-roster percentile bootstrap. A constant sample is not proof of population equality.",
    }


def sampler_comparisons(methods, *, seed=20261020, replicates=20000):
    names = sorted(next(iter(methods.values()))["by_game"])
    reference = next(iter(methods.values()))["by_game"]
    groups = connected_groups([reference[name]["team_rosters"] for name in names])
    draws = np.random.default_rng(seed).integers(0, len(groups), (replicates, len(groups)))
    samples = {}
    summaries = {}
    for method, report in methods.items():
        if sorted(report["by_game"]) != names:
            raise ValueError("Methods did not evaluate the same games")
        for name in names:
            if report["by_game"][name]["team_rosters"] != reference[name]["team_rosters"]:
                raise ValueError("A method changed roster groups")
        families = sorted(report["by_game"][names[0]]["resampling_data"]["fields"])
        point, boot = {}, {}
        for family in families:
            accum = []
            for group in groups:
                rows = np.asarray(
                    [
                        v
                        for i in group
                        for v in report["by_game"][names[i]]["resampling_data"]["fields"][family]
                    ]
                )
                bins = np.minimum((rows[:, 1] * 15).astype(int), 14)
                stats = [len(rows), rows[:, 0].sum(), rows[:, 2].sum(), rows[:, 3].sum()]
                stats.extend(float(np.sum((rows[:, 0] - rows[:, 1])[bins == j])) for j in range(15))
                accum.append(stats)
            totals = np.asarray(accum).sum(0)
            sampled = np.asarray(accum)[draws].sum(1)
            metrics = {
                "modal_accuracy": (totals[1] / totals[0], sampled[:, 1] / sampled[:, 0]),
                "brier": (totals[2] / totals[0], sampled[:, 2] / sampled[:, 0]),
                "sample_accuracy": (totals[3] / totals[0], sampled[:, 3] / sampled[:, 0]),
                "ece": (
                    abs(totals[4:]).sum() / totals[0],
                    abs(sampled[:, 4:]).sum(1) / sampled[:, 0],
                ),
            }
            for key, (value, sample) in metrics.items():
                point[family + "/" + key] = value
                boot[family + "/" + key] = sample
        joint = np.asarray(
            [
                [
                    sum(report["by_game"][names[i]]["resampling_data"][key] for i in group)
                    for key in ["joint_correct", "joint_count", "recall_correct", "recall_count"]
                ]
                for group in groups
            ]
        )
        total = joint.sum(0)
        sampled = joint[draws].sum(1)
        for key, numerator, denominator in [("joint_accuracy", 0, 1), ("joint_recall_at_8", 2, 3)]:
            point[key] = total[numerator] / total[denominator]
            boot[key] = sampled[:, numerator] / sampled[:, denominator]
        summaries[method] = {key: interval(value, boot[key]) for key, value in point.items()}
        samples[method] = (point, boot)
    contrasts = {}
    pairs = [
        ("previous_neural", "prior_only"),
        ("prior_only", "facts_residual"),
        ("prior_only", "native_residual"),
        ("facts_residual", "native_residual"),
        ("expanded_existing_prior", "native_residual"),
    ]
    for left, right in pairs:
        contrasts[right + "_minus_" + left] = {
            key: interval(
                samples[right][0][key] - samples[left][0][key],
                samples[right][1][key] - samples[left][1][key],
            )
            for key in samples[left][0]
        }
    return {
        "games": len(names),
        "roster_groups": len(groups),
        "group_sizes": [len(g) for g in groups],
        "summaries": summaries,
        "contrasts": contrasts,
        "replicates": replicates,
        "seed": seed,
        "method": "Ratio-of-sums connected-roster bootstrap; ECE recomputed from pooled bins.",
    }


def density_comparison(left, right, *, seed=20261021, replicates=20000):
    names = sorted(left["by_game"])
    if sorted(right["by_game"]) != names:
        raise ValueError("Density conditions changed games")
    if any(
        left["by_game"][name]["team_rosters"] != right["by_game"][name]["team_rosters"]
        for name in names
    ):
        raise ValueError("Density conditions changed roster groups")
    groups = connected_groups([left["by_game"][name]["team_rosters"] for name in names])
    draws = np.random.default_rng(seed).integers(0, len(groups), (replicates, len(groups)))
    values = []
    for report in [left, right]:
        result = {}
        families = sorted(
            {family for row in report["by_game"].values() for family in row["marginal"]}
        )
        for route in ["marginal", "conditional"]:
            for family in families:
                totals = np.asarray(
                    [
                        [
                            sum(
                                report["by_game"][names[i]][route].get(family, {}).get(key, 0)
                                for i in group
                            )
                            for key in ["fields", "nll", "correct", "true_probability", "entropy"]
                        ]
                        for group in groups
                    ]
                )
                observed = totals.sum(0)
                sampled = totals[draws].sum(1)
                if observed[0] == 0 or (sampled[:, 0] == 0).any():
                    continue
                for key, col in [("nll", 1), ("top1", 2), ("true_probability", 3), ("entropy", 4)]:
                    result[f"{route}/{family}/{key}"] = (
                        observed[col] / observed[0],
                        sampled[:, col] / sampled[:, 0],
                    )
        outcome = np.asarray(
            [
                [
                    sum(report["by_game"][names[i]][key] for i in group)
                    for key in ["outcome_count", "outcome_sum"]
                ]
                for group in groups
            ]
        )
        observed = outcome.sum(0)
        sampled = outcome[draws].sum(1)
        result["outcome_mse"] = (observed[1] / observed[0], sampled[:, 1] / sampled[:, 0])
        values.append(result)
    return {
        key: interval(values[1][key][0] - values[0][key][0], values[1][key][1] - values[0][key][1])
        for key in sorted(values[0].keys() & values[1].keys())
    }
