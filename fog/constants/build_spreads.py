"""Build SPREADS[species] via k-means on nature-modified stat vectors (spec §1.6).

For each species: cluster the (nature-modified stat vector) from the Smogon chaos *detailed* usage
file, weighted by usage share, into <=16 clusters. Merge clusters whose representative stats differ by
<4 points in every stat. Persist cluster -> representative spread; include <OTHER_SPREAD>.

Verification (test): cluster-representative stats reproduce the listed usage stats within 4 points; and
damage-calc delta between true and clustered spreads has 95th percentile < 6% over 10k sampled
matchups (justifies the only lossy step in the encoding — §2.1). The damage-delta check needs a damage
calculator and runs as part of the M1.5 gate, not here.

Chaos-file spread keys look like "Modest:0/0/0/252/4/252" (nature : HP/Atk/Def/SpA/SpD/Spe EVs);
`parse_chaos_spread` converts them. IVs are not in the chaos files — assume 31 across the board (the
rare 0-Atk/0-Spe sets differ by a few points; that error is inside the merge threshold).

The k-means here is a ~30-line weighted implementation (k-means++ init, fixed seed) rather than
sklearn: n <= a few hundred spreads per species, and determinism from an explicit seed is worth more
than library speed.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from fog.constants.schema import MAX_SPREADS_PER_SPECIES

MERGE_THRESHOLD = 4  # points, per-stat
STAT_ORDER = ("hp", "atk", "def", "spa", "spd", "spe")

# nature -> (+10% stat index, -10% stat index) in STAT_ORDER; neutral natures omitted (1.0 both ways).
# HP (index 0) is never nature-modified.
NATURES: dict[str, tuple[int, int]] = {
    "adamant": (1, 3), "lonely": (1, 2), "brave": (1, 5), "naughty": (1, 4),
    "bold": (2, 1), "impish": (2, 3), "relaxed": (2, 5), "lax": (2, 4),
    "modest": (3, 1), "mild": (3, 2), "quiet": (3, 5), "rash": (3, 4),
    "calm": (4, 1), "gentle": (4, 2), "sassy": (4, 5), "careful": (4, 3),
    "timid": (5, 1), "hasty": (5, 2), "jolly": (5, 3), "naive": (5, 4),
}
NEUTRAL_NATURES = ("hardy", "docile", "serious", "bashful", "quirky")


def parse_chaos_spread(key: str) -> tuple[str, np.ndarray]:
    """"Modest:0/0/0/252/4/252" -> ("modest", evs[6] int array in STAT_ORDER)."""
    nature, evs = key.split(":")
    return nature.strip().lower(), np.array([int(v) for v in evs.split("/")], dtype=np.int64)


def nature_multipliers(nature: str) -> np.ndarray:
    """[6] multiplier vector for a nature name (case-insensitive)."""
    mult = np.ones(6, dtype=np.float64)
    nature = nature.lower()
    if nature in NATURES:
        up, down = NATURES[nature]
        mult[up], mult[down] = 1.1, 0.9
    elif nature not in NEUTRAL_NATURES:
        raise ValueError(f"unknown nature: {nature!r}")
    return mult


def nature_modified_stats(
    base: np.ndarray, evs: np.ndarray, nature: str, level: int = 100, ivs: int | np.ndarray = 31
) -> np.ndarray:
    """Standard Gen 3+ stat formula (Gen 9 uses it unchanged). base/evs in STAT_ORDER.

    HP   = floor((2*base + IV + floor(EV/4)) * L / 100) + L + 10       (Shedinja: base 1 -> HP 1)
    Other = floor( (floor((2*base + IV + floor(EV/4)) * L / 100) + 5) * nature )
    """
    base = np.asarray(base, dtype=np.int64)
    evs = np.asarray(evs, dtype=np.int64)
    iv = np.full(6, ivs, dtype=np.int64) if np.isscalar(ivs) else np.asarray(ivs, dtype=np.int64)
    core = (2 * base + iv + evs // 4) * level // 100
    stats = np.empty(6, dtype=np.int64)
    stats[0] = 1 if base[0] == 1 else core[0] + level + 10
    mult = nature_multipliers(nature)
    stats[1:] = np.floor((core[1:] + 5) * mult[1:]).astype(np.int64)
    return stats


def _weighted_kmeans(
    x: np.ndarray, w: np.ndarray, k: int, iters: int = 100, seed: int = 0
) -> np.ndarray:
    """Weighted k-means with k-means++ init. Returns the assignment vector [n]."""
    n = len(x)
    k = min(k, n)
    rng = np.random.default_rng(seed)
    w = w / w.sum()

    centers = [x[rng.choice(n, p=w)]]
    for _ in range(1, k):
        d2 = np.min(((x[:, None, :] - np.asarray(centers)[None]) ** 2).sum(-1), axis=1)
        probs = d2 * w
        total = probs.sum()
        if total <= 0:
            break  # every point sits on an existing center — fewer clusters than k
        centers.append(x[rng.choice(n, p=probs / total)])
    centers = np.asarray(centers, dtype=np.float64)

    assign = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        assign = ((x[:, None, :] - centers[None]) ** 2).sum(-1).argmin(axis=1)
        new_centers = centers.copy()
        for c in range(len(centers)):
            m = assign == c
            if w[m].sum() > 0:
                new_centers[c] = (x[m] * w[m, None]).sum(0) / w[m].sum()
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return assign


def cluster_species_spreads(
    observed_spreads: list[dict[str, Any]],
    base_stats: np.ndarray,
    max_clusters: int = MAX_SPREADS_PER_SPECIES,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Weighted k-means over nature-modified stat vectors, then merge near-duplicate clusters.

    `observed_spreads`: list of {"nature": str, "evs": [6] ints, "usage_weight": float} from the chaos
    detailed file (see parse_chaos_spread). Returns <= max_clusters representative spreads sorted by
    total cluster weight, each {"nature", "evs", "stats", "weight"} — the representative is the
    highest-weight OBSERVED spread in the cluster (never a fractional centroid: EVs must stay legal).
    """
    if not observed_spreads:
        return []
    stats = np.stack(
        [
            nature_modified_stats(base_stats, np.asarray(sp["evs"]), sp["nature"])
            for sp in observed_spreads
        ]
    ).astype(np.float64)
    weights = np.array([float(sp["usage_weight"]) for sp in observed_spreads], dtype=np.float64)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)

    assign = _weighted_kmeans(stats, weights, k=max_clusters, seed=seed)

    # cluster -> member indices
    clusters: list[list[int]] = [
        list(np.nonzero(assign == c)[0]) for c in np.unique(assign)
    ]

    def rep_of(members: list[int]) -> int:
        return max(members, key=lambda i: weights[i])

    # merge clusters whose REPRESENTATIVE stats differ by < MERGE_THRESHOLD in every stat, to fixpoint
    merged = True
    while merged and len(clusters) > 1:
        merged = False
        reps = [stats[rep_of(m)] for m in clusters]
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if np.all(np.abs(reps[i] - reps[j]) < MERGE_THRESHOLD):
                    clusters[i] = clusters[i] + clusters[j]
                    del clusters[j]
                    merged = True
                    break
            if merged:
                break

    out = []
    for members in clusters:
        r = rep_of(members)
        sp = observed_spreads[r]
        out.append(
            {
                "nature": sp["nature"],
                "evs": [int(v) for v in np.asarray(sp["evs"], dtype=np.int64)],
                "stats": [int(v) for v in stats[r]],
                "weight": float(weights[members].sum()),
            }
        )
    out.sort(key=lambda d: -d["weight"])
    return out
