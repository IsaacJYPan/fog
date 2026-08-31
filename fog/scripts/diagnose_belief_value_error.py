"""Decompose joint belief/value error on the exact Simple-vs-Simple corpus.

This is a read-only diagnostic, not a training job.  It answers two different questions:

* Belief: how much held-out error remains above a train-only, species-conditioned empirical
  posterior after applying the same public reveals?  The empirical posterior is an achievable
  reference, not a claim that it is the true metagame Bayes distribution.
* Value: how much initial-state outcome variance is intrinsic to repeated ordered team matchups,
  and how much model MSE remains above that variance?

No held-out truth is used to form either reference distribution.  Held-out truth is used only for
scoring.  Opponent private fields never enter the model input.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from fog.constants.tables import TABLES
from fog.encoding.reveals import FAMILY_COLS, align_truth_to_views, original_field_hidden
from fog.training.dataset import _build_batch


def _compatible(candidate: np.ndarray, view: np.ndarray) -> bool:
    """Whether one submitted set is compatible with legitimately public fields in ``view``."""
    truth_team = np.full((1, 6, candidate.shape[-1]), -1, dtype=np.int64)
    view_team = np.full_like(truth_team, -1)
    truth_team[0, 0] = candidate
    view_team[0, 0] = view
    aligned, consistent = align_truth_to_views(truth_team, view_team)
    if not bool(consistent[0, 0]):
        return False
    row = aligned[0, 0]
    known = view >= 0
    # <NO_ITEM> is current held-item state, not the submitted item, so it is not a constraint.
    known[1] = known[1] and int(view[1]) != TABLES.item_index("<NO_ITEM>")
    return bool(np.all(row[known] == view[known]))


def _team_truths(manifest) -> dict[int, np.ndarray]:
    truths: dict[int, np.ndarray] = {}
    for row in manifest.iter_rows(named=True):
        team = int(row["p2_team"])
        if team in truths:
            continue
        with np.load(row["shard"]) as shard:
            truth = np.asarray(shard["belief_targets"][0], dtype=np.int64)
        truths[team] = truth
    return truths


def _records_by_species(truths: dict[int, np.ndarray]) -> dict[int, list[np.ndarray]]:
    records: dict[int, list[np.ndarray]] = defaultdict(list)
    seen: set[bytes] = set()
    for team in truths.values():
        for record in team:
            key = record.tobytes()
            if key not in seen:
                records[int(record[0])].append(record.copy())
                seen.add(key)
    return records


def _smoothed_probability(counts: Counter, target: int, classes: int, alpha: float) -> float:
    return (counts[target] + alpha) / (sum(counts.values()) + alpha * classes)


def _belief_reference(
    train_records: dict[int, list[np.ndarray]], heldout_manifest, *, alpha: float
) -> dict:
    sizes = {
        "item": TABLES.n_items,
        "ability": TABLES.n_abilities,
        "move": TABLES.n_moves,
        "nature": 25,
        "ev": 64,
        "tera": 19,
    }
    stats = {
        family: {"count": 0, "supported": 0, "nll_sum": 0.0, "top1": 0,
                 "candidate_sum": 0}
        for family in sizes
    }
    reveal_buckets: dict[str, dict[str, float]] = defaultdict(
        lambda: {"count": 0, "nll_sum": 0.0, "supported": 0}
    )
    for row in heldout_manifest.iter_rows(named=True):
        with np.load(row["shard"]) as shard:
            views = np.asarray(shard["opp_tokens"], dtype=np.int64)
            truths = np.asarray(shard["belief_targets"], dtype=np.int64)
        for view_team, truth_team in zip(views, truths):
            aligned_truth, _ = align_truth_to_views(truth_team[None], view_team[None])
            aligned_truth = aligned_truth[0]
            hidden = original_field_hidden(view_team, TABLES)
            revealed = int((view_team >= 0).sum())
            bucket = "0-6" if revealed <= 6 else "7-12" if revealed <= 12 else "13+"
            for slot in range(6):
                species = int(view_team[slot, 0])
                candidates = [
                    candidate for candidate in train_records.get(species, [])
                    if _compatible(candidate, view_team[slot])
                ]
                aligned_candidates = None
                if candidates:
                    array = np.stack(candidates)
                    candidate_teams = np.full(
                        (len(array), 6, array.shape[-1]), -1, dtype=np.int64
                    )
                    candidate_views = np.full_like(candidate_teams, -1)
                    candidate_teams[:, 0] = array
                    candidate_views[:, 0] = view_team[slot]
                    aligned_candidates, _ = align_truth_to_views(
                        candidate_teams, candidate_views
                    )
                    aligned_candidates = aligned_candidates[:, 0]
                for family, columns in FAMILY_COLS.items():
                    if family == "species":
                        continue
                    for column in columns:
                        if not hidden[slot, column]:
                            continue
                        target = int(aligned_truth[slot, column])
                        if target < 0:
                            continue
                        values = [] if aligned_candidates is None else [
                            int(value) for value in aligned_candidates[:, column] if int(value) >= 0
                        ]
                        counts = Counter(values)
                        probability = _smoothed_probability(
                            counts, target, sizes[family], alpha
                        )
                        datum = stats[family]
                        datum["count"] += 1
                        datum["candidate_sum"] += len(values)
                        datum["supported"] += int(counts[target] > 0)
                        datum["nll_sum"] += -math.log(max(probability, 1e-30))
                        if counts:
                            datum["top1"] += int(target == counts.most_common(1)[0][0])
                        b = reveal_buckets[bucket]
                        b["count"] += 1
                        b["supported"] += int(counts[target] > 0)
                        b["nll_sum"] += -math.log(max(probability, 1e-30))
    def finish(value: dict) -> dict:
        count = max(1, int(value["count"]))
        result = {
            "count": int(value["count"]),
            "support_rate": float(value["supported"] / count),
            "smoothed_nll": float(value["nll_sum"] / count),
        }
        if "top1" in value:
            result["top1"] = float(value["top1"] / count)
            result["mean_compatible_records"] = float(value["candidate_sum"] / count)
        return result
    return {
        "interpretation": (
            "train-only species/reveal-conditioned empirical reference; approximate achievable "
            "baseline, not the true Bayes floor"
        ),
        "dirichlet_alpha": alpha,
        "families": {key: finish(value) for key, value in stats.items()},
        "reveal_buckets": {key: finish(value) for key, value in reveal_buckets.items()},
    }


def _value_decomposition(agent, heldout_manifest, cfg: dict, device: torch.device) -> dict:
    predictions: dict[tuple[int, int], list[float]] = defaultdict(list)
    outcomes: dict[tuple[int, int], list[float]] = defaultdict(list)
    agent.eval()
    with torch.no_grad():
        for row in heldout_manifest.iter_rows(named=True):
            with np.load(row["shard"]) as shard:
                data = {key: shard[key] for key in shard.files}
                batch = _build_batch(data, np.asarray([0]), TABLES, device)
            prediction = float(agent(batch).value.reshape(-1)[0].detach().cpu())
            key = (int(row["p1_team"]), int(row["p2_team"]))
            predictions[key].append(prediction)
            outcomes[key].append(float(row["outcome"]))
    model_sq = []
    oracle_sq = []
    group_variances = []
    weighted_variance_sum = 0.0
    used_games = 0
    for key, ys_list in outcomes.items():
        if len(ys_list) < 2:
            continue
        ys = np.asarray(ys_list, dtype=np.float64)
        ps = np.asarray(predictions[key], dtype=np.float64)
        # Leave-one-out matchup mean uses no evaluated outcome in its own prediction.
        loo = (ys.sum() - ys) / (len(ys) - 1)
        model_sq.extend(((ps - ys) ** 2).tolist())
        oracle_sq.extend(((loo - ys) ** 2).tolist())
        variance = float(np.var(ys, ddof=1))
        group_variances.append(variance)
        weighted_variance_sum += len(ys) * variance
        used_games += len(ys)
    return {
        "interpretation": (
            "initial-state ordered-team repeated-matchup estimate; leave-one-out matchup mean is "
            "an empirical stochastic reference, not an exact per-position Bayes floor"
        ),
        "groups_with_repeats": len(group_variances),
        "games_in_repeated_groups": used_games,
        "model_mse": float(np.mean(model_sq)) if model_sq else None,
        "leave_one_out_matchup_mse": float(np.mean(oracle_sq)) if oracle_sq else None,
        "mean_unbiased_within_matchup_variance": (
            float(np.mean(group_variances)) if group_variances else None
        ),
        "game_weighted_unbiased_within_matchup_variance": (
            float(weighted_variance_sum / used_games) if used_games else None
        ),
        "model_excess_over_estimated_stochastic_variance": (
            float(np.mean(model_sq) - weighted_variance_sum / used_games)
            if model_sq and used_games else None
        ),
        "model_excess_over_loo_reference": (
            float(np.mean(model_sq) - np.mean(oracle_sq)) if model_sq else None
        ),
    }


def main() -> None:
    import polars as pl

    from fog.config import load_config_with_args
    from fog.models.agent import FogAgent

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--heldout-manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--skip-belief-reference", action="store_true")
    args = parser.parse_args()
    cfg = load_config_with_args(args.config)
    train = pl.read_parquet(args.train_manifest)
    heldout = pl.read_parquet(args.heldout_manifest)
    reference = None
    if not args.skip_belief_reference:
        train_truths = _team_truths(train)
        reference = _belief_reference(
            _records_by_species(train_truths), heldout, alpha=args.alpha
        )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    agent = FogAgent.from_tables(cfg).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False))
    report = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "train_manifest": str(args.train_manifest),
        "heldout_manifest": str(args.heldout_manifest),
        "heldout_truth_used_for_reference_fitting": False,
        "belief_empirical_reference": reference,
        "value_stochastic_reference": _value_decomposition(agent, heldout, cfg, device),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
