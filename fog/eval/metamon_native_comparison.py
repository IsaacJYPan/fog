"""Prepare public-only Metamon inputs, then score native outputs in FOG's own environment."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from fog.constants.schema import NATURE_INDEX, NATURES, TYPE_INDEX
from fog.constants.tables import TABLES
from fog.encoding.reveals import to_id
from fog.eval.replay_imputation import align_truth, policy_pilot, score_samples, summarize
from fog.training.replay_candidates import ReplayCandidateBank, replay_digest, revealed_teams
from fog.training.replay_policy_data import complete_private_team, stable_replay_seed

METHODS = ("fog_usage_fixed", "fog_candidates_fixed", "fog_usage_native",
           "fog_candidates_native", "metamon_predictor_fog_evidence", "metamon_end_to_end")


def export_partial(team: np.ndarray) -> list[dict]:
    rows = []
    for row in team:
        if row[0] < 0:
            raise ValueError("native comparison requires complete public preview")
        item = TABLES.items["ids"][row[1]] if row[1] >= 0 else "$missing_item$"
        rows.append({"name": TABLES.species["ids"][row[0]], "gen": 9,
                     "item": "<noitem>" if item == "<NO_ITEM>" else item,
                     "ability": TABLES.abilities["ids"][row[2]] if row[2] >= 0 else "$missing_ability$",
                     "moves": [TABLES.moves["ids"][m] if m >= 0 else "$missing_move$" for m in row[3:7]],
                     "nature": NATURES[row[7]] if row[7] >= 0 else "$missing_nature$",
                     "evs": [int(v) * 4 if v >= 0 else "$missing_ev$" for v in row[8:14]],
                     "ivs": [31] * 6,
                     "tera_type": (*TYPE_INDEX, "stellar")[row[14]] if row[14] >= 0 else "$missing_tera$"})
    return rows


def import_team(rows: list[dict], reference: np.ndarray) -> np.ndarray:
    """Translate native output, preserving errors/missing predictions rather than repairing them."""
    result = np.full((6, 15), -1, dtype=np.int64)
    members = {TABLES.species_index(to_id(row["name"])): row for row in rows}
    for slot, ref in enumerate(reference):
        source = members.get(int(ref[0]))
        if source is None:
            raise ValueError("native output species cannot be aligned with public preview")
        result[slot, 0] = ref[0]
        item = source["item"]
        if not str(item).startswith("$"):
            result[slot, 1] = TABLES.item_index("<NO_ITEM>" if item == "<noitem>" else to_id(item))
        result[slot, 2] = TABLES.ability_index(to_id(source["ability"]), default=-1)
        moves = [TABLES.move_index(to_id(move), default=-1) for move in source["moves"]]
        moves = list(dict.fromkeys(m for m in moves if m >= 0))
        # Move order is irrelevant to scoring; align existing public moves for policy conversion.
        ordered = [int(m) for m in ref[3:7] if m >= 0 and int(m) in moves]
        ordered += [m for m in moves if m not in ordered]
        if len(ordered) > 4:
            raise ValueError("native prediction contains more than four distinct moves")
        result[slot, 3:7] = ordered + [-1] * (4 - len(ordered))
        result[slot, 7] = NATURE_INDEX.get(to_id(source["nature"]), -1)
        result[slot, 8:14] = [int(v) // 4 if isinstance(v, (int, float)) else -1 for v in source["evs"]]
        result[slot, 14] = {**TYPE_INDEX, "stellar": 18}.get(to_id(source["tera_type"]), -1)
    return result


def import_priors(usage: dict) -> tuple[dict, dict]:
    priors = {name: {} for name in ("item_priors", "ability_priors", "move_priors", "tera_priors")}
    spreads = {}
    for name, data in usage["species"].items():
        sid = to_id(name)
        for native, target in (("items", "item_priors"), ("abilities", "ability_priors"),
                               ("moves", "move_priors"), ("tera_types", "tera_priors")):
            priors[target][sid] = {to_id(key): float(value) for key, value in data[native].items()
                                   if key not in {"Other", "Nothing"} and float(value) > 0}
        spreads[sid] = []
        for key, weight in data["spreads"].items():
            if ":" not in key or float(weight) <= 0:
                continue
            nature, evs = key.split(":")
            values = list(map(int, evs.split("/")))
            if len(values) == 6:
                spreads[sid].append({"nature": to_id(nature), "evs": values, "weight": float(weight)})
    return priors, spreads


def prepare(corpus: Path, out: Path) -> None:
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as stream:
        for line in corpus.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            replay = entry["replay"]
            teams = revealed_teams(replay)
            stream.write(json.dumps({"replay": replay, "fog_teams": {
                role: export_partial(team) for role, team in teams.items()}}) + "\n")


def evidence_counts(evidence: np.ndarray, truth: np.ndarray) -> dict:
    counts: Counter = Counter()
    for source, target in zip(evidence, truth):
        for name, cols in (("item", [1]), ("ability", [2]), ("tera", [14])):
            for col in cols:
                if source[col] >= 0:
                    counts[name + "_revealed"] += 1
                    counts[name + "_wrong"] += source[col] != target[col]
        moves = set(int(v) for v in source[3:7] if v >= 0)
        actual = set(int(v) for v in target[3:7] if v >= 0)
        counts["moves_revealed"] += len(moves)
        counts["moves_wrong"] += len(moves - actual)
    return {k: int(v) for k, v in counts.items()}


def paired_comparisons(per_game: list[dict], seed: int) -> dict:
    """Exploratory game bootstrap, conditional on this fixed team pool and controllers."""
    games = [g for g in per_game if g["split"] == "test" and g["all_methods_complete"]]
    if len(games) < 2:
        return {}
    indices = np.random.default_rng(seed).integers(len(games), size=(2000, len(games)))
    result = {}
    for treatment, baseline in ((METHODS[5], METHODS[0]), (METHODS[5], METHODS[4]),
                                (METHODS[3], METHODS[1])):
        metrics = {}
        for metric in ("hidden_move", "item", "joint_public", "joint_full"):
            denominator = "joint_count" if metric.startswith("joint_") else metric + "_count"
            values = []
            totals = []
            for method in (treatment, baseline):
                counts = [g["scores"][method]["counts"] for g in games]
                numerator = np.array([c.get(metric + "_correct", 0) for c in counts])
                divisor = np.array([c.get(denominator, 0) for c in counts])
                values.append(numerator[indices].sum(1) / np.maximum(1, divisor[indices].sum(1)))
                totals.append(float(numerator.sum() / max(1, divisor.sum())))
            lower, upper = np.quantile(values[0] - values[1], [0.025, 0.975])
            metrics[metric] = {"delta": totals[0] - totals[1], "lower": float(lower),
                               "upper": float(upper)}
        result[treatment + "_minus_" + baseline] = metrics
    return result


def score(corpus_path: Path, native_path: Path, bank_path: Path, out: Path, *, seed: int,
          policy_steps: int) -> dict:
    corpus = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines()]
    train = {h for entry in corpus if entry["split"] == "train" for h in entry["team_hashes"]}
    test = {h for entry in corpus if entry["split"] == "test" for h in entry["team_hashes"]}
    if train & test:
        raise ValueError("policy train/test teams overlap")
    native = {record["id"]: record for record in
              (json.loads(line) for line in native_path.read_text(encoding="utf-8").splitlines())}
    if set(native) != {entry["replay"]["id"] for entry in corpus}:
        raise ValueError("native outputs and benchmark games differ")
    bank = ReplayCandidateBank(bank_path)
    scores = {m: Counter() for m in METHODS}
    own_scores = {m: Counter() for m in METHODS}
    evidence = {m: Counter() for m in ("fog", "metamon")}
    failures, per_game, policy_records = [], [], []
    accepted_games, accepted_test_games = 0, 0
    usage_contexts = set()
    for index, entry in enumerate(corpus):
        replay = entry["replay"]
        rid = replay["id"]
        record = native[rid]
        revealed = revealed_teams(replay)
        truth = {r: align_truth(np.asarray(entry["truth"][r]), revealed[r]) for r in revealed}
        counts = {m: Counter() for m in METHODS}
        predictions = {m: {r: [] for r in revealed} for m in METHODS}
        expected_draws = len(record["fog_evidence_predictions"])
        success = {sample["draw"]: sample for sample in record["samples"]}
        complete = len(success) == expected_draws and expected_draws > 0
        failures.extend({"id": rid, **error} for error in record["errors"])
        for role_index, role in enumerate(("p1", "p2")):
            usage = record["usage"][role]
            usage_contexts.add(json.dumps({k: v for k, v in usage.items() if k not in ("species", "errors")},
                                          sort_keys=True))
            priors, spreads = import_priors(usage)
            if usage["errors"]:
                raise ValueError(f"native prior species lookup failed: {rid}, {usage['errors']}")
            if entry["split"] == "test":
                evidence["fog"].update(evidence_counts(revealed[role], truth[role]))
                if "native_revealed" in record:
                    evidence["metamon"].update(evidence_counts(
                        import_team(record["native_revealed"][role], revealed[role]), truth[role]))
            for draw in range(expected_draws):
                sample_seed = stable_replay_seed(rid, role, draw, seed)
                for method in METHODS[:4]:
                    tokens, _ = complete_private_team(
                        revealed[role], seed=sample_seed,
                        candidate_bank=bank if "candidates" in method else None,
                        replay_id=rid, log_hash=replay_digest(replay),
                        timestamp=int(replay["uploadtime"]), rating=int(replay.get("rating") or 0),
                        usage_priors=priors if method.endswith("native") else None,
                        spread_priors=spreads if method.endswith("native") else None,
                    )
                    predictions[method][role].append(tokens)
                external = record["fog_evidence_predictions"][draw].get(role)
                if external:
                    predictions[METHODS[4]][role].append(import_team(external, revealed[role]))
                else:
                    complete = False
                if draw in success:
                    predictions[METHODS[5]][role].append(import_team(
                        success[draw]["sides"][role_index]["predicted"], revealed[role]))
            for method in METHODS:
                counts[method].update(score_samples(predictions[method][role], truth[role], revealed[role]))
        if complete:
            accepted_games += 1
            if entry["split"] == "test":
                accepted_test_games += 1
                for method in METHODS:
                    scores[method].update(counts[method])
        if entry["split"] == "test":
            for method in METHODS:
                own_scores[method].update(counts[method])
        per_game.append({"id": rid, "split": entry["split"], "all_methods_complete": complete,
                         "native_accepted_draws": len(success), "requested_draws": expected_draws,
                         "scores": {m: summarize(c) for m, c in counts.items()}})
        if complete and policy_steps:
            first = {}
            for method in METHODS:
                first[method] = {}
                for role in revealed:
                    tokens = predictions[method][role][0]
                    provenance = np.where(tokens >= 0, 1, 0).astype(np.uint8)
                    exact = (revealed[role] >= 0) & (revealed[role] == tokens)
                    provenance[exact] = 2
                    first[method][role] = tokens, provenance
            policy_records.append((entry, truth, first))
        print(f"native comparison scored {index + 1}/{len(corpus)}", flush=True)
    report = {
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        "native_output_sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
        "bank_sha256": bank.sha256, "seed": seed,
        "native_source_revisions": sorted({r["revision"] for r in native.values()}),
        "native_poke_env_versions": sorted({r["poke_env"] for r in native.values()}),
        "requested_games": len(corpus), "all_methods_complete_games": accepted_games,
        "requested_test_games": sum(entry["split"] == "test" for entry in corpus),
        "all_methods_complete_test_games": accepted_test_games,
        "common_test_scores": {m: summarize(c) for m, c in scores.items()},
        "own_available_test_scores_not_paired": {m: summarize(c) for m, c in own_scores.items()},
        "public_evidence_audit_test": {m: dict(c) for m, c in evidence.items()},
        "native_usage_contexts": [json.loads(context) for context in sorted(usage_contexts)],
        "native_failures": failures, "per_game": per_game,
        "exploratory_paired_game_intervals": paired_comparisons(per_game, seed),
        "scope": "actual native forward/backward parser, ReplayPredictor, sequence checks; FOG-format scoring",
        "limitations": ["fixed scripted team pool; not human ladder accuracy",
                         "native dates/ranks/fallbacks retained, not silently updated",
                         "common-success subset may be selected; failures reported separately",
                         "policy pilot uses FOG state encoding for all arms, not Metamon's policy architecture"],
    }
    if policy_steps and policy_records:
        report["policy_pilot"] = policy_pilot(policy_records, list(METHODS), steps=policy_steps, seeds=(0, 1, 2))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("prepare", "score"))
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--native", type=Path)
    ap.add_argument("--bank", type=Path)
    ap.add_argument("--seed", type=int, default=19)
    ap.add_argument("--policy-steps", type=int, default=0)
    args = ap.parse_args()
    if args.mode == "prepare":
        prepare(args.corpus, args.out)
    else:
        if not args.native or not args.bank:
            ap.error("score requires --native and --bank")
        report = score(args.corpus, args.native, args.bank, args.out,
                       seed=args.seed, policy_steps=args.policy_steps)
        print(json.dumps({k: v for k, v in report.items() if k not in ("per_game", "native_failures")}, indent=2))


if __name__ == "__main__":
    main()
