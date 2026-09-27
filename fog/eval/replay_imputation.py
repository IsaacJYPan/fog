"""Known-sheet Showdown benchmark for replay imputation and a matched BC pilot.

Generate real public protocol logs from disjoint train/test team pools. All imputers receive only
the logs. Truth sheets are used for scoring and a common held-out policy input, never as donors.
The optional BC pilot measures held-out action NLL, not a claim of improved battle win rate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np

from fog.constants.tables import TABLES
from fog.training.replay_candidates import ReplayCandidateBank, revealed_teams, replay_digest
from fog.training.replay_policy_data import complete_private_team, stable_replay_seed


def generate_corpus(pool: Path, out: Path, *, games: int, seed: int, timestamp: int,
                    max_turns: int = 100) -> list[dict]:
    from fog.env.showdown_direct import ShowdownDirectClient
    from fog.env.showdown_local import LocalShowdownBattle
    from fog.env.showdown_local_runner import _choice
    from fog.env.team_truth import truth_tokens_from_paste
    from fog.env.teams import load_pool, to_packed
    from fog.training.replay_candidates import canonical_row

    if out.exists():
        raise FileExistsError(out)
    # Deduplicate complete teams independent of Pokemon and move ordering before splitting.
    teams = {}
    for paste in load_pool(pool):
        truth = truth_tokens_from_paste(paste, TABLES)
        # An absent item in a supplied paste is an exact no-item choice.
        truth[truth[:, 1] < 0, 1] = TABLES.item_index("<NO_ITEM>")
        identity = tuple(sorted(tuple(canonical_row(row)) for row in truth))
        teams.setdefault(identity, (paste, truth))
    members = list(teams.values())
    if len(members) < 4 or games < 4:
        raise ValueError("at least four distinct teams and four games are required")
    rng = random.Random(seed)
    rng.shuffle(members)
    cut = max(2, len(members) // 2)
    pools = {"train": members[:cut], "test": members[cut:]}
    if len(pools["test"]) < 2:
        raise ValueError("not enough held-out teams")
    result = []
    with ShowdownDirectClient() as client:
        for game in range(games):
            split = "test" if game % 3 == 0 else "train"
            p1, p2 = rng.sample(pools[split], 2)
            rid = f"gen9ou-imputation-{seed}-{game}"
            options = {"formatid": "gen9ou", "seed": f"sodium,{rng.getrandbits(256):064x}",
                       "strictChoices": True,
                       "p1": {"name": "ImputeA", "team": to_packed(p1[0])},
                       "p2": {"name": "ImputeB", "team": to_packed(p2[0])}}
            with LocalShowdownBattle.create(client, options, battle_tag=f"battle-{rid}") as local:
                lines = list(local.update.public_messages)
                for _ in range(1000):
                    if local.ended or local.p1.turn >= max_turns:
                        break
                    choices = [_choice("simpleheuristics", local.views[role],
                                       local.update.requests[role], rng) for role in ("p1", "p2")]
                    update = local.step(*choices, include_state=False)
                    if update.choice_errors:
                        raise RuntimeError(f"benchmark controller rejected: {update.choice_errors}")
                    lines.extend(update.public_messages)
                else:
                    raise RuntimeError("benchmark exceeded choice limit")
                entry = {
                    "replay": {"id": rid, "formatid": "gen9ou", "players": ["ImputeA", "ImputeB"],
                               "log": "\n".join(lines), "uploadtime": timestamp,
                               "rating": 0},  # scripted games have no ladder rating
                    "truth": {"p1": p1[1].tolist(), "p2": p2[1].tolist()}, "split": split,
                    "truncated": not local.ended,
                    "team_hashes": [hashlib.sha256(t.tobytes()).hexdigest() for t in (p1[1], p2[1])],
                }
                result.append(entry)
            print(f"benchmark collection: {game + 1}/{games}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(entry) for entry in result), encoding="utf-8")
    return result


def align_truth(truth: np.ndarray, revealed: np.ndarray) -> np.ndarray:
    """Align sheets by species and preserve public move order without consulting hidden values."""
    aligned = np.full_like(revealed, -1)
    for slot, row in enumerate(revealed):
        match = truth[truth[:, 0] == row[0]]
        if len(match) != 1:
            raise ValueError("truth/preview species alignment is ambiguous")
        aligned[slot] = match[0]
        moves = [int(m) for m in row[3:7] if m >= 0]
        moves += [int(m) for m in match[0, 3:7] if m >= 0 and int(m) not in moves]
        if len(moves) > 4 or not set(row[3:7][row[3:7] >= 0]).issubset(set(match[0, 3:7])):
            raise ValueError("public moves contradict submitted sheet")
        aligned[slot, 3:7] = moves + [-1] * (4 - len(moves))
        known = row >= 0
        if not np.array_equal(aligned[slot, known], row[known]):
            raise ValueError("public scalar evidence contradicts submitted sheet")
    return aligned


def score_samples(samples: list[np.ndarray], truth: np.ndarray, revealed: np.ndarray) -> dict:
    totals: Counter = Counter()
    for slot, target in enumerate(truth):
        if target[0] < 0:
            continue
        observed = revealed[slot]
        hidden = (observed < 0) & (target >= 0)
        target_moves = set(int(m) for m in target[3:7] if m >= 0)
        hidden_moves = target_moves - set(int(m) for m in observed[3:7] if m >= 0)
        any_exact = False
        for sample in samples:
            row = sample[slot]
            moves = set(int(m) for m in row[3:7] if m >= 0)
            for name, cols in {"item": [1], "ability": [2], "nature": [7],
                               "ev": list(range(8, 14)), "tera": [14]}.items():
                selected = [c for c in cols if hidden[c]]
                totals[name + "_correct"] += sum(row[c] == target[c] for c in selected)
                totals[name + "_count"] += len(selected)
            totals["hidden_move_correct"] += len(moves & hidden_moves)
            totals["hidden_move_count"] += len(hidden_moves)
            scalar_cols = [c for c in (1, 2, *range(7, 15)) if target[c] >= 0]
            full = moves == target_moves and all(row[c] == target[c] for c in scalar_cols)
            public_cols = [c for c in (1, 2, 14) if target[c] >= 0]
            joint = moves == target_moves and all(row[c] == target[c] for c in public_cols)
            if hidden.any():
                totals["joint_public_correct"] += joint
                totals["joint_full_correct"] += full
                totals["joint_count"] += 1
                any_exact |= full
        if hidden.any():
            totals["full_recall_at_k_correct"] += any_exact
            totals["full_recall_at_k_count"] += 1
    return dict(totals)


def summarize(counts: dict) -> dict:
    counts = {key: int(value) for key, value in counts.items()}
    metrics = {}
    for name in ("item", "ability", "nature", "ev", "tera", "hidden_move", "full_recall_at_k"):
        metrics[name] = counts.get(name + "_correct", 0) / max(1, counts.get(name + "_count", 0))
    for name in ("joint_public", "joint_full"):
        metrics[name] = counts.get(name + "_correct", 0) / max(1, counts.get("joint_count", 0))
    return {"metrics": metrics, "counts": counts}


def paired_intervals(per_game: list[dict], *, seed: int) -> dict:
    """Paired bootstrap over games, never over correlated slots or imputation draws.

    Conditional on the fixed team pool and scripted controllers; not population-level human
    ladder uncertainty. Report null if the test split is too small or has no scored fields.
    """
    games = [entry for entry in per_game if entry["split"] == "test"]
    if len(games) < 2:
        return {}
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(games), size=(2000, len(games)))
    comparisons = {}
    for baseline in games[0]["scores"]:
        if baseline == "replay_candidates":
            continue
        intervals = {}
        for metric, denominator in (("hidden_move", "hidden_move_count"),
                                    ("joint_public", "joint_count"),
                                    ("joint_full", "joint_count"), ("item", "item_count")):
            arrays = []
            for method in ("replay_candidates", baseline):
                arrays.append(np.asarray([[g["scores"][method]["counts"].get(metric + "_correct", 0),
                                           g["scores"][method]["counts"].get(denominator, 0)]
                                          for g in games], dtype=float))
            differences = []
            for array in arrays:
                totals = array[indices].sum(axis=1)
                differences.append(totals[:, 0] / np.maximum(1, totals[:, 1]))
            delta = differences[0] - differences[1]
            intervals[metric] = {"low": float(np.quantile(delta, 0.025)),
                                 "high": float(np.quantile(delta, 0.975))}
        comparisons[baseline] = intervals
    return {"scope": "paired game bootstrap, conditional on fixed team pool; exploratory 95% intervals",
            "comparisons": comparisons}


def evaluate(corpus: list[dict], bank: ReplayCandidateBank, *, samples: int, seed: int,
             metamon=None, policy_steps: int = 0, policy_seeds: tuple[int, ...] = (0,)) -> dict:
    if samples < 1:
        raise ValueError("samples must be positive")
    methods = ["usage", "replay_candidates"] + (["metamon_core_shared_priors"] if metamon else [])
    scores = {method: Counter() for method in methods}
    diagnostics: dict = {}
    per_game, failures, policy_records = [], [], []
    # Generated training and test team identities must be disjoint before any policy fitting.
    train = {h for entry in corpus if entry["split"] == "train" for h in entry["team_hashes"]}
    test = {h for entry in corpus if entry["split"] == "test" for h in entry["team_hashes"]}
    if train & test:
        raise ValueError("policy train/test teams overlap")
    for index, entry in enumerate(corpus):
        replay = entry["replay"]
        rid = replay["id"]
        try:
            revealed = revealed_teams(replay)
            truth = {role: align_truth(np.asarray(entry["truth"][role]), revealed[role])
                     for role in ("p1", "p2")}
        except ValueError as error:
            failures.append({"id": rid, "reason": str(error)})
            continue
        first = {method: {} for method in methods}
        game_scores = {method: Counter() for method in methods}
        for role in ("p1", "p2"):
            predictions = {method: [] for method in methods}
            for draw in range(samples):
                draw_seed = stable_replay_seed(rid, role, draw, seed)
                for method in methods:
                    if method == "metamon_core_shared_priors":
                        tokens, provenance = metamon.complete(revealed[role], seed=draw_seed)
                    else:
                        tokens, provenance = complete_private_team(
                            revealed[role], seed=draw_seed,
                            candidate_bank=bank if method == "replay_candidates" else None,
                            replay_id=rid, log_hash=replay_digest(replay),
                            timestamp=int(replay["uploadtime"]), rating=int(replay.get("rating") or 0),
                            diagnostics=diagnostics if method == "replay_candidates" else None,
                        )
                    predictions[method].append(tokens)
                    if draw == 0:
                        first[method][role] = (tokens, provenance)
            for method in methods:
                game_scores[method].update(score_samples(predictions[method], truth[role], revealed[role]))
        # Reconstruction headline uses test games only. Preserve all individual games for paired
        # analysis and report the denominator rather than treating imputed samples as new games.
        if entry["split"] == "test":
            for method in methods:
                scores[method].update(game_scores[method])
        per_game.append({"id": rid, "split": entry["split"],
                         "scores": {m: summarize(c) for m, c in game_scores.items()}})
        if policy_steps:
            policy_records.append((entry, truth, first))
        print(f"reconstruction evaluation: {index + 1}/{len(corpus)}", flush=True)
    report = {"schema_version": 1, "candidate_bank_sha256": bank.sha256, "seed": seed,
              "samples_per_team": samples, "independent_games": len(per_game),
              "test_games": sum(e["split"] == "test" for e in per_game),
              "truncated_games": sum(e["truncated"] for e in corpus),
              "failed_games": failures, "metrics_scope": "held_out_test_games",
              "methods": {m: summarize(c) for m, c in scores.items()},
              "donor_coverage": diagnostics, "per_game": per_game,
              "metamon": (metamon.metadata | {"coverage": metamon.coverage} if metamon
                          else {"status": "not_run"}),
              "paired_deltas": paired_intervals(per_game, seed=seed),
              "limitations": ["scripted opponents; small controlled team pool",
                              "candidate weights are heuristic, not calibrated probabilities",
                              "usage fallback uses fixed priors; historical donor exclusion alone is not full temporal isolation",
                              "no IV accuracy: FOG submitted-set schema does not encode IVs"]}
    if policy_steps:
        report["policy_pilot"] = policy_pilot(policy_records, methods, steps=policy_steps,
                                               seeds=policy_seeds)
    else:
        report["policy_pilot"] = {"status": "not_run"}
    return report


def policy_pilot(records: list, methods: list[str], *, steps: int, seeds: tuple[int, ...]) -> dict:
    """Equal updates, initialization and common decisions; test on oracle-sheet reconstructions."""
    import torch

    from fog.models.agent import FogAgent
    from fog.scripts.parse_replays import parse_replay, rows_to_arrays
    from fog.training.dataset import _build_batch

    torch.set_num_threads(2)
    training = {method: [] for method in methods}
    testing = []
    for entry, truth, first in records:
        variants = {}
        for method in (methods if entry["split"] == "train" else ["oracle"]):
            sets = truth if method == "oracle" else {r: first[method][r][0] for r in truth}
            provenance = ({r: np.where(t >= 0, 2, 0).astype(np.uint8) for r, t in truth.items()}
                          if method == "oracle" else {r: first[method][r][1] for r in truth})
            rows = parse_replay(entry["replay"], private_sets=sets,
                                private_provenance=provenance)["rows"]
            variants[method] = {(r["role"], r["turn"]): r for r in rows
                                if r["policy_valid"] and r["mask"].sum() >= 2}
        common = set.intersection(*(set(rows) for rows in variants.values()))
        for method, rows in variants.items():
            selected = [rows[key] for key in sorted(common)]
            if method == "oracle":
                testing.extend(selected)
            else:
                training[method].extend(selected)
    if not testing or not all(training.values()):
        return {"status": "insufficient_common_decisions"}
    # Small actual FOG policy, without auxiliary losses, isolates the reconstruction intervention.
    cfg = {"encoding": {"history_encoder": "gru", "state_dim": 64, "dyn_dim": 64,
                        "current_dim": 64, "history_dim": 64, "entity_hidden": 64},
           "model": {"history_gru_hidden": 64, "belief_dim": 32,
                     "dropout": 0.0}}
    arrays = {m: rows_to_arrays(rows) for m, rows in training.items()}
    test_arrays = rows_to_arrays(testing)
    results = {}
    for method in methods:
        runs = []
        for seed in seeds:
            torch.manual_seed(seed)
            agent = FogAgent.from_tables(cfg)
            optimizer = torch.optim.AdamW(agent.parameters(), lr=3e-4)
            rng = np.random.default_rng(seed)
            agent.train()
            for _ in range(steps):
                indices = rng.choice(len(training[method]), size=min(32, len(training[method])),
                                     replace=False)
                batch = _build_batch(arrays[method], indices, TABLES)
                output = agent(batch)
                loss = -output.log_probs.gather(1, batch["action"][:, None]).mean()
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite policy pilot loss")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()
            agent.eval()
            nll, correct, count = 0.0, 0, 0
            with torch.no_grad():
                for start in range(0, len(testing), 32):
                    batch = _build_batch(test_arrays, np.arange(start, min(start + 32, len(testing))), TABLES)
                    output = agent(batch)
                    nll += float(-output.log_probs.gather(1, batch["action"][:, None]).sum())
                    correct += int((output.log_probs.argmax(-1) == batch["action"]).sum())
                    count += len(batch["action"])
            runs.append({"seed": seed, "action_nll": nll / count, "action_top1": correct / count})
            print(f"policy pilot: {method}, seed {seed}, NLL {nll / count:.4f}", flush=True)
        results[method] = runs
    return {"status": "complete", "steps": steps, "train_rows": len(next(iter(training.values()))),
            "test_rows": len(testing), "results": results, "model_config": cfg,
            "scope": "small FOG BC pilot, common decisions, oracle-sheet parsed test inputs; not win rate"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--generate-pool", type=Path)
    parser.add_argument("--games", type=int, default=24)
    parser.add_argument("--timestamp", type=int, default=1789862400)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--metamon-source", type=Path)
    parser.add_argument("--policy-steps", type=int, default=0)
    parser.add_argument("--policy-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    corpus = (generate_corpus(args.generate_pool, args.corpus, games=args.games, seed=args.seed,
                              timestamp=args.timestamp) if args.generate_pool else
              [json.loads(line) for line in args.corpus.read_text(encoding="utf-8").splitlines()])
    metamon = None
    if args.metamon_source:
        from fog.eval.metamon_imputation_baseline import MetamonSharedPriorBaseline

        metamon = MetamonSharedPriorBaseline(args.metamon_source)
    report = evaluate(corpus, ReplayCandidateBank(args.bank), samples=args.samples, seed=args.seed,
                      metamon=metamon, policy_steps=args.policy_steps,
                      policy_seeds=tuple(args.policy_seeds))
    report["corpus_sha256"] = hashlib.sha256(args.corpus.read_bytes()).hexdigest()
    report["python_hash_seed"] = os.environ.get("PYTHONHASHSEED", "unset")
    from fog.constants.tables import GENERATED_DIR

    report["fallback_priors_sha256"] = {
        name: hashlib.sha256((GENERATED_DIR / name).read_bytes()).hexdigest()
        for name in ("priors.json", "spreads.json")
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_game"}, indent=2))


if __name__ == "__main__":
    main()
