"""Paired holdout evaluation for brief 12, with new replacement rows scored separately."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def decision_keys(arrays: dict, metadata: dict) -> dict:
    from fog.training.dataset import _legacy_decision_phases

    phases = metadata.get("row_phase", arrays.get(
        "decision_phase", _legacy_decision_phases(arrays["legality_mask"])))
    counts: Counter = Counter()
    result = {}
    for i, (role, turn, phase) in enumerate(zip(metadata["row_role"], metadata["row_turn"], phases)):
        prefix = (int(role), int(turn), int(phase))
        key = (*prefix, counts[prefix])
        counts[prefix] += 1
        result[key] = i
    return result


def common_indices(old, new, old_meta) -> tuple:
    # The old converter accepted only start-of-turn awaiting_action requests. Its model
    # inferred phase from masks, which can misclassify a normal Struggle request. Align the
    # actual request here while preserving that old inferred phase in the model input.
    old_meta = {**old_meta, "row_phase": np.zeros(len(old["action"]), dtype=np.int64)}
    left, right = decision_keys(old, old_meta), decision_keys(new, new)
    common = sorted(k for k in left.keys() & right.keys() if k[2] == 0)
    label_matches = [k for k in common if old["action"][left[k]] == new["action"][right[k]]]

    def eligible(arrays, i):
        action = int(arrays["action"][i])
        return (bool(arrays["policy_valid"][i])
                and arrays["legal_provenance"][i, action] == 2
                and arrays["action_feature_provenance"][i, action] == 2)

    same = [k for k in label_matches if eligible(old, left[k]) and eligible(new, right[k])]
    replacements = [i for k, i in right.items() if k[2] == 1 and k not in left]
    return (np.asarray([left[k] for k in same], dtype=np.int64),
            np.asarray([right[k] for k in same], dtype=np.int64),
            np.asarray(replacements, dtype=np.int64), len(common) - len(label_matches))


def interval(rows: list[list[float]], seed: int = 12000) -> dict:
    """Resample games, retaining the correlation among all decisions of a game."""
    values = np.asarray(rows, dtype=float)
    totals = values.sum(axis=0)
    if totals[0] == 0:
        return {"rows": 0}
    rng = np.random.default_rng(seed)
    draws = np.asarray([values[rng.integers(0, len(values), len(values))].sum(axis=0)
                        for _ in range(1000)])
    return {"games": int((values[:, 0] > 0).sum()), "rows": int(totals[0]),
            "nll": float(totals[1] / totals[0]), "top1": float(totals[2] / totals[0]),
            "nll_95_game_bootstrap": np.quantile(draws[:, 1] / draws[:, 0], [.025, .975]).tolist(),
            "top1_95_game_bootstrap": np.quantile(draws[:, 2] / draws[:, 0], [.025, .975]).tolist()}


def score(agent, arrays, indices, cfg, batch_size=256) -> list:
    import torch

    from fog.constants.tables import TABLES
    from fog.training.bc import _policy_supervision_mask
    from fog.training.dataset import _build_batch

    total = np.zeros(3, dtype=float)
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch = _build_batch(arrays, indices[start:start + batch_size], TABLES, "cpu")
            valid = _policy_supervision_mask(batch, cfg["loss"])
            output = agent(batch, need_belief=False)
            nll = -output.log_probs.gather(-1, batch["action"][:, None]).squeeze(-1)
            if not torch.isfinite(nll[valid]).all():
                raise ValueError("non-finite holdout action likelihood")
            correct = output.log_probs.argmax(-1) == batch["action"]
            total += [int(valid.sum()), float(nll[valid].sum()), int(correct[valid].sum())]
    return total.tolist()


def load_arrays(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def main() -> None:
    import polars as pl
    import torch

    from fog.config import load_config
    from fog.training.search_distill import build_agent

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-manifest", type=Path, required=True)
    parser.add_argument("--new-manifest", type=Path, required=True)
    parser.add_argument("--old-metadata", type=Path, required=True)
    parser.add_argument("--old-checkpoint", type=Path, required=True)
    parser.add_argument("--new-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/replay_mechanical_cpu.yaml"))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    cfg = load_config(args.config)
    agents = {"old": build_agent(cfg, str(args.old_checkpoint), 20261201),
              "new": build_agent(cfg, str(args.new_checkpoint), 20261201)}
    old = {r["id"]: r for r in pl.read_parquet(args.old_manifest).to_dicts()}
    new = {r["id"]: r for r in pl.read_parquet(args.new_manifest).to_dicts()}
    if old.keys() != new.keys():
        raise ValueError("paired holdouts must contain exactly the same games")
    results = {f"{model}_model_{view}_rows": [] for model in agents for view in agents}
    results.update({f"{model}_model_new_replacements": [] for model in agents})
    results.update({f"{model}_model_nontrivial_replacements": [] for model in agents})
    per_game = []
    for game, replay_id in enumerate(sorted(old), 1):
        first, second = load_arrays(old[replay_id]["shard"]), load_arrays(new[replay_id]["shard"])
        meta = load_arrays(args.old_metadata / f"{replay_id}.npz")
        left, right, forced, disagreements = common_indices(first, second, meta)
        for model, agent in agents.items():
            for view, arrays, indices in (("old", first, left), ("new", second, right)):
                results[f"{model}_model_{view}_rows"].append(score(agent, arrays, indices, cfg))
            results[f"{model}_model_new_replacements"].append(score(agent, second, forced, cfg))
            nontrivial = forced[second["legality_mask"][forced].sum(axis=1) > 1]
            results[f"{model}_model_nontrivial_replacements"].append(
                score(agent, second, nontrivial, cfg))
        per_game.append({"id": replay_id, "common_normal_rows": len(left),
                         "new_replacement_rows": len(forced), "label_disagreements": disagreements})
        if game % 100 == 0:
            print(f"paired holdout {game}/{len(old)}", flush=True)
    report = {"kind": "smoke", "model_config": str(args.config), "seed": 20261201,
              "games": len(old), "results": {key: interval(value) for key, value in results.items()},
              "per_game": per_game, "game_sums": results,
              "checkpoint_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                                    for name, path in (("old", args.old_checkpoint),
                                                       ("new", args.new_checkpoint))}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["results"], indent=2))


if __name__ == "__main__":
    main()
