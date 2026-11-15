"""Check whether replay rows, and a policy trained on them, depend on move-slot position.

Legacy replay rows put moves in first-use order followed by sampled padding. Current conversion
shuffles the completed moves and public preview order independently of the decisions. This
diagnostic accepts either corpus and measures the remaining dependence.

Two measurements:

* **rows**: how often the label is the position-0 move on rows where the acting Pokemon has not
  used a move yet (0.25 to 0.5 would be unremarkable; near 1.0 is the leak), the share of labels
  per position, and two other differences from live play (own PP, replacement decisions).
* **policy** (with ``--checkpoint``): every row is scored as stored and again with the acting
  Pokemon's four move slots shuffled. Ids, PP, legality, label and provenance move together, so
  nothing a player could know changes. On rows where the Pokemon has not moved yet no history
  entry refers to its slots, so there the comparison is exact.

    python -m fog.eval.replay_slot_order_check --config configs/replay_blueprint_cpu.yaml \
        --manifest runs/blueprint/split/holdout_manifest.parquet \
        --checkpoint runs/blueprint/training/checkpoints/bc_best.pt --out slot-order.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fog.encoding.entities import L
from fog.encoding.events import H_MY_MOVE, H_START_MY_SPECIES

_SLOT_KEYS = ("legality_mask", "legal_provenance", "action_feature_provenance")


def not_yet_moved(batch: dict, n_moves: int) -> torch.Tensor:
    """Rows whose acting Pokemon has used no move on any earlier turn in the stored history."""
    history_ids = batch["history_ids"]
    turns = history_ids.shape[1]
    length = batch["history_len"].clamp(max=turns)
    real = torch.arange(turns, device=history_ids.device).unsqueeze(0) >= (turns - length).unsqueeze(1)
    rows = torch.arange(history_ids.shape[0], device=history_ids.device)
    mine = batch["active_idx"][:, 0].clamp(0, 5)
    species = batch["entity_ids"][rows, mine, 0].unsqueeze(1)
    moved = (
        real
        & (history_ids[..., H_START_MY_SPECIES] == species)
        & (history_ids[..., H_MY_MOVE] != n_moves)
    )
    return ~moved.any(dim=1)


def shuffle_move_slots(batch: dict, generator: torch.Generator) -> dict:
    """Return a copy of ``batch`` with the acting Pokemon's move slots permuted per row."""
    out = {key: (value.clone() if isinstance(value, torch.Tensor) else value)
           for key, value in batch.items()}
    count = out["action"].shape[0]
    rows = torch.arange(count)
    mine = out["active_idx"][:, 0].clamp(0, 5)
    # New slot k takes what old slot perm[k] held.
    perm = torch.stack([torch.randperm(4, generator=generator) for _ in range(count)])
    inverse = torch.argsort(perm, dim=1)
    ids = out["entity_ids"][rows, mine]
    ids[:, 3:7] = ids[:, 3:7].gather(1, perm)
    out["entity_ids"][rows, mine] = ids
    raw = out["entities"][rows, mine]
    raw[:, L.move_pp : L.move_pp + 4] = raw[:, L.move_pp : L.move_pp + 4].gather(1, perm)
    raw[:, L.epistemic + 2 : L.epistemic + 6] = raw[
        :, L.epistemic + 2 : L.epistemic + 6
    ].gather(1, perm)
    out["entities"][rows, mine] = raw
    provenance = out.get("own_set_provenance")
    if isinstance(provenance, torch.Tensor):
        moves = provenance[rows, mine, 3:7].gather(1, perm)
        provenance[rows, mine, 3:7] = moves
    features = out.get("action_features")
    if isinstance(features, torch.Tensor):
        feature_perm = perm.unsqueeze(-1).expand(-1, -1, features.shape[-1])
        features[:, :4] = features[:, :4].gather(1, feature_perm)
        features[:, 4:8] = features[:, 4:8].gather(1, feature_perm)
    for key in _SLOT_KEYS:
        table = out.get(key)
        if isinstance(table, torch.Tensor):
            table[:, 0:4] = table[:, 0:4].gather(1, perm)
            table[:, 4:8] = table[:, 4:8].gather(1, perm)
    action = out["action"]
    base = torch.where(action >= 4, 4, 0)
    moved = base + inverse.gather(1, (action % 4).unsqueeze(1)).squeeze(1)
    out["action"] = torch.where(action < 8, moved, action)
    return out


def lead_order_summary(manifest, raw: Path | None = None) -> dict:
    """Use converter metadata or original public logs, never a later retained decision's active."""
    from fog.training.replay_policy_data import replay_lead_positions

    positions = []
    for row in manifest.iter_rows(named=True):
        if "lead_p1" in row and "lead_p2" in row:
            positions.append([row["lead_p1"], row["lead_p2"]])
        elif raw is not None:
            replay = json.loads((raw / f"{row['id']}.json").read_text(encoding="utf-8"))
            lead = replay_lead_positions(replay)
            positions.append([lead["p1"], lead["p2"]])
        else:
            positions.append([None, None])
    result = {"games": len(positions)}
    for side, role in enumerate(("p1", "p2")):
        counts = [sum(p[side] == slot for p in positions) for slot in range(6)]
        observed = sum(counts)
        result[role] = {
            "known_leads": observed, "unknown_leads": len(positions) - observed,
            "counts_by_team_position": counts,
            "share_by_team_position": [n / observed for n in counts] if observed else None,
            "share_position0": counts[0] / observed if observed else None,
        }
    paired = [p for p in positions if all(x is not None for x in p)]
    result["games_with_both_leads_known"] = len(paired)
    result["share_games_either_lead_position0"] = (
        sum(0 in p for p in paired) / len(paired) if paired else None
    )
    result["share_games_both_leads_position0"] = (
        sum(p == [0, 0] for p in paired) / len(paired) if paired else None
    )
    result["source"] = "converter metadata" if "lead_p1" in manifest.columns else (
        "raw public logs" if raw is not None else "unavailable; pass --raw for legacy manifests"
    )
    return result


def game_resampled_intervals(totals: np.ndarray, statistic, *, samples: int, seed: int) -> dict:
    """Percentile intervals for a statistic of additive per-game totals, including both sides.

    Resampling preserves every correlated decision from a selected game. The point estimate is
    row-weighted; game lengths remain part of the statistic rather than becoming new labels.
    """
    if samples < 0:
        raise ValueError("bootstrap samples must be nonnegative")
    if len(totals) < 2 or not samples:
        return {}
    rng = np.random.default_rng(seed)
    estimates: dict[str, list[float]] = {}
    for _ in range(samples):
        draw = totals[rng.integers(len(totals), size=len(totals))].sum(axis=0)
        for name, value in statistic(draw).items():
            if value is not None and np.isfinite(value):
                estimates.setdefault(name, []).append(value)
    return {name: np.quantile(values, [0.025, 0.975]).tolist()
            for name, values in estimates.items()}


def _policy_statistic(total: np.ndarray) -> dict:
    return {name: float(total[i] / total[0]) if total[0] else None
            for i, name in enumerate(("rows", "nll_stored", "nll_shuffled",
                                      "top1_stored", "top1_shuffled")) if i}


def _value_statistic(total: np.ndarray) -> dict:
    count = total[0]
    return {
        "brier": float(total[1] / count) if count else None,
        "ece": float(np.abs(total[12:22] - total[22:32]).sum() / count) if count else None,
    }


def check(cfg: dict, manifest: Path, checkpoint: Path | None, *, batch_size: int = 256,
          seed: int = 0, raw: Path | None = None, bootstrap_samples: int = 2000) -> dict:
    import polars as pl

    from fog.constants.tables import TABLES
    from fog.models.agent import FogAgent
    from fog.training.bc import _policy_supervision_mask
    from fog.training.dataset import StreamingReplayDataset

    table = pl.read_parquet(manifest)
    counts = table[next(c for c in ("n_rows", "rows") if c in table.columns)].to_numpy()
    game_of_row = np.repeat(np.arange(len(counts)), counts)
    agent = None
    if checkpoint is not None:
        agent = FogAgent.from_tables(cfg)
        agent.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        agent.eval()
    generator = torch.Generator().manual_seed(seed)
    groups = ("all", "move", "not_yet_moved")
    scores = {name: np.zeros((len(table), 5), dtype=np.float64) for name in groups}
    # rows, Brier sum, ten calibration-bin counts, prediction sums and outcome sums.
    value_totals = np.zeros((len(table), 32), dtype=np.float64)
    slot_counts = [0, 0, 0, 0]
    switch_counts = [0] * 5
    data = {"rows": 0, "move_rows": 0, "not_yet_moved_rows": 0, "not_yet_moved_label_slot0": 0,
            "switch_only_rows": 0, "largest_own_pp": 0.0}
    favourite = {"stored_slot0": 0, "shuffled_slot0": 0}
    offset = 0
    with torch.no_grad():
        for batch in StreamingReplayDataset(manifest).batches(
            batch_size, shuffle=False, seed=0, device="cpu"
        ):
            game = game_of_row[offset : offset + len(batch["action"])]
            offset += len(game)
            valid = _policy_supervision_mask(batch, cfg.get("loss", {}))
            action = batch["action"]
            is_move = action < 8
            first = is_move & not_yet_moved(batch, TABLES.n_moves)
            legal = batch["legality_mask"]
            data["rows"] += int(valid.sum())
            data["move_rows"] += int((valid & is_move).sum())
            data["not_yet_moved_rows"] += int((valid & first).sum())
            data["not_yet_moved_label_slot0"] += int((valid & first & (action % 4 == 0)).sum())
            data["switch_only_rows"] += int((~legal[:, :8].any(1) & legal[:, 8:13].any(1)).sum())
            data["largest_own_pp"] = max(
                data["largest_own_pp"],
                float(batch["entities"][:, :6, L.move_pp : L.move_pp + 4].abs().max()),
            )
            for slot in range(4):
                slot_counts[slot] += int((valid & is_move & (action % 4 == slot)).sum())
            for slot in range(5):
                switch_counts[slot] += int((valid & (action == 8 + slot)).sum())
            if agent is None:
                continue
            shuffled = shuffle_move_slots(batch, generator)
            output = agent(batch, need_belief=False)
            stored_lp = output.log_probs
            shuffled_lp = agent(shuffled, need_belief=False).log_probs
            p = ((output.value.reshape(-1).double().numpy() + 1) / 2).clip(0, 1)
            target = ((batch["outcome"].double().numpy() + 1) / 2).clip(0, 1)
            bins = np.minimum((p * 10).astype(int), 9)
            np.add.at(value_totals[:, 0], game, 1)
            np.add.at(value_totals[:, 1], game, (p - target) ** 2)
            np.add.at(value_totals, (game, 2 + bins), 1)
            np.add.at(value_totals, (game, 12 + bins), p)
            np.add.at(value_totals, (game, 22 + bins), target)
            for name, chosen in (("all", valid), ("move", valid & is_move),
                                 ("not_yet_moved", valid & first)):
                if not bool(chosen.any()):
                    continue
                score = scores[name]
                selected_games = game[chosen.numpy()]
                observations = torch.stack([
                    torch.ones_like(action[chosen]),
                    -stored_lp[chosen].gather(1, action[chosen].unsqueeze(1)).squeeze(1),
                    -shuffled_lp[chosen].gather(
                        1, shuffled["action"][chosen].unsqueeze(1)).squeeze(1),
                    stored_lp[chosen].argmax(1) == action[chosen],
                    shuffled_lp[chosen].argmax(1) == shuffled["action"][chosen],
                ], dim=1).double().numpy()
                np.add.at(score, selected_games, observations)
            chosen = valid & first
            if bool(chosen.any()):
                favourite["stored_slot0"] += int(
                    (stored_lp[chosen][:, :8].argmax(1) % 4 == 0).sum())
                favourite["shuffled_slot0"] += int(
                    (shuffled_lp[chosen][:, :8].argmax(1) % 4 == 0).sum())

    if offset != len(game_of_row):
        raise ValueError("manifest row counts disagree with stored replay shards")
    moves = max(1, data["move_rows"])
    first_rows = max(1, data["not_yet_moved_rows"])
    report: dict = {
        "smoke": True,
        "manifest": str(manifest),
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "rows": data["rows"],
        "move_rows": data["move_rows"],
        "label_share_by_move_slot": [count / moves for count in slot_counts],
        "not_yet_moved_rows": data["not_yet_moved_rows"],
        "not_yet_moved_label_is_slot0": data["not_yet_moved_label_slot0"] / first_rows,
        "switch_only_rows": data["switch_only_rows"],
        "largest_own_pp": data["largest_own_pp"],
        "switch_rows": sum(switch_counts),
        "switch_label_counts_by_bench_position": switch_counts,
        "label_share_by_bench_position": (
            [count / sum(switch_counts) for count in switch_counts]
            if sum(switch_counts) else None
        ),
        "lead_order": lead_order_summary(table, raw),
        "not_yet_moved_definition": "no move by the acting species in the stored history window",
    }
    if agent is not None:
        policy = {}
        for name, score in scores.items():
            total = score.sum(axis=0)
            policy[name] = {"rows": int(total[0]), **_policy_statistic(total)}
            policy[name]["interval95"] = game_resampled_intervals(
                score, _policy_statistic, samples=bootstrap_samples, seed=seed,
            )
        policy["not_yet_moved"]["favourite_move_in_slot0_stored"] = (
            favourite["stored_slot0"] / first_rows)
        policy["not_yet_moved"]["favourite_move_in_slot0_shuffled"] = (
            favourite["shuffled_slot0"] / first_rows)
        report["policy"] = policy
        report["value"] = {
            "rows": int(value_totals[:, 0].sum()), **_value_statistic(value_totals.sum(0)),
            "interval95": game_resampled_intervals(
                value_totals, _value_statistic, samples=bootstrap_samples, seed=seed,
            ),
        }
        report["interval_method"] = {
            "unit": "game (both perspectives together)", "samples": bootstrap_samples,
            "seed": seed, "method": "percentile bootstrap, row-weighted statistic",
        }
    return report


def main() -> None:
    from fog.config import load_config_with_args

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--raw", type=Path, help="original public logs for legacy lead metadata")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    args = parser.parse_args()
    cfg = load_config_with_args(args.config, args.overrides)
    report = check(cfg, args.manifest, args.checkpoint, batch_size=args.batch_size,
                   seed=args.seed, raw=args.raw, bootstrap_samples=args.bootstrap_samples)
    text = json.dumps(report, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
