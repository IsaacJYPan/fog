"""Check whether replay rows, and a policy trained on them, depend on move-slot position.

The converter fills the acting player's moveset with the moves that player revealed during the
whole game, in the order they were first used, followed by sampled moves. The position of a move
in the moveset therefore says something about the label that a player at the table could not
know: the first move a Pokemon ever uses is always in position 0, and sampled moves, which are
never the label, always come last.

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
    out["entities"][rows, mine] = raw
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


def check(cfg: dict, manifest: Path, checkpoint: Path | None, *, batch_size: int = 256,
          seed: int = 0) -> dict:
    from fog.constants.tables import TABLES
    from fog.models.agent import FogAgent
    from fog.training.bc import _policy_supervision_mask
    from fog.training.dataset import ReplayDataset

    agent = None
    if checkpoint is not None:
        agent = FogAgent.from_tables(cfg)
        agent.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        agent.eval()
    generator = torch.Generator().manual_seed(seed)
    groups = ("all", "move", "not_yet_moved")
    scores = {name: {"rows": 0, "nll_stored": 0.0, "nll_shuffled": 0.0,
                     "top1_stored": 0, "top1_shuffled": 0} for name in groups}
    slot_counts = [0, 0, 0, 0]
    data = {"rows": 0, "move_rows": 0, "not_yet_moved_rows": 0, "not_yet_moved_label_slot0": 0,
            "switch_only_rows": 0, "largest_own_pp": 0.0}
    favourite = {"stored_slot0": 0, "shuffled_slot0": 0}
    with torch.no_grad():
        for batch in ReplayDataset(manifest).batches(batch_size, shuffle=False, seed=0,
                                                     device="cpu"):
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
            if agent is None:
                continue
            shuffled = shuffle_move_slots(batch, generator)
            stored_lp = agent(batch, need_belief=False).log_probs
            shuffled_lp = agent(shuffled, need_belief=False).log_probs
            for name, chosen in (("all", valid), ("move", valid & is_move),
                                 ("not_yet_moved", valid & first)):
                if not bool(chosen.any()):
                    continue
                score = scores[name]
                score["rows"] += int(chosen.sum())
                score["nll_stored"] += float(
                    -stored_lp[chosen].gather(1, action[chosen].unsqueeze(1)).sum())
                score["nll_shuffled"] += float(
                    -shuffled_lp[chosen].gather(1, shuffled["action"][chosen].unsqueeze(1)).sum())
                score["top1_stored"] += int((stored_lp[chosen].argmax(1) == action[chosen]).sum())
                score["top1_shuffled"] += int(
                    (shuffled_lp[chosen].argmax(1) == shuffled["action"][chosen]).sum())
            chosen = valid & first
            if bool(chosen.any()):
                favourite["stored_slot0"] += int(
                    (stored_lp[chosen][:, :8].argmax(1) % 4 == 0).sum())
                favourite["shuffled_slot0"] += int(
                    (shuffled_lp[chosen][:, :8].argmax(1) % 4 == 0).sum())

    moves = max(1, data["move_rows"])
    first_rows = max(1, data["not_yet_moved_rows"])
    report: dict = {
        "manifest": str(manifest),
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "rows": data["rows"],
        "move_rows": data["move_rows"],
        "label_share_by_move_slot": [count / moves for count in slot_counts],
        "not_yet_moved_rows": data["not_yet_moved_rows"],
        "not_yet_moved_label_is_slot0": data["not_yet_moved_label_slot0"] / first_rows,
        "switch_only_rows": data["switch_only_rows"],
        "largest_own_pp": data["largest_own_pp"],
    }
    if agent is not None:
        policy = {}
        for name, score in scores.items():
            rows = max(1, score["rows"])
            policy[name] = {
                "rows": score["rows"],
                "top1_stored": score["top1_stored"] / rows,
                "top1_shuffled": score["top1_shuffled"] / rows,
                "nll_stored": score["nll_stored"] / rows,
                "nll_shuffled": score["nll_shuffled"] / rows,
            }
        policy["not_yet_moved"]["favourite_move_in_slot0_stored"] = (
            favourite["stored_slot0"] / first_rows)
        policy["not_yet_moved"]["favourite_move_in_slot0_shuffled"] = (
            favourite["shuffled_slot0"] / first_rows)
        report["policy"] = policy
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
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    args = parser.parse_args()
    cfg = load_config_with_args(args.config, args.overrides)
    report = check(cfg, args.manifest, args.checkpoint, batch_size=args.batch_size,
                   seed=args.seed)
    text = json.dumps(report, indent=2)
    if args.out is not None:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
