"""Check protocol outcome signs and that submitted labels cannot affect policy/value.

This is a bounded diagnostic, not an exhaustive information-isolation proof.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

from fog.config import load_config_with_args
from fog.constants.tables import TABLES
from fog.models.agent import FogAgent
from fog.scripts.parse_replays import replay_player_names
from fog.training.dataset import StreamingReplayDataset
from fog.training.replay_supervision import _preview_species_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    counts = Counter()
    for row in pl.read_parquet(args.manifest).iter_rows(named=True):
        replay = json.loads((args.raw / f"{row['id']}.json").read_text())
        events = [line.split("|") for line in replay["log"].splitlines() if line.startswith("|")]
        names = replay_player_names(replay)
        winner = next((e[2] for e in events if len(e) > 2 and e[1] == "win"), None)
        p1 = 0 if winner is None else (1 if winner == names["p1"] else -1)
        assert row["p1_win"] == (None if winner is None else winner == names["p1"])
        teams = [[_preview_species_id(e[3]) for e in events
                  if len(e) >= 4 and e[1:3] == ["poke", role]] for role in ("p1", "p2")]
        if any(not 0 < len(team) <= 6 for team in teams):
            raise ValueError("missing or oversized public preview")
        counts["games_with_fewer_than_six_members"] += int(any(len(team) < 6 for team in teams))
        # Ordinary replay labels use -1 for an absent member, including valid five-mon teams.
        ids = [tuple(sorted([TABLES.species_index(s) for s in team] + [-1] * (6-len(team))))
               for team in teams]
        counts["games"] += 1
        with np.load(row["shard"], allow_pickle=False) as arrays:
            own = np.sort(arrays["self_belief_targets"][:, :, 0], axis=1)
            assert len(own) == row["n_rows"]
            for index, composition in enumerate(own):
                roles = [side for side in range(2) if tuple(composition) == ids[side]]
                assert roles
                if len(roles) != 1:
                    counts["ambiguous_mirror_rows_not_sign_checked"] += 1
                    continue
                side = roles[0]
                expected = p1 if side == 0 else -p1
                assert arrays["outcome"][index] == expected
                counts[f"p{side+1}_outcome_{expected}"] += 1
                counts["sign_checked_rows"] += 1
    perturbation = None
    if args.checkpoint is not None:
        cfg = load_config_with_args("configs/replay_mechanical_cpu.yaml")
        model = FogAgent.from_tables(cfg)
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        model.eval()
        batch = next(StreamingReplayDataset(args.manifest).batches(
            256, shuffle=False, seed=0, device="cpu"))

        def changed(value):
            if isinstance(value, dict):
                return {key: changed(item) for key, item in value.items()}
            if isinstance(value, torch.Tensor):
                if value.dtype == torch.bool:
                    return ~value
                return torch.zeros_like(value)
            return value

        labels = {key: changed(value) for key, value in batch.items() if
                  key.startswith(("terminal_", "belief_", "self_belief_")) or key == "outcome"}
        with torch.no_grad():
            before = model(batch, need_belief=False)
            after = model({**batch, **labels}, need_belief=False)
        assert torch.equal(before.log_probs, after.log_probs)
        assert torch.equal(before.value, after.value)
        perturbation = {"rows": len(batch["action"]), "changed_label_keys": sorted(labels),
                        "policy_equal": True, "value_equal": True}
    result = {"scope": "public protocol signs; mirror compositions explicitly excluded from per-row signs",
              "counts": dict(counts), "checkpoint": str(args.checkpoint) if args.checkpoint else None,
              "label_perturbation": perturbation,
              "leakage_metrics_module": "fog.eval.leakage_metrics supplies readability summaries; no standalone privileged-input check"}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
