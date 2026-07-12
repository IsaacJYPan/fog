"""Parsed-replay dataset for behavior cloning (spec §5).

Two loaders over the npz shards listed in a parse manifest:

  * `ReplayDataset` — concatenates everything into memory. Right for small corpora (<~20k games);
    simplest and fastest per-batch.
  * `StreamingReplayDataset` — iterates shard-by-shard with shard-order + within-shard shuffling per
    epoch. Right for the 150k-game M4 corpus, where full concat would blow RAM (history windows are
    the big block). Approximate global shuffle: shards are visited in random order and rows within a
    shard are permuted; with one replay per shard this is close enough to i.i.d. for SGD.

Both yield identical C3+label batches ready for `FogAgent.forward` + the BC losses. Per-decision
belief-net masks are rebuilt from the stored `opp_tokens` at batch time (masks are deterministic
given species, so only tokens are stored).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from fog.encoding.reveals import align_truth_to_views, labels_from_truth, masks_for_tokens

_ARRAYS = (
    "entities", "entity_ids", "field", "history", "active_idx", "legality_mask",
    "opp_tokens", "action", "belief_targets", "self_belief_targets", "outcome",
)


def _build_batch(d: dict[str, np.ndarray], idx: np.ndarray, tables, device=None) -> dict:
    """Assemble one C3+label batch from row indices into in-memory arrays."""
    opp_tokens = d["opp_tokens"][idx]
    masks = _batched_masks(opp_tokens, tables, device)
    batch = {
        "entities": _t(d["entities"][idx], torch.float32, device),
        "entity_ids": _t(d["entity_ids"][idx], torch.long, device),
        "field": _t(d["field"][idx], torch.float32, device),
        "history": _t(d["history"][idx], torch.float32, device),
        "active_idx": _t(d["active_idx"][idx], torch.long, device),
        "legality_mask": _t(d["legality_mask"][idx], torch.bool, device),
        "opp_revealed": {"tokens": _t(opp_tokens, torch.long, device), "masks": masks},
        "action": _t(d["action"][idx], torch.long, device),
        "outcome": _t(d["outcome"][idx], torch.float32, device),
    }
    # Belief targets are stored in paste order; the chain input (opp_tokens) packs revealed moves
    # into the earliest subslots, so align per row before splitting into label dicts.
    aligned, _ = align_truth_to_views(d["belief_targets"][idx], opp_tokens)
    belief = labels_from_truth(aligned)
    batch["belief_targets"] = {k: _t(v, torch.long, device) for k, v in belief.items()}
    batch["belief_valid"] = {k: _t(v >= 0, torch.bool, device) for k, v in belief.items()}
    # self_belief_targets stay paste-order: BC has no mirror input (bc.py — self_belief_coef
    # defaults 0, out.self_belief is None), so nothing consumes them; align against the
    # opponent's-view-of-me tokens at consumption when a BC mirror path exists.
    self_belief = labels_from_truth(d["self_belief_targets"][idx])
    batch["self_belief_targets"] = {k: _t(v, torch.long, device) for k, v in self_belief.items()}
    batch["self_belief_valid"] = {k: _t(v >= 0, torch.bool, device) for k, v in self_belief.items()}
    return batch


class ReplayDataset:
    def __init__(self, manifest_path: str | Path, limit_rows: int | None = None) -> None:
        import polars as pl

        manifest = pl.read_parquet(manifest_path)
        buf: dict[str, list[np.ndarray]] = {k: [] for k in _ARRAYS}
        total = 0
        for shard in manifest["shard"].to_list():
            z = np.load(shard)
            n = len(z["action"])
            for k in _ARRAYS:
                buf[k].append(z[k])
            total += n
            if limit_rows and total >= limit_rows:
                break
        self.data = {k: np.concatenate(v)[:limit_rows] if limit_rows else np.concatenate(v)
                     for k, v in buf.items()}
        self.n = len(self.data["action"])

    def __len__(self) -> int:
        return self.n

    def batches(self, batch_size: int, shuffle: bool = True, seed: int = 0, device=None):
        rng = np.random.default_rng(seed)
        order = rng.permutation(self.n) if shuffle else np.arange(self.n)
        from fog.constants.tables import TABLES

        for start in range(0, self.n, batch_size):
            idx = order[start : start + batch_size]
            yield _build_batch(self.data, idx, TABLES, device)


class StreamingReplayDataset:
    """Shard-streaming loader for corpora that don't fit in RAM (M4 150k-game scale).

    `batches()` visits shards in a per-epoch random order, loads `shard_buffer` shards at a time
    into one in-memory block, and yields shuffled batches from it. Rows spanning shard-buffer
    boundaries are never mixed — acceptable shuffle quality when each shard holds one replay.
    """

    def __init__(self, manifest_path: str | Path, shard_buffer: int = 64) -> None:
        import polars as pl

        manifest = pl.read_parquet(manifest_path)
        self.shards: list[str] = manifest["shard"].to_list()
        self.shard_buffer = max(1, shard_buffer)
        rows_col = next((c for c in ("n_rows", "rows") if c in manifest.columns), None)
        self.n = int(manifest[rows_col].sum()) if rows_col else -1

    def __len__(self) -> int:
        if self.n < 0:
            raise ValueError("manifest lacks a 'rows' column; row count unknown before iteration")
        return self.n

    def batches(self, batch_size: int, shuffle: bool = True, seed: int = 0, device=None):
        from fog.constants.tables import TABLES

        rng = np.random.default_rng(seed)
        shard_order = rng.permutation(len(self.shards)) if shuffle else np.arange(len(self.shards))
        for group_start in range(0, len(self.shards), self.shard_buffer):
            group = [self.shards[i] for i in shard_order[group_start : group_start + self.shard_buffer]]
            buf: dict[str, list[np.ndarray]] = {k: [] for k in _ARRAYS}
            for shard in group:
                z = np.load(shard)
                for k in _ARRAYS:
                    buf[k].append(z[k])
            data = {k: np.concatenate(v) for k, v in buf.items()}
            n = len(data["action"])
            order = rng.permutation(n) if shuffle else np.arange(n)
            for start in range(0, n, batch_size):
                yield _build_batch(data, order[start : start + batch_size], TABLES, device)


def _t(arr: np.ndarray, dtype, device):
    t = torch.as_tensor(np.ascontiguousarray(arr), dtype=dtype)
    return t.to(device) if device is not None else t


def _batched_masks(opp_tokens: np.ndarray, tables, device=None) -> dict:
    """Stack per-row belief masks [B,6,n] from tokens. species/ability/move/spread only (item is a
    learned prior — all-True — so it is omitted, matching reveals.masks_for_tokens)."""
    per = [masks_for_tokens(tok, tables) for tok in opp_tokens]
    out = {}
    for fam in ("species", "ability", "move", "spread"):
        t = torch.as_tensor(np.stack([m[fam] for m in per]), dtype=torch.bool)
        out[fam] = t.to(device) if device is not None else t
    return out
