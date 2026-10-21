"""Parsed-replay dataset for behavior cloning.

Two loaders over the npz shards listed in a parse manifest:

  * `ReplayDataset` — concatenates everything into memory. Right for small corpora (<~20k games);
    simplest and fastest per-batch.
  * `StreamingReplayDataset` — iterates shard-by-shard with shard-order + within-shard shuffling per
    epoch. Useful for large corpora, where full concat would blow RAM (history windows are
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

from fog.encoding.dims import DIMS
from fog.encoding.entities import BASE_ID_FIELDS, ENTITY_RAW_DIM, N_ID_FIELDS, L
from fog.encoding.reveals import align_truth_to_views, labels_from_truth, masks_for_tokens

_ARRAYS = (
    "entities", "entity_ids", "field", "history", "history_ids", "history_state", "history_len",
    "active_idx", "legality_mask", "opp_tokens", "action", "belief_targets",
    "self_belief_targets", "outcome",
    "action_features", "policy_valid",
    "own_set_provenance", "opponent_set_provenance", "legal_provenance",
    "action_feature_provenance", "imputation_id",
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
        "history_ids": _t(d["history_ids"][idx], torch.long, device),
        "history_state": _t(d["history_state"][idx], torch.float32, device),
        "history_len": _t(d["history_len"][idx], torch.long, device),
        "active_idx": _t(d["active_idx"][idx], torch.long, device),
        "legality_mask": _t(d["legality_mask"][idx], torch.bool, device),
        "opp_revealed": {"tokens": _t(opp_tokens, torch.long, device), "masks": masks},
        "action": _t(d["action"][idx], torch.long, device),
        "outcome": _t(d["outcome"][idx], torch.float32, device),
        "action_features": _t(d["action_features"][idx], torch.float32, device),
        "policy_valid": _t(d["policy_valid"][idx], torch.bool, device),
        "own_set_provenance": _t(d["own_set_provenance"][idx], torch.uint8, device),
        "opponent_set_provenance": _t(
            d["opponent_set_provenance"][idx], torch.uint8, device
        ),
        "legal_provenance": _t(d["legal_provenance"][idx], torch.uint8, device),
        "action_feature_provenance": _t(
            d["action_feature_provenance"][idx], torch.uint8, device
        ),
        "imputation_id": _t(d["imputation_id"][idx], torch.long, device),
    }
    # Belief targets are stored in paste order; the chain input (opp_tokens) packs revealed moves
    # into the earliest subslots, so align per row before splitting into label dicts.
    aligned, _ = align_truth_to_views(d["belief_targets"][idx], opp_tokens)
    belief = labels_from_truth(aligned)
    from fog.encoding.reveals import FAMILY_COLS, original_field_hidden

    hidden = original_field_hidden(opp_tokens, tables)
    batch["belief_targets"] = {k: _t(v, torch.long, device) for k, v in belief.items()}
    batch["belief_valid"] = {}
    for family, values in belief.items():
        family_hidden = hidden[..., list(FAMILY_COLS[family])]
        if family not in ("move", "ev"):
            family_hidden = family_hidden.squeeze(-1)
        batch["belief_valid"][family] = _t(
            (values >= 0) & family_hidden, torch.bool, device
        )
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
                if k == "history_len" and k not in z:
                    buf[k].append(_infer_history_lengths(z["history"]))
                elif k == "history_ids" and k not in z:
                    buf[k].append(np.zeros(
                        (*z["history"].shape[:-1], DIMS.hist_id_fields), dtype=np.int64
                    ))
                elif k == "history_state" and k not in z:
                    buf[k].append(np.zeros(
                        (*z["history"].shape[:-1], DIMS.hist_state), dtype=np.float32
                    ))
                elif k == "entities":
                    buf[k].append(_normalize_entities(z[k]))
                elif k == "entity_ids":
                    buf[k].append(_normalize_entity_ids(z[k]))
                elif k in ("opp_tokens", "belief_targets", "self_belief_targets"):
                    buf[k].append(_normalize_set_tokens(z[k]))
                elif k == "action_features" and k not in z:
                    from fog.encoding.action_features import ACTION_FEATURE_DIM

                    buf[k].append(np.zeros((n, 14, ACTION_FEATURE_DIM), dtype=np.float32))
                elif k == "policy_valid" and k not in z:
                    # Legacy shards predate label/feature alignment auditing and must not
                    # silently supervise the policy.  They remain valid for belief/value work.
                    buf[k].append(np.zeros(n, dtype=bool))
                elif k in ("own_set_provenance", "opponent_set_provenance") and k not in z:
                    buf[k].append(np.zeros((n, 6, 15), dtype=np.uint8))
                elif k in ("legal_provenance", "action_feature_provenance") and k not in z:
                    buf[k].append(np.zeros((n, 14), dtype=np.uint8))
                elif k == "imputation_id" and k not in z:
                    buf[k].append(np.full(n, -1, dtype=np.int16))
                else:
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
    """Shard-streaming loader for corpora that don't fit in RAM.

    `batches()` visits shards in a per-epoch random order, loads `shard_buffer` shards at a time
    into one in-memory block, and yields shuffled batches from it. Rows spanning shard-buffer
    boundaries are never mixed — acceptable shuffle quality when each shard holds one replay.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        shard_buffer: int = 64,
        *,
        force_policy_valid: bool = False,
    ) -> None:
        import polars as pl

        manifest = pl.read_parquet(manifest_path)
        self.shards: list[str] = manifest["shard"].to_list()
        self.shard_buffer = max(1, shard_buffer)
        self.force_policy_valid = bool(force_policy_valid)
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
                    if k == "history_len" and k not in z:
                        buf[k].append(_infer_history_lengths(z["history"]))
                    elif k == "history_ids" and k not in z:
                        buf[k].append(np.zeros(
                            (*z["history"].shape[:-1], DIMS.hist_id_fields), dtype=np.int64
                        ))
                    elif k == "history_state" and k not in z:
                        buf[k].append(np.zeros(
                            (*z["history"].shape[:-1], DIMS.hist_state), dtype=np.float32
                        ))
                    elif k == "entities":
                        buf[k].append(_normalize_entities(z[k]))
                    elif k == "entity_ids":
                        buf[k].append(_normalize_entity_ids(z[k]))
                    elif k in ("opp_tokens", "belief_targets", "self_belief_targets"):
                        buf[k].append(_normalize_set_tokens(z[k]))
                    elif k == "action_features" and k not in z:
                        from fog.encoding.action_features import ACTION_FEATURE_DIM

                        buf[k].append(np.zeros((len(z["action"]), 14, ACTION_FEATURE_DIM), dtype=np.float32))
                    elif k == "policy_valid" and k not in z:
                        buf[k].append(np.zeros(len(z["action"]), dtype=bool))
                    elif k in ("own_set_provenance", "opponent_set_provenance") and k not in z:
                        buf[k].append(np.zeros((len(z["action"]), 6, 15), dtype=np.uint8))
                    elif k in ("legal_provenance", "action_feature_provenance") and k not in z:
                        buf[k].append(np.zeros((len(z["action"]), 14), dtype=np.uint8))
                    elif k == "imputation_id" and k not in z:
                        buf[k].append(np.full(len(z["action"]), -1, dtype=np.int16))
                    else:
                        buf[k].append(z[k])
            data = {k: np.concatenate(v) for k, v in buf.items()}
            if self.force_policy_valid:
                # This is only valid for authoritative simulator corpora whose stored actions
                # were actually executed by the requested teacher.  It avoids duplicating many
                # gigabytes of otherwise identical exact-truth shards.
                data["policy_valid"] = np.ones(len(data["action"]), dtype=bool)
            n = len(data["action"])
            order = rng.permutation(n) if shuffle else np.arange(n)
            for start in range(0, n, batch_size):
                yield _build_batch(data, order[start : start + batch_size], TABLES, device)


def _t(arr: np.ndarray, dtype, device):
    t = torch.as_tensor(np.ascontiguousarray(arr), dtype=dtype)
    return t.to(device) if device is not None else t


def _infer_history_lengths(history: np.ndarray) -> np.ndarray:
    """Backward compatibility for fixed-width replay shards created before history_len was stored."""
    return np.any(np.asarray(history) != 0, axis=-1).sum(axis=-1).astype(np.int64)


def _normalize_entities(entities: np.ndarray) -> np.ndarray:
    """Insert native nature/EV epistemic flags into legacy replay entity rows.

    The old block had item, ability, four moves, spread-cluster, and Tera flags.  A cluster flag
    cannot establish any exact nature/EV fact, so the seven native flags are inserted as unknown
    while the old Tera flag and all following public-state fields retain their positions.
    """
    values = np.asarray(entities, dtype=np.float32)
    if values.shape[-1] == ENTITY_RAW_DIM:
        return values
    old_width = ENTITY_RAW_DIM - 6
    if values.shape[-1] != old_width:
        raise ValueError(f"unsupported entities width {values.shape[-1]}")
    migrated = np.zeros((*values.shape[:-1], ENTITY_RAW_DIM), dtype=np.float32)
    ep = L.epistemic
    migrated[..., : ep + 6] = values[..., : ep + 6]
    migrated[..., ep + 13] = values[..., ep + 7]  # old Tera-known flag
    migrated[..., ep + 14 :] = values[..., ep + 8 :]
    return migrated


def _normalize_entity_ids(entity_ids: np.ndarray) -> np.ndarray:
    """Migrate legacy spread-cluster entity ids to native nature/EV ids."""
    ids = np.asarray(entity_ids, dtype=np.int64)
    if ids.shape[-1] == N_ID_FIELDS:
        return ids
    from fog.constants.schema import N_EV_UNITS, N_NATURES
    from fog.constants.tables import TABLES

    if ids.shape[-1] in (9, 11):
        migrated = np.empty((*ids.shape[:-1], N_ID_FIELDS), dtype=np.int64)
        migrated[..., :8] = ids[..., :8]  # species/item/ability/moves/last-used
        migrated[..., 8] = N_NATURES
        migrated[..., 9:15] = N_EV_UNITS
        if ids.shape[-1] == 11:
            migrated[..., 15] = ids[..., 9]
            migrated[..., 16] = ids[..., 10]
        else:
            migrated[..., 15] = TABLES.n_items
            migrated[..., 16] = TABLES.n_abilities
        return migrated
    if ids.shape[-1] == BASE_ID_FIELDS:
        extra = np.empty((*ids.shape[:-1], 2), dtype=np.int64)
        extra[..., 0] = TABLES.n_items
        extra[..., 1] = TABLES.n_abilities
        return np.concatenate([ids, extra], axis=-1)
    raise ValueError(f"unsupported entity_ids width {ids.shape[-1]}")


def _normalize_set_tokens(tokens: np.ndarray) -> np.ndarray:
    """Expand legacy [..,6,9] grids; old spread ids are not recoverable labels."""
    values = np.asarray(tokens, dtype=np.int64)
    if values.shape[-1] == 15:
        return values
    if values.shape[-1] != 9:
        raise ValueError(f"unsupported set-token width {values.shape[-1]}")
    migrated = np.full((*values.shape[:-1], 15), -1, dtype=np.int64)
    migrated[..., :7] = values[..., :7]
    migrated[..., 14] = values[..., 8]  # Tera moved; legacy spread column 7 is discarded
    return migrated


def _batched_masks(opp_tokens: np.ndarray, tables, device=None) -> dict:
    """Stack per-row belief masks [B,6,n] from tokens. Species/ability/move only (item is a
    learned prior — all-True — so it is omitted, matching reveals.masks_for_tokens)."""
    per = [masks_for_tokens(tok, tables) for tok in opp_tokens]
    out = {}
    for fam in ("species", "ability", "move"):
        t = torch.as_tensor(np.stack([m[fam] for m in per]), dtype=torch.bool)
        out[fam] = t.to(device) if device is not None else t
    return out
