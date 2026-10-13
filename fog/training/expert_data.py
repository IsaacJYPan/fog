"""Hash-bound, portable teacher data with separate labels and replay-level sampling."""

from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path

import numpy as np

from fog.training.teacher_replay import digest


class ExpertData:
    def __init__(self, manifest: Path, *, cache_size=16):
        self.root = manifest.resolve().parent
        self.source_sha256 = digest(manifest)
        self.manifest = json.loads(manifest.read_text(encoding="utf-8"))
        if self.manifest.get("format") != "fog.teacher-replay-data" or self.manifest.get("version") != 1 or not self.manifest.get("complete"):
            raise ValueError("Require complete, versioned teacher data")
        self.records = self.manifest["records"]
        self.splits = {name: [row for row in self.records if row["split"] == name] for name in ("train", "development")}
        groups = {name: {key for row in rows for key in row["compositions"]} for name, rows in self.splits.items()}
        if not all(self.splits.values()) or groups["train"] & groups["development"]:
            raise ValueError("Require nonempty composition-disjoint teacher splits")
        if len({row["id"] for row in self.records}) != len(self.records):
            raise ValueError("Duplicate teacher game identity")
        self.cache_size = cache_size
        self.cache = OrderedDict()
        for row in self.records:
            path = (self.root / row["shard"]).resolve()
            if not path.is_relative_to(self.root) or digest(path) != row["shard_sha256"]:
                raise ValueError("Teacher shard changed or escaped its data folder")
        self.weights = np.asarray([row["policy_rows"] for row in self.splits["train"]], dtype=np.float64)
        if np.any(self.weights <= 0):
            raise ValueError("A teacher training game has no strong-policy labels")
        self.weights /= self.weights.sum()

    def arrays(self, record):
        from fog.encoding.public_action_features import public_action_features
        from fog.training.dataset import _ARRAYS

        key = record["id"]
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        path = self.root / record["shard"]
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(_ARRAYS):
                raise ValueError("Teacher shard schema differs")
            arrays = {name: saved[name] for name in _ARRAYS}
        n = len(arrays["action"])
        if n != record["rows"] or any(len(value) != n for value in arrays.values()):
            raise ValueError("Teacher shard row counts differ")
        if int(arrays["policy_valid"].sum()) != record["policy_rows"]:
            raise ValueError("Teacher policy eligibility differs")
        if not np.array_equal(arrays["action_features"], public_action_features(arrays["action_features"])):
            raise ValueError("Teacher-answer feature reached stored policy inputs")
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise ValueError("Teacher arrays contain nonfinite values")
        actions = arrays["action"]
        if np.any(actions < 0) or np.any(actions >= 14) or not arrays["legality_mask"][np.arange(n), actions].all():
            raise ValueError("Teacher action is outside the authoritative mask")
        if not (arrays["legal_provenance"] == 2).all() or not (arrays["action_feature_provenance"] == 2).all():
            raise ValueError("Teacher legality or feature provenance is not exact")
        if not (arrays["imputation_id"] == -1).all():
            raise ValueError("Exact teacher data cannot contain imputation variants")
        self.cache[key] = arrays
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return arrays

    def sample(self, rng, *, games=8, rows_per_game=32):
        """Sample strong decisions from several games, weighted by eligible rows."""
        chosen = rng.choice(len(self.splits["train"]), size=games, p=self.weights)
        specifications = []
        for index in chosen:
            record = self.splits["train"][int(index)]
            eligible = np.flatnonzero(self.arrays(record)["policy_valid"])
            selected = rng.choice(eligible, size=rows_per_game, replace=True)
            specifications.extend((record, int(i)) for i in selected)
        return specifications

    def development(self):
        """Fixed first/middle/last decisions from each teacher seat in each game."""
        result = []
        for record in self.splits["development"]:
            arrays = self.arrays(record)
            offset = 0
            for role in ("p1", "p2"):
                count = record["decisions"][role]
                if role in record["teachers"]:
                    result.extend((record, offset + i) for i in sorted({0, count // 2, count - 1}))
                offset += count
            if offset != len(arrays["action"]):
                raise ValueError("Teacher seat offsets differ from the shard")
        return result

    def batch(self, specifications, device="cpu"):
        from fog.constants.tables import TABLES
        from fog.training.dataset import _build_batch

        selected = [(self.arrays(record), index) for record, index in specifications]
        arrays = {key: np.stack([row[key][index] for row, index in selected]) for key in selected[0][0]}
        return _build_batch(arrays, np.arange(len(selected)), TABLES, device=device)
