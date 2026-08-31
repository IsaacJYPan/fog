"""Validate a converted replay-policy corpus without loading it all into memory."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from fog.encoding.action_features import ACTION_FEATURE_NAMES


def main() -> None:
    import polars as pl

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    frame = pl.read_parquet(args.manifest)
    legal_index = ACTION_FEATURE_NAMES.index("legal")
    provenance = {
        name: Counter()
        for name in (
            "own_set_provenance",
            "opponent_set_provenance",
            "legal_provenance",
            "action_feature_provenance",
        )
    }
    slot_counts: Counter[int] = Counter()
    imputation_counts: Counter[int] = Counter()
    failures: list[str] = []
    counts = Counter()

    for record in frame.iter_rows(named=True):
        path = Path(record["shard"])
        try:
            with np.load(path, allow_pickle=False) as shard:
                n = int(shard["action"].shape[0])
                counts["rows"] += n
                counts["shards"] += 1
                if n != int(record["n_rows"]):
                    failures.append(f"{path}: {n} rows != manifest {record['n_rows']}")
                action = shard["action"].astype(np.int64, copy=False)
                legality = shard["legality_mask"].astype(bool, copy=False)
                if ((action < 0) | (action >= legality.shape[1])).any():
                    counts["action_out_of_range"] += int(
                        ((action < 0) | (action >= legality.shape[1])).sum()
                    )
                    continue
                chosen_legal = legality[np.arange(n), action]
                counts["chosen_illegal"] += int((~chosen_legal).sum())
                feature_legal = shard["action_features"][:, :, legal_index].astype(bool)
                counts["action_feature_legal_mismatch"] += int(
                    np.not_equal(feature_legal, legality).sum()
                )
                counts["nonfinite_action_features"] += int(
                    (~np.isfinite(shard["action_features"])).sum()
                )
                counts["invalid_policy_rows"] += int((~shard["policy_valid"]).sum())
                for value in action.tolist():
                    slot_counts[int(value)] += 1
                for value in shard["imputation_id"].tolist():
                    imputation_counts[int(value)] += 1
                for name, counter in provenance.items():
                    values, value_counts = np.unique(shard[name], return_counts=True)
                    for value, count in zip(values.tolist(), value_counts.tolist(), strict=True):
                        counter[int(value)] += int(count)
        except Exception as error:  # noqa: BLE001 - preserve every failing shard
            failures.append(f"{path}: {type(error).__name__}: {error}")

    counts["opponent_future_inferred_fields"] = provenance[
        "opponent_set_provenance"
    ][1]
    required_zero = (
        "action_out_of_range",
        "chosen_illegal",
        "action_feature_legal_mismatch",
        "nonfinite_action_features",
        "invalid_policy_rows",
        "opponent_future_inferred_fields",
    )
    report = {
        "status": "passed"
        if not failures and all(counts[key] == 0 for key in required_zero)
        else "failed",
        "manifest": str(args.manifest),
        "shards": counts["shards"],
        "rows": counts["rows"],
        "counts": {key: int(value) for key, value in sorted(counts.items())},
        "provenance_codes": {"unknown": 0, "inferred": 1, "exact": 2},
        "provenance_counts": {
            name: {str(key): int(value) for key, value in sorted(counter.items())}
            for name, counter in provenance.items()
        },
        "action_slot_counts": {
            str(key): int(value) for key, value in sorted(slot_counts.items())
        },
        "imputation_counts": {
            str(key): int(value) for key, value in sorted(imputation_counts.items())
        },
        "failures": failures,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
