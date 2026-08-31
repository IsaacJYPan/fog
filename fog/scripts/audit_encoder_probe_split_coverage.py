"""Audit exact-value coverage shifts across encoder-probe train/holdout splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fog.training.shared_latent_reconstruction import _RAW_BLOCK_NAMES, _RAW_BLOCKS

_IDENTITY_NAMES = (
    "species", "item", "ability", "move0", "move1", "move2", "move3", "move4",
    "nature", "ev0", "ev1", "ev2", "ev3", "ev4", "ev5",
    "original_item", "original_ability",
)


def _identity_stats(train: torch.Tensor, holdout: torch.Tensor) -> dict:
    result = {}
    present_train = train[..., 0] != 0
    present_holdout = holdout[..., 0] != 0
    for column, name in enumerate(_IDENTITY_NAMES):
        train_values = train[..., column][present_train]
        holdout_values = holdout[..., column][present_holdout]
        train_known = train_values[train_values != 0]
        holdout_known = holdout_values[holdout_values != 0]
        train_unique = {int(value) for value in train_known.unique().tolist()}
        holdout_unique = {int(value) for value in holdout_known.unique().tolist()}
        unseen = holdout_unique - train_unique
        unseen_rows = sum(int((holdout_known == value).sum()) for value in unseen)
        result[name] = {
            "train_known_rows": int(train_known.numel()),
            "holdout_known_rows": int(holdout_known.numel()),
            "train_unique": len(train_unique),
            "holdout_unique": len(holdout_unique),
            "unseen_unique": len(unseen),
            "unseen_row_fraction": unseen_rows / max(1, int(holdout_known.numel())),
            "unseen_values": sorted(unseen),
        }
    return result


def _quantized_column_stats(train: torch.Tensor, holdout: torch.Tensor) -> dict:
    # Exact input values are normalized. A 0.01 bin is stricter than the probe's0.05 tolerance
    # while avoiding meaningless float serialization differences.
    train_bins = torch.round(train.float() * 100).to(torch.int64)
    holdout_bins = torch.round(holdout.float() * 100).to(torch.int64)
    train_unique = {int(value) for value in train_bins.unique().tolist()}
    holdout_unique = {int(value) for value in holdout_bins.unique().tolist()}
    unseen = holdout_unique - train_unique
    unseen_rows = sum(int((holdout_bins == value).sum()) for value in unseen)
    return {
        "train_mean": float(train.float().mean()),
        "holdout_mean": float(holdout.float().mean()),
        "train_nonzero_fraction": float((train != 0).float().mean()),
        "holdout_nonzero_fraction": float((holdout != 0).float().mean()),
        "train_unique_bins": len(train_unique),
        "holdout_unique_bins": len(holdout_unique),
        "unseen_unique_bins": len(unseen),
        "unseen_row_fraction": unseen_rows / max(1, int(holdout_bins.numel())),
    }


def _numeric_stats(train: torch.Tensor, holdout: torch.Tensor) -> dict:
    result = {}
    for name, (_, offset, width) in zip(_RAW_BLOCK_NAMES, _RAW_BLOCKS):
        columns = []
        for index in range(width):
            columns.append(_quantized_column_stats(train[..., offset + index], holdout[..., offset + index]))
        result[name] = {
            "width": width,
            "max_unseen_row_fraction": max(column["unseen_row_fraction"] for column in columns),
            "max_nonzero_fraction_shift": max(
                abs(column["train_nonzero_fraction"] - column["holdout_nonzero_fraction"])
                for column in columns
            ),
            "columns": columns,
        }
    return result


def _field_stats(train: torch.Tensor, holdout: torch.Tensor) -> dict:
    columns = [_quantized_column_stats(train[:, index], holdout[:, index]) for index in range(train.shape[1])]
    return {
        "width": int(train.shape[1]),
        "max_unseen_row_fraction": max(column["unseen_row_fraction"] for column in columns),
        "max_nonzero_fraction_shift": max(
            abs(column["train_nonzero_fraction"] - column["holdout_nonzero_fraction"])
            for column in columns
        ),
        "columns": columns,
    }


def run(args: argparse.Namespace) -> dict:
    splits = {
        name: torch.load(args.data / f"{name}.pt", map_location="cpu", weights_only=False)
        for name in ("train", "team_holdout", "battle_holdout")
    }
    train = splits["train"]
    report = {
        "status": "complete",
        "data": str(args.data),
        "rows": {name: int(data["entities"].shape[0]) for name, data in splits.items()},
        "holdouts": {},
    }
    for name in ("team_holdout", "battle_holdout"):
        holdout = splits[name]
        report["holdouts"][name] = {
            "identities": _identity_stats(train["entity_ids"], holdout["entity_ids"]),
            "raw": _numeric_stats(train["entities"], holdout["entities"]),
            "field": _field_stats(train["field"], holdout["field"]),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
