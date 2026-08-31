"""Shared kernels for the project's distinct supervised training objectives.

Objectives remain separate, but data isolation and parameter-scope rules must not be reimplemented
slightly differently in every trainer.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GroupedSplit:
    """Indices for a deterministic group-disjoint train/holdout split."""

    train: np.ndarray
    holdout: np.ndarray
    holdout_groups: tuple[int, ...]
    all_groups: tuple[int, ...]


def holdout_groups(groups: Iterable[int], fraction: float, seed: int) -> set[int]:
    """Choose a deterministic nonempty proper subset of unique integer groups."""
    unique = sorted({int(group) for group in groups})
    if len(unique) < 2:
        raise ValueError("at least two groups are required")
    if not 0.0 < fraction < 1.0:
        raise ValueError("holdout fraction must be between zero and one")
    random.Random(seed).shuffle(unique)
    count = min(len(unique) - 1, max(1, round(len(unique) * fraction)))
    return set(unique[:count])


def grouped_row_split(
    rows: list[dict],
    group: str | Callable[[dict], int],
    *,
    holdout_fraction: float = 0.1,
    seed: int,
    eligible: Callable[[dict], bool] | None = None,
) -> GroupedSplit:
    """Split rows without allowing one battle/team identity to cross the boundary."""
    getter = (lambda row: int(row[group])) if isinstance(group, str) else group
    groups = tuple(sorted({int(getter(row)) for row in rows}))
    held = holdout_groups(groups, holdout_fraction, seed)
    use = eligible or (lambda _row: True)
    train = np.asarray(
        [index for index, row in enumerate(rows) if use(row) and int(getter(row)) not in held],
        dtype=np.int64,
    )
    holdout = np.asarray(
        [index for index, row in enumerate(rows) if use(row) and int(getter(row)) in held],
        dtype=np.int64,
    )
    if not len(train) or not len(holdout):
        raise ValueError("group-disjoint split produced an empty side")
    return GroupedSplit(train, holdout, tuple(sorted(held)), groups)


def set_trainable_prefixes(module: Any, prefixes: tuple[str, ...]) -> list[Any]:
    """Freeze a module, then enable exactly the parameters under ``prefixes``."""
    selected = []
    for name, parameter in module.named_parameters():
        enabled = name.startswith(prefixes)
        parameter.requires_grad_(enabled)
        if enabled:
            selected.append(parameter)
    if not selected:
        raise ValueError(f"no parameters matched trainable prefixes {prefixes!r}")
    return selected


def changed_outside_prefixes(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    allowed_prefixes: tuple[str, ...],
) -> list[str]:
    """Return state tensors changed outside an explicitly allowed training scope."""
    import torch

    if set(before) != set(after):
        missing = sorted(set(before) ^ set(after))
        raise ValueError(f"state dictionaries have different keys: {missing[:8]}")
    return [
        name
        for name, value in before.items()
        if not name.startswith(allowed_prefixes)
        and not torch.equal(value.detach().cpu(), after[name].detach().cpu())
    ]
