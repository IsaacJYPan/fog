"""Leakage / readability metrics (spec §7 headline measurements, §9).

Headline measurements:
  (a) win-rate of L vs N at each β, ≥3000 games/cell
  (b) opponent-side readability: how well a FRESH predictor reads L vs N agents at matched turns
  (c) behavioral case studies: tera-timing distributions, choice-lock concealment (does L delay
      revealing a Choice item?), move-order-of-reveal entropy

(c) is what makes the writeup interesting even if (a) is a null result.

These are the pure-math helpers; the drivers that produce their inputs (trajectory collection, the
fresh-probe fit, the case-study extractors, the L-vs-N match and the β-grid frontier) live in
fog/eval/readability.py and are exposed through `python -m fog.eval.run_eval`.
"""
from __future__ import annotations

import numpy as np


def readability_curve(ce_by_turn: list[tuple[int, float]], bucket: int = 3) -> dict[int, float]:
    """Fresh predictor's mean CE (lower = more readable) over an agent's still-hidden attrs, by turn
    bucket (§7b). Input: list of (turn, summed-CE) pairs from running a freshly-trained self-belief
    predictor over the evaluated agent's trajectories. Compare L vs N at matched buckets."""
    from collections import defaultdict

    acc: dict[int, list[float]] = defaultdict(list)
    for turn, ce in ce_by_turn:
        acc[(turn // bucket) * bucket].append(float(ce))
    return {b: float(np.mean(v)) for b, v in sorted(acc.items())}


def tera_timing_distribution(tera_turns: list[int], max_turn: int = 30) -> np.ndarray:
    """Histogram of the turn on which the agent Terastallizes (§7c case study + collapse symptom:
    tera-turn-1-always / tera-never). `tera_turns`: one entry per game (>=1; use 0 for 'never')."""
    hist = np.zeros(max_turn + 1, dtype=np.int64)
    for t in tera_turns:
        hist[min(max(int(t), 0), max_turn)] += 1
    return hist


def choice_lock_reveal_delay(delays: list[int]) -> float:
    """Mean turns between first using a Choice-locked move and the item being inferred/revealed
    (§7c: does L conceal a Choice item longer than N?). `delays`: per-game turn gaps."""
    return float(np.mean(delays)) if delays else float("nan")


def move_reveal_entropy(reveal_orders: list[tuple[int, ...]]) -> float:
    """Entropy of the order in which an agent reveals its moves (§7c). `reveal_orders`: per game, the
    sequence of move-slot indices in reveal order; returns mean per-position entropy in nats."""
    from collections import Counter

    if not reveal_orders:
        return float("nan")
    max_len = max(len(o) for o in reveal_orders)
    total = 0.0
    for pos in range(max_len):
        counts = Counter(o[pos] for o in reveal_orders if len(o) > pos)
        n = sum(counts.values())
        if n:
            total += -sum((c / n) * np.log(c / n) for c in counts.values())
    return float(total)
