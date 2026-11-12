"""Pool several ``showdown_direct_selection`` reports that differ only in seed.

One evaluation process plays one game at a time. To use several cores, run the same command in
parallel with different ``--seed`` and ``--out`` values (each shard is its own set of matched
four-game blocks), then merge the reports into one row per pairing:

    python -m fog.eval.merge_selection_reports runs/g1/shard*.json --out runs/g1/merged.json

Shards must share every protocol field except the seed, and must be fixed-length runs: reports
made with ``--sequential`` are refused, because pooling runs that each stopped on their own
sequential test does not keep that test's error rates. Each merged row prints a ledger row whose
command names the shard count and seeds; give ``--command`` to record the shard command itself.
Per-game scores and controller counters are concatenated in shard order.
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Any

from fog.eval.showdown_direct_selection import (
    MatchResult,
    git_short_hash,
    ledger_row,
    match_result_from_dict,
)

_SUMMED = ("requested_games", "completed_games", "a_wins", "b_wins", "draws", "truncated",
           "failures", "turns", "wall_s")


def _comparable(protocol: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in protocol.items() if key != "seed"}


def merge_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Merged report for shard reports of one protocol. Raises ``ValueError`` on a mismatch."""
    if not reports:
        raise ValueError("no reports to merge")
    first = reports[0]
    seeds = []
    for report in reports:
        if report.get("status") != "complete":
            raise ValueError("every shard must be a complete report; resume unfinished shards")
        protocol = report["protocol"]
        if "sequential" in protocol:
            raise ValueError("sequential (SPRT) shards cannot be pooled; use fixed --games")
        if _comparable(protocol) != _comparable(first["protocol"]):
            raise ValueError("shards differ in more than their seed")
        if report.get("checkpoints") != first.get("checkpoints"):
            raise ValueError("shards use different checkpoints")
        seeds.append(protocol["seed"])
    if len(set(seeds)) != len(seeds):
        raise ValueError("two shards share a seed; they would be the same games")

    merged: dict[str, MatchResult] = {}
    for report in reports:
        for row in report["rows"]:
            part = match_result_from_dict(row)
            if part.label not in merged:
                merged[part.label] = MatchResult(label=part.label, a=part.a, b=part.b,
                                                 requested_games=0)
            total = merged[part.label]
            for name in _SUMMED:
                setattr(total, name, getattr(total, name) + getattr(part, name))
            total.game_scores.extend(part.game_scores)
            total.game_counters.extend(part.game_counters)
    protocol = {**_comparable(first["protocol"]), "seeds": seeds, "shards": len(reports),
                "games_per_row": sum(r["protocol"]["games_per_row"] for r in reports)}
    return {"status": "complete", "protocol": protocol,
            "checkpoints": first.get("checkpoints", {}),
            "rows": [result.as_dict() for result in merged.values()]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--command", default=None,
                        help="the shard command to record in the ledger row")
    parser.add_argument("--commit", default=None,
                        help="commit the shards ran at (default: the current checkout)")
    args = parser.parse_args()
    merged = merge_reports([json.loads(path.read_text(encoding="utf-8"))
                            for path in args.reports])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    seeds = ", ".join(str(seed) for seed in merged["protocol"]["seeds"])
    command = (args.command or "python -m fog.eval.showdown_direct_selection ...") + (
        f" ({merged['protocol']['shards']} shards merged, seeds {seeds})")
    for row in merged["rows"]:
        for side in ("a", "b"):
            if row.get(f"{side}_fidelity"):
                print(f"FIDELITY {row['label']} ({side}): {json.dumps(row[f'{side}_fidelity'])}")
        print(ledger_row(row, backend="showdown_direct", commit=args.commit or git_short_hash(),
                         command=command, date=datetime.date.today().isoformat()))


if __name__ == "__main__":
    main()
