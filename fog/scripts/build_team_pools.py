"""Validate + package the PoC team pools.

Team pastes live as .txt files under data/teams/pool_{a,b}/ (hand-authored). This script validates
every team against the PINNED simulator (the legality oracle — a rejected team hangs battles) and
emits a packaged JSON ({"teams": [pastes]}) for tooling that prefers a single file. Training itself
reads the paste directory directly via env.team_pool.

  * POOL-A: teams sharing a fixed species core but differing in sets (preview reveals species; sets
    stay hidden) — the M1/M1.5/M3 pool. Shared species are what make belief/leakage non-vacuous.
  * POOL-B: diverse full OU sample teams (unique species multisets) — the open-OU self-play pool
    (train_ou / M4). The trainer refuses it for leakage runs (nothing to conceal after preview).

    python -m fog.scripts.build_team_pools --pool all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fog.env.teams import load_pool, validate_pool

TEAMS_DIR = Path("data/teams")


def build(pool: str, battle_format: str = "gen9ou") -> Path:
    src = TEAMS_DIR / f"pool_{pool}"
    if not src.is_dir():
        raise FileNotFoundError(f"{src} not found (author .txt pastes there first, §4.4)")
    errors = validate_pool(src, battle_format)
    if errors:
        raise ValueError(f"pool_{pool} has illegal teams:\n  " + "\n  ".join(errors))
    teams = load_pool(src)
    out = TEAMS_DIR / f"pool_{pool}.json"
    out.write_text(json.dumps({"teams": teams, "format": battle_format}, indent=2), encoding="utf-8")
    print(f"pool_{pool}: {len(teams)} teams validated -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate + package PoC team pools (§4.4).")
    ap.add_argument("--pool", choices=["a", "b", "all"], default="all")
    ap.add_argument("--format", default="gen9ou")
    args = ap.parse_args()
    pools = ["a", "b"] if args.pool == "all" else [args.pool]
    for p in pools:
        build(p, args.format)


if __name__ == "__main__":
    main()
