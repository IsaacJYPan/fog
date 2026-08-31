"""Collect exact public Gen 9 OU pastes into a provenance-aware self-play bank.

Examples:

    python -m fog.scripts.collect_team_bank --sources configs/team_sources_gen9ou.yaml
    python -m fog.scripts.collect_team_bank --target 150 --min-quality 70

This never reconstructs hidden sets from replays. Only public Pokepaste ``/raw`` imports that parse
as six-Pokemon teams and pass the pinned Showdown validator are exported.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from fog.env.teams import to_packed, validate_team
from fog.team_bank import HttpFetcher, SourceSpec, collect_bank


def load_sources(path: Path) -> list[SourceSpec]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [SourceSpec(**entry) for entry in data.get("sources", [])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources", type=Path, default=Path("configs/team_sources_gen9ou.yaml")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/teams/public_gen9ou")
    )
    parser.add_argument("--cache", type=Path, default=Path("data/team_bank_cache"))
    parser.add_argument("--format", default="gen9ou")
    parser.add_argument("--target", type=int, default=200)
    parser.add_argument("--min-quality", type=int, default=60)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--no-robots", action="store_true")
    args = parser.parse_args()

    sources = load_sources(args.sources)
    if not sources:
        raise ValueError(f"no sources configured in {args.sources}")
    fetcher = HttpFetcher(
        args.cache,
        delay=args.delay,
        respect_robots=not args.no_robots,
    )
    manifest = collect_bank(
        sources,
        args.out,
        fetch=fetcher.get_text,
        canonicalize=to_packed,
        validate=validate_team,
        battle_format=args.format,
        min_quality=args.min_quality,
        target=args.target,
    )
    print(
        f"public team bank: {manifest['team_count']}/{args.target} legal unique teams "
        f"-> {args.out / 'pool.json'}"
    )
    if manifest["source_errors"]:
        print(f"warning: {len(manifest['source_errors'])} sources failed; see manifest.json")

if __name__ == "__main__":
    main()
