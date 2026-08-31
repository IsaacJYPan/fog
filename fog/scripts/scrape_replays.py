"""Scrape the Showdown replay search API for gen9ou (spec §4.2).

Stores raw replay JSON IMMUTABLY under data/raw_replays/ (never mutated; re-parse from raw). Resumable
(skips ids already present) and rate-limited (polite fixed delay). Filters on the replay rating.

Pagination: the search API returns the most recent ~50 replays; `&before=<uploadtime>` walks backward
in time. We page until the target count of kept replays is reached or the feed is exhausted.

    python -m fog.scripts.scrape_replays --target-games 2000 --min-rating 1300
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

SEARCH_URL = "https://replay.pokemonshowdown.com/search.json?format={fmt}"
REPLAY_URL = "https://replay.pokemonshowdown.com/{replay_id}.json"
UA = {"User-Agent": "fog/0.1 (research; contact via github)"}


def _get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def search_page(fmt: str, before: int | None = None) -> list[dict]:
    url = SEARCH_URL.format(fmt=fmt)
    if before is not None:
        url += f"&before={before}"
    try:
        return json.loads(_get(url))
    except (json.JSONDecodeError, OSError, TimeoutError, urllib.error.URLError) as e:
        print(f"  search error: {type(e).__name__} {e}")
        return []


def fetch_replay(replay_id: str) -> dict | None:
    try:
        return json.loads(_get(REPLAY_URL.format(replay_id=replay_id)))
    except (json.JSONDecodeError, OSError, TimeoutError, urllib.error.URLError) as e:
        print(f"  fetch error {replay_id}: {type(e).__name__}")
        return None


def scrape(fmt: str, min_rating: int, target: int, out_dir: Path, delay: float = 1.0,
           resume_cursor: bool = True) -> int:
    """Walk the search feed backward in time until `target` kept replays exist under `out_dir`.

    150k-scale resume support:
      * `_cursor.json` persists the oldest uploadtime reached; a resumed run continues from there
        instead of re-walking every already-seen page from the top of the feed
        (`--fresh-cursor` restarts from the most recent replays, e.g. after a long gap).
      * `_rejected.txt` caches ids fetched but rejected (rating/no-log), so resumes never re-fetch
        them. Kept replays are already deduped by presence on disk.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cursor_file = out_dir / "_cursor.json"
    rejected_file = out_dir / "_rejected.txt"
    have = {p.stem for p in out_dir.glob("*.json")}
    have.discard("_cursor")
    rejected: set[str] = set()
    if rejected_file.exists():
        rejected = set(rejected_file.read_text(encoding="utf-8").split())
    kept = len(have)
    before: int | None = None
    if resume_cursor and cursor_file.exists():
        before = json.loads(cursor_file.read_text(encoding="utf-8")).get("before")
        print(f"  resuming cursor at uploadtime<{before}")
    empty_pages = 0

    def _save_cursor() -> None:
        if before is not None:
            cursor_file.write_text(json.dumps({"before": before}), encoding="utf-8")

    while kept < target and empty_pages < 3:
        page = search_page(fmt, before)
        if not page:
            empty_pages += 1
            time.sleep(delay)
            continue
        empty_pages = 0
        before = min(int(e["uploadtime"]) for e in page) - 1  # walk backward
        _save_cursor()

        for entry in page:
            rid = entry["id"]
            if rid in have or rid in rejected:
                continue
            if int(entry.get("rating") or 0) < min_rating:
                continue
            replay = fetch_replay(rid)
            time.sleep(delay)
            # confirm rating from the full record too (search entry can lag)
            if not replay or "log" not in replay or int(replay.get("rating") or 0) < min_rating:
                rejected.add(rid)
                with open(rejected_file, "a", encoding="utf-8") as f:
                    f.write(rid + "\n")
                continue
            (out_dir / f"{rid}.json").write_text(json.dumps(replay), encoding="utf-8")
            have.add(rid)
            kept += 1
            if kept % 25 == 0:
                print(f"  kept {kept}/{target}")
            if kept >= target:
                break

    _save_cursor()
    print(f"done: {kept} replays in {out_dir}")
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description="Scrape gen9ou replays (§4.2).")
    ap.add_argument("--format", default="gen9ou")
    ap.add_argument("--min-rating", type=int, default=1300)
    ap.add_argument("--target-games", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=Path("data/raw_replays"))
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--fresh-cursor", action="store_true",
                    help="restart from the most recent replays instead of the saved cursor")
    args = ap.parse_args()
    scrape(args.format, args.min_rating, args.target_games, args.out, args.delay,
           resume_cursor=not args.fresh_cursor)


if __name__ == "__main__":
    main()
