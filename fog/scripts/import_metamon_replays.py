"""Import bounded, revision-pinned Metamon raw public replays into FOG's native JSON pipeline.

No Metamon observations, rewards, imputed teams, or model code are trusted as FOG labels.
Use --parquet for an offline import, or download up to --max-shards immutable HF Parquet shards.
Resume uses the same filters/revision and records the next unconsumed row, including mid-batch stops.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.parse
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

REPOSITORY = "jakegrigsby/metamon-raw-replays"


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _request(url: str):
    return urllib.request.urlopen(urllib.request.Request(
        url, headers={"User-Agent": "fog/0.1 research-data-import"}), timeout=60)


def download_shards(cache: Path, *, revision: str | None = None, max_shards: int = 1,
                    max_bytes: int = 256 * 1024**2) -> tuple[str, list[dict]]:
    """Resolve an immutable revision and verify each cached/downloaded shard's content hash."""
    if max_shards < 1 or max_bytes < 1:
        raise ValueError("download budgets must be positive")
    ref = urllib.parse.quote(revision or "main", safe="")
    url = f"https://huggingface.co/api/datasets/{REPOSITORY}/revision/{ref}?blobs=true"
    with _request(url) as response:
        metadata = json.load(response)
    pinned = metadata["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", pinned):
        raise ValueError("dataset did not resolve to a commit SHA")
    cache = cache / pinned
    cache.mkdir(parents=True, exist_ok=True)
    files = sorted((entry for entry in metadata["siblings"]
                    if re.fullmatch(r"data/train-\d+-of-\d+\.parquet", entry["rfilename"])),
                   key=lambda entry: entry["rfilename"])[:max_shards]
    if not files:
        raise ValueError("no raw Parquet shards in the pinned Metamon revision")
    used = 0
    sources = []
    for entry in files:
        name = entry["rfilename"]
        expected = (entry.get("lfs") or {}).get("sha256")
        path = cache / Path(name).name
        if not path.exists():
            size = int(entry.get("size") or (entry.get("lfs") or {}).get("size") or 0)
            if size > max_bytes - used:
                raise ValueError("selected shard exceeds remaining download budget")
            temporary = path.with_suffix(".partial")
            with _request(f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{pinned}/{name}") as source:
                with temporary.open("wb") as destination:
                    while chunk := source.read(1024**2):
                        used += len(chunk)
                        if used > max_bytes:
                            raise ValueError("download byte budget exhausted; partial shard was not imported")
                        destination.write(chunk)
            digest = _sha256(temporary)
            if expected and digest != expected:
                raise ValueError(f"Metamon shard SHA256 mismatch: {name}")
            temporary.replace(path)
        digest = _sha256(path)
        if expected and digest != expected:
            raise ValueError(f"cached Metamon shard SHA256 mismatch: {name}")
        sources.append({"name": name, "path": str(path.resolve()), "sha256": digest})
    return pinned, sources


def import_shards(sources: list[dict], out: Path, *, revision: str, max_games: int = 1000,
                  battle_format: str = "gen9ou", min_rating: int = 1650,
                  after: int | None = None, before: int | None = None) -> dict:
    import pyarrow.parquet as pq

    if max_games < 1 or min_rating < 0 or not re.fullmatch(r"[a-z0-9]+", battle_format):
        raise ValueError("invalid import limit, rating or format")
    if after is not None and before is not None and after >= before:
        raise ValueError("after must precede before")
    metadata_path = out / ".metamon" / "import.json"
    filters = {"format": battle_format, "min_rating": min_rating, "after": after, "before": before}
    manifest = (json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists()
                else {"schema_version": 1, "repository": REPOSITORY, "revision": revision,
                      "filters": filters, "sources": {}, "progress": {}, "filter_counts": {}})
    if manifest["revision"] != revision or manifest["filters"] != filters:
        raise ValueError("import revision/filters changed; choose a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    kept = len(list(out.glob(f"{battle_format}-*.json")))
    counts = Counter(manifest["filter_counts"])

    def save(status):
        manifest.update(status=status, kept_games=kept, target_games=max_games,
                        filter_counts=dict(counts), private_truth=False,
                        observation_source="public_spectator_protocol")
        _atomic_json(metadata_path, manifest)

    save("in_progress")
    for source in sources:
        name, path, digest = source["name"], Path(source["path"]), source["sha256"]
        if _sha256(path) != digest:
            raise ValueError(f"input shard changed: {path}")
        previous = manifest["sources"].get(name)
        if previous is not None and previous["sha256"] != digest:
            raise ValueError(f"resume shard changed: {name}")
        manifest["sources"][name] = source
        row_number = 0
        cursor = int(manifest["progress"].get(name, 0))
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            for row in batch.to_pylist():
                current = row_number
                row_number += 1
                if current < cursor:
                    continue
                if kept >= max_games:
                    save("target_reached")
                    return manifest
                reason = None
                rid = str(row.get("id", ""))
                if (row.get("formatid") != battle_format
                        or not rid.startswith(battle_format + "-")):
                    reason = "other_format"
                elif not re.fullmatch(r"[a-z0-9-]+", rid):
                    reason = "invalid_id"
                else:
                    try:
                        rating = int(row.get("rating") or 0)
                        timestamp = int(row["uploadtime"])
                    except (ValueError, TypeError, KeyError):
                        reason = "invalid_metadata"
                    if reason is None:
                        if rating < min_rating:
                            reason = "below_rating"
                        elif ((after is not None and timestamp < after)
                              or (before is not None and timestamp >= before)):
                            reason = "outside_dates"
                        elif not isinstance(row.get("log"), str) or not row["log"]:
                            reason = "missing_log"
                        elif not any(line.startswith(("|win|", "|tie")) for line in row["log"].splitlines()):
                            reason = "unfinished"
                if reason is not None:
                    counts[reason] += 1
                else:
                    replay = {**row, "rating": rating, "_fog_source": {
                        "provider": "metamon_raw_replays", "repository": REPOSITORY,
                        "revision": revision, "shard": name, "shard_sha256": digest,
                        "row": current, "private_truth": False,
                    }}
                    destination = out / f"{rid}.json"
                    if destination.exists():
                        old = json.loads(destination.read_text(encoding="utf-8"))
                        # A duplicate ID may occur in another shard; its public record must agree.
                        if {k: v for k, v in old.items() if k != "_fog_source"} != row | {"rating": rating}:
                            raise ValueError(f"conflicting immutable replay ID: {rid}")
                        counts["duplicate_id"] += 1
                    else:
                        _atomic_json(destination, replay)
                        kept += 1
                manifest["progress"][name] = current + 1
            save("in_progress")
    save("target_reached" if kept >= max_games else "selected_shards_exhausted")
    return manifest


def _timestamp(value: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=UTC).timestamp())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("data/metamon_cache"))
    parser.add_argument("--revision")
    parser.add_argument("--parquet", type=Path, action="append", help="local raw Parquet shard; repeatable")
    parser.add_argument("--max-games", type=int, default=1000)
    parser.add_argument("--max-shards", type=int, default=1)
    parser.add_argument("--max-download-mb", type=int, default=256)
    parser.add_argument("--format", default="gen9ou")
    parser.add_argument("--min-rating", type=int, default=1650)
    parser.add_argument("--after", type=_timestamp, help="inclusive UTC date, YYYY-MM-DD")
    parser.add_argument("--before", type=_timestamp, help="exclusive UTC date, YYYY-MM-DD")
    args = parser.parse_args()
    revision = args.revision
    prior = args.out / ".metamon" / "import.json"
    if revision is None and prior.exists():
        revision = json.loads(prior.read_text(encoding="utf-8"))["revision"]
    if args.parquet:
        if revision is None:
            parser.error("offline imports require --revision to identify their source release")
        sources = [{"name": path.name, "path": str(path.resolve()), "sha256": _sha256(path)}
                   for path in args.parquet]
    else:
        revision, sources = download_shards(args.cache, revision=revision,
                                           max_shards=args.max_shards,
                                           max_bytes=args.max_download_mb * 1024**2)
    result = import_shards(sources, args.out, revision=revision, max_games=args.max_games,
                           battle_format=args.format, min_rating=args.min_rating,
                           after=args.after, before=args.before)
    print(json.dumps({key: result[key] for key in ("status", "revision", "kept_games", "filter_counts")},
                     indent=2))


if __name__ == "__main__":
    main()
