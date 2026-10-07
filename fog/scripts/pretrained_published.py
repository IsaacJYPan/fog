"""Freeze a bounded Gen9OU subset without extracting the complete published pile."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile

from fog.pretrained.data import composition, file_hashes
from fog.pretrained.actions import identifier


def roster_key(names) -> str:
    return hashlib.sha256(
        json.dumps(sorted(identifier(n) for n in names), separators=(",", ":")).encode()
    ).hexdigest()


def held_out_rosters(state: dict, protected: set[str]) -> bool:
    own = [state["player_active_pokemon"]["name"]] + [
        p["name"] for p in state["available_switches"]
    ]
    opponent = state.get("opponent_teampreview", [])
    # Both rosters must be available so either viewpoint can exclude a protected matchup.
    if len(own) != 6 or len(opponent) != 6:
        return True
    return roster_key(own) in protected or roster_key(opponent) in protected


def freeze(archive: Path, pools: Path, out: Path, *, max_records=100000, byte_cap=15_000_000_000):
    import lz4.frame

    if archive.suffix == ".part":
        raise ValueError(
            "Only a completely downloaded, publisher-hash-verified archive can be frozen"
        )
    source = json.loads(archive.with_suffix(archive.suffix + ".json").read_text())
    if file_hashes([archive])[str(archive)] != source["sha256"]:
        raise ValueError("Published archive changed after download verification")
    if out.exists():
        raise FileExistsError(out)
    protected = set()
    for name in ("development", "final"):
        protected.update(
            composition(t) for t in json.loads((pools / (name + ".json")).read_text())["teams"]
        )
    out.mkdir(parents=True)
    destination = out / "train/gen9ou"
    destination.mkdir(parents=True)
    counters, records, total_bytes = Counter(), [], 0
    with lz4.frame.open(archive, "rb") as stream, tarfile.open(fileobj=stream, mode="r|") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".json.lz4"):
                continue
            if member.size > 32 * 1024 * 1024:
                raise ValueError("Unexpected oversized published trajectory")
            counters["seen"] += 1
            payload = tar.extractfile(member).read()
            data = json.loads(lz4.frame.decompress(payload))
            states, actions = data["states"], data["actions"]
            if len(states) != len(actions) or len(states) < 3 or actions[-1] != -1:
                counters["invalid_trajectory"] += 1
                continue
            if held_out_rosters(states[0], protected):
                counters["protected_or_unknown_roster"] += 1
                continue
            terminal = states[-1]
            if not terminal["battle_won"] and not terminal["battle_lost"]:
                counters["unverified_terminal"] += 1
                continue
            # Gen9OU now bans these moves. Reject the whole trajectory even when its
            # recorded chosen action would be legal on the historical server.
            banned = {"terablast", "batonpass", "lastrespects", "shedtail"}
            if any(
                identifier(m["name"]) in banned
                for state in states
                for pokemon in [state["player_active_pokemon"], *state["available_switches"]]
                for m in pokemon["moves"]
            ):
                counters["historically_banned_move"] += 1
                continue
            if total_bytes + len(payload) > byte_cap:
                counters["byte_cap_reached"] += 1
                break
            digest = hashlib.sha256(payload).hexdigest()
            name = digest + ".json.lz4"
            path = destination / name
            if path.exists():
                counters["duplicate_payload"] += 1
                continue
            path.write_bytes(payload)
            total_bytes += len(payload)
            records.append(
                {
                    "path": str(path.relative_to(out)).replace("\\", "/"),
                    "sha256": digest,
                    "member": member.name,
                    "bytes": len(payload),
                    "decisions": len(actions) - 1,
                }
            )
            if len(records) % 1000 == 0:
                print(
                    "Frozen",
                    len(records),
                    "published perspectives",
                    total_bytes,
                    "bytes",
                    flush=True,
                )
            if len(records) >= max_records:
                break
    manifest = {
        "source": source,
        "records": records,
        "bytes": total_bytes,
        "counts": dict(counters),
        "perspectives": len(records),
        "max_matches": max_records,
        "byte_cap": byte_cap,
        "split": "all selected published perspectives are training-only",
        "match_identity_limit": "Upstream assigns random IDs independently to saved POVs. "
        "No published POV enters development or final evaluation.",
        "historical_exposure": "These matches may overlap Kakuna pretraining.",
        "belief_label_provenance": "No complete submitted teams in these records; mask all set labels.",
        "outcome_label_provenance": "Only explicit won/lost terminal records are accepted.",
        "protected_rosters": sorted(protected),
        "pool_manifest_sha256": file_hashes([pools / "manifest.json"]),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--pools", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=100000)
    args = parser.parse_args()
    result = freeze(args.archive, args.pools, args.out, max_records=args.max_records)
    print(json.dumps({k: result[k] for k in ("perspectives", "bytes", "counts")}))


if __name__ == "__main__":
    main()
