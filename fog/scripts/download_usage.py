"""Download Smogon chaos usage files for the pinned month and build derived tables.

Produces, under the constants generated dir + data/usage/:
  * data/usage/gen9ou-<rating>-<month>.json  -- raw chaos file (immutable cache)
  * constants/generated/spreads.json          -- SPREADS[species] via build_spreads (§1.6)
  * constants/generated/priors.json           -- item/move usage priors + teammate co-occurrence

The chaos file schema (per species under `data`):
  { "Moves": {move: weight}, "Items": {item: weight}, "Abilities": {ability: weight},
    "Spreads": {"Nature:hp/atk/def/spa/spd/spe": weight}, "Teammates": {species: weight},
    "usage": float, "Raw count": int, ... }

Run:
    python -m fog.scripts.download_usage --month 2026-06 --rating 1695
(then record the month in configs/default.yaml::pins.usage_month)
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from fog.constants.build import to_id
from fog.constants.build_spreads import cluster_species_spreads, parse_chaos_spread
from fog.constants.schema import OTHER_SPREAD
from fog.constants.tables import GENERATED_DIR, Tables

CHAOS_URL = "https://www.smogon.com/stats/{month}/chaos/gen9ou-{rating}.json"
DATA_USAGE = Path("data/usage")


def fetch_chaos(month: str, rating: int, out_dir: Path = DATA_USAGE) -> Path:
    """Download (and cache) one chaos file. Returns the local path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"gen9ou-{rating}-{month}.json"
    if dest.exists():
        return dest
    url = CHAOS_URL.format(month=month, rating=rating)
    req = urllib.request.Request(url, headers={"User-Agent": "fog/0.1 (research)"})
    with urllib.request.urlopen(req, timeout=120) as r:
        dest.write_bytes(r.read())
    return dest


def build_spreads(chaos: dict[str, Any], tables: Tables) -> dict[str, Any]:
    """Cluster each species' chaos spreads into <=16 representatives (§1.6).

    Only species present in the generated SPECIES table get an entry; the rest fall back to
    <OTHER_SPREAD> at encode time. Returns the spreads.json payload {"species": {sid: [clusters]}}.
    """
    data = chaos.get("data", chaos)
    species_feats = tables.species["features"]
    out: dict[str, list] = {}
    for name, row in data.items():
        sid = to_id(name)
        if sid not in species_feats:
            continue
        base = np.array(species_feats[sid]["base_stats"], dtype=np.int64)
        observed = []
        for key, weight in (row.get("Spreads") or {}).items():
            try:
                nature, evs = parse_chaos_spread(key)
            except (TypeError, ValueError):
                continue
            observed.append({"nature": nature, "evs": list(evs), "usage_weight": float(weight)})
        if not observed:
            continue
        clusters = cluster_species_spreads(observed, base)
        if clusters:
            out[sid] = clusters
    return {"species": out, "_other": OTHER_SPREAD}


def build_priors(chaos: dict[str, Any], tables: Tables) -> dict[str, Any]:
    """Per-species item/move/ability/tera usage priors + a teammate co-occurrence matrix (skip-gram
    warm-start, §4.1). Normalized to distributions; only in-table ids kept. Ability priors are
    filtered to the species' legal abilities and tera priors to real types (the chaos file carries
    a "nothing" tera bucket for non-terastallized games — not a sheet attribute; dropped)."""
    from fog.constants.schema import TYPE_INDEX

    data = chaos.get("data", chaos)
    species_ids = [s for s in tables.species["ids"] if s in tables.species["features"]]
    sp_index = {s: i for i, s in enumerate(species_ids)}

    item_priors: dict[str, dict[str, float]] = {}
    move_priors: dict[str, dict[str, float]] = {}
    ability_priors: dict[str, dict[str, float]] = {}
    tera_priors: dict[str, dict[str, float]] = {}
    cooc = np.zeros((len(species_ids), len(species_ids)), dtype=np.float32)

    def norm(d: dict[str, float]) -> dict[str, float]:
        tot = sum(d.values()) or 1.0
        return {k: v / tot for k, v in d.items()}

    for name, row in data.items():
        sid = to_id(name)
        if sid not in sp_index:
            continue
        legal_abilities = set(tables.species["features"][sid].get("abilities") or [])
        item_priors[sid] = norm({to_id(k): float(v) for k, v in (row.get("Items") or {}).items()})
        move_priors[sid] = norm({to_id(k): float(v) for k, v in (row.get("Moves") or {}).items()})
        ability_priors[sid] = norm({
            to_id(k): float(v) for k, v in (row.get("Abilities") or {}).items()
            if not legal_abilities or to_id(k) in legal_abilities
        })
        tera_priors[sid] = norm({
            to_id(k): float(v) for k, v in (row.get("Tera Types") or {}).items()
            if to_id(k) in TYPE_INDEX
        })
        for mate, w in (row.get("Teammates") or {}).items():
            mid = to_id(mate)
            if mid in sp_index:
                cooc[sp_index[sid], sp_index[mid]] = float(w)

    return {
        "species_order": species_ids,
        "item_priors": item_priors,
        "move_priors": move_priors,
        "ability_priors": ability_priors,
        "tera_priors": tera_priors,
        "teammate_cooccurrence": cooc.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Download Smogon chaos usage + build derived tables (§4.1).")
    ap.add_argument("--month", required=True, help="YYYY-MM (pinned)")
    ap.add_argument("--rating", type=int, default=1695)
    ap.add_argument("--out", type=Path, default=DATA_USAGE)
    ap.add_argument("--gen-dir", type=Path, default=GENERATED_DIR)
    args = ap.parse_args()

    path = fetch_chaos(args.month, args.rating, args.out)
    print(f"chaos: {path} ({path.stat().st_size/1e6:.1f} MB)")
    chaos = json.loads(path.read_text(encoding="utf-8"))

    tables = Tables(gen_dir=args.gen_dir)
    commit = tables.species["_meta"]["showdown_commit"]

    spreads = build_spreads(chaos, tables)
    spreads["_meta"] = {"showdown_commit": commit, "usage_month": args.month, "table": "spreads"}
    (args.gen_dir / "spreads.json").write_text(json.dumps(spreads), encoding="utf-8")
    print(f"spreads: {len(spreads['species'])} species clustered -> {args.gen_dir/'spreads.json'}")

    priors = build_priors(chaos, tables)
    priors["_meta"] = {"showdown_commit": commit, "usage_month": args.month, "table": "priors"}
    (args.gen_dir / "priors.json").write_text(json.dumps(priors), encoding="utf-8")
    print(f"priors: {len(priors['item_priors'])} species -> {args.gen_dir/'priors.json'}")
    print(f"NOTE: set pins.usage_month: {args.month} in configs/default.yaml")


if __name__ == "__main__":
    main()
