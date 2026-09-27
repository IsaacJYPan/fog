"""Generate metagame-dependent lookup tables from the pinned Showdown data files.

Pipeline: `server/dump-dex.cjs` (Node, run against the BUILT pinned pokemon-showdown checkout) dumps
the dex to one JSON blob; the build_* functions here filter/shape it into the versioned tables under
`constants/generated/`; `tables.py` loads those into frozen numpy arrays + masks at runtime.

Generated JSON schemas (all carry `_meta: {showdown_commit, table}`):

  species.json   { ids: [species_id..., "<RARE_SPECIES>"],
                   features: { id: { name, base_stats: [6], types: [..], weightkg, weight_bucket,
                                     abilities: [ability_id..], learnset: [move_id..] } } }
  moves.json     { ids: [move_id...],
                   features: { id: { name, type, category, base_power, accuracy, bypass_accuracy,
                                     priority, flags: [..], is_hazard, is_recovery, is_pivot,
                                     is_setup, inflicts_status } } }
  items.json     { ids: [item_id..., "<OTHER_ITEM>", "<NO_ITEM>", "<UNKNOWN>"],
                   usable: { species_id: [item_id..] } }        # nonzero usage on that species
  abilities.json { ids: [ability_id...] }
  spreads.json   { species: { species_id: [ {nature, evs, stats, weight}... ] } }  # + <OTHER_SPREAD>
                                                                # implicit last index per species

Usage filtering (chaos file, §4.1): species >= 0.1% usage (plus anything in `extra_species`, e.g. from
scraped replays); items >= 0.5% usage share on at least one included species. Without a usage file
(early PoC), everything gen9-legal is included — bigger tables, same code paths.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from fog.constants.schema import NO_ITEM, OTHER_ITEM, RARE_SPECIES, UNKNOWN_ITEM

GENERATED_DIR = Path(__file__).resolve().parent / "generated"
BRIDGE = Path(__file__).resolve().parents[2] / "server" / "dump-dex.cjs"

SPECIES_USAGE_CUT = 0.001   # >= 0.1% (§1.1)
ITEM_USAGE_CUT = 0.005      # >= 0.5% share on any included species (§1.4)

# Grass Knot / Low Kick weight classes (kg cutoffs), §1.2. Bucket = index into this list.
WEIGHT_CUTOFFS_KG = (10.0, 25.0, 50.0, 100.0, 200.0)  # <10, <25, <50, <100, <200, >=200 -> buckets 0-5

HAZARD_MOVES = {"stealthrock", "spikes", "toxicspikes", "stickyweb", "stoneaxe", "ceaselessedge"}


def to_id(name: str) -> str:
    """Showdown id normalization: lowercase, strip everything non-alphanumeric."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def weight_bucket(weightkg: float) -> int:
    for i, cut in enumerate(WEIGHT_CUTOFFS_KG):
        if weightkg < cut:
            return i
    return len(WEIGHT_CUTOFFS_KG)


def load_showdown_data(showdown_root: Path) -> dict[str, Any]:
    """Run the Node bridge against the pinned checkout and parse its JSON dump.

    The checkout must be built (`node build`) first. The dump carries the checkout's commit sha in
    `commit`; it is written into every generated table's `_meta` for traceability.
    """
    proc = subprocess.run(
        ["node", str(BRIDGE), str(showdown_root)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"dump-dex.cjs failed:\n{proc.stderr}")
    return json.loads(proc.stdout)


# ------------------------------------------------------------------------------------------- helpers
def _gen9_learnset(dump: dict[str, Any], species_id: str) -> set[str]:
    """Gen-9-legal move ids for a species: sources prefixed '9', walking the forme/prevo fallback
    chain (formes fall back to baseSpecies/changesFrom; evolutions inherit prevo learnsets)."""
    moves: set[str] = set()
    seen: set[str] = set()
    stack = [species_id]
    while stack:
        sid = stack.pop()
        if sid in seen:
            continue
        seen.add(sid)
        for move_id, sources in dump.get("learnsets", {}).get(sid, {}).items():
            if any(str(src).startswith("9") for src in sources):
                moves.add(move_id)
        entry = dump["pokedex"].get(sid, {})
        for key in ("changesFrom", "baseSpecies", "prevo"):
            nxt = entry.get(key)
            if nxt:
                stack.append(nxt)
    return moves


def _usage_species_shares(usage: dict[str, Any] | None) -> dict[str, float]:
    """chaos json -> {species_id: usage share}. Accepts the raw chaos file ({"data": {...}})."""
    if not usage:
        return {}
    data = usage.get("data", usage)
    return {to_id(name): float(row.get("usage", 0.0)) for name, row in data.items()}


def _usage_item_shares(usage: dict[str, Any] | None) -> dict[str, dict[str, float]]:
    """chaos json -> {species_id: {item_id: share}} (chaos 'Items' values are raw weights)."""
    out: dict[str, dict[str, float]] = {}
    if not usage:
        return out
    data = usage.get("data", usage)
    for name, row in data.items():
        items = row.get("Items", {})
        total = sum(items.values()) or 1.0
        out[to_id(name)] = {to_id(i): v / total for i, v in items.items()}
    return out


# ------------------------------------------------------------------------------------- table builders
def build_species_table(
    dex: dict[str, Any],
    usage: dict[str, Any] | None,
    extra_species: set[str] | None = None,
) -> dict[str, Any]:
    """id<->index + per-species static features. Filter to >=0.1% usage OR appears in scraped data;
    without a usage file, include everything gen9-legal."""
    shares = _usage_species_shares(usage)
    extra = {to_id(s) for s in (extra_species or set())}

    ids = []
    for sid, entry in dex["pokedex"].items():
        if entry.get("isNonstandard") or entry.get("num", 0) <= 0:
            continue
        if shares and shares.get(sid, 0.0) < SPECIES_USAGE_CUT and sid not in extra:
            continue
        ids.append(sid)
    ids.sort()

    features: dict[str, Any] = {}
    for sid in ids:
        entry = dex["pokedex"][sid]
        bs = entry["baseStats"]
        features[sid] = {
            "name": entry["name"],
            "base_stats": [bs["hp"], bs["atk"], bs["def"], bs["spa"], bs["spd"], bs["spe"]],
            "types": [t.lower() for t in entry["types"]],
            "weightkg": entry["weightkg"],
            "weight_bucket": weight_bucket(entry["weightkg"]),
            "abilities": sorted({to_id(a) for a in entry["abilities"].values() if a}),
            "learnset": sorted(_gen9_learnset(dex, sid)),
        }
    return {"ids": ids + [RARE_SPECIES], "features": features}


def build_moves_table(dex: dict[str, Any], species_table: dict[str, Any]) -> dict[str, Any]:
    """Index over all moves appearing in any included species' learnset (§1.3), with static features."""
    in_learnsets: set[str] = set()
    for feats in species_table["features"].values():
        in_learnsets.update(feats["learnset"])

    ids = sorted(m for m in in_learnsets if m in dex["moves"] and not dex["moves"][m].get("isNonstandard"))
    features: dict[str, Any] = {}
    for mid in ids:
        m = dex["moves"][mid]
        boosts = m.get("boosts")
        secondary = m.get("secondary") or {}
        features[mid] = {
            "name": m["name"],
            "type": m["type"].lower(),
            "category": m["category"].lower(),
            "base_power": m["basePower"],
            "accuracy": 100 if m["accuracy"] is True else m["accuracy"],
            "bypass_accuracy": m["accuracy"] is True,
            "priority": m["priority"],
            "flags": sorted(m.get("flags", {}).keys()),
            "is_hazard": mid in HAZARD_MOVES or m.get("sideCondition") in
                         ("stealthrock", "spikes", "toxicspikes", "stickyweb"),
            "is_recovery": bool(m.get("heal")) or "heal" in m.get("flags", {}),
            "is_pivot": bool(m.get("selfSwitch")),
            "is_setup": bool(boosts) and m["category"].lower() == "status"
                        and m.get("target") == "self",
            "inflicts_status": bool(m.get("status") or secondary.get("status")),
        }
    return {"ids": ids, "features": features}


def build_items_table(
    dex: dict[str, Any], usage: dict[str, Any] | None, species_table: dict[str, Any]
) -> dict[str, Any]:
    """Items with >=0.5% usage share on any included species (§1.4) + sentinels, plus the per-species
    nonzero-usage lists that back the belief model's usable-items mask (§3.2)."""
    item_shares = _usage_item_shares(usage)
    included_species = [s for s in species_table["ids"] if s != RARE_SPECIES]

    if item_shares:
        keep: set[str] = set()
        for sid in included_species:
            for iid, share in item_shares.get(sid, {}).items():
                if share >= ITEM_USAGE_CUT and iid in dex["items"]:
                    keep.add(iid)
        ids = sorted(keep)
    else:
        ids = sorted(i for i, e in dex["items"].items() if not e.get("isNonstandard"))

    id_set = set(ids)
    usable: dict[str, list[str]] = {}
    for sid in included_species:
        shares = item_shares.get(sid)
        if shares:
            lst = sorted(i for i, v in shares.items() if v > 0 and i in id_set)
            if lst:
                usable[sid] = lst
        # species absent from usage -> omitted; tables.py falls back to all-legal for them

    return {"ids": ids + [OTHER_ITEM, NO_ITEM, UNKNOWN_ITEM], "usable": usable}


def build_abilities_table(dex: dict[str, Any], species_table: dict[str, Any]) -> dict[str, Any]:
    """Global ability index over abilities of included species (§1.5); prediction is always masked to
    the species' legal 1-3 anyway."""
    keep: set[str] = set()
    for feats in species_table["features"].values():
        keep.update(feats["abilities"])
    ids = sorted(a for a in keep if a in dex["abilities"])
    return {"ids": ids}


def write_table(name: str, table: dict[str, Any], commit: str, out_dir: Path = GENERATED_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    table = {"_meta": {"showdown_commit": commit, "table": name}, **table}
    out = out_dir / f"{name}.json"
    with open(out, "w") as f:
        json.dump(table, f, indent=2)
    return out


def build_all(
    dump: dict[str, Any],
    usage: dict[str, Any] | None = None,
    extra_species: set[str] | None = None,
    out_dir: Path = GENERATED_DIR,
) -> dict[str, Path]:
    """Build + write every table (except spreads — see scripts/download_usage + build_spreads)."""
    commit = dump.get("commit", "unknown")
    species = build_species_table(dump, usage, extra_species)
    moves = build_moves_table(dump, species)
    items = build_items_table(dump, usage, species)
    abilities = build_abilities_table(dump, species)
    return {
        "species": write_table("species", species, commit, out_dir),
        "moves": write_table("moves", moves, commit, out_dir),
        "items": write_table("items", items, commit, out_dir),
        "abilities": write_table("abilities", abilities, commit, out_dir),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build FOG constant tables from Showdown data.")
    ap.add_argument("--showdown", type=Path, required=True, help="path to pinned pokemon-showdown checkout")
    ap.add_argument("--usage", type=Path, help="Smogon chaos usage json for the pinned month")
    ap.add_argument("--out", type=Path, default=GENERATED_DIR)
    args = ap.parse_args()

    dump = load_showdown_data(args.showdown)
    usage = None
    if args.usage:
        with open(args.usage, encoding="utf-8") as f:
            usage = json.load(f)
    paths = build_all(dump, usage, out_dir=args.out)
    for name, p in paths.items():
        print(f"wrote {name}: {p}")


if __name__ == "__main__":
    main()
