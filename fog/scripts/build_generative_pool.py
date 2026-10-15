"""Generative set-variant team pool (audit follow-up 2026-07-05): usage-weighted procedural sets
on a fixed species core.

WHY: POOL-A's 8 hand-built variants carry ~3 bits of team identity — enough to validate the belief
pipeline (M1.5) but thin oxygen for the leakage experiment, and a 3.5M-param probe can memorize a
finite 8-team pool. This script enlarges the hidden-information space: every team shares the SAME
6-species core (preview reveals nothing about identity), but each slot's item / ability / 4 moves /
EV-spread cluster / tera type are sampled from the pinned month's real usage distributions
(priors.json + spreads.json). The result is a bank of N validator-clean teams the existing pipeline
consumes unchanged (PoolTeambuilder samples per battle; the recording teambuilder is the exact
belief-truth source; `PoolTruth.hidden_info_warning` passes because species multisets are shared).

HONESTY NOTES (write these into any M3 report that uses this pool):
  * Attributes are sampled from MARGINAL usage distributions — chaos files carry no joint set data,
    so item/move coherence is only as good as the marginals are peaked (a rare Choice-item +
    setup-move draw can happen; the simulator validates legality, not wisdom). The hand-built
    POOL-A stays the coherence-controlled pool; this one is the entropy-scaled pool.
  * The per-slot entropy report printed at build time is an UPPER BOUND on concealable information
    (truncated-marginal entropies; the belief net exploits usage correlations the bound ignores).
  * Sampling truncates each marginal to `--mass` cumulative probability (default 0.95) — the long
    tail of one-off sets is noise at these sample sizes, and dropping it keeps every sampled
    attribute inside the belief vocabulary (no <OTHER_ITEM>/<OTHER_SPREAD> truth tokens).

Build (deterministic given seed + pinned constants; the bank is COMMITTED like the other pools —
regenerability is not a substitute for the exact experiment definition):

    python -m fog.scripts.build_generative_pool --n 128 --seed 0
    # then point a run at it: configs/archive/m3_leakage_gen.yaml
    # (env.team_pool: data/teams/pool_gen)
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from fog.constants.schema import MAX_SPREADS_PER_SPECIES, TYPE_INDEX
from fog.constants.tables import Tables

TEAMS_DIR = Path("data/teams")
DEFAULT_CORE_POOL = TEAMS_DIR / "pool_a"

EV_LABELS = ["HP", "Atk", "Def", "SpA", "SpD", "Spe"]


def species_core_from_pool(path: str | Path) -> list[str]:
    """The shared species core (ids, slot order) of an existing pool — read from its first paste."""
    from fog.env.team_truth import _to_id
    from fog.env.teams import load_pool

    paste = load_pool(path)[0]
    from poke_env.teambuilder import Teambuilder

    class _T(Teambuilder):
        def yield_team(self) -> str:  # pragma: no cover - never battled
            return ""

    from fog.env.teams import normalize_paste_newlines

    mons = _T().parse_showdown_team(normalize_paste_newlines(paste))
    core = [_to_id(m.species or m.nickname or "") for m in mons[:6]]
    if len(core) != 6 or any(not s for s in core):
        raise ValueError(f"could not read a 6-species core from {path}")
    return core


def truncate_prior(dist: dict[str, float], mass: float, min_keep: int = 1) -> dict[str, float]:
    """Keep the head of a usage distribution up to cumulative `mass`, renormalized."""
    items = sorted(dist.items(), key=lambda kv: -kv[1])
    kept, acc = [], 0.0
    for k, v in items:
        kept.append((k, v))
        acc += v
        if acc >= mass and len(kept) >= min_keep:
            break
    tot = sum(v for _, v in kept) or 1.0
    return {k: v / tot for k, v in kept}


def _weighted_choice(rng: random.Random, dist: dict[str, float]) -> str:
    return rng.choices(list(dist), weights=list(dist.values()), k=1)[0]


def _weighted_sample_distinct(rng: random.Random, dist: dict[str, float], k: int) -> list[str]:
    """k distinct weighted draws (sequential without replacement)."""
    pool = dict(dist)
    out: list[str] = []
    for _ in range(k):
        pick = _weighted_choice(rng, pool)
        out.append(pick)
        del pool[pick]
    return out


class SetSampler:
    """Per-species samplers over table-vocabulary-filtered, mass-truncated usage marginals."""

    def __init__(self, core: list[str], tables: Tables, mass: float = 0.95) -> None:
        priors = json.loads((Path(tables.gen_dir) / "priors.json").read_text(encoding="utf-8"))
        if "ability_priors" not in priors or "tera_priors" not in priors:
            raise ValueError(
                "priors.json predates ability/tera priors — regenerate it: "
                "python -m fog.scripts.download_usage --month <pinned> (cached chaos file, no network)"
            )
        self.tables = tables
        self.mass = mass
        self.core = core
        self.moves: dict[str, dict[str, float]] = {}
        self.items: dict[str, dict[str, float]] = {}
        self.abilities: dict[str, dict[str, float]] = {}
        self.teras: dict[str, dict[str, float]] = {}
        self.spread_clusters: dict[str, list[dict]] = {}
        for sid in core:
            feats = tables.species["features"].get(sid)
            if feats is None:
                raise ValueError(f"core species {sid!r} not in the generated SPECIES table")
            learnset = set(feats.get("learnset") or [])
            mv = {
                m: w for m, w in (priors["move_priors"].get(sid) or {}).items()
                if m and m in learnset and tables.move_index(m, default=-1) >= 0
            }
            it = {
                i: w for i, w in (priors["item_priors"].get(sid) or {}).items()
                if i and i != "nothing" and i in tables._item_idx
            }
            ab = {
                a: w for a, w in (priors["ability_priors"].get(sid) or {}).items()
                if tables.ability_index(a, default=-1) >= 0
            }
            te = {t: w for t, w in (priors["tera_priors"].get(sid) or {}).items() if t in TYPE_INDEX}
            clusters = (tables.spreads.get("species") or {}).get(sid, [])[:MAX_SPREADS_PER_SPECIES]
            if len(mv) < 4 or not it or not ab or not te or not clusters:
                raise ValueError(
                    f"{sid}: insufficient usage data to sample sets "
                    f"(moves={len(mv)}, items={len(it)}, abilities={len(ab)}, "
                    f"teras={len(te)}, spreads={len(clusters)})"
                )
            self.moves[sid] = truncate_prior(mv, mass, min_keep=6)
            self.items[sid] = truncate_prior(it, mass)
            self.abilities[sid] = truncate_prior(ab, mass)
            self.teras[sid] = truncate_prior(te, mass)
            self.spread_clusters[sid] = clusters

    def sample_set(self, sid: str, rng: random.Random) -> dict:
        weights = [float(c.get("weight", 1.0)) for c in self.spread_clusters[sid]]
        spread_i = rng.choices(range(len(weights)), weights=weights, k=1)[0]
        cluster = self.spread_clusters[sid][spread_i]
        return {
            "species": sid,
            "item": _weighted_choice(rng, self.items[sid]),
            "ability": _weighted_choice(rng, self.abilities[sid]),
            "moves": _weighted_sample_distinct(rng, self.moves[sid], 4),
            "nature": cluster["nature"],
            "evs": list(cluster["evs"]),
            "spread_index": spread_i,
            "tera": _weighted_choice(rng, self.teras[sid]),
        }

    def sample_team(self, rng: random.Random) -> list[dict]:
        return [self.sample_set(sid, rng) for sid in self.core]

    def entropy_report(self) -> dict[str, dict[str, float]]:
        """Per-species per-attribute entropies (bits) of the truncated sampling marginals.
        Moves are reported as log2 C(n_candidates, 4) — a loose upper bound (weighted sampling
        concentrates mass on the head)."""

        def h(dist: dict[str, float]) -> float:
            return -sum(p * math.log2(p) for p in dist.values() if p > 0)

        report: dict[str, dict[str, float]] = {}
        for sid in self.core:
            weights = [float(c.get("weight", 1.0)) for c in self.spread_clusters[sid]]
            tot = sum(weights) or 1.0
            spread_h = h({str(i): w / tot for i, w in enumerate(weights)})
            n_mv = len(self.moves[sid])
            report[sid] = {
                "item": round(h(self.items[sid]), 2),
                "ability": round(h(self.abilities[sid]), 2),
                "moves_upper": round(math.log2(math.comb(n_mv, 4)), 2) if n_mv >= 4 else 0.0,
                "spread": round(spread_h, 2),
                "tera": round(h(self.teras[sid]), 2),
            }
        return report


def render_paste(team: list[dict], tables: Tables) -> str:
    """Sampled sets -> Showdown paste. Display names where the tables have them (species, moves);
    raw ids elsewhere — the importer/validator normalizes ids identically either way."""
    blocks = []
    for s in team:
        name = tables.species["features"][s["species"]].get("name", s["species"])
        move_feats = tables.moves.get("features", {})
        lines = [
            f"{name} @ {s['item']}",
            f"Ability: {s['ability']}",
            f"Tera Type: {s['tera'].capitalize()}",
            "EVs: " + " / ".join(
                f"{ev} {lab}" for ev, lab in zip(s["evs"], EV_LABELS) if ev
            ),
            f"{s['nature'].capitalize()} Nature",
        ]
        lines += [f"- {move_feats.get(m, {}).get('name', m)}" for m in s["moves"]]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


def _team_key(team: list[dict]) -> tuple:
    return tuple(
        (s["species"], s["item"], s["ability"], tuple(sorted(s["moves"])), s["spread_index"], s["tera"])
        for s in team
    )


def generate_pool(
    n: int,
    seed: int,
    core_pool: str | Path = DEFAULT_CORE_POOL,
    mass: float = 0.95,
    tables: Tables | None = None,
    validate: bool = True,
    battle_format: str = "gen9ou",
    max_attempts_per_team: int = 20,
) -> list[str]:
    """N distinct, (optionally) validator-clean pastes on the shared core. Deterministic in
    (seed, pinned constants); validation rejections advance the RNG, so the committed bank — not
    regeneration — is the experiment definition."""
    tables = tables or Tables()
    sampler = SetSampler(species_core_from_pool(core_pool), tables, mass=mass)
    rng = random.Random(seed)
    pastes: list[str] = []
    seen: set[tuple] = set()
    rejected = 0
    for i in range(n):
        for attempt in range(max_attempts_per_team):
            team = sampler.sample_team(rng)
            key = _team_key(team)
            if key in seen:
                continue
            paste = render_paste(team, tables)
            if validate:
                from fog.env.teams import validate_team

                ok, out = validate_team(paste, battle_format)
                if not ok:
                    rejected += 1
                    print(f"[gen] team {i} attempt {attempt}: validator rejected -> resample\n  {out}")
                    continue
            seen.add(key)
            pastes.append(paste)
            break
        else:
            raise RuntimeError(
                f"team {i}: {max_attempts_per_team} consecutive rejections — the sampler is "
                "producing illegal sets; inspect the validator output above."
            )
    if rejected:
        print(f"[gen] {rejected} validator rejections across {n} teams (resampled)")
    return pastes


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a usage-sampled set-variant team pool.")
    ap.add_argument("--n", type=int, default=128, help="teams in the bank")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--core", type=Path, default=DEFAULT_CORE_POOL,
                    help="pool whose first team defines the shared species core")
    ap.add_argument("--mass", type=float, default=0.95, help="cumulative usage mass kept per marginal")
    ap.add_argument("--out", type=Path, default=TEAMS_DIR / "pool_gen")
    ap.add_argument("--format", default="gen9ou")
    ap.add_argument("--no-validate", action="store_true",
                    help="skip the pinned-simulator validation (tests only — never for a real bank)")
    args = ap.parse_args()

    tables = Tables()
    sampler = SetSampler(species_core_from_pool(args.core), tables, mass=args.mass)
    report = sampler.entropy_report()
    total = 0.0
    print("hidden-information upper bound (bits/slot, truncated marginals):")
    for sid, attrs in report.items():
        slot_total = sum(attrs.values())
        total += slot_total
        print(f"  {sid:16s} {attrs}  = {slot_total:.1f} bits")
    print(f"  TEAM TOTAL <= {total:.1f} bits (vs ~3 bits of team identity on the 8-team POOL-A)")

    pastes = generate_pool(
        args.n, args.seed, core_pool=args.core, mass=args.mass, tables=tables,
        validate=not args.no_validate, battle_format=args.format,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    for f in args.out.glob("gen_*.txt"):
        f.unlink()
    for i, paste in enumerate(pastes):
        (args.out / f"gen_{i:03d}.txt").write_text(paste, encoding="utf-8")
    meta = {
        "seed": args.seed,
        "n": args.n,
        "mass": args.mass,
        "core_pool": str(args.core),
        "core_species": sampler.core,
        "entropy_bits_upper": report,
        "validated": not args.no_validate,
        "format": args.format,
        "source": {
            "priors": "fog/constants/generated/priors.json",
            "spreads": "fog/constants/generated/spreads.json",
        },
    }
    out_json = args.out.with_suffix(".json")
    out_json.write_text(
        json.dumps({"teams": pastes, "format": args.format, "_meta": meta}, indent=2),
        encoding="utf-8",
    )
    print(f"pool_gen: {len(pastes)} teams -> {args.out} + {out_json}")


if __name__ == "__main__":
    main()
