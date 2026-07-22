"""Phase H instrumentation (PROPOSAL_IMPL_PLAN; proposal §10.4 backend note): measure, before
engineering, the two known chance-fidelity gaps of the poke-engine branch enumeration.

1. **Damage-roll straddles.** The engine collapses the 16 damage rolls to the average roll
   (branch damage = floor(base * 0.925), base = the max roll from `pe.calculate_damage`), and
   decides knockouts BY that average. When the defender's HP falls strictly inside the roll span,
   the real outcome is a mixture (some rolls KO, some do not) but the engine emits ONE
   deterministic verdict. We reconstruct the 16 rolls per matched Damage instruction
   (floor(base * r / 100), r = 85..100 — probe fact 3) and log: the straddle rate over matched
   move-damage instructions, and the probability mass the average-roll verdict gets wrong.

2. **Crit rate.** The engine must be compiled with its Gen 9 feature, which branches ordinary
   critical hits at 1/24 (~4.17%).
   Crit branches are identified by matching branch damage against the crit base's average roll;
   the per-cell crit share is aggregated so a global constant shows up as a tight distribution
   around one value (high-crit moves and abilities would legitimately widen it — report the
   spread, don't assume).

Method: sample pool positions by short random playouts, then for every legal REAL action pair at
each position, enumerate `pe.generate_instructions` on the true state (no forks needed — this is
K = J = 1 instrumentation) and classify Damage instructions. Damage amounts that match no
computed base average (hazards, recoil, chip, multi-hit) are counted as unmatched and skipped —
this instrumentation targets exactly the single-hit move-damage verdicts the grid consumes.
A side whose active changes mid-branch (a Switch instruction) stops classification for that side
in that branch (conservative: unknown target HP).

Usage:
    python -m fog.eval.chance_fidelity --pool data/teams/pool_a --positions 20 \
        --out runs/chance_fidelity.json
"""
from __future__ import annotations

import argparse
import json
import random
import re

_DAMAGE_RE = re.compile(r"^Damage Side(One|Two): (\d+)$")
_SWITCH_RE = re.compile(r"^Switch Side(One|Two)\b")


def _avg_roll(base: int) -> int:
    return int(base * 0.925)


def _rolls(base: int) -> list[int]:
    return [base * r // 100 for r in range(85, 101)]


def pair_damage_stats(state, m1: str, m2: str) -> dict:
    """Instrument one committed action pair on one engine state. Returns counters + per-family
    crit shares (family = one side's move-damage pair (no-crit, crit) in this cell)."""
    import poke_engine as pe

    sets = pe.generate_instructions(state, m1, m2)
    if not sets:
        return {"matched": 0, "unmatched": 0, "straddles": 0, "wrong_mass": 0.0,
                "crit_families": []}

    # candidate (no-crit, crit) bases per damaged side, both move orders (order can change
    # damage through e.g. weather or screens set first)
    bases: dict[str, set[tuple[int, int]]] = {"One": set(), "Two": set()}
    for first in (True, False):
        try:
            d1, d2 = pe.calculate_damage(state, m1, m2, first)
        except Exception:
            continue
        if len(d1) == 2 and d1[0] > 0:
            bases["Two"].add((int(d1[0]), int(d1[1])))    # side one's move damages side two
        if len(d2) == 2 and d2[0] > 0:
            bases["One"].add((int(d2[0]), int(d2[1])))

    hp0 = {"One": int(state.side_one.pokemon[int(state.side_one.active_index)].hp),
           "Two": int(state.side_two.pokemon[int(state.side_two.active_index)].hp)}

    matched = unmatched = straddles = 0
    wrong_mass = 0.0
    matched_mass = 0.0
    # crit accounting: per family, probability mass seen at the crit vs no-crit average
    fam_mass: dict[tuple[str, int, int], dict[str, float]] = {}

    for branch in sets:
        p = float(branch.percentage) / 100.0
        hp = dict(hp0)
        alive = {"One": True, "Two": True}
        for ins in branch.instruction_list:
            r = repr(ins)
            msw = _SWITCH_RE.match(r)
            if msw:
                alive[msw.group(1)] = False           # unknown target from here on
                continue
            mdm = _DAMAGE_RE.match(r)
            if not mdm:
                continue
            side, amount = mdm.group(1), int(mdm.group(2))
            if not alive[side]:
                continue
            hit = None
            for (nc, cr) in bases[side]:
                if amount == _avg_roll(nc):
                    hit = (nc, False)
                elif cr > 0 and amount == _avg_roll(cr):
                    hit = (cr, True)
                if hit:
                    fam = (side, nc, cr)
                    fam_mass.setdefault(fam, {"crit": 0.0, "plain": 0.0})
                    fam_mass[fam]["crit" if hit[1] else "plain"] += p
                    break
            if hit is None:
                unmatched += 1
                hp[side] = max(0, hp[side] - amount)  # chip still moves the target's HP
                continue
            matched += 1
            matched_mass += p
            base = hit[0]
            rolls = _rolls(base)
            ko_p = sum(1 for x in rolls if x >= hp[side]) / len(rolls)
            verdict_ko = amount >= hp[side]
            if 0.0 < ko_p < 1.0:
                straddles += 1
                wrong_mass += p * ((1.0 - ko_p) if verdict_ko else ko_p)
            hp[side] = max(0, hp[side] - amount)
            if hp[side] == 0:
                alive[side] = False                   # engine's verdict: this mon is down

    crit_families = [
        {"side": side, "base": nc, "crit_base": cr,
         "crit_share": m["crit"] / (m["crit"] + m["plain"]),
         # families missing one variant are usually KO-merged branches (a dead target cannot
         # flinch, so crit and non-crit collapse) or a pruned/unmatched twin — their share is
         # not evidence about the crit RATE, so the aggregator reports them separately
         "both_present": m["crit"] > 0 and m["plain"] > 0}
        for (side, nc, cr), m in fam_mass.items() if m["crit"] + m["plain"] > 0 and cr > 0
    ]
    return {"matched": matched, "unmatched": unmatched, "straddles": straddles,
            "wrong_mass": wrong_mass, "matched_mass": matched_mass,
            "crit_families": crit_families}


def measure_position(battle, role: str = "p1") -> dict:
    """All legal real action pairs at one battle state (K = J = 1 instrumentation)."""
    import numpy as np

    totals = {"matched": 0, "unmatched": 0, "straddles": 0, "wrong_mass": 0.0,
              "matched_mass": 0.0, "pairs": 0}
    crit_shares: list[float] = []
    degenerate = 0
    opp = "p2" if role == "p1" else "p1"
    for a in np.flatnonzero(battle.action_mask(role)):
        for b in np.flatnonzero(battle.action_mask(opp)):
            try:
                m1 = battle.engine_action("p1", int(a if role == "p1" else b))
                m2 = battle.engine_action("p2", int(b if role == "p1" else a))
                s = pair_damage_stats(battle.state, m1, m2)
            except Exception:
                continue
            totals["pairs"] += 1
            for k in ("matched", "unmatched", "straddles"):
                totals[k] += s[k]
            totals["wrong_mass"] += s["wrong_mass"]
            totals["matched_mass"] += s["matched_mass"]
            for f in s["crit_families"]:
                if f["both_present"]:
                    crit_shares.append(f["crit_share"])
                else:
                    degenerate += 1
    totals["crit_shares"] = crit_shares
    totals["degenerate_families"] = degenerate
    return totals


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pool", required=True)
    ap.add_argument("--positions", type=int, default=20)
    ap.add_argument("--max-random-turns", type=int, default=6,
                    help="positions are sampled along random playouts up to this depth")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import numpy as np

    from fog.env.engine_battle import EngineBattle
    from fog.env.teams import load_pool

    pastes = load_pool(args.pool)
    rng = random.Random(args.seed)
    agg = {"matched": 0, "unmatched": 0, "straddles": 0, "wrong_mass": 0.0,
           "matched_mass": 0.0, "pairs": 0, "degenerate_families": 0}
    crit_shares: list[float] = []

    for p in range(args.positions):
        i, j = rng.randrange(len(pastes)), rng.randrange(len(pastes))
        b = EngineBattle(pastes[i], pastes[j], seed=rng.randrange(2**31),
                         battle_tag=f"chance-fid-{p}")
        for _ in range(rng.randrange(args.max_random_turns + 1)):
            if b.finished:
                break
            moves = {}
            for r in b.pending_sides():
                legal = np.flatnonzero(b.action_mask(r))
                moves[r] = b.engine_action(r, int(rng.choice(list(legal))))
            b.step(moves)
        if b.finished:
            continue
        s = measure_position(b)
        for k in ("matched", "unmatched", "straddles", "pairs", "degenerate_families"):
            agg[k] += s[k]
        agg["wrong_mass"] += s["wrong_mass"]
        agg["matched_mass"] += s["matched_mass"]
        crit_shares.extend(s["crit_shares"])

    shares = np.asarray(crit_shares) if crit_shares else np.zeros(0)
    summary = {
        "positions": args.positions, "pairs": agg["pairs"],
        "damage_instructions_matched": agg["matched"],
        "damage_instructions_unmatched": agg["unmatched"],
        "straddle_rate_over_matched": (agg["straddles"] / agg["matched"]
                                       if agg["matched"] else 0.0),
        "wrong_verdict_mass_over_matched_mass": (agg["wrong_mass"] / agg["matched_mass"]
                                                 if agg["matched_mass"] else 0.0),
        "crit_share_families_both_present": int(shares.size),
        "crit_share_families_degenerate": agg["degenerate_families"],
        "crit_share_mean": float(shares.mean()) if shares.size else None,
        "crit_share_min": float(shares.min()) if shares.size else None,
        "crit_share_max": float(shares.max()) if shares.size else None,
        "gen9_reference_crit_rate": 1.0 / 24.0,
        "config": vars(args),
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        from pathlib import Path

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
