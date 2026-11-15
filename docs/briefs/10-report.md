# Brief 10 report — 2026-11-14

**Answer.** The committed replay-derived pool has **600 train, 141 validation and 62 test
teams**, one distinct composition per team, with no composition shared across splits. Average
team evidence is **27.28 revealed / 62.72 sampled of 90 scalar fields** (six exact species
included), with **15.74/24 moves revealed**. Unrevealed items, abilities, Tera types and moves,
and all spreads, are usage samples; a team can be less coherent than one a person built.
The A0–A2 validation round-robin completed **1,000 matched games per pair**, with zero failures,
draws or truncations. MaxDamage beat Random **89.8%**; SimpleHeuristics beat Random **96.6%**
and MaxDamage **82.5%**. P0.7 is complete under this brief. A3 is unavailable.

| Validation pair | Matched games | First agent wins [95% Wilson] |
| --- | --- | --- |
| A0 Random vs A1 MaxDamage | 1,000 | 102 (10.2% [8.5%, 12.2%]) |
| A0 Random vs A2 SimpleHeuristics | 1,000 | 34 (3.4% [2.4%, 4.7%]) |
| A1 MaxDamage vs A2 SimpleHeuristics | 1,000 | 175 (17.5% [15.3%, 20.0%]) |

**Verified (pool statistics are smoke construction diagnostics).** Pinned import: 22,000 games;
44,000 sides → 9,850 rated ≥1700 → 9,849 complete known previews → 1,839 with ≥14 moves →
1,837 renderable → 1,439 validator-clean → 999 compositions → 803 capped teams. Two Ditto
sheets with unfillable move columns were dropped. Test compositions are absent from all older
raw previews, even below the rating/move thresholds. Every committed team validates. A fresh
full rebuild reproduces all 807 files exactly. The 34 focused tests and repository lint pass.
No player metadata or original nicknames are exported. Detailed counts, commands, split manifest
hashes and the printed match rows are in the [ledger](../results-ledger.md), supported by
[pool evidence](10-artifacts/pool-summary.json) and [match results](10-artifacts/anchors.json).
Pool implementation: `3506a39`; evaluation: `66a1e29`. CPU only; no forum fetch or hosted workflow.

**Not verified.** Exact submitted human spreads and unrevealed set fields, and resulting team
coherence. A3 is unavailable, so G0's full anchor requirement remains open; throughput work is
also unfinished. G1 and learned-agent/tier claims were not attempted. Scripted baselines expose
no per-decision timing/fallback counters (printed as `n/a`). Test teams underwent construction
checks only and were never played. No GPU or training was used.
