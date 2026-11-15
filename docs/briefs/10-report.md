# Brief 10 report — 2026-11-14

**Answer.** The committed replay-derived pool has **600 train, 141 validation and 62 test
teams**, one distinct composition per team, with no composition shared across splits. Average
team evidence is **27.28 revealed / 62.72 sampled of 90 scalar fields** (six exact species
included), with **15.74/24 moves revealed**. Items, abilities, Tera types and unrevealed moves
are usage samples, as are all spreads; a team can be less coherent than one a person built.
The A0–A2 validation round-robin is pending. A3 is unavailable in this fresh checkout.

| Validation pair | Matched games | Result |
| --- | --- | --- |
| A0 Random vs A1 MaxDamage | pending | pending |
| A0 Random vs A2 SimpleHeuristics | pending | pending |
| A1 MaxDamage vs A2 SimpleHeuristics | pending | pending |

**Verified (smoke construction diagnostics).** Pinned import: 22,000 games; 44,000 sides →
9,850 rated ≥1700 → 9,849 complete known previews → 1,839 with ≥14 moves → 1,837 renderable →
1,439 validator-clean → 999 compositions → 803 capped teams. Test compositions are absent from
all older raw previews, even below the rating/move thresholds. Every committed team validates.
A fresh full rebuild reproduces all 807 files exactly. The 34 focused tests and repository lint
pass. No player metadata or original nicknames are exported. Detailed counts, commands, split
manifest hashes and limits are in the [ledger](../results-ledger.md), supported by
[artifacts](10-artifacts/pool-summary.json). Pool implementation: `3506a39`.

**Not verified.** Stage 2 has not run yet; P0.7 is unchecked. A3, G0 and agent-strength/tier
claims remain unmeasured. Test teams have only undergone construction checks, never play.
