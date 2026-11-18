# Brief 12: replay/live parity — complete

**Answer.** On 200 games (smoke), own PP differed on every baseline row and now matches exactly; all 2,566 live replacements align. First-action history overwrites are fixed. Remaining own-HP/history differences are explained row by row by public HP rounding and hidden or overridden submitted choices; every other requested field exceeds 99.5% exact agreement. On identical corrected holdout inputs, new training improves NLL 1.7762→1.7178 and top-1 34.34%→36.60%. Play scores rise 39.00%→45.00% against SimpleHeuristics and 75.75%→79.50% against MaxDamage; the latter difference is uncertain. All results are smoke.

| Exact rows, smoke | Before | After |
| --- | ---: | ---: |
| Own PP | 0% | 100% |
| Own HP | 3.76% | 10.92% |
| History floats | 4.77% | 5.13% |
| History IDs / length | 40.89% | 100% |
| Legality | 94.37% | 99.64% |

Aligned decisions: 24,149→26,143; missing replay rows: 3,195→1,201; extras: zero.
Own-HP error is below 0.01. The tool captures actual forward tensors and uses true own sheets
only for this diagnostic. Current-turn history stays excluded, matching live. Three rollout
digests change solely because replacements preserve the first action. Raw unused candidate
columns and implicit all-true item masks have documented residual explanations.

Reconverted 22,000 games: 1,244,604 rows, zero quarantines. Same 1,000 training / 2,000
holdout games, eight-epoch R4 recipe, seed 20261201; selected epochs old/new: 7/6.
CPU only, old/new threads 2/4; numerical equivalence and allocation checks are recorded.
Showdown `e440c4a`. Normal comparison: 95,988 common decisions, zero label disagreements.
Cells below are NLL / top-1 %; full game-bootstrap intervals are in the artifacts.

| Model, smoke | Old normal inputs | Corrected normal inputs |
| --- | ---: | ---: |
| Old | 1.7803 / 34.13 | 1.7762 / 34.34 |
| New | 1.9013 / 33.05 | 1.7178 / 36.60 |

Replacement rows, old→new: all 19,514, **1.2182 / 38.35→1.1646 / 42.56**;
17,941 nontrivial, **1.3250 / 32.94→1.2667 / 37.52**. One-option rows have zero policy CE.
The new model worsens on old inputs; input versions are not interchangeable.

| Greedy play, smoke; 400 games/cell | Old W–L / score | New W–L / score | Paired gain, points [95%] |
| --- | ---: | ---: | ---: |
| SimpleHeuristics | 156–244 / 39.00% | 180–220 / 45.00% | +6.00 [0.74, 11.25] |
| MaxDamage | 303–97 / 75.75% | 318–82 / 79.50% | +3.75 [−2.00, 9.76] |

Both actors use corrected history, `pool_b`, seed 0, crossed teams/sides. No draws,
truncations or failures. Play intervals resample matched four-game blocks.
Evidence: `12-artifacts/`, including full parity tables/residuals, holdout cells, checkpoint
hashes, CSVs, play outcomes and `comparison-summary.json`; commands are in the ledger.
The 52-test covering run and added interruption test pass; ruff is clean.

**Not verified:** additional training seeds, imputation accuracy, held-out playing strength or
phase gates. Pool B play and imitation metrics are diagnostics; no gate threshold changed.
