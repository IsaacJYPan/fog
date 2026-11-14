# Brief 07: move-order correction (smoke; in progress)

**Answer.** In the original R0 checkpoint, held-out top-1 is 79.10% as stored versus 25.43%
with move slots shuffled on the retained-history group: a 53.67-point gap (smoke). Its value
Brier is 0.3905, worse than constant 0.5's 0.25. The converter now shuffles completed move
sets and preview order while preserving imputation draws and identity alignment. R1 and its
second seed are pending; an order-blind model, revised value targets, mechanical features,
lead training and play have not been implemented or measured.

| Work | State | Evidence |
| --- | --- | --- |
| Stage 1 converter and diagnostics | 84 focused tests pass; final corpus comparison pending | `tests-final.txt`, `ruff-final.txt` |
| Original baseline | Eight epochs complete; best policy epoch 7 | `R0-metrics.csv`, `R0-slot-check-extended.json`; commit `d603920` |
| Corrected corpus | Complete: 22,000 games, zero quarantines, fixed IDs unchanged | `converter-after-quality.json`, `fixed-split.json` |
| Stages 2–5 | Not started | No model, loss, search, PPO or belief source changed |

Lead ordering needs the extra team shuffle requested by stage 1: first public send-outs favour
preview position zero in the imported logs (`lead-order-before.json`). Both replay observations
and live `FogPlayer` observations call the same encoder: own entities follow `battle.team`
insertion order, opponents follow public preview order, and switches follow own order minus the
active. Replay registration inserts own preview entries before play; live insertion comes from
the private request. These are the same indexing convention, but the raw logs and live team
sheets need not put the same species at the same position. The public-fact pass and imputation
use original preview slots, keeping draws paired with R0. Completed sheets and provenance then
move with the shuffled preview before decision reconstruction; no opponent hidden facts enter
inputs. Species unable to fill four moves keep empty columns trailing, matching installed
request order (`test_unfillable_move_slots_stay_after_the_installed_moves`).

The slot check now moves IDs, PP, known flags, candidate features and provenance together. It
reports switch-label positions, per-side lead positions, and policy/value intervals resampling
entire games with both perspectives together. Its existing “not yet moved” test observes only
the stored history window; it does not establish that a species never moved earlier in a long
game. The history one-hot caveat still applies when comparing shuffled, already-moved rows.

The corpus comparison exposed an existing Illusion alignment defect: the acting object can
change after the snapshot, and its current move dictionary can label a different encoded move.
The converter now rejects mismatches between the protocol label and the encoded candidate,
and does not copy their stale slot index into history. `row-count-changes.json` retains the
first corrected-corpus comparison; that corpus is superseded by a rebuild with the guards.
This adds a filtering difference beyond the order correction.

The original baseline source is isolated in a detached checkout. Corrected conversion has
per-game atomic recovery records and rejects changed resume settings. The import and both
conversions use the specified pinned release; fixed holdout IDs match. CPU only;
no hosted account is used. Converter timing overlaps other CPU work and is not a throughput
benchmark. The initial R0 attempt was interrupted before a completed epoch because CPU
OpenMP busy waiting made evaluation unexpectedly slow. All completed runs use
`OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0`, retaining five threads and the prescribed recipe.
On recorded rows with perturbed weights, all 241 checked output/gradient tensors were
bit-identical (`runtime-equivalence.json`). The original checkpoint strictly loads, and all
three rollout digests match before/after on this runtime (`checkpoint-compatibility.json`,
`digests-summary.json`). The historical references do not reproduce, including a separate
CPU-only build repeat and a repeat at the historical reference commit; their discrepancy
remains unisolated.

**Not verified.** R1 results and seed noise, any stage 2–5 result, rare-move transfer, value
repair, play, and complete Illusion/Transform reconstruction beyond the label-alignment guard.
The historical rollout references remain unreproduced. No roadmap task or gate is claimed.
