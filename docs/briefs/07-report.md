# Brief 07: move-order correction (smoke; in progress)

**Answer.** The converter now independently shuffles completed move sets and public preview
order, with move provenance, action labels and history kept aligned. The effect on holdout
accuracy is not measured yet: R0 is being preserved on the unchanged source before comparing
R1. An order-blind model, revised value targets, mechanical features and trained leads have
not been implemented or measured. No play or strength claim follows from this preparation.

| Work | State | Evidence |
| --- | --- | --- |
| Stage 1 converter and diagnostics | 81 focused tests pass; final corpus comparison pending | `tests-identity-history.txt`, `ruff-identity-history.txt` |
| Original baseline | Fixed split verified; R0 training and checkpointing | `R0-execution.json`; baseline commit `d603920` |
| Corrected corpus | Final rebuild includes snapshot identity guards | `conversion-resume.json`, `guarded-partial-attempt.json` |
| Stages 2–5 | Not started | No model, loss, search, PPO or belief source changed |

Lead ordering needs the extra team shuffle requested by stage 1: first public send-outs favour
preview position zero in the imported logs (`lead-order-before.json`). Both replay observations
and live `FogPlayer` observations call the same encoder: own entities follow `battle.team`
insertion order, opponents follow public preview order, and switches follow own order minus the
active. Replay registration inserts own preview entries before play; live insertion comes from
the private request. These are the same indexing convention, but the raw logs and live team
sheets need not put the same species at the same position. The correction shuffles only replay
preview messages before both chronological passes; it uses no opponent hidden facts.

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
conversions use the specified pinned release; fixed holdout verification is pending. CPU only;
no hosted account is used. Converter timing overlaps other CPU work and is not a throughput
benchmark. The initial R0 attempt was interrupted before a completed epoch because CPU
OpenMP busy waiting made evaluation unexpectedly slow. All completed runs use
`OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0`, retaining five threads and the prescribed recipe.
On recorded rows with perturbed weights, all 241 checked output/gradient tensors were
bit-identical (`runtime-equivalence.json`). The original checkpoint strictly loads, and all
three rollout digests match before/after on this runtime (`checkpoint-compatibility.json`,
`digests-summary.json`). The historical references do not reproduce, including a separate
CPU-only build repeat; their discrepancy remains unisolated.

**Not verified.** R0/R1 training results, the completed fixed split and post-conversion row
statistics, any stage 2–5 result, seed noise, rare-move transfer, value improvement, and play.
The unchanged checkout already fails the historical rollout digest references on this runtime;
before/after reproduction is being recorded separately. No roadmap task or gate is claimed.
