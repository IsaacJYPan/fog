# Brief 07: move-order correction (smoke; in progress)

**Answer.** The converter now independently shuffles completed move sets and public preview
order, with move provenance, action labels and history kept aligned. The effect on holdout
accuracy is not measured yet: R0 is being preserved on the unchanged source before comparing
R1. An order-blind model, revised value targets, mechanical features and trained leads have
not been implemented or measured. No play or strength claim follows from this preparation.

| Work | State | Evidence |
| --- | --- | --- |
| Stage 1 converter and diagnostics | Code and focused tests complete; corpus comparison pending | `07-artifacts/tests-stage1.txt`, `ruff-stage1.txt` |
| Original baseline | Import complete; original conversion and R0 pipeline running | `R0-execution.json` when each command finishes; baseline commit `d603920` |
| Corrected corpus | Resumed after interruption; running | `conversion-resume.json` |
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

The original baseline source is isolated in a detached checkout. Corrected conversion has
per-game atomic recovery records and rejects changed resume settings. The import and both
conversions use the specified pinned release; fixed holdout verification is pending. CPU only;
no hosted account is used. Converter timing overlaps other CPU work and is not a throughput
benchmark.

**Not verified.** R0/R1 training results, the completed fixed split and post-conversion row
statistics, any stage 2–5 result, seed noise, rare-move transfer, value improvement, and play.
The unchanged checkout already fails the historical rollout digest references on this runtime;
before/after reproduction is being recorded separately. No roadmap task or gate is claimed.
