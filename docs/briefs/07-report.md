**Answer (smoke; standalone stage 1 only).** The leak inflated R0 top-1 by 53.67
percentage points on the retained-history move group: 79.10% stored versus 25.43% shuffled.
Reconversion removes the first-use column bias. With the unchanged model, R1's shuffled
all-row NLL improves by 0.2895, above twice its two-seed gap (0.0102); filtering
also changed, so this is not a pure ordering ablation. The value head still fails the
constant-prediction Brier reference of 0.25. Architectural invariance, value repair,
rare-move features and play effects are unknown: stages 2–5 were not implemented.

All results below are smoke. `configs/replay_blueprint_cpu.yaml` trains on the same 1,000 games for eight epochs,
with early stopping off and the whole fixed chronological 2,000-game holdout evaluated each
epoch. Seeds are 20261201 and 20261202. Checkpoints minimize policy NLL. CPU only; five Torch
threads, passive OpenMP waiting. Recorded-input outputs and gradients match the original
waiting setting bit-for-bit (`runtime-equivalence.json`).

Stored metrics: each policy cell is **NLL / top-1 (%)**. H means no earlier move by the acting
species in the retained history window, **not** necessarily its first move in the game.

| Run, best epoch | All | Move | H | Value Brier / ECE |
| --- | --- | --- | --- | --- |
| R0, 7 | 1.7717 / 36.52 | 1.5419 / 43.46 | 0.8328 / 79.10 | 0.3905 / 0.3604 |
| R1, 8 | 1.9239 / 26.48 | 1.7657 / 29.14 | 1.6733 / 33.71 | 0.3920 / 0.3660 |
| R1-seed2, 7 | 1.9293 / 26.67 | 1.7886 / 29.12 | 1.7022 / 32.97 | 0.3975 / 0.3712 |

Compare R0 and R1 **only in this shuffled table**; R0's stored holdout shares its leak.

| Run | All | Move | H |
| --- | --- | --- | --- |
| R0 | 2.2135 / 22.69 | 2.1601 / 24.09 | 2.5077 / 25.43 |
| R1 | 1.9239 / 26.41 | 1.7660 / 28.98 | 1.6733 / 33.52 |
| R1-seed2 | 1.9290 / 26.68 | 1.7877 / 29.15 | 1.7024 / 32.77 |

R1's shuffled all-row top-1 point difference is 3.72 points; it exceeds the
twice-seed-gap threshold of 0.54 points. The tables do not establish model invariance.
All policy/value 95% intervals resample games with both perspectives together; full intervals
are in the per-run slot-check JSONs. Value intervals (Brier; ECE) are:

- R0: [0.3761, 0.4044]; [0.3449, 0.3755].
- R1: [0.3773, 0.4058]; [0.3507, 0.3807].
- R1-seed2: [0.3828, 0.4118]; [0.3558, 0.3865].

**Converter and defects.** Final move tokens and provenance are shuffled together after
completion, retaining the original imputation draws by species slot. Completed sheets and
public previews then move together. Empty moves remain trailing to match installed request
order. Measured lead-position bias justified the team shuffle: full-corpus first-public-lead
position-zero shares change from 24.46% / 25.01% to 16.75% / 17.34% (p1 / p2).
Switch-label bench shares are near one fifth. Replay and live observations use the same
encoder indexing convention (own insertion order, opponent preview order, bench minus active),
but replay preview insertion and live private-request insertion can differ.

The corrected corpus retains all 22,000 games, zero quarantines, and 1,126,792 rows. Fixed
training/holdout IDs are unchanged. Snapshot identity guards reject stale move/switch labels
around replaced actors and prevent their slot numbers entering history; raw holdout rows
change from 104,037 to 103,944. The H slot-zero label share falls from 98.40% to 24.70%
(game-bootstrap 95% interval 24.09–25.30%). This point is just below the expected 25% lower
bound; the interval includes it, and the seed was not retuned. Conversion resumes per game;
the interrupted attempt and superseded corpora are retained as negative evidence.

99 tests pass (84 focused plus 15 shared-imputer/supervision tests); Ruff is clean.
Old trained checkpoints load strictly. Model, encoding,
loss, search, PPO and belief source are unchanged. The three final rollout hashes match their
unchanged-source counterparts on this runtime, but **do not reproduce the historical hashes**.
Unchanged historical source, a CPU-only build, and alternate CPU kernels also fail the
historical sequential reference (`digests-summary.json`); the discrepancy is unisolated.

**Not verified.** Stages 2–5 (R2–R4, order-blind acceptance tests, logistic/auxiliary value
targets, mechanical features, rare-move transfer, trained leads and play); lifetime-first-move
statistics; complete Illusion/Transform reconstruction beyond identity guards. Replayed own
PP remains zero. Historical digest compatibility remains open. This handoff stops at the
standalone converter stage; no roadmap task or strength gate is claimed.

Evidence and exact commands: `07-artifacts/README.md`, `stage1-summary.json`, per-run CSV/JSON,
`R0-execution.json` and `stage1-execution.json`. R0 uses original source `d603920`; R1 uses the
paired converter at `531f29d`. Checkpoints remain under ignored `runs/order_blind/`.
