# Brief 01 report: does the model learn from human replays? (ROADMAP P2)

Branch `work/p2-replay-curve`, started from `work/roadmap` at `41cc7b9`. Machine: 4 vCPU Intel
Xeon @ 2.10 GHz, 15.7 GB memory, no GPU, `torch 2.13.0+cu130` running on CPU. All files named
below are in `docs/briefs/01-artifacts/`.

## Answer

_In progress: the 1,000-game run is training._

## Curve table

Holdout: the 2,000 newest games, 104,037 decision rows (103,981 policy rows), uploaded
2026-02-20 to 2026-05-19. Chance top-1 lift is 1.0; a constant 0.5 value has Brier 0.25.

| Games | Rows | Epochs (best) | Wall | Top-1 | Lift | NLL | Brier | ECE | Train top-1 | Train action loss | File |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 (untrained) | – | – | – | 0.100 | 1.047 | 2.307 | 0.253 | 0.058 | – | – | `untrained-holdout.json` |
| 1,000 | 52,983 | pending | | | | | | | | | |
| 3,000 | 156,597 | pending | | | | | | | | | |
| 10,000 | 514,103 | not started (projected 7.9 h) | | | | | | | | | |
| 20,000 | 1,023,525 | not started (projected 15.5 h) | | | | | | | | | |

## Per stage

| Stage | Status | Wall-clock | Notes |
| --- | --- | --- | --- |
| Setup | done | `uv sync` 9 min 28 s; Showdown build 7 s | |
| 1 Import + convert | done | import 34 s; convert 28 min 29 s | 22,000 games from 2 of 4 downloaded shards (104,736 source rows scanned: 77,225 below rating, 4,176 invalid metadata, 840 other format, 495 outside dates); 22,000 converted, 0 quarantined, 1,127,562 decision rows, 2,920 excluded (2,366 called or unaligned moves, 554 forced or unaligned switches); 624 MB; 12.9 games/s. `stage1-*.txt/json` |
| 2 Split | done | 2 s | holdout 2,000 games / 104,037 rows (2026-02-20 to 2026-05-19); training 20,000 games / 1,023,525 rows (2024-02-01 to 2026-02-20); nested 40 / 1,000 / 3,000 / 10,000 / 20,000 games = 2,020 / 52,983 / 156,597 / 514,103 / 1,023,525 rows. `stage2-split-summary.txt` |
| 3a Policy rows | done | 1 s | 100% `policy_valid`; 99.97% exact chosen features, 100% exact chosen legality: flags kept. Mean 10.45 legal actions per row; 28.2% switches; chosen action always in the mask. `stage3a.json` |
| 3b Overfit 40 games | **failed the pass rule**, then a diagnostic passed | 11 min 6 s; diagnostic 19 min 48 s | top-1 0.614 (< 0.90), action loss 2.286 → 1.055. Diagnostic at batch 32: top-1 0.988, loss 0.036. See below. |
| 3c Timing | done | 8 min 9 s | ~112 training rows/s, 2.3 GB peak, 5,853,084 parameters; full holdout evaluation 138 s. `stage3c-timing.txt` |
| 4 Holdout tool | done | 140 s per full holdout | `fog/eval/replay_holdout_report.py`, tests in `tests/test_replay_holdout_report.py` |
| 5 Curve | in progress | | |
| 6 Reading | pending | | |
| 7 Play smoke | pending | | |

### Stage 3b in detail

With the recipe as written (batch 256, 40 epochs on 2,020 rows), training is 8 optimizer steps
per epoch, 320 in total, with cosine decay to 3e-5. Top-1 on the 40 games rose steadily
(0.08 → 0.34 at epoch 10 → 0.48 at 20 → 0.61 at 40) but did not reach 0.90 (`overfit40-metrics.csv`).
The value head memorised the outcomes (Brier 0.009, `overfit40-report-train40.json`).
Checks on the 2,020 rows: the chosen action is always inside the legality mask; its action
features are never all-zero and never identical to another legal action's; top-1 is 0.58 on
move rows and 0.70 on switch rows.

As a diagnostic (not a re-run to obtain a pass), the same command with `bc.batch_size=32`
(8× the steps, nothing else changed) reached top-1 0.988 and action loss 0.036 at epoch 40
(`overfit40-b32-diagnostic-metrics.csv`). The pipeline can memorise; the batch-256 failure was
a step-count limit. The brief says a 3b failure stops stages 5–7. I asked the owner, and they
chose to run the curve with the recipe unchanged (batch 256).

## Deviations and corrections

- Stage 3b failed its literal pass rule. The curve was run anyway on the owner's instruction,
  after the batch-32 diagnostic above.
- The brief and the 2026-11-12 ledger entry say about 3.5 legal actions per row and 29% random
  top-1. On this conversion the mean is 10.45 legal actions per row (`stage3a.json`), and the
  untrained model scores 0.100 top-1 (lift 1.05). Judge by the lift.
- `/usr/bin/time` is not installed. Peak memory was read with a wrapper that runs the command and
  reports `getrusage(RUSAGE_CHILDREN).ru_maxrss`.
- Game phase: each shard stores each player's rows as a separate block, so position in the
  shard is not game time. The report reads the turn from the field vector, which is capped at 50.
- The commit message of `Record the brief 01 stage 3b diagnostic` quotes top-1 0.985 and loss
  0.049. Those are epoch 37 values; the final epoch-40 values are 0.988 and 0.036.

## Not verified

- Training rows/s excludes start-up and two short evaluations by estimate (~16 s); the trainer
  does not time its phases.

## Checkpoints

_Pending._

## Commands for a GPU run

_Pending._

## Suggestions

_Pending._
