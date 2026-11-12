# Brief 01 report: does the model learn from human replays? (ROADMAP P2)

Branch `work/p2-replay-curve`, started from `work/roadmap` at `41cc7b9`. Machine: 4 vCPU Intel
Xeon @ 2.10 GHz, 15.7 GB memory, no GPU, `torch 2.13.0+cu130` running on CPU. All files named
below are in `docs/briefs/01-artifacts/`.

## Answer

It learns. On a fixed holdout of the 2,000 newest games, policy NLL falls from 2.307 (untrained)
to 1.841 at 1,000 games and 1.635 at 3,000 (−0.43 per tenfold data), and top-1 lift rises from 1.05
to 3.44 and 4.32. At 3,000 games the value head is calibrated (Brier 0.229 against 0.25 for a
constant, ECE 0.032). It plays weakly. With no search, it beats MaxDamage (0.623 [0.574, 0.669] greedy)
but loses clearly to SimpleHeuristics (0.258 [0.217, 0.303]). Confidence: the curve has only two
trained points, one seed each, and 10,000 and 20,000 games were not run (over the 6-hour limit on
this CPU). The 40-game overfit check failed its literal rule at batch 256 (passed at batch 32, see
below). The play rows are 400-game smoke on `pool_b`, not a gate.

## Curve table

Holdout: the 2,000 newest games, 104,037 decision rows (103,981 policy rows), uploaded
2026-02-20 to 2026-05-19. Chance top-1 lift is 1.0; a constant 0.5 value has Brier 0.25.

Holdout columns score `bc_best.pt` (chosen by total eval loss); train columns are the last epoch's
row-weighted training means from `metrics.csv`.

| Games | Rows | Epochs (best) | Wall | Top-1 | Lift | NLL | Brier | ECE | Train top-1 | Train action loss | File |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 (untrained) | – | – | – | 0.100 | 1.047 | 2.307 | 0.253 | 0.058 | – | – | `untrained-holdout.json` |
| 1,000 | 52,983 | 6 (4) | 55.8 min | 0.328 | 3.44 | 1.841 | 0.264 | 0.131 | 0.366 | 1.731 | `g1000-holdout.json`, `g1000-metrics.csv` |
| 3,000 | 156,597 | 6 (4) | 2 h 6 min | 0.413 | 4.32 | 1.635 | 0.229 | 0.032 | 0.449 | 1.523 | `g3000-holdout.json`, `g3000-metrics.csv` |
| 10,000 | 514,103 | not started: projected 7.9 h from stage 3c, 6.3 h from the g3000 rate | | | | | | | | | |
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
| 5 Curve | partly done: 1,000 and 3,000 games | g1000 55 min 45 s; g3000 2 h 6 min 4 s (2.4 GB peak each) | both ran all 6 epochs (best epoch 4 by total eval loss); 10,000 and 20,000 not started (projection over 6 h) |
| 6 Reading | done | – | below |
| 7 Play smoke | done (smoke) | 14 min 19 s for 1,600 games | g3000 `bc_best.pt`, loaded natively (see deviations). `play_t0.json`, `play_t0.5.json`, `play-output.txt` |

### Stage 6: reading the curve

- Holdout NLL: 2.307 (untrained), 1.841 (1,000), 1.635 (3,000). The change from 1,000 to 10,000
  cannot be given: 10,000 was not run. Fitted over the two trained points, the slope is
  **−0.433 NLL per tenfold data** (1,000 → 3,000 is −0.207).
- Roadmap rule P2.4: an improvement of more than 0.02 per tenfold data means **more data is the
  next step**. This holds by a wide margin, though from two points at small sizes.
- Lift at the largest size, 4.32, is clearly above chance and above the brief author's 1.3
  orientation line.
- Generalisation gap at 3,000 games (last epoch): training top-1 0.449 and action loss 1.523
  against holdout 0.419 and 1.612 for the same epoch (`g3000-metrics.csv`). The gap is small.
  The 1,000-game run's value head overfitted (holdout value MSE rose from 1.01 to 1.24 over 6
  epochs).
- Breakdown at 3,000 games (`g3000-holdout.json`): move rows top-1 0.497 (lift 5.13, NLL 1.398);
  switch rows top-1 0.196 (lift 2.12, NLL 2.244). Switches are the weak part. Value Brier by
  phase: early 0.248, middle 0.235, late 0.204. 4,297 rows sit at the turn cap of 50.
- Reliability, 3,000 games (bin: count, mean predicted, observed win rate): 0: 244, 0.095, 0.102 ·
  1: 6,725, 0.155, 0.211 · 2: 10,052, 0.252, 0.322 · 3: 14,239, 0.353, 0.403 · 4: 22,395, 0.454,
  0.480 · 5: 23,540, 0.545, 0.542 · 6: 12,170, 0.646, 0.616 · 7: 8,152, 0.748, 0.711 · 8: 5,833,
  0.845, 0.807 · 9: 687, 0.912, 0.897. ECE 0.032, which is under G2's 0.05 at this scale.

### Stage 7: play smoke (400 games per row, `pool_b`, not a held-out team split)

| Temperature | vs SimpleHeuristics | vs MaxDamage |
| --- | --- | --- |
| 0 | 0.258 [0.217, 0.303] (103–297) | 0.623 [0.574, 0.669] (249–151) |
| 0.5 | 0.223 [0.184, 0.266] (89–311) | 0.517 [0.469, 0.566] (207–193) |

Gate G2 is not decided by these rows. They show that the offline gains do not yet carry over
against SimpleHeuristics, the possibility the brief anticipates (roadmap answer: P3).

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
- Stage 7: the brief's command loads the checkpoint through `load_posterior_agent`, which turns
  on posterior features and widens a checkpoint trained without them. That is not the model that
  was trained and scored offline. I added `--checkpoint-config blueprint=configs/replay_blueprint_cpu.yaml`
  so it loads natively and strictly. The largest model trained is g3000, so stage 7 used
  `runs/blueprint/g3000/checkpoints/bc_best.pt` instead of the g20000 path in the brief.
- The first launch of g3000 was stopped within a minute (during its baseline evaluation),
  because the session's background-job limit (2 h) was shorter than the run. It was restarted
  from scratch as a detached process; nothing was spliced.
- Batch 256 (the roadmap asks for ≥ 1,024) and 3,000 games instead of 100,000 at the largest
  point, as the brief anticipated.
- Checkpoint selection uses total eval loss (all heads). For both sizes the best epoch was 4,
  while holdout policy NLL kept improving to epoch 6 (g3000: 1.635 at epoch 4 against 1.612 at
  6). The curve uses `bc_best.pt` as the brief says.
- The commit message of `Record the brief 01 stage 3b diagnostic` quotes top-1 0.985 and loss
  0.049. Those are epoch 37 values; the final epoch-40 values are 0.988 and 0.036.

## Not verified

- Training rows/s excludes start-up and two short evaluations by estimate (~16 s); the trainer
  does not time its phases. The g3000 run implies about 141 rows/s.
- That the slope would hold beyond 3,000 games: two points, one seed, no repeat runs.
- The holdout's newer upload dates (2026-02 to 2026-05) may differ in metagame from training;
  the size of that effect was not measured separately.

## Checkpoints

Git-ignored and kept only on the session machine, a cloud container that is expected to be
reclaimed. Treat them as lost and re-create them with the commands in the ledger.

| Path | Bytes | SHA-256 |
| --- | --- | --- |
| `runs/blueprint/g1000/checkpoints/bc_best.pt` | 23,548,503 | `a5a5b89e0bdc0e60f8d44f437e980f1ee8156027d58372c7a925e632a32d23e4` |
| `runs/blueprint/g3000/checkpoints/bc_best.pt` | 23,548,503 | `2cef18e97d9298d95320d78e9485736e12544bd6dc9d54e62b33adb112213be2` |

## Commands for a GPU run

After stages 1 and 2 as in the brief, which recreate the same split from the pinned revision:

```sh
for S in 10000 20000; do
  uv run --no-sync python -m fog.training.bc --config configs/replay_blueprint_cpu.yaml --set \
    bc.manifest=runs/blueprint/split/train_$S.parquet \
    bc.holdout_manifest=runs/blueprint/split/holdout_manifest.parquet \
    bc.streaming=true log.dir=runs/blueprint/g$S device=cuda mixed_precision=true \
    > runs/blueprint/g$S.txt 2>&1
  uv run --no-sync python -m fog.eval.replay_holdout_report --config configs/replay_blueprint_cpu.yaml \
    --checkpoint runs/blueprint/g$S/checkpoints/bc_best.pt \
    --manifest runs/blueprint/split/holdout_manifest.parquet --out runs/blueprint/g$S-holdout.json
done
```

For 100,000 games, import more with the same filters into the same directory (the importer
resumes). Shard and byte budgets are a guess at about 11,000 qualifying games per shard. Then
convert into a new directory and split. A fresh chronological split of the larger corpus moves
the holdout to the newest 2,000 games of the larger import, so the 100k point would not share the
holdout above. Re-score the small models on it, or split 100k with the existing holdout ids excluded.

```sh
uv run --no-sync python -m fog.scripts.import_metamon_replays --out data/metamon_gen9ou_r1500 \
  --revision 792a86b5cdbcab9696032e947483b218453a0d4f --min-rating 1500 --after 2024-02-01 \
  --max-games 102000 --max-shards 16 --max-download-mb 2400
uv run --no-sync python -m fog.scripts.convert_replay_decisions --raw data/metamon_gen9ou_r1500 \
  --out data/metamon_gen9ou_r1500_decisions_100k --imputations 1 --workers 4
uv run --no-sync python -m fog.scripts.split_replay_manifest \
  --manifest data/metamon_gen9ou_r1500_decisions_100k/manifest.parquet --out runs/blueprint/split100k \
  --chronological-holdout-games 2000 --train-sizes 1000 3000 10000 20000 100000
uv run --no-sync python -m fog.training.bc --config configs/replay_blueprint_cpu.yaml --set \
  bc.manifest=runs/blueprint/split100k/train_100000.parquet \
  bc.holdout_manifest=runs/blueprint/split100k/holdout_manifest.parquet \
  bc.streaming=true log.dir=runs/blueprint/g100000 device=cuda mixed_precision=true
```

## Suggestions

1. Run 10,000 and 20,000 on a GPU next: the slope says data is the limit at these sizes.
2. Fixed epochs at batch 256 left the small runs short of steps (stage 3b). At the roadmap's batch ≥ 1,024 this gets worse; an owner decision on a step budget would help.
3. Select the policy checkpoint on holdout `action_ce` rather than total loss: policy NLL was still falling when `bc_best.pt` was taken.
4. Switch decisions are the weak part (lift 2.1 against 5.1 for moves at 3,000 games).
5. The SimpleHeuristics gap is the covariate-shift case the roadmap sends to P3.
