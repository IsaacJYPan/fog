# Brief 05: the replay blueprint on a GPU, at 20,000 and 100,000 games

You are a coding agent working alone on the `fog` repository. Nobody will watch you work. A
reviewer reads what you push and decides the next step from it.

## 1. Why

Briefs 01 and 03 trained the replay blueprint on a CPU, up to 3,000 games. They found
(`docs/briefs/03-report.md`, ledger section "review of brief 03"):

- The policy is limited by data. At 1,000 and at 3,000 games the holdout loss is best at epoch 8
  of 18 and then rises. The 3,000-game model plays better than any before it (0.360 against
  SimpleHeuristics, 0.7725 against MaxDamage, smoke).
- Training runs at 157 to 194 rows per second on four CPU cores. 20,000 games would take half a
  day per run and 100,000 games several days. Nothing has ever run on a GPU.
- The value head memorises games. At the best policy epoch it is confident and wrong: holdout
  Brier 0.3445 where a constant scores 0.25. 3,000 games are 3,000 outcome labels.

This brief makes the two missing points of the learning curve (ROADMAP P2.1 to P2.5) on a hosted
GPU and reports, for each, what the value head does.

## 2. Rules

- Branch **`work/p2-gpu-scale`** from `work/roadmap`. Push after every stage.
- Commits are authored as the owner (`git config user.name "Isaac Pan"`,
  `git config user.email "154495180+IsaacJYPan@users.noreply.github.com"`); no assistant or
  vendor names, co-author lines or session links anywhere; no pull requests.
- Read `AGENTS.md` first. Setup is in `docs/briefs/03-p2-scale-measure.md` section 3, with the
  branch name above.
- **GPU.** Private Kaggle jobs under the account already configured on your machine
  (`KAGGLE_TRAINING.md`, `docs/kaggle-workflow.md`). If the client is not configured, do not
  look for another GPU: do stage 0, push, and say so. `fog.scripts.kaggle_experiment` was
  built for the play-training recipe; if it does not fit a replay run, submit a plain private
  script job with the official client that follows cells 2 to 4 of
  `docs/replay-blueprint-scale-run.md`, and keep the script under `docs/briefs/05-artifacts/`.
- **Budget: 20 GPU-hours in total, one GPU job at a time.** Before each job, state its expected
  GPU time from what you measured. Import, conversion and splitting run on CPU and must not use
  GPU time.
- This repository is public. Every Kaggle dataset and job is private. Commit no account name,
  no token and no `configs/kaggle_results.json`; name jobs and datasets by their slug without
  the account.
- No change to the model, the losses, the learning rate schedule or the data filters. Additive
  options that default to off are allowed where a stage names them, each with a unit test.
- Every number is labelled smoke. No gate is claimed. Never fill in a number you did not
  measure.

## 3. Stage 0: data, on CPU

1. Add `--prepare-only` to `fog.scripts.run_replay_blueprint`: import, convert and split, then
   exit before training. A later run of the same command without the flag must continue into
   training without redoing them (brief 03 stage 4 showed the skip works after a kill).
2. Prepare two work directories with the fixed holdout
   (`data/replay_splits/p2_holdout_2000_ids.txt`): one from `--games 22000` and one from
   `--games 112000`. If fewer than 100,000 training games pass the filters, apply ROADMAP P2.1's
   rule (rating floor 1300 before wider dates) and record the counts. The 22,000-game set
   must reproduce the row counts in the ledger (1,023,525 training rows at 20,000 games,
   104,037 holdout rows).
3. Upload each as a private dataset the GPU job can read. Start stage 1 as soon as the first
   is ready; the second can convert meanwhile (about two hours on four cores).

## 4. Stage 1: what a GPU does (at most one GPU-hour)

Run **G3: 3,000 games, 18 epochs, batch 1,024, `--device cuda`**, otherwise as brief 03's run B.
Record: device name, install time, training rows per second, seconds per validation pass, peak
GPU memory (`torch.cuda.max_memory_allocated`), and the per-epoch metrics.

- Compare G3's best holdout NLL with run B's 1.579 (CPU, batch 256). If they differ by more
  than 0.031, run 3,000 games once more on the GPU at batch 256 and say whether batch size or
  device explains it. Use the better batch size from here on.
- Once: stop a job before it finishes, and continue it in a second job from the first job's
  saved work directory. Show the metrics rows before and after. The 100,000-game run may need
  this.
- Add an option, default off, that also keeps the checkpoint with the lowest **holdout value
  MSE** (`bc_best_value.pt`) beside `bc_best.pt`. Use it in every run of this brief.

## 5. Stages 2 and 3: 20,000 and 100,000 games

- **G20: 20,000 games, 18 epochs.**
- **G100: all prepared training games.** Epochs: the largest of 18, 12 and 6 that fits in the
  remaining GPU budget at the measured rate with two hours to spare. If not even 6 fit, train on
  the largest nested subset for which 6 do, and say so.

For each run keep the per-epoch metrics CSV, the last 60 log lines, and
`fog.eval.replay_holdout_report` outputs for `bc_best.pt` and for `bc_best_value.pt`.

**The checkpoints must outlive your machine and the job.** Put `bc_best.pt` and
`bc_best_value.pt` of G3, G20 and G100 in one private Kaggle dataset, and commit
`docs/briefs/05-artifacts/checkpoints.json` with file names, sizes and SHA-256. The report says
in one command how the owner downloads them.

## 6. Stage 4: the learning curve and the value head (no new training)

1. **Curve (ROADMAP P2.4).** For run B (from brief 03's files), G3, G20 and G100, at the best
   policy epoch: holdout top-1, policy NLL, value Brier, value ECE and the ten-bin reliability
   table. Apply P2.4's decision rule to the step from 20,000 to 100,000 and state which branch
   of the rule holds.
2. **Value head.** For each run, at the best policy epoch and at the best value epoch: holdout
   value MSE, Brier and ECE overall and for the early, middle and late third of a game, the
   training value MSE of that epoch, and the share of holdout rows predicted below 0.1 or
   above 0.9 with their observed win rates.
   **Reading.** The value head has learned from replays if, at 100,000 games and at its own
   best epoch, the holdout Brier is below 0.24 and the late third is at least 0.02 below the
   early third. If it has, say how many epochs separate the best value epoch from the best
   policy epoch. If it has not, run the two checks ROADMAP names under gate G2
   (`fog/eval/leakage_metrics.py`, and outcome sign per side) on G100 and report them. Change
   nothing in the losses either way.

## 7. Stage 5: play smoke, on CPU

`bc_best.pt` of G20 and of G100: brief 02 stage 5 unchanged (400 greedy games per opponent on
`pool_b` against SimpleHeuristics and MaxDamage), one output file per model. Put the rates
beside brief 03's 0.360 and 0.7725 with their intervals.

## 8. Order if time or budget is short

Stage 0 (22,000 set) → stage 1 → G20 → stage 4 for what exists → stage 5 for G20 → G100 → the
rest. A complete G20 with its curve and value tables is worth more than a G100 cut short.

## 9. What to hand back

On the pushed branch:

1. `--prepare-only` and the value-checkpoint option, each with its test.
2. One new section in `docs/results-ledger.md`
   (`## <date>: replay blueprint on a GPU at 3,000 to 100,000 games (smoke)`): one row per run
   and per play result, with command and commit. Do not edit older sections.
3. `docs/briefs/05-artifacts/`: the job script, per run the files listed in section 5, the
   holdout reports, the play outputs, `checkpoints.json`, and one file with the GPU time each
   job used. Text and JSON only; no checkpoints.
4. A dated line under P2.1 to P2.5 in `ROADMAP.md`. Check no box.
5. `docs/replay-blueprint-scale-run.md`: replace "not run on a hosted GPU" and the CPU
   projections with what you measured.
6. `docs/briefs/05-report.md`, at most two pages: **Answer** (rows per second on the GPU; the
   curve at 3,000, 20,000 and 100,000 games; which branch of P2.4's rule holds; whether the
   value head learned; play rates), the two tables of section 6, GPU-hours used, deviations,
   not verified.

Final message: the **Answer** paragraph, the branch name and the last commit hash.

## 10. How the reviewer will check this

- Every number in the report is in a file under `docs/briefs/05-artifacts/`.
- The 22,000-game set reproduces the ledger's row counts, and the holdout is the fixed one.
- The model, losses, schedule and filters are unchanged; both new options default to off and
  have tests; `ruff check fog tests` is clean.
- The value reading follows the rule in section 6, not a looser one.
- GPU-hours add up to 20 or less, and no account name or token is in the repository.
- Rows are labelled smoke; no gate is claimed; no commit carries an assistant or vendor
  attribution.
