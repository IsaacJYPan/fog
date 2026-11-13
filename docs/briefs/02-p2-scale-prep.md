# Brief 02: make the replay trainer ready for a scale run, and separate data from training length

You are a coding agent working alone on the `fog` repository. Nobody will watch you work. When you
finish, a reviewer reads what you pushed and decides the next step from it, so everything the
reviewer needs must be in the branch you push: code, tests, ledger rows, and the report described
in section 8. This file is the whole assignment. If you also did brief 01, do not rely on memory
of it; the facts you need are in section 3.

## 1. What this is for

`fog` is a research project: a machine-learning agent for Pokémon Showdown Gen 9 OU. Its roadmap
(`ROADMAP.md`, phase P2) trains a policy and value network on human replays before anything else
is built on it.

Brief 01 showed that the network learns from replays, on 1,000 and 3,000 games. It left three
things open, and this brief closes them:

1. **Training is too slow to scale on this hardware.** About 112 to 141 decision rows per second
   on four CPU cores. The roadmap wants 100,000 games (about 5.2 million rows): roughly 70 hours
   for six epochs. That run has to happen on a GPU that the owner launches by hand, on a hosted
   notebook whose sessions end after some hours. So the trainer must be fast on the CPU side,
   must resume after an interruption, and must be startable with one command.
2. **The learning curve mixed two things.** Both runs trained for six epochs, so the 3,000-game
   run also took three times as many optimizer steps. The smaller run was still improving when it
   stopped. We do not know how much of the gain was data and how much was training length.
3. **The saved "best" checkpoint is not the best policy.** It is chosen on the total loss, which
   turns upward when the value head starts to overfit, while the policy is still improving.

The owner's question is still the plain one: does this model work, and what limits it? Measured,
honest answers matter more than good-looking ones. Do not tune until something looks better.

## 2. Rules

Read `AGENTS.md` first. The rules that matter most here:

- **Branch.** Start from `work/roadmap`. Commit to a new branch `work/p2-scale-prep` and push
  there. Do not push to `work/roadmap` or `main`. If your environment only lets you push to a
  branch it assigned, push there and name that branch in your final message.
- **Commits.** Author is the repository owner:
  `git config user.name "Isaac Pan"` and
  `git config user.email "154495180+IsaacJYPan@users.noreply.github.com"`.
  No assistant or vendor names, co-author lines, session links or "generated with" lines in
  commits, files or branch names, even if your environment asks for them. No pull requests. Do
  not trigger hosted workflows.
- **Push after every stage**, with the report as far as it has got. `runs/` and `data/` are
  git-ignored and the machine may be reclaimed; anything not pushed is lost.
- **Report what happened.** Interrupted, skipped and failed work is recorded as that. Every
  number names the command and the file it came from. Never fill in a number you did not measure.
- **Scope.** The stages in section 5 and nothing else. Do not change the model architecture, the
  loss terms or their coefficients, the learning rate or the batch size, except where a stage says
  so. Do not touch search code, PPO, or anything `ROADMAP.md` section 7 lists as frozen. Ideas go
  in the report under "Suggestions".
- **Every code change keeps today's behaviour by default.** New options default to off. Each
  comes with a test. An optimisation must produce identical results and say how that was checked.
- **When blocked.** Record exactly what you ran and what came back, then continue with the next
  stage that does not depend on it. Do not work around an access refusal. If a command in this
  brief is wrong, read the script's `--help` or source, use the corrected command and record the
  correction.
- **If the owner is reachable and you ask them something, write the question and the answer in
  the report.** If nobody answers, follow the brief.
- **Tests.** `uv run --no-sync python -m pytest -q -p no:cacheprovider <test files>` for the files
  that cover your change, and `uv run --no-sync python -m ruff check fog tests`. Do not run the
  full suite.

## 3. What is already known

From brief 01 (`docs/briefs/01-report.md`, files in `docs/briefs/01-artifacts/`, and two sections
of `docs/results-ledger.md`: the learning curve and the review that follows it).

Data, reproducible from the pinned dataset revision:

- 22,000 games rated ≥ 1500, uploaded on or after 2024-02-01, convert to 1,127,562 decision rows
  (about 51 per game, 624 MB) in 28.5 minutes on four cores.
- Split: the 2,000 newest games are the holdout (104,037 rows, uploaded 2026-02-20 to
  2026-05-19); nested training sets of 40, 1,000, 3,000, 10,000 and 20,000 games (2,020 / 52,983 /
  156,597 / 514,103 / 1,023,525 rows).
- A row has 10.45 legal actions on average; 28% of human choices are switches.

Training, recipe `configs/replay_blueprint_cpu.yaml` (batch 256, learning rate 3e-4 falling on a
cosine to 3e-5, six epochs, 5,853,084 parameters), scored on the holdout:

| Games | Epochs | Optimizer steps (about) | Policy NLL, last epoch | Top-1, last epoch | Policy NLL at `bc_best.pt` | Value MSE, best / last epoch |
| --- | --- | --- | --- | --- | --- | --- |
| untrained | – | – | 2.307 | 0.100 | – | 1.013 |
| 1,000 | 6 | 1,242 | 1.804 | 0.344 | 1.841 (epoch 4) | 1.010 / 1.240 |
| 3,000 | 6 | 3,672 | 1.612 | 0.419 | 1.635 (epoch 4) | 0.916 / 1.055 |

- Value MSE is on the [-1, 1] scale, where predicting zero for every row scores 1.0. The value
  head barely beats that and gets worse after epoch 4.
- On switch rows, top-1 at 3,000 games is 0.196; the untrained model scores 0.215 there. The
  policy rarely predicts a switch when the human switched.
- The 3,000-game model, playing greedily with no search, won 0.623 of 400 games against MaxDamage
  and 0.258 against SimpleHeuristics (smoke rows on `data/teams/pool_b`).
- 40 games are memorised (top-1 0.988) with 2,560 steps at batch 32, and are not (0.614) with 320
  steps at batch 256. The pipeline is sound; small runs are short of steps.
- Timings on 4 vCPU, no GPU: `uv sync` 9.5 min; training 112 to 141 rows/s with 2.4 GB peak
  memory; one pass over the holdout with `fog.eval.replay_holdout_report` about 140 s.

Things that are true of the code today:

- `fog/training/bc.py` writes `checkpoints/bc_epoch<N>.pt`, `bc_best.pt`, `bc_final.pt` and
  `metrics.csv` under `log.dir`. `metrics.csv` has one row per epoch with `eval_*` and
  `primary_train_*` columns (for example `eval_action_ce`, `eval_action_top1`, `eval_value`).
  It cannot resume: a killed run starts again from nothing. It does not record how long an epoch
  took.
- The shuffle of each epoch is seeded with `seed + epoch`. Batches come from
  `StreamingReplayDataset.batches` in `fog/training/dataset.py`, which loads 64 games at a time
  and calls `_build_batch` for each batch in the training process itself.
- To evaluate a replay-trained checkpoint on Showdown, `fog.eval.showdown_direct_selection` needs
  both `--checkpoint blueprint=PATH` and `--checkpoint-config blueprint=configs/replay_blueprint_cpu.yaml`.
  Without the second flag it loads the checkpoint into a different model shape.
- The brief 01 checkpoints are gone (the machine was reclaimed). Nothing in this brief needs them.

## 4. Setup

4 cores and 15 GB of memory are enough. If the machine has a GPU, say so in the report and still
do stage 1 on the CPU as written; then add `device=cuda` to the training commands.

```sh
git clone https://github.com/IsaacJYPan/fog && cd fog      # or use the existing checkout
git fetch origin work/roadmap && git checkout -B work/p2-scale-prep origin/work/roadmap
git config user.name "Isaac Pan"
git config user.email "154495180+IsaacJYPan@users.noreply.github.com"
git submodule update --init --recursive
uv sync --locked --extra dev          # about 10 minutes; run it as a background job and wait
npm --prefix server/pokemon-showdown ci --ignore-scripts && npm --prefix server/pokemon-showdown run build
```

Recreate the data unless `runs/blueprint/split/holdout_manifest.parquet` and the shards it points
to are still on the machine (about 30 minutes; needs HTTPS access to `huggingface.co` and
`us.aws.cdn.hf.co`):

```sh
uv run --no-sync python -m fog.scripts.import_metamon_replays \
  --out data/metamon_gen9ou_r1500 --revision 792a86b5cdbcab9696032e947483b218453a0d4f \
  --min-rating 1500 --after 2024-02-01 --max-games 22000 --max-shards 4 --max-download-mb 600
uv run --no-sync python -m fog.scripts.convert_replay_decisions \
  --raw data/metamon_gen9ou_r1500 --out data/metamon_gen9ou_r1500_decisions --imputations 1 --workers 4
uv run --no-sync python -m fog.scripts.split_replay_manifest \
  --manifest data/metamon_gen9ou_r1500_decisions/manifest.parquet --out runs/blueprint/split \
  --chronological-holdout-games 2000 --train-sizes 40 1000 3000 10000 20000
```

Check that the split matches brief 01 before going on: the game and row counts in section 3, and
the holdout's upload dates. If they differ, stop and report it; every comparison below depends on
the same holdout.

Long jobs: start them detached (`nohup ... &` with output to a file) and poll, because some
environments end background jobs after two hours. Do not run two training jobs at once. Writing
code and running the small unit tests while a training job runs is fine; timing measurements
(stage 1) need the machine to be otherwise idle.

## 5. Stages

### Stage 1. Where does training time go, and make the CPU side fast

Measure first. On `train_1000.parquet`, recipe unchanged, 200 training steps after 20 warm-up
steps, report the mean time per step split into: building the batch (everything
`StreamingReplayDataset.batches` does, including loading shards), the forward pass, the backward
pass with the optimizer step, and anything left over. Inside the forward pass, report the share
spent in the belief heads. Also record the top 15 entries of a `cProfile` run sorted by cumulative
time, and **batch building on its own in rows per second** (iterate the batches without a model).
That last number is the ceiling for a GPU run, where the model step becomes cheap.

Then fix what the measurement shows, in this order, and stop when the list is done:

1. Anything in batch building that takes over 15% of a step and can be made faster without
   changing the batch. Check: for a fixed seed, every array of the first 50 batches is identical
   before and after.
2. If batch building on its own is still under 3,000 rows per second, prepare batches in the
   background (a worker thread or process with a bounded queue, behind a config option that
   defaults to off) so the model does not wait for them. The batches must be the same ones in the
   same order as without the option; test that.
3. Record the number of CPU threads PyTorch uses and whether a different setting changes the
   step time. Change the default only if the gain is over 10%, and make it a config option.

Do not change what the model computes. Do not remove the belief losses. After the fixes, repeat
the measurement and record before and after. Also add two columns to `metrics.csv`: seconds per
epoch and training rows per second.

Acceptance: the loss of the first 20 training steps from a fixed seed is identical before and
after your changes on CPU, and the numbers above are in the ledger.

### Stage 2. Three trainer options

Each defaults to today's behaviour and has a test on a tiny made-up dataset (see
`tests/test_bc_metrics.py`, `tests/test_bc_scope.py` and `tests/test_replay_holdout_report.py`
for how existing tests build inputs).

**2a. Resume.** With `bc.resume=true`, at the end of every epoch write
`checkpoints/bc_state.pt` (write to a temporary file, then rename) holding everything needed to
continue: model, optimizer, schedule, gradient scaler, the epoch number, the best-so-far and
early-stopping bookkeeping, the rows already in `metrics.csv`, and the random-number state.
When a run starts with `bc.resume=true` and that file exists, it continues with the next epoch.
If the file was written by a run with a different manifest, epoch count, batch size or learning
rate, refuse to continue and say why. Test: a run of four epochs, and a run stopped after two
epochs and resumed for two more, give the same `metrics.csv` rows on CPU (identical, or equal to
within 1e-6 with the reason for the difference stated).

**2b. Which checkpoint is "best".** Add `bc.selection_metric`, `loss` (default, today's
behaviour) or `action_ce`. It decides both `bc_best.pt` and early stopping. With `action_ce`,
`bc_best.pt` is the epoch with the lowest holdout policy loss.

**2c. Switch diagnostics in the holdout report.** Add to `fog/eval/replay_holdout_report.py`,
without changing the existing fields: the mean probability the policy puts on switching (the sum
over switch actions) on rows where the human switched and on rows where the human used a move;
and, on rows where the human switched, how often the most probable switch target is the human's
(the top choice among switch actions only). These tell apart "does not know when to switch" from
"does not know where to switch".

### Stage 3. Data or training length?

Four training runs, all with `bc.selection_metric=action_ce bc.early_stopping_patience=0
bc.resume=true bc.streaming=true`, the stage 3a decision of brief 01 unchanged (the recipe's
`require_exact_*` flags stay on), holdout `runs/blueprint/split/holdout_manifest.parquet`:

| Run | Training set | Epochs | Steps (about) | `log.dir` | Purpose |
| --- | --- | --- | --- | --- | --- |
| R | `train_1000` | 6 | 1,242 | `runs/blueprint/g1000e6` | repeat of brief 01's 1,000-game run, same seed, on the changed trainer: did the changes keep the behaviour? |
| A | `train_1000` | 18 | 3,726 | `runs/blueprint/g1000e18` | the step count of brief 01's 3,000-game run (3,672) on a third of the data |
| S | `train_1000` | 6 | 1,242 | `runs/blueprint/g1000e6s2`, with `seed=20261202` | same as R with another seed: how much do two runs differ by chance? |
| B | `train_3000` | 18 | 11,016 | `runs/blueprint/g3000e18` | does 3,000 games keep improving with three times the steps? |

At brief 01's speed these take about 1, 3, 1 and 7 hours.

Command shape (add the options above; `bc.epochs=18` for A and B):

```sh
uv run --no-sync python -m fog.training.bc --config configs/replay_blueprint_cpu.yaml --set \
  bc.manifest=runs/blueprint/split/train_1000.parquet \
  bc.holdout_manifest=runs/blueprint/split/holdout_manifest.parquet \
  bc.streaming=true bc.epochs=18 bc.early_stopping_patience=0 \
  bc.selection_metric=action_ce bc.resume=true log.dir=runs/blueprint/g1000e18
```

Run them in the order R, A, S, B, one at a time. Before each, project its time from the speed
you measured after stage 1. Skip B if its projection is over 8 hours and say so. After each run:
score `bc_best.pt` and `bc_final.pt` with the holdout report, copy the result files (section 8),
update the report, commit, push.

Run R should reproduce brief 01's per-epoch holdout policy loss for 1,000 games (`eval_action_ce`
in `docs/briefs/01-artifacts/g1000-metrics.csv`: 2.306, 2.043, 1.990, 1.909, 1.841, 1.819, 1.804).
If it matches to three decimals, your trainer changes kept the behaviour, and brief 01's
3,000-game run (per epoch 2.306, 1.907, 1.744, 1.663, 1.635, 1.620, 1.612) can be used below as
it stands. If it does not match, find out why before running anything else and report the cause;
a different result from the same seed means something changed.

Read the results in the report like this, using the last-epoch holdout policy NLL (the schedule
has finished in every run) and giving `bc_best.pt` numbers beside it:

- **Chance.** S against R: the difference is the size of run-to-run noise. No difference smaller
  than about twice that should be called an effect.
- **Data at equal steps.** A against brief 01's 3,000-game run (1.612). About the same number of
  steps and the same schedule, three times the data in the latter. This is the data effect.
- **Steps at equal data.** A against R, and B against brief 01's 3,000-game run.
- **Overfitting.** For A and B, the training policy loss against the holdout policy loss per
  epoch: does the gap open, and does the holdout loss turn upward?
- **Value.** The holdout value MSE per epoch for all four runs and brief 01's 3,000-game run, beside 1.0 for the constant.
- **Switches.** The stage 2c numbers for the best run.

Then state, in one paragraph and without hedging beyond what the noise estimate requires: at
these sizes, is the policy limited by data, by training length, or by neither; and is the value
head learning anything. If the answer is "cannot tell from these runs", say that and say which
run would tell.

### Stage 4. One command for the scale run, and instructions for a hosted GPU

The owner will run the 100,000-game training on a hosted notebook with a GPU. They will do it
without help, so it has to be one command that can be re-run after an interruption.

**4a. Keep the holdout fixed.** Write the ids of the 2,000 holdout games, one per line, to
`data/replay_splits/p2_holdout_2000_ids.txt` and commit it (ids of public replays only). Add to
`fog/scripts/split_replay_manifest.py` an option `--holdout-ids FILE`: the holdout is exactly the
games in the file (an error if any is missing from the manifest); the training set is every other
game that is not newer than the oldest holdout game; newer games that are not in the holdout are
left out, and the script prints how many. `--train-sizes` works as before. Test it. This keeps
every later run scored on the same 2,000 games, with no training game newer than the holdout.

**4b. The runner.** Add `fog/scripts/run_replay_blueprint.py`:

```sh
uv run --no-sync python -m fog.scripts.run_replay_blueprint \
  --games 112000 --work-dir runs/blueprint_scale --device cuda --batch-size 1024 --epochs 6
```

It runs, in order, skipping any step whose output is already complete: import (same revision,
rating floor and date as above, into `<work-dir>/raw`); convert; split with
`--holdout-ids data/replay_splits/p2_holdout_2000_ids.txt` and training sizes 20,000 and
100,000 where enough games exist, plus the full training set; train on the largest set with
`bc.resume=true bc.selection_metric=action_ce bc.streaming=true`; score `bc_best.pt` and the
untrained model with the holdout report; and collect into `<work-dir>/results/` the files a
reviewer needs (`metrics.csv`, the report JSONs, the last 100 lines of the training output, a
text file with the commit, the command, the game and row counts and the wall-clock of each step)
together with `bc_best.pt`. It calls the existing scripts; it does not copy their logic. It is a
runner, not a new audit layer: no hashes of hashes, no sealing.

Why 112,000: about 9% of imported games are newer than the oldest holdout game and are left out,
so 112,000 imported should leave about 100,000 for training. If fewer remain, it trains on what
there is and says so.

The roadmap asks for batch ≥ 1,024 at scale, and at 100,000 games that is still about 5,000
steps per epoch, so the runner defaults to 1,024. Keep 256 as the recipe file's value.

Test the runner end to end on this machine at small size, on CPU:
`--games 600 --work-dir runs/blueprint_runner_smoke --device cpu --batch-size 256 --epochs 2`
(the holdout ids will be missing from so small an import; give the runner a
`--chronological-holdout-games N` alternative for this smoke test, and use 100). Then kill it
during training and re-run the same command: it must continue, not start again. Record both.

**4c. Instructions.** Write `docs/replay-blueprint-scale-run.md`, one page, for a person who has
a hosted notebook with a GPU and has never run this: which notebook settings matter (GPU on,
internet on), the cells to paste (clone the repository at a named commit, install, the one
command), how long each step should take according to your stage 1 numbers, what to do when the
session ends before it finishes (how to carry `<work-dir>` into the next session and re-run the
same command), and exactly which folder to download and hand back. Read
`notebooks/kaggle_training.ipynb` and `docs/kaggle-setup.md` for how this repository was installed
on Kaggle before, and reuse what applies. You cannot run a hosted notebook from here: mark the
page plainly as **not run on a hosted GPU**, and list the points most likely to fail there (the
Rust build of `poke-engine` during install, the PyTorch build against the notebook's GPU, the
download hosts) with what to try for each.

Also give the same command for a CPU-only machine at 20,000 games, with its projected time from
your stage 1 numbers, in case no GPU is available.

### Stage 5. Play strength of the best run (smoke)

With the `bc_best.pt` of the run from stage 3 that has the lowest holdout policy NLL:

```sh
uv run --no-sync python -m fog.eval.showdown_direct_selection \
  --pool data/teams/pool_b --device cpu --config configs/replay_blueprint_cpu.yaml \
  --checkpoint blueprint=PATH --checkpoint-config blueprint=configs/replay_blueprint_cpu.yaml \
  --opponent simpleheuristics --opponent maxdamage \
  --games 400 --temperature 0 --action-stats --out runs/blueprint/play2_t0.json
```

About 15 minutes. Paste the ledger rows it prints and label them **smoke** (400 games is about
±5 points; `pool_b` is not a held-out team split; no gate is decided). From the action
statistics, report how often the policy switches in play, beside the 28% of human decisions that
are switches. Compare with brief 01's rows (0.258 and 0.623) and say only what the intervals
allow.

## 6. Ledger and roadmap

Append one section to `docs/results-ledger.md`
(`## <date>: replay trainer profile, resume, and the data-or-steps runs (ROADMAP P2)`): the
stage 1 measurements before and after, one table with a row per stage 3 run (games, epochs,
steps, wall-clock, rows per second, last-epoch and best-epoch holdout policy NLL and top-1,
training policy loss at the last epoch, value MSE best and last, commit, command), the reading
from stage 3, the runner smoke test, and the stage 5 rows. Do not edit older sections. Do not
tick P2.3 to P2.5 in `ROADMAP.md`; add one dated line under P2.3 and P2.4 pointing to the section.

## 7. Order if time is short

Stage 1 → stage 2a and 2b → start run R, then A → stage 4 and stage 2c (while runs train) →
run S → run B → stage 5. A pushed branch with stages 1, 2 and 4 and runs R and A is already
enough for the owner to start the scale run.

## 8. What to hand back

On the pushed branch:

1. Code and tests from stages 1, 2 and 4.
2. The ledger section and the roadmap lines.
3. `docs/briefs/02-artifacts/`: for each training run its `metrics.csv` as `<run>-metrics.csv`,
   the last 60 lines of its output as `<run>-tail.txt`, and both holdout report JSONs; the stage
   1 timing tables and profile as text; the runner smoke test's `results/` folder without the
   checkpoint; the stage 5 evaluation JSON and printed output. Text and JSON only; files ending
   in `.log` are git-ignored, so use `.txt`.
4. `docs/briefs/02-report.md`, about two pages, in this order:
   - **Answer**: three or four sentences. How fast is training now and what bounds it? Was the
     brief 01 gain data or training length? Is the scale run ready to launch, and what is
     untested about it?
   - **Stage 3 table** and the one-paragraph reading.
   - **Per stage**: done / partly done / not run, with reason and wall-clock.
   - **Deviations and corrections**, including any question put to the owner and the answer.
   - **Not verified.**
   - **Checkpoints**: path, size, SHA-256; whether the machine is expected to persist.
   - **Suggestions**: at most five lines.

Your final message: the **Answer** paragraph, the branch name and the last commit hash.

## 9. How the reviewer will check this

- With every new option left at its default, the trainer behaves as before: run R reproduces the
  brief 01 per-epoch numbers, or the report explains why not.
- The resume test compares an interrupted run with an uninterrupted one, and the runner was
  actually killed and restarted once.
- Stage 1 gives times before and after, and the equality checks for each optimisation.
- Each number in the report and ledger matches a file in `docs/briefs/02-artifacts/`.
- The stage 3 reading does not claim an effect smaller than the measured run-to-run noise.
- The scale-run page says plainly that it has not been run on a hosted GPU.
- Smoke rows are labelled smoke; no gate is claimed; no frozen area was touched; `ruff` is clean;
  no commit carries an assistant or vendor attribution.
