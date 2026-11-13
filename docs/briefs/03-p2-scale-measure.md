# Brief 03: run the measurements brief 02 could not

You are a coding agent working alone on the `fog` repository. Nobody will watch you work. A
reviewer reads what you push and decides the next step from it.

Brief 02 (`docs/briefs/02-p2-scale-prep.md`) asked for trainer changes and for measurements. The
code was written and reviewed. **None of the measurements ran**, because that worker's machine
could not reach the dataset host. This brief is the measurements. Brief 02 remains the
specification of each stage; read its sections 1, 2, 3 and 5 before starting. This file says
what is already done, what is left, and what is different.

## 0. Check access before anything else

```sh
curl -sS -o /dev/null -w "%{http_code}\n" -m 30 https://huggingface.co/api/datasets/jakegrigsby/metamon-raw-replays
```

This must print `200`. If it prints anything else, or the import in section 3 is refused:
**stop**. Do not build stand-in data, do not write more tooling, do not try another host. Tell
the owner in your final message that this machine cannot reach `huggingface.co` and
`us.aws.cdn.hf.co` and that the brief needs a machine that can. That one sentence is the whole
deliverable in that case.

A download that is cut off partway (connection reset) is not a refusal: run the same command
again; finished shards are cached.

## 1. Rules

Brief 02 section 2 applies unchanged, with one difference: commit to a new branch
**`work/p2-scale-measure`**, started from `work/roadmap`. The points that most need repeating:

- Commits are authored as the owner (`git config user.name "Isaac Pan"`,
  `git config user.email "154495180+IsaacJYPan@users.noreply.github.com"`); no assistant or
  vendor names, co-author lines or session links anywhere; no pull requests.
- Push after every stage. Report what happened; never fill in a number you did not measure.
- Do not change the model, the losses, the learning rate or the batch size.
- One training job at a time. Timing measurements need an otherwise idle machine.

## 2. What is already done

On `work/roadmap`, reviewed (82 tests pass, lint clean):

- `bc.resume=true`: the trainer writes `checkpoints/bc_state.pt` after every epoch and continues
  from it when started again with the same settings.
- `bc.selection_metric=action_ce`: `bc_best.pt` and early stopping follow the holdout policy
  loss. The default, `loss`, is the old behaviour.
- `metrics.csv` has `seconds_per_epoch` and `train_rows_per_second`. `bc.torch_threads` sets the
  PyTorch thread count when given.
- `fog.eval.replay_holdout_report` reports switch diagnostics (probability put on switching on
  switch rows and on move rows; switch-target accuracy).
- `fog.scripts.profile_replay_training` measures a training step by phase (stage 1 below).
- `fog.scripts.split_replay_manifest --holdout-ids FILE` and the runner
  `fog.scripts.run_replay_blueprint` (import → convert → split → train → score → collect results,
  skipping finished steps).
- `data/replay_splits/p2_holdout_2000_ids.txt`: the ids of the 2,000 holdout games, verified in
  review against brief 01 (same upload-date range, 104,037 decision rows).
- `docs/replay-blueprint-scale-run.md`: a first draft of the owner's instructions. It was
  written without any real measurement and is out of date; stage 5 replaces it.

All of this has only been exercised on made-up data. Treat the first real run of each piece as a
test of it: if something fails on real data, fix it with a test and record what was wrong.

## 3. Setup and data

Setup commands are in brief 02 section 4; use the branch name above. Then recreate the data
(about 30 minutes) with the three commands in that section (import 22,000 games, convert, split
with `--chronological-holdout-games 2000 --train-sizes 40 1000 3000 10000 20000`).

Check before going on, and record the output:

```sh
uv run --no-sync python - <<'PY'
import polars as pl
h = pl.read_parquet("runs/blueprint/split/holdout_manifest.parquet")
ids = set(open("data/replay_splits/p2_holdout_2000_ids.txt").read().split())
print(len(h), h["n_rows"].sum(), set(h["id"].cast(pl.String).to_list()) == ids)
for s in (1000, 3000, 20000):
    t = pl.read_parquet(f"runs/blueprint/split/train_{s}.parquet")
    print(s, len(t), t["n_rows"].sum())
PY
```

Expected: `2000 104037 True`, then `1000 1000 52983`, `3000 3000 156597`, `20000 20000 1023525`.
If any differs, stop and report it; every comparison below depends on the same data.

Then check the fixed-id split on the same manifest:

```sh
uv run --no-sync python -m fog.scripts.split_replay_manifest \
  --manifest data/metamon_gen9ou_r1500_decisions/manifest.parquet --out runs/blueprint/split_fixed \
  --holdout-ids data/replay_splits/p2_holdout_2000_ids.txt --train-sizes 20000
```

It should report no dropped games, and its holdout and `train_20000` should hold the same games
as the chronological split's. Record the result.

## 4. Stages

**Stage 1. Profile, then speed up.** Brief 02 stage 1, unchanged. The measurement command
exists:

```sh
uv run --no-sync python -m fog.scripts.profile_replay_training --out runs/blueprint/profile-before
```

Read its output and its source first to see what it reports. The numbers the owner needs:
training rows per second, the share of a step spent building the batch, in the forward pass
(and in the belief heads inside it), and in the backward pass; and batch building on its own in
rows per second. Then apply brief 02's three-item fix list in order, each with its equality
check, and measure again (`--out runs/blueprint/profile-after`). If nothing on the list
qualifies, say so; that is a result.

**Stage 2. Data or training length.** Brief 02 stage 3, unchanged: runs R, A, S, B in that
order, with the reading it specifies.

| Run | Training set | Epochs | `log.dir` | Extra |
| --- | --- | --- | --- | --- |
| R | `train_1000` | 6 | `runs/blueprint/g1000e6` | must reproduce brief 01 per epoch: 2.306, 2.043, 1.990, 1.909, 1.841, 1.819, 1.804 |
| A | `train_1000` | 18 | `runs/blueprint/g1000e18` | |
| S | `train_1000` | 6 | `runs/blueprint/g1000e6s2` | `seed=20261202` |
| B | `train_3000` | 18 | `runs/blueprint/g3000e18` | skip if projected over 8 hours |

All four with `bc.streaming=true bc.early_stopping_patience=0 bc.selection_metric=action_ce
bc.resume=true`, the holdout `runs/blueprint/split/holdout_manifest.parquet`, and no other
change to `configs/replay_blueprint_cpu.yaml`. Because `bc.resume=true`, a job that the
environment ends early is continued by running the same command again; record every such
restart. After each run, score `bc_best.pt` and `bc_final.pt` with
`fog.eval.replay_holdout_report`, copy the result files, update the report, commit, push.

If R does not reproduce brief 01 to three decimals, stop the other runs and find out why. Run R
once more with `bc.resume` and `bc.selection_metric` left out to see whether an option or the
machine is the cause, and report what you find.

**Stage 3. Stop and continue, on real data.** During run A, after at least two epochs have
finished, kill the training process and start the same command again. Record that it continued
from the last finished epoch and that the earlier rows of `metrics.csv` are unchanged. This
replaces a separate resume experiment.

**Stage 4. The runner on real data.** Brief 02 stage 4b's small test, with real games:

```sh
uv run --no-sync python -m fog.scripts.run_replay_blueprint \
  --games 600 --work-dir runs/blueprint_runner_smoke --device cpu --batch-size 256 --epochs 2 \
  --chronological-holdout-games 100
```

Let it finish once. Then delete only its `training` folder and `results` folder, start it again,
kill it during training after the first epoch, and start it once more: it must skip import,
conversion and split, and continue training. Record the wall-clock of each step.

**Stage 5. Rewrite the owner's instructions.** Replace `docs/replay-blueprint-scale-run.md` with
one page a person can follow without knowing the code:

- numbered steps, each a cell to paste or a setting to click, in the order they are done;
- the repository is public, so the clone needs no login; pin the clone to the last code commit
  of your branch (the reviewer merges by fast-forward, so the hash stays valid);
- the holdout id file is in the repository now; remove the paragraph about restoring it;
- time and disk for each step at 112,000 games, projected from **your** stage 1 and stage 2
  numbers, with the arithmetic shown; say which figures are projections from CPU measurements
  and that GPU speed is unmeasured;
- what to do when the session ends before the run finishes, as three or four concrete steps;
- the one folder to download and hand back;
- the same command for a CPU-only machine at 22,000 games, with its projected time;
- at the top, plainly: **not run on a hosted GPU**, and the three most likely failure points
  with what to try for each.

**Stage 6. Play smoke.** Brief 02 stage 5, unchanged, with the `bc_best.pt` of the stage 2 run
that has the lowest holdout policy loss, output to `runs/blueprint/play3_t0.json`.

## 5. Order if time is short

Section 3 → stage 1 measurement → run R → run A (with stage 3 inside it) → stage 4 → stage 5 →
run S → stage 1 fixes and second measurement → run B → stage 6.

Runs R and A with stages 4 and 5 are what the owner needs to launch the scale run. If stage 1's
fixes would delay them by more than an hour, do the fixes after run A.

## 6. What to hand back

On the pushed branch:

1. Any code fix, with its test.
2. One new section in `docs/results-ledger.md`
   (`## <date>: replay trainer profile and the data-or-steps runs (ROADMAP P2)`), as brief 02
   section 6 describes. Do not edit older sections.
3. `docs/briefs/03-artifacts/`: the profile outputs as text; for each run `<run>-metrics.csv`,
   `<run>-tail.txt` (last 60 lines) and both holdout report JSONs; the section 3 check outputs;
   the runner test's `results/` folder without the checkpoint; the play evaluation JSON and
   printed output. Text and JSON only.
4. `docs/briefs/03-report.md`, about two pages, laid out as brief 02 section 8 asks: **Answer**
   (how fast is training and what bounds it; was the brief 01 gain data or training length; is
   the scale run ready to launch and what about it is untested), the stage 2 table and reading,
   per-stage status with wall-clock, deviations, not verified, checkpoints, suggestions.

Final message: the **Answer** paragraph, the branch name and the last commit hash.

## 7. How the reviewer will check this

- The section 3 check printed the expected numbers.
- Run R reproduces brief 01, or the report explains why not.
- The stage 2 reading claims no effect smaller than twice the difference between R and S.
- Each number in the report and the ledger matches a file in `docs/briefs/03-artifacts/`.
- The kill-and-continue in stages 3 and 4 actually happened, with the before and after shown.
- The instructions page states its projections' arithmetic and that no hosted GPU run was made.
- Smoke rows are labelled smoke; no gate is claimed; lint is clean; no commit carries an
  assistant or vendor attribution.
