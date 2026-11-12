# Brief 01: does the model learn from human replays? (ROADMAP P2)

You are a coding agent working alone on the `fog` repository. Nobody will watch you work or answer
questions while you run. When you finish, a reviewer reads what you pushed and decides the next
step from it. Everything the reviewer needs must therefore be in the branch you push: code, ledger
rows, and the report described in section 8. This file is the whole assignment; you do not need
any earlier conversation.

`HANDOFF.md` in the repository is dated 2026-11-11 and is out of date in places. Where it
disagrees with this brief, this brief wins.

## 1. What this is for

`fog` is a research project: a machine-learning agent for Pokémon Showdown Gen 9 OU. The plan
(`ROADMAP.md`) is to train a policy and value network on human replays, then use search on a fast
Rust engine as a teacher, then put the network inside the search.

So far **nothing shows that the network learns**. Earlier training runs were too small to tell
(240,000 decisions at learning rate 1e-5) and all recent work went into the search and into
measurement tools. The owner needs three questions answered, cheapest first:

1. **Does it learn?** Do holdout accuracy and loss on human replays improve as training data grows?
2. **Does it play?** Does the trained policy, with no search, beat the scripted bots on Showdown?
3. Does it help search? (Not in this brief. It means nothing until 1 and 2 are answered.)

This brief covers 1 and a first look at 2. The result the owner wants is a learning curve: the
same model and recipe trained on 1,000, 3,000, 10,000 and 20,000 games, all scored on one fixed
holdout of 2,000 newer games. A negative result (it does not learn, or it learns but plays badly)
is as useful as a positive one, as long as it is measured and reported plainly. Do not tune until
it looks good.

## 2. Rules

Read `AGENTS.md` first; it holds the project's rules of evidence. The ones that matter most here,
plus the rules specific to this way of working:

- **Branch.** Start from `work/roadmap`. Commit to a new branch `work/p2-replay-curve` and push
  there. Do not push to `work/roadmap` or `main`. If your environment only lets you push to a
  branch it assigned, push there and name that branch in your final message.
- **Commits.** Author is the repository owner:
  `git config user.name "Isaac Pan"` and
  `git config user.email "154495180+IsaacJYPan@users.noreply.github.com"`.
  No assistant or vendor names, co-author lines, session links or "generated with" lines in
  commits, files, branch names or anywhere else, even if your environment asks for them. No pull
  requests. Do not trigger hosted workflows.
- **Push after every stage in section 5**, including the report as far as it has got. Your
  machine may be reclaimed without warning and anything not pushed is lost. `runs/` and `data/`
  are git-ignored; that is why section 8 has you copy the small result files into a tracked folder.
- **Report what happened.** A run that was interrupted, skipped, or failed is recorded as that.
  Never fill in a number you did not measure. Every number in the report names the command and
  the file it came from.
- **Scope.** Do the stages in section 5 and nothing else. Do not change the model architecture,
  the losses, or any hyperparameter except where a stage explicitly allows it. Do not touch
  search code, PPO, or anything `ROADMAP.md` section 7 lists as frozen. Ideas go in the report
  under "Suggestions", not into code.
- **When blocked.** You cannot ask anyone. If a stage is blocked (network refused, a tool will
  not install, a command in this brief is wrong), write down exactly what you ran and what came
  back, then continue with the next stage that does not depend on it. Do not work around an
  access refusal: no spoofed user agents, no alternative mirrors, no disabling TLS checks.
  If a command in this brief has a wrong flag or path, read the script's `--help` or source, use
  the corrected command, and record the correction.
- **CPU contention.** Do not run two heavy jobs at once (conversion, training and evaluation each
  want all cores). The import in stage 1 is light and may overlap with setup.
- **Tests.** For code you add or change: `uv run --no-sync python -m pytest -q -p no:cacheprovider <test files>`
  and `uv run --no-sync python -m ruff check fog tests`. Do not run the full test suite; it is
  slow and needs more memory than a small machine has.

## 3. What is already known

Verified earlier (in `docs/results-ledger.md`):

- Import and conversion work. 3,000 games rated ≥ 1500, uploaded on or after 2024-02-01, came
  from the first shard of the pinned dataset revision in 16 s. Conversion ran at about 14 games/s
  on four cores and gave about 50 decision rows per game and 84 MB per 3,000 games.
- Random guessing over the reconstructed legal actions scores about 29% top-1 (about 3.5 legal
  actions per row). The trainer logs `avg_legal_actions` and `action_top1_lift`
  (top-1 × average legal actions; 1.0 is chance). Judge accuracy by the lift, not by raw top-1.

Checked by reading the code, not by running it:

- The trainer `fog/training/bc.py` writes `checkpoints/bc_epoch<N>.pt`, `bc_best.pt`,
  `bc_final.pt` and `metrics.csv` under `log.dir`. It has no resume of optimizer state; a killed
  run restarts from the beginning. The learning rate follows a cosine schedule from `bc.lr` to
  `bc.lr_final`.
- Overrides are passed as `--set key.sub=value ...` and are type-coerced.
- `bc.eval_batches=0` evaluates on the whole holdout.
- `bc.streaming=true` loads 64 games at a time; `false` loads every row into memory and
  `bc.limit_rows` only works in that mode.
- The trainer does **not** report a value Brier score or a calibration table. Stage 4 adds that.

Never run, so treat as unknown:

- `configs/replay_blueprint_cpu.yaml` is an untested draft. Whether it runs at all, how many rows
  per second it trains on CPU, and how much memory it needs are unknown.
- The recipe sets `loss.require_exact_chosen_action_features` and
  `loss.require_exact_chosen_legality`. They keep a row for policy training only if the chosen
  action's provenance code is 2 (exact). From `fog/scripts/parse_replays.py` it looks as if nearly
  all rows should pass (switches are always exact; a move the player used is revealed), but this
  has not been measured. Stage 3 measures it.

## 4. Setup

A machine with 4 cores, 15 GB of memory and no GPU is enough. A GPU is welcome: add
`device=cuda mixed_precision=true` to the `--set` list of every training command and say so in the
report.

```sh
git clone https://github.com/IsaacJYPan/fog && cd fog      # or use the existing checkout
git fetch origin work/roadmap && git checkout -B work/p2-replay-curve origin/work/roadmap
git config user.name "Isaac Pan"
git config user.email "154495180+IsaacJYPan@users.noreply.github.com"
git submodule update --init --recursive
uv sync --locked --extra dev
```

`uv sync` downloads about 1.5 GB and builds the Rust engine; on a slow link it takes more than
15 minutes. Run it as a background job and wait for it. Do not kill and restart it because it
looks stalled.

The Showdown build is only needed for stage 7. Plain `npm ci` has been refused in some sandboxes;
this form worked:

```sh
npm --prefix server/pokemon-showdown ci --ignore-scripts
npm --prefix server/pokemon-showdown run build
```

The import needs HTTPS access to `huggingface.co` and `us.aws.cdn.hf.co`. If either is refused,
stages 1 to 6 are blocked: record it, and do stage 4's code and tests (they need no data) so the
session is not wasted.

Record in the report: cores, memory, GPU or not, `torch.__version__`, and the commit you started
from.

## 5. Stages

Run every command from the repository root. Record wall-clock time for each stage.

### Stage 1. Import 22,000 games and convert them

```sh
uv run --no-sync python -m fog.scripts.import_metamon_replays \
  --out data/metamon_gen9ou_r1500 \
  --revision 792a86b5cdbcab9696032e947483b218453a0d4f \
  --min-rating 1500 --after 2024-02-01 \
  --max-games 22000 --max-shards 4 --max-download-mb 600

uv run --no-sync python -m fog.scripts.convert_replay_decisions \
  --raw data/metamon_gen9ou_r1500 --out data/metamon_gen9ou_r1500_decisions \
  --imputations 1 --workers 4
```

Use exactly these directory names (they match the git-ignore patterns). If the import stops short
of 22,000 games because it ran out of shards or download budget, re-run the same command with
`--max-shards` and `--max-download-mb` raised (the importer continues where it stopped); go up to
8 shards and 1,200 MB. If it still has fewer than 22,000, carry on with what there is and scale
the training sizes in stage 2 down in proportion, keeping the holdout at 2,000 games.

Do not change the revision, the rating floor or the date, and do not run
`fog.scripts.download_usage`: the conversion fills in unseen team details from the usage prior
already in the repository, and changing that prior would change the data between curve points.

Conversion should take about half an hour. Record: games imported, source rows scanned, shards
used, games converted, decision rows, rows excluded and why (the converter prints these), disk
size, games per second.

### Stage 2. Split: a newest-games holdout and nested training sets

`fog/scripts/split_replay_manifest.py` only does a random split by fraction. Extend it (keep the
existing behaviour and flags working) with:

- `--chronological-holdout-games N`: the holdout is the `N` games with the largest `uploadtime`
  in the manifest (break ties by `id`). Every other game is training.
- `--train-sizes S1 S2 ...`: shuffle the training games once with `--seed`, then write
  `train_<S>.parquet` holding the first `S` games of that order for each size. This makes the
  sets nested: every game in a smaller set is in every larger one.

The outputs are `holdout_manifest.parquet`, `train_manifest.parquet` (all training games) and
the `train_<S>.parquet` files, all with the same columns as the input manifest (one row per game;
the trainer reads the `shard` column). Keep the refusal to overwrite an existing split.

Add a test (see `tests/test_bc_metrics.py` and `tests/test_replay_supervision.py` for the style)
on a small made-up manifest: the holdout and training sets share no game, the newest training
game is not newer than the oldest holdout game, the sets are nested, and the sizes are right.

Then:

```sh
uv run --no-sync python -m fog.scripts.split_replay_manifest \
  --manifest data/metamon_gen9ou_r1500_decisions/manifest.parquet \
  --out runs/blueprint/split \
  --chronological-holdout-games 2000 --train-sizes 40 1000 3000 10000 20000
```

Record the number of games and decision rows in each file and the upload-date range of the
training set and of the holdout. Commit the code and test.

### Stage 3. Three cheap checks before any real training

**3a. How many rows may train the policy.** Over the shards of `train_1000.parquet`, compute:
the share of rows with `policy_valid` true; of those, the share where
`action_feature_provenance[row, action[row]] == 2`; and the share where
`legal_provenance[row, action[row]] == 2`. (Each shard is an `.npz` with arrays `action`,
`policy_valid`, `action_feature_provenance`, `legal_provenance`.) Also record the mean number of
legal actions per row (`legality_mask` summed over the last axis) and the share of rows whose
chosen action is a switch (action index 8 or higher).

Decision: if at least 90% of rows pass both provenance tests, keep the recipe's two
`require_exact_*` flags. If fewer pass, add
`loss.require_exact_chosen_action_features=false loss.require_exact_chosen_legality=false` to the
`--set` list of **every** training command below, and say so in the report. Make this choice once
and use it for all runs.

**3b. Can the model memorise 40 games?** This catches a broken pipeline (labels not lined up with
inputs, a mask that hides the answer) in minutes instead of hours.

```sh
uv run --no-sync python -m fog.training.bc --config configs/replay_blueprint_cpu.yaml --set \
  bc.manifest=runs/blueprint/split/train_40.parquet \
  bc.holdout_manifest=runs/blueprint/split/train_40.parquet \
  bc.streaming=true bc.epochs=40 bc.early_stopping_patience=0 \
  log.dir=runs/blueprint/overfit40
```

The "holdout" here is the training set on purpose. Read `runs/blueprint/overfit40/metrics.csv`
(look at its header; the column names are not listed in this brief). Pass: top-1 on these rows
reaches at least 0.90 and the action loss falls far below its epoch-0 value. If it does not pass,
**stop the training stages**: do not run stages 5 to 7. Report the curve of the metrics over the
40 epochs and what you can find about why (for example: is the chosen action always inside the
legality mask; are the action features of the chosen action non-zero; does top-1 on switch rows
differ from move rows). Then finish stage 4 and the report.

If this command fails to start because of the config itself (a missing key, a checkpoint path, a
preflight requirement), fix `configs/replay_blueprint_cpu.yaml` with the smallest change that
makes it run from scratch with the model defined by `configs/gen9ou_typed_token_trunk.yaml`,
commit that with an explanation, and record it. If `bc.streaming=true` is the cause, use
`bc.streaming=false` for sizes that fit in memory and record that.

**3c. How fast is training?** One epoch on 1,000 games with a short evaluation:

```sh
uv run --no-sync python -m fog.training.bc --config configs/replay_blueprint_cpu.yaml --set \
  bc.manifest=runs/blueprint/split/train_1000.parquet \
  bc.holdout_manifest=runs/blueprint/split/holdout_manifest.parquet \
  bc.streaming=true bc.epochs=1 bc.eval_batches=8 \
  log.dir=runs/blueprint/timing
```

Record training rows per second, peak memory (`/usr/bin/time -v`, where installed, gives "Maximum
resident set size"), and the model's parameter count (count it yourself if the trainer does not
print it). Separately time one evaluation over the whole
holdout (stage 4's tool gives you this).

### Stage 4. A holdout report tool

Add `fog/eval/replay_holdout_report.py`:

```sh
uv run --no-sync python -m fog.eval.replay_holdout_report --config configs/replay_blueprint_cpu.yaml \
  --checkpoint PATH --manifest runs/blueprint/split/holdout_manifest.parquet --out PATH.json \
  [--set key=value ...]
```

It builds the model the way `fog/training/bc.py` does, loads the checkpoint, runs every row of the
manifest once in evaluation mode, and writes JSON with:

- `rows`, `policy_rows` (rows the policy mask keeps; use `_policy_supervision_mask` from
  `fog/training/bc.py` with the config's `loss` section so it matches training);
- policy: `top1`, `nll` (mean negative log-probability of the human's action), `avg_legal_actions`,
  `top1_lift` (top-1 × average legal actions), and top-1 and NLL separately for move rows and
  switch rows;
- value: the model's value is in [-1, 1] and the target `outcome` is +1 win, -1 loss, 0 draw for
  the acting player. Convert both to a win probability `p = (v + 1) / 2`, clipped to [0, 1].
  Report `value_mse` on the raw scale, `brier` = mean of `(p - target_p)²`, a ten-bin reliability
  table (equal-width bins of `p`; per bin: count, mean predicted `p`, observed win rate) and
  `ece` = Σ over bins of (count / total) × |mean predicted − observed|;
- value by game phase: Brier for rows in the first third, middle third and last third of their
  game, if the row's turn or position in the game can be read from the shard; if it cannot, leave
  this out and say so;
- the checkpoint path, manifest path, row counts and the commit.

Put the arithmetic (Brier, reliability table, ECE) in a pure function and unit-test it on small
hand-made arrays whose answers you can state. A constant prediction of 0.5 has Brier 0.25; keep
that as the reference line in the report.

Run it on the untrained model too (a freshly initialised model saved to a file, or a flag that
skips loading) so the curve has a zero point. Commit the code and test.

### Stage 5. The learning curve

Four runs, identical except for the training set:

```sh
for S in 1000 3000 10000 20000; do
  uv run --no-sync python -m fog.training.bc --config configs/replay_blueprint_cpu.yaml --set \
    bc.manifest=runs/blueprint/split/train_$S.parquet \
    bc.holdout_manifest=runs/blueprint/split/holdout_manifest.parquet \
    bc.streaming=true log.dir=runs/blueprint/g$S \
    > runs/blueprint/g$S.txt 2>&1
done
```

(Create `runs/blueprint/` first; add the stage 3a flags if that stage said so. Run them one after
another as background jobs and wait; do not run two at once.)

Rules:

- Smallest first. After each run, score `bc_best.pt` with the stage 4 tool, copy the result
  files (section 8), update the report, commit and push. Then start the next.
- Before each run, project its time from stage 3c: rows × 6 epochs ÷ rows per second, plus six
  holdout evaluations. **If the projection is over 6 hours, do not start that run or any larger
  one.** Write the exact skipped commands in the report so they can be run on a GPU. A curve
  with three points and an honest note is a good outcome.
- Do not change epochs, batch size, learning rate or model between sizes. The recipe uses batch
  256 where the roadmap asks for ≥ 1,024; that deviation is known, keep it and note it.
- One fallback is allowed, once: if a run's loss becomes NaN, or its holdout policy NLL after
  the first epoch is worse than the untrained model's, stop it, re-run **all** sizes with
  `bc.lr=1.0e-4` instead, and report both attempts. No other tuning.
- If a run is killed by the machine, record that and start it again from the beginning. Do not
  splice partial runs together.

For each size report: games, decision rows, epochs run and which epoch was best, wall-clock,
then on the holdout: top-1, top-1 lift, NLL, value Brier, ECE; and the final training top-1 and
action loss from `metrics.csv` (the gap between training and holdout matters for the next
decision).

### Stage 6. Read the curve

Write these down in the report, with the numbers:

- Holdout NLL at each size, the change from 1,000 to 10,000 games, and the slope of NLL against
  log10(games) fitted over all the points you have (that slope is the change per tenfold data).
- The roadmap's rule (P2.4), applied to the points you have: an improvement of more than 0.02
  NLL per tenfold data means more data is the next step; less than 0.005 with training NLL close
  to holdout NLL means the model is the limit.
- Whether the largest model's lift is clearly above chance. For orientation only, and marked as
  the brief author's estimate, not a gate: a top-1 lift under 1.3 at the largest size, or no
  improvement in NLL from the smallest size to the largest, means it is not learning usefully.
  In that case do not start changing things; list what the stage 4 breakdown shows (move rows
  versus switch rows, value calibration by phase) and stop there.

Do not tick P2.1 to P2.4 in `ROADMAP.md`: their acceptance asks for 100,000 games. Add one line
under each saying what has been done at this scale, with the ledger section's name.

### Stage 7. First look at play strength (smoke)

Needs the Showdown build. Use the `bc_best.pt` of the largest size trained. Policy-only games are
fast (minutes for hundreds of games). Four rows, 400 games each:

```sh
for T in 0 0.5; do
  uv run --no-sync python -m fog.eval.showdown_direct_selection \
    --pool data/teams/pool_b --device cpu \
    --config configs/replay_blueprint_cpu.yaml \
    --checkpoint blueprint=runs/blueprint/g20000/checkpoints/bc_best.pt \
    --opponent simpleheuristics --opponent maxdamage \
    --games 400 --temperature $T --out runs/blueprint/play_t$T.json
done
```

This command line was assembled from the script's argument list and has not been run with a
replay-trained checkpoint; if it needs adjusting (for example `--checkpoint-config`), read
`fog/eval/showdown_direct_selection.py` and record what you changed. The script prints one
ready-to-paste Markdown ledger row per opponent; paste those rows.

Label every row **smoke**: 400 games is about ±5 points, and `pool_b` is not a held-out team
split. Gate G2 in the roadmap (≥ 65% against SimpleHeuristics, and beating the older imitation
checkpoint, which is not in this clone) is **not** decided by these rows, whichever way they
fall. Report the win rates with their intervals and say only what they show.

If the offline numbers in stage 5 are good and these win rates are poor, that is a known
possibility (the replays never show the player's full team, and the model has only ever seen
states that humans reached). The roadmap's answer is its phase P3. Record the result; do not try
to fix it here.

If Showdown cannot be built, record that stage 7 was not run and why.

## 6. Ledger and roadmap

Append one new section to `docs/results-ledger.md`
(`## <date>: replay blueprint learning curve at 1k–20k games (ROADMAP P2)`), never editing older
sections. It holds: the data counts from stages 1 and 2, the stage 3 checks, one table with a
row per training size (columns as listed at the end of stage 5, plus the commit and the command),
the untrained-model row, the reliability table of the largest model, the stage 6 reading, and the
stage 7 rows marked smoke. State the deviations from the roadmap: batch 256, 20,000 games instead
of 100,000, and anything skipped.

## 7. Time and order

There is no deadline, but assume the machine may disappear. If time is short, the order of value
is: stage 3b (does it memorise) → stage 5 at 1,000 and 3,000 games → stage 4 → larger sizes →
stage 7. A pushed report covering three stages beats an unpushed one covering seven.

## 8. What to hand back

Everything on the pushed branch:

1. Code and tests from stages 2 and 4, and any fix to the recipe.
2. The ledger section and the roadmap notes.
3. `docs/briefs/01-artifacts/` with the small result files copied out of `runs/` (text and JSON
   only, no checkpoints; files ending in `.log` are git-ignored, so use `.txt`):
   each run's `metrics.csv` as `g<S>-metrics.csv`, the last 60 lines of each training output as
   `g<S>-tail.txt`, every holdout report JSON, the overfit run's `metrics.csv`, and the two
   evaluation JSON reports from stage 7 (leave those out if either is over 2 MB and say so).
4. `docs/briefs/01-report.md`, at most about two pages, in this order:
   - **Answer**: two or three sentences. Does it learn? Does it play? How sure are these numbers?
   - **Curve table**: one row per size plus the untrained model.
   - **Per stage**: done / partly done / not run, with the reason and the wall-clock time.
   - **Deviations and corrections**: every place you departed from this brief, and every command
     in it that was wrong, with what you ran instead.
   - **Not verified**: things you believe but did not check.
   - **Checkpoints**: path, size and SHA-256 of each `bc_best.pt`. They are git-ignored and stay
     on the machine; say whether that machine is expected to persist.
   - **Commands for a GPU run**: the exact lines for any size you skipped, and for 100,000 games.
   - **Suggestions**: at most five lines.

Your final message should be the **Answer** paragraph, the branch name and the last commit hash.

## 9. How the reviewer will check this

So you know what will be looked at:

- The split test proves the holdout is newer than, and disjoint from, every training set.
- Each number in the report and the ledger matches a file in `docs/briefs/01-artifacts/`.
- All curve runs used the same recipe and flags; the only difference is the training manifest.
- The overfit check ran before the curve, and its outcome is stated.
- Smoke rows are labelled smoke, and no gate is claimed.
- Skipped or failed work is listed as such, with the command to finish it.
- `ruff` is clean and the new tests pass; no frozen area was touched; no commit carries an
  assistant or vendor attribution.
