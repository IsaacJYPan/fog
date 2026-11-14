# Brief 06: collection and play-training speed with a GPU

You are a coding agent working alone on the `fog` repository. Nobody will watch you work. A
reviewer reads what you push and decides the next step from it.

## 1. Why

Collection on headless Showdown was made faster on 2026-11-14 (ledger section "faster rollout
collection on headless Showdown"): the trainer's rollout processes can play many battles at once
(`env.rollout_batch`) and can run the acting forward on a GPU (`env.actor_device`). On four CPU
cores without a GPU it collects 451 learner decisions per second in self-play and 647 against
MaxDamage. **The GPU path has never run on a GPU**, and nobody has measured how fast the PPO
update is on one. This brief measures both. The answer decides how many games a week of GPU
time can train on, and what to speed up next.

## 2. Rules

- Branch **`work/collect-gpu`** from `work/roadmap`. Push after every stage.
- Commits are authored as the owner (`git config user.name "Isaac Pan"`,
  `git config user.email "154495180+IsaacJYPan@users.noreply.github.com"`); no assistant or
  vendor names, co-author lines or session links anywhere; no pull requests.
- Read `AGENTS.md` first. Setup is in `docs/briefs/02-p2-scale-prep.md` section 4, with the
  branch name above.
- **GPU.** Private Kaggle jobs under the account already configured on your machine
  (`KAGGLE_TRAINING.md`, `docs/kaggle-workflow.md`); a plain private script job that clones this
  public repository at your pushed commit and builds Showdown is fine. If the client is not
  configured, stop and say so. **Budget: 3 GPU-hours in total.** If brief 05 is using the same
  account, run when it is not: one GPU job at a time.
- This repository is public: commit no account name, token or `configs/kaggle_results.json`.
- Fix a defect in the GPU path if one stops you, with a unit test that fails without the fix,
  and say so. Change nothing else in the model, the losses or the trainer.
- Every number is labelled smoke. Never fill in a number you did not measure.

## 3. Stage 1: collection with the forward on the GPU

In one GPU job, on the job's own CPU cores (record how many and the CPU model):

```sh
python -m fog.eval.showdown_rollout_probe --workers W --batch B --games 512 \
  --opponent O --pool data/teams/pool_a --device D
```

for `W` = the number of CPU cores, `O` in {`self`, `MaxDamage`}, and
(`D`, `B`) in {(`cpu`, 32), (`cuda`, 32), (`cuda`, 64), (`cuda`, 128)}. Run each twice and
report both. Also record GPU memory in use with all workers running (`nvidia-smi`).

If the `cuda` rows fail, find out why before anything else: this path was only tested on the
CPU. The expected weak points are a CUDA start-up problem in the spawned worker processes and
any tensor left on the wrong device in `fog/env/showdown_batch_runner.py` or
`FogPlayer.complete_decision`.

Check once that `--device cuda --batch 32` and `--device cpu --batch 32` play the same games:
same number of learner decisions is not expected (sampled actions differ once
log-probabilities differ in the last bits), so compare instead the first battle's first ten
recorded log-probability rows from each device and report the largest absolute difference.

## 4. Stage 2: the whole training loop

With a config that extends `configs/gen9ou_typed_token_selfplay.yaml`, turns off the preflight
requirement, the replay-policy term and the starting checkpoint as brief 04 did
(`docs/briefs/04-report.md`, "Config changes"), and sets `device: cuda`,
`env.actor_device: cuda`, `env.rollout_batch` to the best value from stage 1,
`env.rollout_workers` to the number of cores, `env.actor_threads: 1`,
`ppo.games_per_chunk` to at least workers × batch, `ppo.batch_decisions: 8000`,
`ppo.minibatch_size: 1024`, `ppo.epochs: 3`:

1. Train 100,000 decisions against MaxDamage only on the first team of `data/teams/pool_a`
   (brief 04 stage 1's problem, so the learning curve can be compared with its CPU run).
2. Report, from the trainer's own log and wall clock: decisions per second for collection
   alone, rows per second for the update alone, end to end, the share of wall time in each,
   and peak GPU memory.
3. Evaluate the final checkpoint with 200 greedy games against MaxDamage on the same team
   (`fog.eval.showdown_direct_selection`) and put the win rate beside brief 04's value at
   106,889 decisions. This is a check that the batched GPU path still learns, not a strength
   claim.

## 5. Stage 3: what limits it

From the numbers above, no new runs: state what a week of 30 GPU-hours trains at this speed, in
decisions and in games; which of collection and update takes more of the hour; and for the
larger one, its split (for collection: `--batch` stats from a one-worker
`python -m fog.eval.showdown_rollout_probe --batch B --games 128 --opponent MaxDamage`, which
prints seconds for prepare, forward, choose and step).

## 6. What to hand back

On the pushed branch:

1. The config, any fix with its test.
2. One new section in `docs/results-ledger.md`
   (`## <date>: collection and play-training speed on a GPU (smoke)`): one row per measurement
   with command and commit. Do not edit older sections.
3. `docs/briefs/06-artifacts/`: the job script, probe outputs, the trainer's metrics CSV and
   last 60 log lines, the evaluation JSON. Text and JSON only.
4. A dated line under P0.5 in `ROADMAP.md`; tick the box only if the `cuda` rows ran and stage 2
   learned.
5. `docs/briefs/06-report.md`, at most one page: **Answer** (collection rate on CPU and on GPU;
   update rate; end to end; what a 30-hour week trains; what limits it), the tables, fixes,
   not verified.

Final message: the **Answer** paragraph, the branch name and the last commit hash.

## 7. How the reviewer will check this

- Every number in the report is in a file under `docs/briefs/06-artifacts/`.
- GPU-hours add up to 3 or less; no account name or token is in the repository.
- Any fix has a test; nothing else in the model, losses or trainer changed; `ruff check fog tests`
  is clean.
- Rows are labelled smoke; no commit carries an assistant or vendor attribution.
