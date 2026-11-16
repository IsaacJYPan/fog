# Brief 17: search distillation at scale (P3.2 to P3.4, gate G3)

You are a coding agent working alone on the `fog` repository. Nobody will watch you work. A
reviewer reads what you push and decides the next step from it.

## 1. Why

The pilot (brief 15; ledger, "search distillation pilot") worked. A network trained from
random weights on 58,160 decisions labelled by EngineMCTS scored 44.8% against
SimpleHeuristics and 81.5% against MaxDamage without search; adding 60,000 decisions from
positions that network itself reached moved it to 61.5% and 90.9% (1,008 games each,
validation teams). Its teacher scores 92.9% and 97.3%. Strength was still rising steeply at
120,000 rows. The roadmap's P3.2 asks for one million. This brief takes the same loop there
and measures gate G3.

**Start when the ledger section "search distillation pilot" has its last round recorded.**
Read that section first: it has every command, seed and cost.

## 2. Rules

- Branch **`work/distil-scale`** from `work/roadmap`. Push after every round.
- Commits are authored as the owner (`git config user.name "Isaac Pan"`,
  `git config user.email "154495180+IsaacJYPan@users.noreply.github.com"`); no assistant or
  vendor names, co-author lines or session links anywhere; no pull requests.
- Read `AGENTS.md`, then `fog/training/search_distill.py` and its tests. Setup is in
  `HANDOFF.md` section 9.
- Generate only on `data/teams/replay_v1/train`; evaluate on `data/teams/replay_v1/validation`;
  never read the test split.
- The teacher stays EngineMCTS at K = 8 and 200 ms, one search thread, one collection
  process per idle core and nothing else on the machine while collecting (the budget is
  wall-clock). Do not change search code or the model.
- This repository is public: commit no account name, token or private dataset name.
- **Budgets.** CPU: 150 core-hours of collection. GPU: 6 hours, one job at a time on the
  account, and only when brief 13 is not using it; read `kaggle quota --format json` before
  every job and write the reading down. Keep a running total of both in the artifacts.
- Collected rows and checkpoints are not in git. Keep every round in the owner's private
  hosted dataset (the pilot's rounds are already there; `kaggle datasets list --mine` shows
  it) and record SHA-256 of every checkpoint you report.

## 3. The loop

One round is: collect about 100,000 labelled rows with the current best network acting
(`--beta 0.25`, a new seed each round), train a network on all rows so far, play it.

- **Collect** with `python -m fog.training.search_distill collect ... --agreement-every 40`.
  About 0.35 s a row on one core: 100,000 rows is about ten core-hours.
- **Train** with `python -m fog.training.search_distill train --device cuda` on a hosted GPU
  (`docs/briefs/15-artifacts/gpu_train_job.py` is the job the pilot ran: eight epochs over
  118,518 rows finished in under eleven minutes including setup, against 58 minutes on four
  CPU cores). Random initial weights each round, eight epochs, as in the pilot, so the rounds
  form one learning curve. If the GPU is not available, train on the CPU and say so.
- **Play** each round's network alone, greedy, 1,008 matched games against each of
  SimpleHeuristics and MaxDamage (about three minutes; the command is in the ledger).

Stop when one million rows are stored, or when two consecutive rounds each fail to raise the
score against SimpleHeuristics by more than the width of its 95% interval, or when the CPU
budget is spent; say which.

## 4. Three questions to answer on the way, one change at a time

Each costs one GPU training and one six-minute play run. Do them on the data of the second
round you collect (about 380,000 rows with the pilot's), and keep a change only if the
1,008-game score against SimpleHeuristics rises by more than its interval.

1. **Sharper targets.** The teacher's visit distributions are flat (about 1.77 nats; it
   agrees with its own second seed on 87% of choices and the network matched it on 57%).
   Add a default-off `--target-power P` to the trainer (targets raised to the power P and
   renormalised; a test for P = 1 being the identity) and try P = 2.
2. **Initial weights.** Train once from the brief 13 blueprint's best-policy checkpoint
   instead of random weights, if brief 13 has delivered one that loads into
   `configs/search_distill_cpu.yaml`'s model (same keys except the outcome auxiliary heads;
   load with those dropped and say so). The roadmap's P3.3 asks for this.
3. **The value head.** In the pilot it was no better than a constant on held-out outcomes
   after the first epoch. Save the checkpoint with the lowest held-out value error
   separately (as `bc.save_best_value` does for replays) and report its Brier score and
   calibration. Nothing in this brief plays from the value head; P4 will need it.

## 5. Gate G3 and the other P3.4 rows

For the final network:

- at least 1,008 matched games against EngineMCTS at the default budget (K = 8, 1,600 ms)
  with `fog.eval.run_shards`, four shards, nothing else on the machine (about three and a
  half hours on four cores). G3 asks for at least 30%;
- 1,008 games against its initial weights if question 2 was kept (G3 asks for more than
  55%), otherwise record that clause as trivially met against random weights and not yet
  measured against a blueprint;
- 1,008 games against the pilot's network 1 if its checkpoint can be regenerated or was
  kept, as the measure of what the extra rows bought.

State each G3 clause as met, not met or not measurable. If G3 fails, do what its "if it
fails" line in `ROADMAP.md` says, and report; do not change the model.

## 6. What to hand back

1. Code changes (the target-power switch, the value checkpoint) with tests;
   `.venv/bin/ruff check fog tests` clean.
2. One ledger section (`## <date>: search distillation at scale (P3)`): one row a round
   (rows, label-check agreement, teacher top-1 agreement, both play scores with intervals),
   the three questions, the G3 rows, commands and commits.
3. `ROADMAP.md`: dated lines under P3.2 to P3.4; tick what its acceptance text allows; the
   G3 result under the gate.
4. `docs/briefs/17-artifacts/` (round summaries, metrics, play JSONs, budget record; no
   checkpoints or rows) and `docs/briefs/17-report.md`, at most one page: **Answer** (the
   learning curve in one table; G3; which of the three changes were kept), then what was not
   verified.

Final message: the **Answer** paragraph, the branch name and the last commit hash.

## 7. How the reviewer will check this

- Generation used train teams and evaluation validation teams (the commands show it).
- Every play row matches its JSON file and has at least 1,008 games.
- The GPU and CPU totals are within budget; no account or dataset name is in the repository.
- `tests/test_search_distill.py` passes on the reviewer's machine.
