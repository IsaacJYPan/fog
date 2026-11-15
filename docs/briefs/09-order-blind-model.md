# Brief 09: the order-blind model, value targets, move features and leads

You are a coding agent working alone on the `fog` repository. Nobody will watch you work. A
reviewer reads what you push and decides the next step from it.

## 1. Why

Brief 07 had five stages. Stage 1, the converter, is done and merged (ledger, "review of
brief 07, stage 1"): replay rows no longer give away the label through move or team order.
**This brief is stages 2 to 5 of `docs/briefs/07-order-blind-encoder.md`, sections 5 to 8,
exactly as written there**, with the changes below. Read that brief first; its section 1
explains each defect and its section 2 rules apply here.

## 2. What is different from brief 07

- Branch **`work/order-blind-model`** from `work/roadmap`. CPU only; no hosted GPU.
- **Data.** Convert the 22,000 games with the merged converter
  (`convert_replay_decisions --imputations 1 --workers 4 --resume`; the conversion now resumes
  per game) and make the same split. Expected: 22,000 games, no quarantines, 103,944 holdout
  rows from the fixed 2,000 holdout games, 52,955 training rows in the 1,000-game split.
- **Baseline.** Run R1 once yourself (corrected rows, current model, seed 20261201). Brief
  07 measured holdout NLL 1.9239 and top-1 26.48% at epoch 8; a second seed gave 1.9293. Yours
  should be within about 0.01 NLL; if it is not, stop and report it. There is no R0 here.
- **Digests.** The three rollout digests depend on the CPU model. Record them on your machine
  at `work/roadmap` before you change anything, and show they are the same at the end with all
  new keys off. Do not compare with the values in `HANDOFF.md`.
- **Value (brief 07 section 6).** Brief 05 found on 20,000 games that the value head is best
  at epoch 4 (Brier 0.215) and the policy at epoch 12, by which point value Brier is 0.254.
  So `bc.save_best_value` is expected to matter; report both checkpoints for every run.
- **Play smoke (brief 07 section 7).** R1 and R4 only.
- Report file `docs/briefs/09-report.md`, artifacts `docs/briefs/09-artifacts/`, ledger
  section `## <date>: order-blind model, value targets and move features (smoke)`.

## 3. Order of work

Stage 2 (order-blind model and its permutation test, run R2), stage 3 (value targets, R3),
stage 4 (move features, R4, rare-move split, play smoke), stage 5 (leads). Push after each.
If you run out of time, stop after a finished stage and say which.

The brief 07 acceptance test for stage 2 is the part most likely to be skipped and the part
the reviewer checks first: weights perturbed before testing, all four permutations, and the
same test failing with the key off.

Final message: the **Answer** paragraph of your report, the branch name and the last commit
hash.
