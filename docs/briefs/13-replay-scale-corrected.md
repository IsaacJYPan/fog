# Brief 13: the replay blueprint at scale on corrected rows (GPU)

You are a coding agent working alone on the `fog` repository. Nobody will watch you work. A
reviewer reads what you push and decides the next step from it.

## 1. Why

Brief 05 trained on 20,000 and 40,955 games on a GPU, but on rows that gave away the label
through move order, so its accuracy numbers are inflated and its models lean on a shortcut
that does not exist in play (ledger, "review of brief 07, stage 1"). Since then the converter
is corrected, the model can be made blind to slot order with better value targets and move
features (brief 09), and a slow loop in training is gone (ledger, "per-row Python removed
from the belief training path"). This brief trains the blueprint again, properly, and takes
it to gate G2.

**Start only when briefs 08 and 09 are merged.** The review of brief 09 names the model keys
to switch on; use exactly those. If it names none, use the current model.

## 2. Rules

- Branch **`work/p2-scale-corrected`** from `work/roadmap`. Push after every stage.
- Commits are authored as the owner (`git config user.name "Isaac Pan"`,
  `git config user.email "154495180+IsaacJYPan@users.noreply.github.com"`); no assistant or
  vendor names, co-author lines or session links anywhere; no pull requests.
- Read `AGENTS.md` first, then `docs/briefs/05-p2-gpu-scale.md` and `docs/briefs/05-report.md`:
  reuse that brief's job scripts, resume logic and budget accounting.
- **GPU budget: 20 GPU-hours**, one job at a time on the account. Data preparation and play
  run on the CPU.
- This repository is public: commit no account name, token or private dataset name.
- Change no model, loss or trainer code. If something must change, stop and report.
- Every number is smoke unless it meets the roadmap's 1,000-game rule.

## 3. Stage 1: data

Import and convert with the merged converter (`--resume`): the fixed 2,000-game holdout, and
nested training sets of 20,000 and 100,000 games (as many as pass the filters; brief 05
prepared 101,928). Report games, rows, row counts per split, and the row check of
`fog.eval.replay_slot_order_check` on the holdout (label share per move slot near 25%).

## 4. Stage 2: training

1. One epoch on 20,000 games first, to measure rows per second and seconds per validation.
   Brief 05 measured 583 rows/s before the loop fix. From the measurement, choose the largest
   nested training set and epoch count that finish inside the budget with two hours spare,
   and write the arithmetic down before starting.
2. Train 20,000 games to its best holdout policy NLL, then the larger set. Batch 1,024. Save
   the best-policy and the best-value checkpoint separately.
3. Report per run: best policy epoch, holdout NLL and top-1 (all rows, move rows, rows where
   the acting Pokémon has not moved), value Brier and ECE at both checkpoints with intervals
   that resample games, the ten-bin reliability table, and the slot check (stored against
   shuffled).

Apply ROADMAP P2.4's decision rule to the 20,000-to-larger step and state the outcome.

## 5. Stage 3: play

For the best model: 400 greedy games each against SimpleHeuristics and MaxDamage on
`data/teams/pool_b` (comparable with briefs 03 and 05), then, if
`data/teams/replay_v1/validation` exists, at least 1,000 matched games against each on that
split, greedy and at temperature 0.5 (P2.5).

Gate G2 as written in `ROADMAP.md`. State each clause as passed, failed or not measurable
(the head-to-head against `A3` is not measurable without that checkpoint).

## 6. What to hand back

1. One ledger section (`## <date>: replay blueprint at scale on corrected rows`).
2. `ROADMAP.md`: dated lines under P2.1 to P2.5; tick only what is complete; the G2 result.
3. `docs/briefs/13-artifacts/` (metrics, reliability tables, slot checks, play JSONs, budget
   record; no checkpoints) and where the checkpoints are kept, with their SHA-256.
4. `docs/briefs/13-report.md`, at most one page: **Answer** (NLL and top-1 at each size; the
   P2.4 decision; value at both checkpoints; play; each G2 clause), then not verified.

Final message: the **Answer** paragraph, the branch name and the last commit hash.

## 7. How the reviewer will check this

- Rows come from the corrected converter (the slot check on the holdout shows it).
- The holdout is the fixed 2,000 games; intervals resample games.
- GPU-hours are 20 or less; no account or dataset name is in the repository.
- Play rows match their JSON files; no gate clause is claimed on fewer than 1,000 games.
