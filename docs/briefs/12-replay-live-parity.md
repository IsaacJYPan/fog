# Brief 12: make a replay row look like the observation the agent gets in play

You are a coding agent working alone on the `fog` repository. Nobody will watch you work. A
reviewer reads what you push and decides the next step from it.

## 1. Why

The replay policy is trained on rows rebuilt from public logs and then plays from live
observations. Wherever the two differ, the policy acts on inputs it never saw. Four
differences were found on 2026-11-14 (`HANDOFF.md`); none is fixed:

| Difference | Evidence |
| --- | --- |
| Own PP is 0 in every replay row; live play has real fractions | Largest own PP over 42,391 rows was 0.0 |
| Forced replacement choices are not training rows | 2 switch-only rows of 42,391; `parse_replays.py` accepts a label only when the request is `awaiting_action` |
| Live, a replacement choice overwrites the move recorded for that turn in the history; replays keep the first action | `record_history_decision` in `fog/encoding/events.py` against `record_action` in `fog/scripts/parse_replays.py` |
| At a replacement decision the current turn's events, including the faint, are not in the history | `encode_history` uses turns before the current one |

There is no test that compares the two paths. This brief builds that test first and then
uses it to close the differences.

**Start after brief 09 is merged**: both touch the converter.

## 2. Rules

- Branch **`work/replay-parity`** from `work/roadmap`. Push after every stage.
- Commits are authored as the owner (`git config user.name "Isaac Pan"`,
  `git config user.email "154495180+IsaacJYPan@users.noreply.github.com"`); no assistant or
  vendor names, co-author lines or session links anywhere; no pull requests.
- Read `AGENTS.md` first. CPU only. Setup and data are in `docs/briefs/09-order-blind-model.md`.
- Live behaviour is the reference. Change the converter to match live play. Change live code
  only where it is plainly wrong (the history overwrite), behind a test, and say so.
- Rollout digests are compared before and after on your own machine; a live-side change that
  alters them must be named in the report with the reason.
- Every number is labelled smoke.

## 3. Stage 1: the parity check

Write `fog/eval/replay_live_parity.py`:

1. Play N self-play games on pinned Showdown with any checkpoint or random weights, on
   `data/teams/pool_b`. Record every learner observation exactly as the model receives it,
   the action taken, and each side's true team sheet. Save each game's public log.
2. Convert each saved log with the replay converter, giving it the true sheets in place of
   imputation (so imputation is not what is being measured) and team and move order as played.
3. Align rows by turn and decision and compare field by field: entity floats by named block
   (`fog/encoding/entities.py` has the layout), entity ids, field vector, legality mask,
   active index, decision phase, history record and history ids, label.
4. Report, per field, the share of rows that match exactly and the largest difference, and
   the number of live decisions with no replay row and the reverse.

Run it on 200 games before changing anything. That table is the baseline and the to-do list.

## 4. Stage 2: close the differences

In the order of how many rows each affects. Expected from the table above:

- **PP.** Count each own move's uses from the log (2 per use against a Pressure ability that
  the log has shown) and write the same fraction live play shows.
- **Replacement decisions.** Emit a training row for a forced switch, labelled with the
  Pokémon sent in, with the request phase live play would show.
- **History.** Make both paths record the same thing for a turn in which a move and then a
  replacement were chosen, and include what live play includes at a replacement decision.

Each fix has a unit test, and the parity table is re-run after each. Stop when every field
matches on at least 99.5% of aligned rows or the remainder is explained row by row.

## 5. Stage 3: does it matter

Reconvert the 22,000 games. Train the 1,000-game recipe of brief 09 twice: on rows as they
were before this brief and on the new rows, same model keys as the best run of brief 09, same
seed. Compare holdout NLL and top-1 on the rows both have, report the new replacement rows
separately, and play 400 greedy games each against SimpleHeuristics and MaxDamage on
`data/teams/pool_b` for both models.

## 6. What to hand back

1. The parity tool, fixes, tests; `ruff check fog tests` clean.
2. One ledger section (`## <date>: replay rows against live observations (smoke)`): the
   parity table before and after, and the stage 3 rows.
3. `docs/briefs/12-artifacts/` and `docs/briefs/12-report.md`, at most one page: **Answer**
   (which fields differed and on what share of rows; what is fixed; what remains and why;
   whether play changed), then not verified.

Final message: the **Answer** paragraph, the branch name and the last commit hash.

## 7. How the reviewer will check this

- The parity tool compares what the model receives, not an intermediate structure.
- True sheets are used only by the parity tool, never by the training converter.
- The before table shows the four known differences (if it does not, the tool is wrong).
