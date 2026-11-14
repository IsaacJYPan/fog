# Brief 07: remove the move-order leak, make the model blind to slot order, and fix what the replay model cannot learn

You are a coding agent working alone on the `fog` repository. Nobody will watch you work. A
reviewer reads what you push and decides the next step from it.

## 1. Why

Four defects were found on 2026-11-14 by reading the code and by one measurement (`HANDOFF.md`
section 5). Each stage below fixes one and measures the effect on the same data.

1. **Replay rows give away the answer through move order.** `complete_private_team`
   (`fog/training/replay_policy_data.py`) puts the acting Pokémon's moves in order of first use.
   On rows where the Pokémon has not moved yet the label is the first move slot in 99.8% of
   150 games. A 900-game model scored 0.573 top-1 on such rows as stored and 0.303 with the four
   slots shuffled. Live play lists moves in team-sheet order, so the shortcut is wrong there.
2. **The model can read slot order at all.** `EntityEmbeddings.forward`
   (`fog/models/three_encoder.py`) concatenates the four move embeddings in slot order; the
   entity record carries four PP values in slot order and a `party_slot` one-hot;
   `SharedBattleLatent` adds `token_position` and `party_position` to every Pokémon token; the
   history record stores our action as a 14-way slot one-hot (`hist_my_action`). Move order and
   bench order mean nothing in the game, so any dependence on them is an error waiting to happen.
3. **The value head memorises games.** 3,000 games are 3,000 outcome labels. At the best policy
   epoch the holdout Brier score was 0.3445 (a constant 0.5 scores 0.25). The loss is
   `tanh` + MSE on win or loss only, and the saved checkpoint is chosen by policy loss.
4. **Moves are bare ids.** A move candidate is described by a 24-wide id embedding
   (`FogAgent.action_context`), so type, power and category must be learned per move and a move
   that is rare in training carries nothing. Brief 04 found no transfer to unseen teams; whether
   this is the cause is not known.

A fifth, smaller one is stage 5: the lead choice (`TeamPreviewHead`) is never trained
(`preview_coef: 0.0`), so leads are close to random.

## 2. Rules

- Branch **`work/order-blind`** from `work/roadmap`. Push after every stage. If you run out of
  time, stop after a finished stage; each stage stands alone.
- Commits are authored as the owner (`git config user.name "Isaac Pan"`,
  `git config user.email "154495180+IsaacJYPan@users.noreply.github.com"`); no assistant or
  vendor names, co-author lines or session links anywhere; no pull requests; do not trigger
  hosted workflows.
- Read `AGENTS.md` first. Setup and data commands are in `docs/briefs/02-p2-scale-prep.md`
  section 4, with the branch name above. CPU only (4 cores, 15 GB is enough). **Do not use the
  hosted GPU account**: briefs 05 and 06 are using it.
- **Every model change is behind a config key that defaults to off.** With all new keys off,
  old checkpoints load and the three rollout digests in `HANDOFF.md` section 3 are reproduced
  (`0efd4319547a4b3b`, `23405989f4665988`, `338cfca82f31b181`). Show this once at the end.
- The converter fix in stage 1 is not behind a key: the old order is a defect.
- Keep information isolation: nothing about the opponent's hidden team may enter a policy or
  value input. Training-only targets may use what is public at the end of the game.
- Do not touch search, PPO or belief code. If a belief head needs slot order, leave its inputs
  as they are and say so.
- Every number is labelled smoke. Never fill in a number you did not measure. Claim no
  difference smaller than twice the difference between two seeds of the same run.
- Intervals for holdout numbers resample **games**, not rows (rows of one game are correlated).

## 3. The fixed comparison

Data: the 22,000-game import and the chronological 2,000-game holdout of brief 02 (ids tracked
in `data/replay_splits/p2_holdout_2000_ids.txt`), the 1,000-game training split, one imputation.

Recipe for every run: `configs/replay_blueprint_cpu.yaml`, `bc.epochs=8`, early stopping off,
seed as in the file. A run is about an hour on four cores [estimated from brief 03's 157 to 194
rows/s, not measured for this recipe]. Evaluate every epoch on the whole holdout.

Report for every run, at its best policy epoch: holdout policy NLL and top-1 on all rows, on
move rows, and on rows where the acting Pokémon has not moved yet; value Brier and ECE; and the
output of

```sh
python -m fog.eval.replay_slot_order_check --config <config> \
  --manifest runs/blueprint/split/holdout_manifest.parquet --checkpoint <bc_best.pt> --out <file>
```

| Run | Rows | Model |
| --- | --- | --- |
| R0 | as converted today (leaky) | current |
| R1 | stage 1 converter | current |
| R2 | stage 1 converter | stage 2 keys on |
| R3 | stage 1 converter | stages 2 and 3 on |
| R4 | stage 1 converter | stages 2, 3 and 4 on |

Run R1 and R4 with a second seed. R0's holdout is leaky too, so R0 and R1 are compared only on
the shuffled column of the slot check, which is the same question for both.

## 4. Stage 1: converter

1. Run R0 first, before changing anything, and keep its checkpoint and slot-check output.
2. In `complete_private_team`, after the four moves of a Pokémon are final and before
   `apply_private_set`, permute the four move columns with a generator seeded from the replay
   seed and the slot. Token and provenance columns move together. Everything downstream reads
   the installed order, so labels and history stay consistent; confirm this, do not assume it.
3. Tests: the same seed gives the same sheet; over many seeds each move lands in each column
   about a quarter of the time; exact and sampled provenance travel with their move; a
   converted fixture game still has every label on a legal action.
4. Extend `replay_slot_order_check` to report the share of **switch** labels per bench position
   and the share of games whose lead is team position 0. Say whether replay rows and a live
   `FogPlayer` observation order the team the same way (read `fog/env/player.py` and the
   encoder). Report only; fix nothing here unless the share is far from uniform, in which case
   shuffle team order in the converter the same way and test it the same way.
5. Reconvert the 22,000 games and remake the split. The holdout must be the same 2,000 games.
   Expected in the row check: label share per move slot near 25% each and
   `not_yet_moved_label_is_slot0` between 0.25 and 0.5.
6. Run R1.

## 5. Stage 2: a model that cannot see slot order

Key `model.order_blind_entities` (default false). When on:

- **Moves as a set.** Each of a Pokémon's four moves becomes one vector from a small shared
  network over (move embedding, that move's PP fraction, that move's known/unknown flag, and in
  stage 4 its features). The four vectors are pooled with a symmetric function (sum, or mean
  and max together) and take the place of the four concatenated embeddings, the four PP floats
  and the four move flags in the entity encoder's input. `last_move` stays as it is.
- **No slot position on Pokémon tokens.** No `token_position` or `party_position` for the twelve
  Pokémon tokens, and the `party_slot` one-hot is zeroed. `token_kind`, `token_side` and
  `is_active` stay. Field, control, history and global tokens keep their positions.
- **No arbitrary id numbers.** Drop `normalized_ids` from the entity encoder input.
- **History by identity.** Our own past action enters as its kind (move, Tera move, switch,
  forced) plus the move id or the species switched to, which the id sidecar already holds
  (`H_MY_MOVE` and the species fields in `fog/encoding/events.py`); the 14-way slot one-hot is
  zeroed. Read how `hist_opp_action` indexes the opponent's moves and report whether it is an
  order a player at the table can see; do not change it.
- **Candidates.** A move candidate is described by the same per-move vector as above instead of
  the bare embedding, so the scorer sees that move's PP. Switch candidates are unchanged.

Acceptance test (`tests/test_order_blind_model.py`), in eval mode, with **every parameter
perturbed by random noise first** (output layers are zero-initialised, so an untrained model
passes trivially): on at least 64 observations from holdout rows and 64 from a live rollout,
each of these changes the 14 logits only by the matching permutation and leaves the value
unchanged to 1e-5:

1. permuting the acting Pokémon's four moves (ids, PP, flags, legality together);
2. permuting the moves of any other Pokémon on either side;
3. permuting our bench (switch candidates, legality and `active_idx` remapped);
4. permuting the opponent's team.

The same test with the key off must fail for 1 and 3; assert that too, so the test is known to
bite. Then run R2. Expected: slot check stored equals shuffled to four decimals.

## 6. Stage 3: value targets

Keys under `loss`, all default off:

- `value_loss: logistic`: the win probability is `(value + 1) / 2`; train its logit with
  cross-entropy against win 1, loss 0, tie 0.5. The value the rest of the code reads keeps its
  range and meaning.
- `outcome_aux_coef`: two training-only heads. (a) For each of the twelve Pokémon tokens, one
  shared linear layer predicts whether that Pokémon is still alive when the game ends. (b) From
  the value reader's summary, the final difference in remaining HP fractions, (ours − theirs)/6.
  The converter adds both labels to every row from the public end state of the game. They are
  never read in play or search.
- `bc.save_best_value`: also save the checkpoint with the lowest holdout value loss and record,
  per epoch, value Brier, ECE and the two auxiliary losses.

Tests: labels for a fixture game match its log; the new heads get gradient and the policy and
value outputs are unchanged when the coefficients are zero. Run R3. Report value Brier and ECE
at the best policy epoch and at the best value epoch, with game-resampled intervals, and the
share of predictions below 0.1 or above 0.9 with how often those games were won. A Brier score
at or above 0.25 means the head still carries nothing; say so plainly if that is the result.

## 7. Stage 4: what a move does

Key `model.mechanical_features` (default false). When on:

- Per move, from the tables the repository already loads (no hand-written lists): type,
  category, base power, accuracy, priority, and any boolean flags the tables carry. These join
  the per-move vector of stage 2, for both sides' known moves.
- Per move candidate, these columns of `fog/encoding/action_features.py` and no others:
  `move_power`, `move_accuracy`, `expected_hits`, `physical`, `special`, `status_move`,
  `same_type_bonus`, `type_effectiveness`, `physical_stat_ratio`, `special_stat_ratio`,
  `estimated_damage`. **Not** `slot_fraction` (it is a slot position) and **not** the columns
  that are a rule player's judgement (`hazard_setup_ready` through `simple_rule_pick`). A unit
  test must fail if any excluded column reaches the model.

Run R4. Besides the standard report, split holdout move rows by how often the labelled move
appears as a label in the training split (never, 1 to 19 times, 20 or more) and report NLL and
top-1 for R3 and R4 in each group. That is the cheap test of whether features help where ids
cannot.

Then one play smoke, with the command and teams of brief 03's play smoke
(`docs/briefs/03-report.md`; 400 games each against SimpleHeuristics and MaxDamage on
`data/teams/pool_b`), for R0, R1 and R4 at their best policy epoch. This answers whether the
leak helped or hurt play. It is a smoke, not a strength claim.

## 8. Stage 5: leads (only after stages 1 to 4 are pushed)

The converter emits one preview row per player per game: the preview observation and, as label,
which of the six was sent out first (public in the log). Check that the label is not readable
from team position (share per position near one sixth; otherwise shuffle as in stage 1). Train
`TeamPreviewHead` with cross-entropy through `loss.preview_coef`. Report holdout lead top-1
against one sixth and against "always the species most often led in training". No play run.

## 9. What to hand back

On the pushed branch:

1. The code and tests; `ruff check fog tests` clean with `.venv/bin/ruff`; the tests covering
   each change pass; the three digests with all new keys off.
2. One new section in `docs/results-ledger.md`
   (`## <date>: move-order leak removed and order-blind encoder (smoke)`): one row per run with
   command and commit. Do not edit older sections.
3. `docs/briefs/07-artifacts/`: per-run metrics CSV, slot-check JSON, play summaries, the
   converter's row-check output before and after. Text and JSON only; no checkpoints.
4. `docs/briefs/07-report.md`, at most two pages: **Answer** first (did the leak inflate
   holdout accuracy and by how much; does the order-blind model cost or gain accuracy; does the
   value head now beat a constant; do features help on rare moves; did play change), then the
   tables, then defects found, then **not verified**.

Final message: the **Answer** paragraph, the branch name and the last commit hash.

## 10. How the reviewer will check this

- Every number in the report is in a file under `docs/briefs/07-artifacts/`.
- The order-blind test perturbs the weights, covers all four permutations, and fails with the
  key off.
- With all new keys off the three digests are unchanged and an old checkpoint loads.
- The holdout is the same 2,000 games; intervals resample games.
- No excluded action-feature column and no opponent truth reaches a policy or value input.
- Nothing in search, PPO or belief code changed; no commit carries an assistant or vendor
  attribution.
