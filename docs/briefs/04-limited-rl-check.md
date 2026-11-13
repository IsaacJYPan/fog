# Brief 04: can this model learn from play at all? A limited test on headless Showdown

You are a coding agent working alone on the `fog` repository. Nobody will watch you work. A
reviewer reads what you push and decides the next step from it.

## 1. Why

Nobody has ever seen this model get better by playing. The earlier self-play runs trained for
240,000 decisions at `lr 1e-5`, `clip 0.05`, one epoch, and the policy did not move (ROADMAP
section 2). The replay work since then shows the policy head learns from human games, and the
value head barely does (`docs/briefs/01-report.md`, `docs/briefs/03-report.md`).

Before more is built for self-play at scale, the owner wants the smallest test that answers:

1. Can this model and this training loop learn from play when the problem is made small: one
   team, one fixed opponent?
2. Does play improve the policy trained on human games, on a small set of teams?
3. How many decisions does a visible gain cost, and where does the time go?

ROADMAP section 7 freezes PPO self-play until gate G3. **The owner asked for this test on
2026-11-13 as a diagnostic.** It un-freezes nothing: no checkpoint from it is a candidate for
anything, no gate is claimed, and nothing is tuned beyond the settings listed here.

Battles already run headless: `env.backend: showdown_direct` drives pinned Showdown in a local
Node process with no server and no web socket (`fog/env/showdown_direct.py`,
`fog/env/showdown_worker.js`). For scale: pinned Showdown driven directly with random choices
runs a few thousand player decisions per second on one core; FOG's rollouts ran at 87 to 194
decisions per second when last measured (ledger, 2026-11-11), most of it FOG's own observation
building and forward pass. Part of this brief is to say where the time goes during training.

## 2. Rules

- Branch **`work/rl-check`** from `work/roadmap`. Push after every stage.
- Commits are authored as the owner (`git config user.name "Isaac Pan"`,
  `git config user.email "154495180+IsaacJYPan@users.noreply.github.com"`); no assistant or
  vendor names, co-author lines or session links anywhere; no pull requests.
- Use the existing trainer: `python -m fog.train --config <file> --trainer ppo --set key=value ...`
  (`fog/training/ppo.py`). Do not change the model, the losses or the trainer's algorithm. Fix a
  bug only if it stops the test, with a unit test, and say so.
- Training happens on `showdown_direct`. All evaluation is on pinned Showdown with
  `fog.eval.showdown_direct_selection`.
- One training job at a time. Use `ppo.resume_ckpt` when a job is cut off, and record each
  restart.
- Every number is labelled smoke. Report what happened; never fill in a number you did not
  measure.
- Read `AGENTS.md` first. Setup is in `docs/briefs/02-p2-scale-prep.md` section 4, with the
  branch name above.

## 3. Stage 0: does the training path run, and how fast

Make two configs under `configs/` that extend `configs/gen9ou_typed_token_selfplay.yaml`
(`rl_check_scratch.yaml`, `rl_check_blueprint.yaml`). That config carries a preflight
requirement, a replay-policy term and a starting checkpoint that will not exist on your
machine. Turn each off in the new configs by the means the config schema offers, and list in the
report exactly which keys you changed and why. Team pools here are the folders
`data/teams/pool_a` (8 teams sharing six species) and `data/teams/pool_b` (16 teams); find out
what form `env.team_pool` accepts and convert if needed.

Run 20,000 decisions against MaxDamage with four rollout workers and record:

- decisions per second for collection alone, for the update alone (rows per second and epochs),
  and end to end;
- the share of collection time spent in the Node process, in observation building and in the
  forward pass (the cProfile command is in the ledger, 2026-11-11);
- whether belief heads are computed while acting (they should not be).

If end-to-end speed is under 40 decisions per second, spend at most one hour on the cause, then
continue with what you have and scale the budgets below in proportion, saying so.

## 4. Stage 1: the small problem that must be learnable

- Start: **random weights** (no starting checkpoint).
- Teams: one team for both sides, the first file in `data/teams/pool_a`.
- Opponent: MaxDamage only (`league.scripted_frac: 1.0`, no latest or past self).
- Settings: `ppo.lr: 3.0e-4`, `ppo.clip: 0.2`, `ppo.epochs: 3`, `ppo.target_kl: 0.03`,
  `ppo.batch_decisions: 8000`, `ppo.minibatch_size: 1024`, entropy coefficient `0.01` falling
  to `0.003`. Everything else as inherited.
- Budget: 600,000 decisions.
- Before training and after every 100,000 decisions: 200 games against MaxDamage with the same
  team, greedy play, and the same against SimpleHeuristics.
- From the trainer's own log at every update: policy entropy, measured KL, clip fraction, value
  loss and **value explained variance**, mean episode return against the opponent.

**Reading.** Learned: the win rate against MaxDamage rises by at least 30 points over the
untrained start, with 95% intervals that do not overlap. Not learned: anything else.

If not learned, do not go on to stage 2. Run this grid instead, 300,000 decisions each, same
evaluation at the end only: `lr` in {1e-4, 1e-3} × `epochs` in {1, 3}. Then report the curves of
all five quantities above for every run. Which of them moved and which stayed flat is the
finding: for example, value explained variance that never leaves zero points at the value head
or its inputs, while entropy that never falls points at the policy update.

## 5. Stage 2: does play improve the human-data policy

Only if stage 1 learned.

- Start: the replay blueprint. Check access first
  (`curl -sS -o /dev/null -w "%{http_code}\n" -m 30 https://huggingface.co/api/datasets/jakegrigsby/metamon-raw-replays`
  must print `200`). If it does, build the 1,000-game blueprint as brief 03's run A did and
  take its `bc_best.pt` (about 2.5 hours on four cores; commands in
  `docs/briefs/03-p2-scale-measure.md` section 3 and its stage 2 table, run A, stopped after
  epoch 8). If the host is not reachable, skip stage 2 and say so; do not substitute another
  starting point.
- Teams: the 8 teams of `data/teams/pool_a`.
- Opponents: SimpleHeuristics 50%, MaxDamage 10%, latest self 40%.
- Settings: as stage 1 but `ppo.lr: 1.0e-4`.
- Budget: 1,000,000 decisions, or what fits in 10 hours.
- Before training and after every 200,000 decisions: 400 games against SimpleHeuristics and 400
  against MaxDamage on `pool_a`, and the same on `pool_b` (teams never trained on).

**Reading.** Improved: the win rate against SimpleHeuristics on `pool_a` rises by at least 10
points over the blueprint with non-overlapping 95% intervals. Report the `pool_b` curve beside
it: a rise on `pool_a` with none on `pool_b` is learning the eight teams, not the game.

## 6. Stage 3: what it costs

From the runs you made, no new runs:

1. Decisions per 10 points of win rate, for each stage that learned.
2. The time split of a training hour: collection (Node, observation building, forward pass,
   the rest) against the update.
3. The end-to-end decisions per second, and what it would be if each part alone were free.
   That says which part a faster run would have to fix first.

## 7. Order if time is short

Stage 0 → stage 1 → (grid, or stage 2) → stage 3. Stage 1 with its reading is the result the
owner needs most.

## 8. What to hand back

On the pushed branch:

1. The two configs, any fix with its test.
2. One new section in `docs/results-ledger.md`
   (`## <date>: limited learning-from-play check (diagnostic, smoke)`): one row per evaluation
   point with command and commit. Do not edit older sections.
3. `docs/briefs/04-artifacts/`: per run the trainer's metrics as CSV, the last 60 log lines,
   every evaluation JSON and the profile output. Text and JSON only; no checkpoints.
4. `docs/briefs/04-report.md`, at most two pages: **Answer** (did the small problem get
   learned; did play improve the blueprint; decisions per 10 points; where the time goes), one
   table per stage, the keys changed in the configs, deviations, not verified.

Final message: the **Answer** paragraph, the branch name and the last commit hash.

## 9. How the reviewer will check this

- Every win rate in the report is in an evaluation JSON under `docs/briefs/04-artifacts/`.
- Evaluation used pinned Showdown and greedy play, with the game counts above.
- The reading follows the rule stated in this brief, not a looser one.
- No model, loss or algorithm change; config changes are listed.
- Rows are labelled smoke; no gate is claimed; lint is clean; no commit carries an assistant or
  vendor attribution.
