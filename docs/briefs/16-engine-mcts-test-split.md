# Brief 16: the search-only agent on the held-out test teams (tier T1)

You are a coding agent working alone on the `fog` repository. Nobody will watch you work. A
reviewer reads what you push and decides the next step from it.

## 1. Why

Gate G1 was measured on the validation teams, which every search fix so far has looked at.
Tier T1 in `ROADMAP.md` section 1 ("clearly beats scripted bots": at least 90% against
SimpleHeuristics and at least 97% against MaxDamage, held-out teams, pinned Showdown, at least
1,000 matched games each) is a claim about teams nobody has tuned on. This brief measures
EngineMCTS once on the 62 test teams of `data/teams/replay_v1/test`. It is a measurement and
nothing else.

**Start only if `docs/briefs/QUEUE.md` says gate G1 passed.**

**2026-11-16 22:40 UTC: the planner is running this measurement itself (results go to
`work/roadmap`). Do not start it again: the test teams are run once.**

## 2. Rules

- Branch **`work/t1-search`** from `work/roadmap`. Push after each opponent finishes.
- Commits are authored as the owner (`git config user.name "Isaac Pan"`,
  `git config user.email "154495180+IsaacJYPan@users.noreply.github.com"`); no assistant or
  vendor names, co-author lines or session links anywhere; no pull requests.
- Read `AGENTS.md`. CPU only. Setup is in `HANDOFF.md` section 9; no replay data is needed.
- **Change no code.** If something breaks, record it and stop.
- **One attempt.** The test split is run once per agent version. Whatever the numbers are,
  they are the result. Do not re-run with another seed, budget or game count, and do not
  inspect individual test-team games to tune anything.
- Nothing else may run on the machine during the matches: the search budget is wall-clock.

## 3. The measurement

Default budget, four shards, 1,008 games an opponent (252 a shard), no sequential stop:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python -m fog.eval.run_shards --shards 4 --games 1008 --seed 16000 \
  --out-dir docs/briefs/16-artifacts/test -- \
  --pool data/teams/replay_v1/test --device cpu \
  --controller mcts=fog.search.engine_mcts:showdown_controller \
  --controller-kwargs 'mcts={"k":8,"budget_ms":1600,"threads":1}' \
  --opponent simpleheuristics --opponent maxdamage --max-turns 800 --progress-every 4
```

Re-running the same command resumes unfinished shards from their last complete block. Brief
11 needed about 3.9 hours for SimpleHeuristics and 3.2 for MaxDamage on four cores. Commit
the shard JSONs and `merged.json` as brief 11 did; the shard `.log` files are git-ignored, so
copy each to a `.txt` beside it.

## 4. What to hand back

1. One ledger section (`## <date>: EngineMCTS on the test teams (tier T1)`) with the two
   printed rows, the command, the commit, the CPU model and the measured seconds per
   decision.
2. `docs/briefs/16-report.md`, half a page: **Answer**: the two win rates with Wilson
   intervals, fallback rate, rejected actions, failures and truncations; then each T1 clause as
   met or not met (a clause is met when the measured rate is at or above its threshold); then
   the validation numbers from brief 11 beside them.
3. `ROADMAP.md`: one dated line under the tier table in section 1 with the result. It must
   say that the teams are the replay-derived pool (usage-sampled sets, see the ledger section
   "review of brief 10") and that the claim is to be repeated on the forum bank's test split
   if that is ever collected.

Final message: the **Answer** paragraph, the branch name and the last commit hash.

## 5. How the reviewer will check this

- One run, the command above, 1,008 games an opponent, test teams, no code change.
- The rows match `merged.json`.
