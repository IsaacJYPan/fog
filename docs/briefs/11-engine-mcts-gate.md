# Brief 11: run gate G1 for the search-only agent

You are a coding agent working alone on the `fog` repository. Nobody will watch you work. A
reviewer reads what you push and decides the next step from it.

## 1. Why

EngineMCTS (`fog/search/engine_mcts.py`) is the strongest agent in the repository: at a
quarter of its default budget it scored 0.88 against SimpleHeuristics and 0.99 against
MaxDamage in 100-game smokes (ledger, 2026-11-11 and 2026-11-12). It has never been run to
its gate because there was no held-out team pool. Brief 10 provides one
(`data/teams/replay_v1`). Passing G1 gives the project its first gated result, the anchor `A4`
that every later model is compared with, and the teacher for distillation (P3).

**Start only when `data/teams/replay_v1/validation` exists on `work/roadmap`.**

## 2. Rules

- Branch **`work/g1`** from `work/roadmap`. Push after every stage.
- Commits are authored as the owner (`git config user.name "Isaac Pan"`,
  `git config user.email "154495180+IsaacJYPan@users.noreply.github.com"`); no assistant or
  vendor names, co-author lines or session links anywhere; no pull requests.
- Read `AGENTS.md`, then `docs/search-and-belief-design.md`. CPU only. Setup is in
  `docs/briefs/02-p2-scale-prep.md` section 4 (no replay data is needed).
- Use the validation split only. Never the test split.
- Do not change a gate threshold. Do not add a search variant. Fixes are limited to state
  reconstruction and action mapping, each with a test.
- Search is single-threaded per decision; you may run one evaluation shard per CPU core.
  Record seconds per decision from the shards as they ran.

## 3. Stage 1: fidelity on the new teams (P1.3)

Run `fog/scripts/fuzz_recon_showdown.py` and `fog/scripts/fuzz_recon.py` on the validation
teams. The new pool has species, items and abilities the 16-team pool never exercised. Fix
the most frequent reconstruction differences, each with a test, until a 200-game smoke at the
default budget has fallbacks on under 5% of decisions and no rejected action. Record what was
found and what was left.

## 4. Stage 2: budget curve (P1.4)

Win rate against SimpleHeuristics for `budget_ms` in {200, 800, 3200} and `K` in {1, 8, 32},
400 games per cell. This is about 60 core-hours if every cell is run; run the cells in this
order and stop when 40 core-hours are spent, reporting the rest as not run: (800, 8),
(200, 8), (3200, 8), (800, 1), (800, 32), (200, 1), (200, 32), (3200, 1), (3200, 32).

## 5. Stage 3: the gate

At the default budget (`K = 8`, `budget_ms = 1600`), with `--sequential` off: at least 1,000
matched games against SimpleHeuristics and at least 1,000 against MaxDamage on the validation
split. Gate G1: at least 85% and at least 95%, fallbacks on under 5% of decisions, zero
rejected actions.

If it fails, do what the roadmap's "if it fails" line under G1 says and nothing else: inspect
losses with `fog/eval/search_trace.py`, compare engine and Showdown outcomes with
`fog/eval/cross_sim.py` and `fog/eval/chance_fidelity.py`, fix the largest mechanical
difference, and re-run. Report every attempt, not only the last.

## 6. What to hand back

1. Fixes with tests; `ruff check fog tests` clean.
2. One ledger section (`## <date>: EngineMCTS fidelity, budget curve and gate G1`) with the
   printed rows, commands and commits.
3. `ROADMAP.md`: tick P1.3 and P1.4 if done; write the G1 result under the gate.
4. `docs/briefs/11-artifacts/` (evaluation JSONs, fuzzer summaries) and
   `docs/briefs/11-report.md`, at most one page: **Answer** (G1 passed or not, with the two win
   rates, intervals, fallback rate and seconds per decision; the best cell of the budget
   curve), then defects fixed, then not verified.

Final message: the **Answer** paragraph, the branch name and the last commit hash.

## 7. How the reviewer will check this

- Gate rows have at least 1,000 games each, validation teams, default budget, no sequential
  stop, and match the JSON files.
- No threshold changed; no new search variant; every fix has a test.
