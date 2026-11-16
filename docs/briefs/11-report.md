# Brief 11: EngineMCTS gate G1

**Completed in review, 2026-11-16.** The worker's session ended with the MaxDamage half of
the gate at 504 of 1,008 games. The reviewer re-ran the recorded gate command, which resumed
each shard from its last complete block, on a different machine (Intel Xeon 2.10 GHz, four
cores). **G1 passes: 936 of 1,008 against SimpleHeuristics (0.9286 [0.9110, 0.9429]) and 981
of 1,008 against MaxDamage (0.9732 [0.9613, 0.9815]); fallbacks 53 of 28,020 and 41 of 24,112
decisions; no rejected action, failure or truncation.** MaxDamage by machine: 490 of 504 and
491 of 504. Files: `11-artifacts/gate/merged.json`, `gate-complete-summary.json`. The text
below is the worker's report as it stood.

**Answer:** G1 not yet measured; gate matches running. Gate: not measured. Budget curve: best observed complete cell: 800 ms, K=8, 91.750%; intervals overlap, so this does not rank close cells.

**Defects fixed.** Teal-Tera Ogerpon matches its submitted-world slot after re-entry;
active identity, HP and used Tera survive. Pending public charge turns reach the engine,
with Power Herb / immediate animation completion respected. Tests cover both defects.
On the identical 200-game Showdown fuzzer schedule: 14,362 positions, zero exceptions,
1,214 → 1,043 mismatching positions. Opponent HP/Tera/active and missing Phantom Force /
Meteor Beam differences disappear. Counts overlap. Most residual positions are hidden
Regenerator healing (821). Engine round-trip: 3,526 positions, 11 residual core mismatches;
extended last-move, Wish, Future Sight, toxic counts and timers remain approximate.

**Measurements.** Default-budget fidelity smoke: simpleheuristics: 200 games, 93.500% [95% 89.198%, 96.162%], fallbacks 2/5341, rejected 0, 1.776 s/decision.
Budget allocated so far: 40.000 core-hours (40-hour cap).
All runs use validation teams, pinned Showdown `e440c4a`, CPU, one search thread, fixed
matched blocks, no sequential stop. The gate requests 1,008 games per opponent so all four
shards have equal whole-block sizes (252 each); the merger requires equal shard protocols.
Commands, launch commits, completed/partial JSONs, counters and printed rows are in
`11-artifacts/`; actual seconds per decision are pooled from the shards as run.

**Validation / not verified.** 45 focused and 84 integration tests passed; ruff clean.
Source `aa38e46`; no test-split files read, no threshold or search variant changed.
Unfinished stages and not-run cells are explicit in `measurement-progress.json`.
Residual Flash Fire, Unburden, locked moves, screen/item bookkeeping and engine chance
approximations remain. Smoke failure reasons: {'PanicException: Encore should not be active when last used move is not a move': 16, 'ValueError: no world with positive weight and kept visits': 2}. A failed gate requires manual loss/mechanics diagnosis before a retry.

Resume from the repository root: `.venv/bin/python docs/briefs/11-artifacts/continue_measurements.py`.
