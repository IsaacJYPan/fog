# Brief 11: EngineMCTS gate G1

**Answer:** G1 not yet measured; budget curve running. Gate: not measured. Budget curve: best observed complete cell: 800 ms, K=8, 91.750%; intervals overlap, so this does not rank close cells.

**Defects fixed.** Teal-Tera Ogerpon matches its submitted-world slot after re-entry;
active identity, HP and used Tera survive. Pending public charge turns reach the engine,
with Power Herb / immediate animation completion respected. Tests cover both defects.
On the identical 200-game Showdown fuzzer schedule: 14,362 positions, zero exceptions,
1,214 → 1,043 mismatching positions. Opponent HP/Tera/active and missing Phantom Force /
Meteor Beam differences disappear. Counts overlap. Most residual positions are hidden
Regenerator healing (821). Engine round-trip: 3,526 positions, 11 residual core mismatches;
extended last-move, Wish, Future Sight, toxic counts and timers remain approximate.

**Measurements.** Default-budget fidelity smoke: simpleheuristics: 200 games, 93.500% [95% 89.198%, 96.162%], fallbacks 2/5341, rejected 0, 1.776 s/decision.
Budget allocated so far: 15.460 core-hours (40-hour cap).
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
