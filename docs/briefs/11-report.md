# Brief 11: EngineMCTS gate G1

**Answer:** G1 is not yet measured. The default-budget validation fidelity smoke is running;
no gate win rate, interval, fallback rate or seconds per decision is claimed yet. The budget
curve has not started, so no best cell exists. Source: `aa38e46`, branch `work/g1`.

**Defects fixed.** Teal-Tera Ogerpon now matches its original world slot after re-entry,
preserving active identity, HP and used Tera. Public pending charge turns now reach the engine;
Power Herb and immediate animations clear them. Both have regression tests. The identical
200-game Showdown fuzzer schedule has 14,362 positions and zero reconstruction exceptions;
any mismatch fell from 1,214 to 1,043. The removed fields include opponent HP (96), Tera (56),
active identity (10), Phantom Force (52) and Meteor Beam (34). Counts overlap by position.

**Validation.** 45 focused and 84 affected integration tests pass; `ruff check fog tests` is
clean. The engine fuzzer checked 3,526 positions: 11 core mismatches (opponent missing mon,
three also wrong active), with the larger documented extended-field tail retained. Artifacts
contain exact commands, pinned Showdown `e440c4a`, environment and before/after summaries.

**Not verified.** Stage 1 smoke, all budget cells and both 1,000-game gate rows remain open.
The test split has not been read. Remaining Showdown differences include hidden Regenerator
healing (821 positions), unavailable-move flags during replacement (53), Metronome item
bookkeeping, screen durations, Flash Fire, Unburden and locked moves. Engine last-move,
Future Sight, Wish and toxic counters remain approximate. No threshold or searcher changed.

Resume the smoke with the `fidelity_smoke` command in `11-artifacts/commands.json`; the existing
shard runner skips completed shards and resumes unfinished ones from complete four-game blocks.
