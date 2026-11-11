# Working rules for the FOG project

For people and coding agents. Nothing outside this repository is needed to continue.

1. Read [ROADMAP.md](ROADMAP.md). It lists the phases, the next task, and the gates.
2. Read [docs/results-ledger.md](docs/results-ledger.md). It says what has actually been measured.
3. Before changing search or belief code, read
   [docs/search-and-belief-design.md](docs/search-and-belief-design.md).

## Repository conventions

- Do not put assistant or vendor product names in files, commits, branch names or pull requests.
- Use neutral branch names such as `work/<topic>`. Do not trigger hosted workflows.
- Push to the branch the session was given. Open a pull request only when the owner asks.

## Choosing work

- Take the first unchecked task in the earliest phase whose prerequisites are met. P1 and P2 are
  independent; everything else is ordered.
- Do not start a phase whose gate has not passed. If a gate fails, do what its "if it fails"
  line says. Do not invent a new architecture, loss, solver or simulator in response.
- Section 7 of the roadmap lists frozen areas. Keep them working; do not extend them.
- New ideas go in the roadmap's parking lot, not into code.
- Finish a task before starting the next: code, tests, measurement, ledger row, roadmap tick.

## Rules of evidence

- Strength claims come from matches on pinned Showdown with held-out teams, in matched blocks
  (teams and sides crossed), run to a sequential-test decision or at least 1,000 games.
  Fewer games is a smoke test and must be labelled as one.
- Compare searchers at equal wall-clock per move. Record seconds per decision.
- A result exists when it is a row in the ledger with its command, `fog` commit, game count and
  interval. Append rows; never edit or delete old ones. Record failures and negative results
  the same way.
- Training loss, imitation accuracy and engine-only win rates are diagnostics, not gates.
- Report what happened. An interrupted, partial or skipped run is recorded as such.
- Do not lower a gate threshold to pass it. A threshold changes only with a ledger entry that
  explains the new evidence, and never in the same session that would benefit from it.

## Engineering

- Setup is in `README.md`: `uv sync --locked --extra dev`, then build Showdown.
  Run `uv run --no-sync python -m pytest -q <files>` for the tests that cover the change; the
  full suite is slow and partly needs the Showdown build.
- Optimisations must be behaviour-preserving and say how that was checked (identical arrays on
  recorded inputs, or an equivalence test).
- Any long job must be resumable from its last checkpoint.
- Keep information isolation intact: see the checklist at the end of the design document.
- One coherent change per commit, with a message that says what changed and why.
- Do not add audit, sealing or provenance layers. The ledger and the commit hash are enough.

## When stuck

- A sandbox cannot build Showdown (`npm ci` blocked or no network): develop on the engine
  backend, and record plainly that the Showdown gate is still open.
- A task needs an owner decision (roadmap section 8): stop that task, note it in the ledger,
  and continue with the next independent task.
- A task in the roadmap turns out to be wrong or impossible as written: record why in the
  ledger, make the smallest change to the roadmap that restores a clear next step, and keep the
  gate thresholds.
- Unsure whether something is in scope: it is not. Put it in the parking lot.
