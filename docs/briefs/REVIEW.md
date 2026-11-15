# How to review a finished brief

For a coding agent or a person acting as reviewer. A worker has pushed a branch and says a
brief is finished or stopped. Your job is to decide what is true, merge it, and say what
happens next. Budget: about an hour. You do not redo the worker's long runs.

Prompt to start a review:

> In IsaacJYPan/fog, check out work/roadmap, read AGENTS.md and docs/briefs/REVIEW.md, then
> review branch `<branch>` for brief `<NN>`.

## Rules

- Commits are authored as the owner (`git config user.name "Isaac Pan"`,
  `git config user.email "154495180+IsaacJYPan@users.noreply.github.com"`); no assistant or
  vendor names, co-author lines or session links anywhere; no pull requests; do not trigger
  hosted workflows. Work on `work/roadmap`.
- The reviewer changes no application code. If a fix is needed, it is a new brief or a line
  in the queue.
- Report what you checked and what you did not. "Not re-run" is a normal thing to write.

## Steps

1. **Fetch and read.** `git fetch origin <branch>`; read `docs/briefs/<NN>-report.md` on the
   branch, then the brief itself, then `git diff --stat origin/work/roadmap...origin/<branch>`.
2. **Check every number.** Each number in the report must be in a file under
   `docs/briefs/<NN>-artifacts/`. Open the files and compare, with a short script when there
   are many. A number with no file behind it is reported as unsupported, not repeated.
3. **Read the code diff** outside `docs/`, line by line. Check against the brief's rules:
   new behaviour behind keys that default to off; nothing changed that the brief said not to
   touch; no opponent truth in a policy or value input (design document, section 7).
4. **Check the repository rules.** Every commit authored as the owner; no attribution lines;
   no account name, token, private dataset name or player name in any file; text and JSON
   artifacts only.
5. **Merge locally** (`git merge --no-ff --no-commit origin/<branch>`). When
   `docs/results-ledger.md` conflicts, keep both blocks, older first.
6. **Run** on the merged tree: the test files that cover the change
   (`uv run --no-sync python -m pytest -q <files>`), `.venv/bin/ruff check fog tests`, and, if
   anything under `fog/encoding`, `fog/env` or `fog/models` changed, the three rollout digests
   from `HANDOFF.md` **before and after the merge on your own machine**. The digests depend on
   the CPU model, so only before-and-after on one machine means anything.
7. **Write the review** as a new ledger section, `## <date>: review of brief <NN>`: what was
   checked, what was not re-run, a table of findings with their evidence, and what follows for
   other briefs. Label everything below 1,000 matched games as smoke.
8. **Roadmap.** Tick a task only when its acceptance text is met. Write a gate result only
   from rows that meet the gate's own conditions. Never change a threshold.
9. **Queue.** Update the status table in `docs/briefs/QUEUE.md`, and apply any "the review
   decides" line there (for example which model keys brief 13 uses).
10. **Commit the merge, push `work/roadmap`,** and end with: what is true now, the next prompt
    from the queue, and any decision the owner must make.

## When the work is partial or wrong

- Partial: merge what is finished and proven, say which stages remain, and add a line to the
  queue (a short follow-up brief that points at the unfinished sections is enough; brief 09 is
  an example).
- A number does not match its file, or a rule was broken: do not merge. Write the ledger
  section anyway, saying what failed, and put the brief back in the queue.
- The result contradicts an earlier ledger row: keep both, say which you believe and why.
