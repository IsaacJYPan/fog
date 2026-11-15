# Brief 14: make the PPO update cheaper, and fix two things it gets wrong

You are a coding agent working alone on the `fog` repository. Nobody will watch you work. A
reviewer reads what you push and decides the next step from it.

## 1. Why

Self-play is frozen until gate G3 (ROADMAP section 7), but it has to be ready and affordable
when the gate opens. Brief 06 measured 37.5 decisions per second end to end on a T4, with the
update taking 72% of the time. One loop has been removed since; brief 08 measures what is
left. **Start only if the review of brief 08 says the update still takes more than half of
the wall time**, or the owner asks for it.

Known from reading the code and one CPU profile (ledger, 2026-11-14):

- The two belief models are about 60% of the update's forward pass on the CPU. They are
  auxiliary: the policy does not read them when acting. Counting what only they need (the
  opponent-view pass through the history encoder and the trunk, which feeds the self-belief
  model), **about 80% of the update's forward is spent on beliefs and 20% on policy and
  value** (ledger, "the policy-only training step", from the profile of 2026-11-14: belief
  passes 12.1 s, two trunk passes 3.7 s and two history passes 3.7 s of a 20.3 s forward,
  heads 0.3 s).
- `fog/training/ppo.py` does not use mixed precision; `fog/training/bc.py` does.
- When one minibatch exceeds `target_kl`, the update stops altogether, so the value head
  stops training too (1.39 of 3 epochs in brief 04).
- How a game cut off at `max_turns` is scored in the update has never been checked.

## 2. Rules

- Branch **`work/ppo-update`** from `work/roadmap`. Push after every stage.
- Commits are authored as the owner (`git config user.name "Isaac Pan"`,
  `git config user.email "154495180+IsaacJYPan@users.noreply.github.com"`); no assistant or
  vendor names, co-author lines or session links anywhere; no pull requests.
- Read `AGENTS.md` first. **GPU budget: 2 GPU-hours.**
- Every change is behind a config key that defaults to today's behaviour. With the keys off
  the update is bit-identical on a recorded batch; show it.
- Do not change the model, the collection path or the meaning of any loss.
- Every number is labelled smoke.

## 3. Stages

1. **Truncated games.** Read how `truncated=True` games from
   `fog/env/showdown_batch_runner.py` reach `ppo_update`: what return and advantage their
   last steps get. Write a test that pins the behaviour down. If a cut-off game is scored as
   a loss or a win, report it and add a key that scores it as a draw; do not change the default.
2. **`ppo.belief_row_fraction`** (default 1.0): compute the belief and self-belief losses on a
   random fraction of each minibatch's rows and scale the loss so its expectation is the same.
   At 0, skip the belief passes and the opponent-view pass altogether (`need_belief=False`,
   as `loss.skip_unweighted_belief` does in `fog/training/bc.py`) and guard the belief
   diagnostics the update prints.
3. **`ppo.mixed_precision`** (default false): autocast and a gradient scaler in the update,
   as `bc.py` does it.
4. **`ppo.value_after_kl_stop`** (default false): after the KL stop, the remaining
   minibatches of that update still train the value head (and nothing else).
5. **Measure.** Brief 06's short run (two complete updates, `configs/collect_gpu_check.yaml`)
   on a T4: all keys off; fraction 0.25; fraction 0; fraction 0 with mixed precision. Report seconds
   per optimizer step, rows per second, end-to-end decisions per second and peak memory.
   Then on the CPU, brief 04's fixed-team run for 100,000 decisions with keys off and with
   fraction 0.25 and `value_after_kl_stop`: the win-rate curve against MaxDamage must not be
   worse than brief 04's (0.160 to 1.000 by 106,889 decisions) beyond its interval.

## 4. What to hand back

1. Code, tests, `ruff check fog tests` clean, the bit-identity check with keys off.
2. One ledger section (`## <date>: PPO update levers (smoke)`).
3. `docs/briefs/14-artifacts/` and `docs/briefs/14-report.md`, at most one page: **Answer**
   (how truncated games are scored; step time and end-to-end rate for each setting; whether
   learning held up), then not verified.

Final message: the **Answer** paragraph, the branch name and the last commit hash.

## 5. How the reviewer will check this

- Keys off: bit-identical update on a recorded batch.
- The scaled belief loss has the same expectation (a test averages many draws).
- GPU-hours are 2 or less; no account name or token is in the repository.
