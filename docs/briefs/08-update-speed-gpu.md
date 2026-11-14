# Brief 08: is the PPO update fast on a GPU now?

You are a coding agent working alone on the `fog` repository. Nobody will watch you work. A
reviewer reads what you push and decides the next step from it.

## 1. Why

Brief 06 measured 6.08 s per 1,024-row optimizer step on a T4 and 37.5 decisions per second
end to end, with 87% of optimization time in the forward pass. A per-row Python loop in
`BeliefModel._permute_hidden_moves` has since been replaced by one vectorised pass (ledger,
"per-row Python removed from the belief training path"). On the CPU that saved 8% of an
update. On a GPU it should save much more, but that has not been measured. This brief measures
it and says what limits the update next.

## 2. Rules

- Branch **`work/update-speed`** from `work/roadmap`. Push after every stage.
- Commits are authored as the owner (`git config user.name "Isaac Pan"`,
  `git config user.email "154495180+IsaacJYPan@users.noreply.github.com"`); no assistant or
  vendor names, co-author lines or session links anywhere; no pull requests.
- Read `AGENTS.md` first. Reuse brief 06's job scripts in `docs/briefs/06-artifacts/`.
- **GPU budget: 1 GPU-hour in total**, one job at a time on the account. Prefer a runtime with
  four CPU cores; record the CPU model and core count either way.
- This repository is public: commit no account name or token.
- **Change no code** in the model, losses or trainer. This brief only measures.
- Every number is labelled smoke. Never fill in a number you did not measure.

## 3. Stage 1: the same short run

Run brief 06's short training again at the current `work/roadmap` commit: the same command and
`configs/collect_gpu_check.yaml` (set `env.rollout_workers` to the core count), stopped after
two complete updates. Report next to brief 06's numbers: seconds per optimizer step, processed
rows per second, collection decisions per second, end-to-end decisions per second, the three
wall-time shares, and peak GPU memory.

## 4. Stage 2: where the update's time goes on the GPU

During the second update, time one 1,024-row minibatch by part with
`torch.cuda.synchronize()` before and after each part: the shared trunk, the opponent belief
model, the self-belief model, policy and value heads, loss assembly, backward, optimizer step.
Do it from a small script that loads the saved checkpoint and one saved batch, or with a
timing shim like brief 06's `training_timing.py`; do not edit the trainer. Also take three
`py-spy dump` stacks of the main process during an update and keep them.

## 5. What to hand back

1. One new section in `docs/results-ledger.md`
   (`## <date>: PPO update speed on a GPU after the belief-loop fix (smoke)`).
2. `docs/briefs/08-artifacts/`: metrics CSV, the per-part timing JSON, the stacks, the hardware
   record. Text and JSON only.
3. `docs/briefs/08-report.md`, at most one page: **Answer** (step time before and after; end
   to end before and after; the largest remaining part of the update and its share; whether
   collection or the update now takes more of the hour), the tables, not verified.

Final message: the **Answer** paragraph, the branch name and the last commit hash.

## 6. How the reviewer will check this

- Every number is in a file under `docs/briefs/08-artifacts/`; GPU-hours are 1 or less.
- No code under `fog/` changed; no account name or token is in the repository.
- No commit carries an assistant or vendor attribution.
