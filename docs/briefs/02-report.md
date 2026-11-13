# Brief 02: replay scale preparation

Continuation in progress from reviewed `work/roadmap` commit `4f9d75c`, on
`work/p2-scale-prep`. The pinned dataset metadata endpoint now returns HTTP 200;
the corpus import is running. The fixed holdout IDs are already tracked by that
review commit. A fresh locked install and pinned Showdown build succeeded; the
82 focused tests passed in 20.26 s, and Ruff is clean. This machine has four CPU
cores of quota, 16 GiB memory and no CUDA device. See `continuation-commands.txt`,
`continuation-environment.json`, `continuation-tests.txt` and `continuation-ruff.txt`
in `02-artifacts/`. The prior attempt's report below is retained until these
measurements finish; its access/holdout blockers describe that earlier attempt.

## Answer

Training speed and its bottleneck on the pinned replay corpus remain unmeasured because
this environment refuses the dataset host and the previous shards are absent. The brief 01
gain still cannot be separated into data and training length because R, A, S and B are
blocked. Resume, policy-loss selection, switch diagnostics and the restartable runner are
implemented: 82 focused checks passed, including a real subprocess interruption test, and
a generated 600-game pipeline recovered after termination. The scale command still needs
the exact fixed holdout IDs and real-data performance qualification; it has not run on a
hosted GPU.

## Stage 3 table

| Run | Training games | Epochs requested | Steps / wall / rows/s | Last / best holdout NLL and top-1 | Final train NLL | Value MSE best / last | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R | 1,000 | 6 | unmeasured | unmeasured | unmeasured | unmeasured | blocked: corpus unavailable |
| A | 1,000 | 18 | unmeasured | unmeasured | unmeasured | unmeasured | blocked: corpus unavailable |
| S | 1,000 | 6 | unmeasured | unmeasured | unmeasured | unmeasured | blocked: corpus unavailable |
| B | 3,000 | 18 | unmeasured | unmeasured | unmeasured | unmeasured | blocked: corpus unavailable |

The runs cannot distinguish data from training length, estimate seed noise, or update the
value/switch reading. R must first reproduce brief 01 to three decimals; then A versus
brief 01's 3,000-game run separates data at approximately equal steps, and S quantifies
run-to-run variation. There is no measured runtime projection for B and no new play result.

## Per stage

- **Setup: done except data.** Existing checkout at `f0dc3d1`; branched through
  `work/roadmap` to `work/p2-scale-prep`. Pinned Showdown submodule built. Locked dependency
  sync succeeded after moving caches to writable directories. Environment: 4 CPU cores of
  cgroup quota, 16 GiB memory limit, Python 3.12.14, PyTorch 2.13.0+cu130, no CUDA device,
  PyTorch intra/inter-op threads 5/5. See `02-artifacts/setup-*.txt` and `environment.txt`.
- **Stage 1: partial.** The import command below failed with `Tunnel connection failed:
  403 Forbidden`; `stage0-import-output.txt` retains the traceback and `dataset-access.txt`
  retains the HTTP probe. No replay splits exist. The new
  `fog.scripts.profile_replay_training` command measures 200 CPU steps after 20 warm-ups,
  batch loading alone, model forward, belief-head share, backward/optimizer, remainder,
  and a separate cumulative top-15 cProfile. Its real-corpus invocation failed on the
  absent manifest (`stage1-profile-output.txt`). No loader optimisation, prefetch or
  thread-default change is justified by measurements. Four profiler tests passed in 3.66 s
  (`stage1-tests.txt`), including identical first 20 losses and final parameters against
  an independent CPU update loop with dropout. This made-up-data check is not the brief's
  real-data acceptance check or a throughput measurement.
- **Stage 2: done on made-up inputs.** `bc.resume=true` writes atomic epoch state, restores
  model/optimizer/schedule/scaler/all RNG streams/metric history/best model/patience, and
  rejects changed manifests or training settings. Its four-epoch interruption test has
  exactly equal numeric CSV fields and state after resume; the measured timing columns
  are excluded because real elapsed time differs. Default `bc.selection_metric=loss`
  preserves selection; `action_ce` selects/stops on policy NLL. New switch diagnostics
  retain old report fields. Metrics add `seconds_per_epoch` (train+eval, excludes file writes)
  and `train_rows_per_second` (train only). Optional `bc.torch_threads` leaves the runtime
  setting untouched unless supplied. Trainer+scope tests: 44 passed, 3.70 s
  (`stage2-trainer-tests.txt`); switch/split tests in `stage2-report-split-tests.txt`;
  Ruff in `stage2-ruff.txt`. Implementation wall-clock was not timed.
  A further full-recipe check on the generated fixture runs the original `f0dc3d1` trainer
  and the new trainer with all new options absent: all 24 losses (including the first 20)
  and every final model tensor match exactly (`stage2-default-equivalence.json`, command
  `uv run --no-sync python docs/briefs/02-artifacts/check_default_updates.py`). This is
  supplemental default-behaviour evidence, not run R.
- **Stage 3: not run.** Dataset access blocks every run; no losses or runtimes invented.
- **Stage 4: partial.** Fixed-ID splitter, restartable runner and
  `docs/replay-blueprint-scale-run.md` are implemented. Exact replay IDs remain blocked;
  the runner fails clearly before import when the file is absent. Resolved recipe/holdout
  changes are refused, existing import revision/filters are checked, completed work is
  skipped, and runner-owned shard paths survive moving the work directory. The genuine
  600-game import smoke is blocked. A **generated-data** 600-game smoke at code commit
  `ddd83638f8068a85c740eb90029e8201cd98279c` actually ran import→convert→split→full-model
  training→scoring: 2,400 rows, train 500 games/2,000 rows, holdout 100/400. SIGTERM after
  epoch one exited 143 (16.890 s); the identical command resumed epoch two and exited 0
  (15.349 s), with CSV epochs `[0,1]` then `[0,1,2]`. Outputs:
  `runner-smoke/interruption-test.json` and `runner-smoke/results/` without checkpoint.
  Step walls in `run-summary.txt`: import 0.210 s, convert 3.232 s, split 1.067 s,
  training attempts combined 24.157 s, best/untrained scoring 1.620/1.656 s.
  These short duplicate protocols do not measure real replay throughput or learning.
- **Stage 5: not run.** No stage 3 checkpoint exists to evaluate.

Exact blocked import command, at `f0dc3d1` plus the profiler changes:

```sh
UV_CACHE_DIR=/tmp/fog-uv-cache uv run --no-sync python -m fog.scripts.import_metamon_replays \
  --out data/metamon_gen9ou_r1500 --revision 792a86b5cdbcab9696032e947483b218453a0d4f \
  --min-rating 1500 --after 2024-02-01 --max-games 22000 --max-shards 4 --max-download-mb 600
UV_CACHE_DIR=/tmp/fog-uv-cache uv run --no-sync python -m fog.scripts.profile_replay_training \
  --out runs/blueprint/profile-before
```

## Deviations and corrections

The provided checkout had a local `work` branch and fetched only `main`; renamed that clean
local branch to `environment-base` and fetched an explicit remote-tracking ref before
creating the requested branches. No owner question was needed. Dataset access refusal was
recorded, with no alternate host or access workaround. The first dependency installation
failed on a read-only home cache; retrying with writable XDG/Rust/uv caches succeeded.
Likewise npm needed `npm_config_cache=/tmp/fog-npm-cache`.
The runner exposes `--parquet` and `--config` for offline testing. The generated source uses
the importer's required revision argument only to exercise its interface; it did not come
from that revision. Reproduce the fixture with `runner-fixture-setup.py` then
`runner-fixture-interrupt.py` in `02-artifacts/` from the repository root. CPU fallback
instructions import 22k games to retain the fixed 2k holdout and train on 20k; importing
only 20k could omit fixed IDs. Historical estimates are explicitly separated from missing
stage 1 projections. All validation commands are in `02-artifacts/test-commands.txt`:
81 tests passed in 15.40 s plus the final timing test in 1.63 s, Ruff clean. The pytest
warning about `cache_dir` follows the requested disabled cache provider; no full suite ran.

## Not verified

The 22,000-game split, real-data profiles/equality, R reproduction, A/S/B, exact fixed holdout
IDs, real 600-game runner smoke and play strength remain blocked. Hosted GPU/CUDA speed,
memory, and determinism are not verified. No prefetch or thread tuning was made without
the stage 1 measurements. No roadmap gate is claimed.

## Checkpoints

No checkpoint trained on the pinned replay corpus exists. The generated fixture's
`runs/blueprint_runner_fixture_final/training/checkpoints/bc_best.pt` is 23,548,503 bytes,
SHA-256 `67c31faec47317b2e0a6900cd2bf3179057779f5cca419911a553752c02b6859`.
`02-artifacts/checkpoints.json` also lists the final weights and resumable state with
their paths, byte sizes and SHA-256 values. They are engineering fixtures, not research
models, and are not pushed. This machine may be reclaimed; the branch retains text
evidence and reproducible fixture scripts, not ignored checkpoints.

## Suggestions

Restore access to the pinned corpus, recover the fixed IDs, then finish stage 1 and R/A/S/B.
Do not infer a data effect or change the loss/model from these blocked experiments.
