# Brief 02: replay scale preparation

## Answer

Training throughput on the pinned replay corpus is unmeasured: this environment refuses
the dataset host, and the previous shards are absent. The brief 01 gain still cannot be
separated into data and training length because R, A, S and B need that corpus. Trainer and
runner implementation is in progress; the scale run is not yet ready to launch, and no
hosted GPU execution has been verified.

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
- **Stage 2: in progress.** Resume, metric selection, epoch timings, switch diagnostics.
- **Stage 3: not run.** Dataset access blocks every run; no losses or runtimes invented.
- **Stage 4: in progress.** Fixed-ID splitter and restartable runner; the actual 2,000
  replay IDs cannot be recovered without the pinned import. No substitute IDs are committed.
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

## Not verified

The 22,000-game split, real-data profiles/equality, R reproduction, A/S/B, exact fixed holdout
IDs, real 600-game runner smoke and play strength remain blocked. Hosted GPU/CUDA speed,
memory, and determinism are not verified. No roadmap gate is claimed.

## Checkpoints

No replay-trained checkpoint exists. Generated test inputs/checkpoints are local test
fixtures and not trained research results. This machine may be reclaimed; committed text
evidence is retained on `work/p2-scale-prep`, not in ignored `runs/`.

## Suggestions

Restore access to the pinned corpus, recover the fixed IDs, then finish stage 1 and R/A/S/B.
Do not infer a data effect or change the loss/model from these blocked experiments.
