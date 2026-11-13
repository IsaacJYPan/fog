# Replay blueprint scale run

**Not run on a hosted GPU.** CPU engineering tests pass; real-data timing and the
brief 01 holdout recovery are blocked in the development environment. Do not treat
the generated-data smoke as a learning curve or a GPU speed estimate.

Use a private GPU notebook with **GPU accelerator and Internet enabled**, enough
disk for raw replays, compressed decision shards and checkpoints, and a way to
retain its working directory between sessions. Keep this private repository and
outputs private. On Kaggle use `/kaggle/working`, not read-only `/kaggle/input`, for
the active run. Repository access must use the owner's configured authentication;
if cloning is unavailable, attach a private checkout archive at the same commit.

Paste these cells (the setup follows `notebooks/kaggle_training.ipynb` and
`docs/kaggle-setup.md`; it preserves the notebook's CUDA PyTorch):

```python
!git clone https://github.com/IsaacJYPan/fog /kaggle/working/fog
%cd /kaggle/working/fog
!git checkout ddd83638f8068a85c740eb90029e8201cd98279c
!git submodule update --init --recursive
!python -m pip install uv
!npm --prefix server/pokemon-showdown ci --ignore-scripts
!npm --prefix server/pokemon-showdown run build
!python -m fog.scripts.kaggle_setup
!.venv/bin/python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
```

**Required before the scale command:** restore
`data/replay_splits/p2_holdout_2000_ids.txt`. It is deliberately absent from this
commit: the dataset host refused access, so the exact IDs could not be recovered.
On a machine allowed to reach the dataset, reproduce brief 02 section 4's pinned
22,000-game import, conversion and chronological split. Verify all its row counts
and the holdout dates before exporting the IDs; do not create a fresh scale holdout.
Then run from that checkout, and copy the resulting text file into this notebook:

```sh
uv run --no-sync python - <<'PY'
from pathlib import Path
import polars as pl
m = pl.read_parquet('runs/blueprint/split/holdout_manifest.parquet')
assert len(m) == 2000 and m['n_rows'].sum() == 104037
assert m['id'].n_unique() == 2000
p = Path('data/replay_splits/p2_holdout_2000_ids.txt')
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text('\n'.join(sorted(m['id'].to_list())) + '\n')
PY
```

Launch, or rerun after interruption, with this **same command**:

```sh
uv run --no-sync python -m fog.scripts.run_replay_blueprint \
  --games 112000 --work-dir runs/blueprint_scale --device cuda --batch-size 1024 --epochs 6
```

The runner imports the pinned revision (rating ≥1500, from 2024-02-01), converts,
splits using exactly the fixed IDs, reports excluded newer games, and trains on all
eligible games. It also saves nested 20k/100k subsets when enough games exist. Fewer
than 100k eligible games is reported, not hidden. Completed steps are skipped;
training restores optimizer, schedule, scaler, RNG, metrics and best model from the
last complete epoch. An incomplete epoch is repeated. Keep epochs, batch size,
recipe and holdout unchanged; incompatible reruns fail. Scoring currently runs on
CPU, including when training uses CUDA.

When a session ends, save **all of `runs/blueprint_scale/`**, including `run.json`,
`raw/`, `decisions/`, `split/`, `training/checkpoints/bc_state.pt` and `output/`, to
private persistent storage. Download/archive this folder before the notebook's
temporary storage disappears. Restore it writable in the next session, recreate
the environment at the same commit, restore the same holdout file, and rerun the
same command. Runner-owned shard paths are rebased if the restored work directory
moves. Retain external local-source files too if using `--parquet` for a smoke.
Hard termination can lose the unfinished attempt's timing, which the summary says.

For review download **`runs/blueprint_scale/results/`** in full: `bc_best.pt`,
`metrics.csv`, `holdout-best.json`, `holdout-untrained.json`, `train-tail.txt` and
`run-summary.txt`. That results folder is sufficient for review, not for resume.

No new real-corpus runtime projection is available: stage 1 could not load its
manifest. Brief 01's measurements were 28.5 minutes to convert 22k games on four
cores (about 2.4 hours for 112k if linear), training 112–141 rows/s, and about 140 s
per 104,037-row holdout score. These are **historical context**, not stage 1 or GPU
measurements. After data recovery, run
`python -m fog.scripts.profile_replay_training --out runs/blueprint/profile-before`
on an idle CPU; estimate training seconds as `6 * training_rows / measured_rows_per_second`
and add evaluation/checkpoint time. Measure CUDA speed before budgeting a session.

CPU fallback (22k imported includes the fixed 2k holdout, leaving 20k training games):

```sh
uv run --no-sync python -m fog.scripts.run_replay_blueprint \
  --games 22000 --work-dir runs/blueprint_cpu20k --device cpu --batch-size 1024 --epochs 6
```

Its stage 1 projection is unmeasured. At the historical brief 01 rate, six passes
over 1,023,525 training rows would take about 12–15 hours **for training alone**,
with a different batch size here; this is only a planning estimate. Importing only
20k may omit some fixed holdout IDs and is therefore insufficient. Measure first.

Likely notebook failures: (1) `uv sync` may fail building Rust `poke-engine`; the
notebook installer above omits that unused engine for this replay-only pipeline;
for the complete locked environment install Rust first and retry `uv sync --locked
--extra dev`. (2) If the CUDA assertion fails, enable GPU/restart and use the
notebook's supported PyTorch/CUDA build; do not silently fall back to CPU. (3)
Downloads require `huggingface.co` and its redirect/CDN hosts, including
`us.aws.cdn.hf.co`; enable Internet and use an environment where those hosts are
permitted. Retain error output and fix access through that environment's settings.
None of these hosted-notebook steps has been verified here.
