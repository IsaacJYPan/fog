# Replay blueprint scale run

**Not run on a hosted GPU.** The settings below reuse this repository's notebook
installer; GPU throughput and memory use still need a real notebook check. Likely
setup failures: a full `uv sync` may need Rust to build `poke-engine` (install Rust
and retry, or use the replay-only installer below, which omits that engine); an
unavailable CUDA device needs the notebook GPU enabled and its supported PyTorch
build; replay downloads need Internet access to `huggingface.co` and
`us.aws.cdn.hf.co`. Keep the error output when fixing a failed setup.

1. Create a Kaggle Python notebook. In **Settings**, enable a **GPU accelerator**
   and **Internet**. Use writable `/kaggle/working` for the checkout and active run.
   Arrange persistent storage before starting: the whole work folder is needed to
   resume, and temporary notebook storage can disappear when a session ends.

2. Paste this cell to clone the public repository at the pinned code revision.
   Cloning needs no login. Run it once in a fresh session.

   ```bash
   %%bash
   set -e
   git clone https://github.com/IsaacJYPan/fog /kaggle/working/fog
   cd /kaggle/working/fog
   git checkout 4f9d75c6f2a2562153809263ab7dc2d292e04540
   git submodule update --init --recursive
   ```

3. Install and check the runtime. The notebook installer keeps the notebook's
   CUDA PyTorch, installs FOG into `.venv`, and omits the unused Rust engine. If
   `node` or `npm` is missing, install the notebook platform's Node.js/npm packages
   first. The final assertion must succeed before training.

   ```bash
   %%bash
   set -e
   cd /kaggle/working/fog
   python -m pip install uv
   npm --prefix server/pokemon-showdown ci --ignore-scripts
   npm --prefix server/pokemon-showdown run build
   python -m fog.scripts.kaggle_setup
   .venv/bin/python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
   test -f data/replay_splits/p2_holdout_2000_ids.txt
   ```

4. Start the scale run with this cell. The fixed 2,000-game holdout ID file is
   already in the repository; no data upload is needed. Keep this command and the
   recipe unchanged for every restart.

   ```bash
   %%bash
   set -e
   cd /kaggle/working/fog
   uv run --no-sync python -m fog.scripts.run_replay_blueprint \
     --games 112000 --work-dir runs/blueprint_scale --device cuda --batch-size 1024 --epochs 6
   ```

   The runner imports public games from the pinned dataset revision (rating
   ≥1500, uploaded from 2024-02-01), converts them, excludes games newer than the
   fixed holdout, and trains on all eligible games. It prints the count if fewer
   than 100,000 remain. Completed steps are skipped. Training continues from its
   last complete epoch, restoring optimizer, schedule, random state, metrics and
   best model; an unfinished epoch is repeated. Holdout scoring runs on CPU.
   Import, conversion, training/evaluation and disk projections from the current
   real-data measurements are pending; GPU speed is unmeasured.

5. Before a session ends, preserve the run and restart it as follows:

   - Stop the running cell normally, then save **all of
     `/kaggle/working/fog/runs/blueprint_scale/`** to persistent storage. Retain
     `run.json`, `raw/`, `cache/`, `decisions/`, `split/`, `training/` and `output/`.
     Saving only `results/` loses the state needed to resume. Save intermediate
     work before the platform's hard session limit.
   - In a fresh GPU/Internet session, repeat cells 2–3 at the same commit.
   - Restore the saved folder as writable
     `/kaggle/working/fog/runs/blueprint_scale/`, preserving its contents. For
     example, copy an attached saved folder from `/kaggle/input` into this path;
     do not train in read-only `/kaggle/input`.
   - Run cell 4 again. Look for completed stages being skipped and
     `[bc] resume: completed epoch N/6`. A hard termination can lose timing for
     the unfinished attempt; the results summary records that limitation.

6. When the runner prints `Results:`, download and hand back exactly
   **`/kaggle/working/fog/runs/blueprint_scale/results/`** in full. It contains
   `bc_best.pt`, `metrics.csv`, `holdout-best.json`, `holdout-untrained.json`,
   `train-tail.txt` and `run-summary.txt`. Keep the larger work folder separately
   if further training or recovery may be needed.

For a CPU-only machine, install the locked local environment with
`uv sync --locked --extra dev`, build Showdown as in cell 3, then run from the
repository root:

```sh
uv run --no-sync python -m fog.scripts.run_replay_blueprint \
  --games 22000 --work-dir runs/blueprint_cpu20k --device cpu --batch-size 1024 --epochs 6
```

Importing 22,000 games retains the fixed 2,000-game holdout and leaves 20,000 for
training; importing only 20,000 can omit holdout IDs. Its runtime projection from
the current stage 1 measurements is pending. The recipe itself retains batch 256;
the runner uses batch 1,024 for the scale command.
