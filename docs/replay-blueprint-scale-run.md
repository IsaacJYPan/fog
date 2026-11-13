# Replay blueprint scale run

**Not run on a hosted GPU.** GPU speed and memory use are unmeasured. Three likely
setup failures: **Rust** — a full `uv sync` builds `poke-engine`; cell 3 avoids that
unused engine. **CUDA** — enable the GPU, restart the notebook if needed, and keep
its supplied CUDA PyTorch using cell 3. **Downloads** — enable Internet; the run
needs `huggingface.co` and `us.aws.cdn.hf.co`. If access is refused, retain the error
and use a machine with access.

1. Create a Kaggle Python notebook. Turn **GPU accelerator** and **Internet** on.
   Use writable `/kaggle/working`, and plan how to save the run before the session
   limit. Allow **25 GB free disk** for environment, working data and a resume
   archive; the arithmetic below is a projection, not a hosted storage test.

2. Paste this cell in a fresh session. The repository is public; no login is needed.

   ```bash
   %%bash
   set -e
   git clone https://github.com/IsaacJYPan/fog /kaggle/working/fog
   cd /kaggle/working/fog
   git checkout 0e937552905ac93b36938f2d765384651d7b2185
   git submodule update --init --recursive
   ```

3. Install and check the runtime. This reuses the repository's Kaggle installer,
   preserving preinstalled CUDA PyTorch and omitting the unused Rust engine.
   Hosted installation time is unmeasured.

   ```bash
   %%bash
   set -e
   cd /kaggle/working/fog
   python -m pip install uv
   if ! command -v node >/dev/null || ! command -v npm >/dev/null; then
     apt-get update
     apt-get install -y nodejs npm
   fi
   npm --prefix server/pokemon-showdown ci --ignore-scripts
   npm --prefix server/pokemon-showdown run build
   python -m fog.scripts.kaggle_setup
   .venv/bin/python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
   test -f data/replay_splits/p2_holdout_2000_ids.txt
   df -h /kaggle/working
   ```

4. Start the run with this cell. Keep this command unchanged for every restart.
   The fixed 2,000 holdout IDs are included. The runner imports the pinned public
   corpus, converts it, excludes newer games outside that holdout, and trains on
   all eligible games. If fewer than 100,000 remain, it prints the actual count.

   ```bash
   %%bash
   set -e
   cd /kaggle/working/fog
   uv run --no-sync python -m fog.scripts.run_replay_blueprint \
     --games 112000 --work-dir runs/blueprint_scale --device cuda --batch-size 1024 --epochs 6
   ```

   CPU measurements on four cores give these **projections**, assuming similar
   game lengths and acceptance rates; hosted CPU/network speeds can differ:

   | Step | Arithmetic for 112,000 imported games | Projected time / disk |
   | --- | --- | --- |
   | Import | `194 s × 112/22`; raw `209,345,372 bytes × 112/22` | 16.5 min; 1.07 GB raw |
   | Convert | `(2,225 − 758 paused) s × 112/22`; `608,097,221 bytes × 112/22` | 2.07 h active; 3.10 GB decisions |
   | Download cache | `481,396,201 bytes × 112/22` | about 2.45 GB; grows in whole shard chunks |
   | Split | 2,000 fixed holdout games; about 100,000 eligible training games | not separately timed; manifests are small |
   | Six training passes | `1,023,525/20,000 × 100,000 = 5,117,625 rows`; `6 × rows / {194.04 profile, 157.72–187.06 full runs} rows/s` | 44.0–54.1 h CPU reference at batch 256; GPU time unknown |
   | CPU batch preparation alone | `6 × 5,117,625 / 7,100.46 rows/s` | 1.20 h feed floor; **not a GPU runtime estimate** |

   Conversion's active time still includes initial CPU contention. The training
   reference excludes evaluation and file writes; throughput at batch 1,024 is
   unmeasured. Validation averaged `154.66–172.19 s` per pass in R/A/S. Six
   validations plus initial validation/startup/writes give
   `6 × {154.66, 172.19} + {172.37, 209.31} = 1,100–1,242 s` CPU overhead;
   A's outside-epoch time includes its deliberate restart. Two standalone
   reports took `218.62–262.13 s`, used as a proxy for the runner's best/untrained
   CPU scoring. Adding import, conversion and those overheads projects to
   **46.7–56.8 h total on CPU**. This planning range includes observed runtime
   variation; it is not a guaranteed bound. GPU validation time is unknown. Raw data, decisions and
   cache project to `1.07 + 3.10 + 2.45 = 6.61 GB`; the measured local environment
   is 5.68 GB. Keeping a second full data copy for resume gives
   `5.68 + 2 × 6.61 = 18.90 GB` before checkpoints, source and logs, hence 25 GB
   headroom. Backup creation and handback download time are unmeasured. Inputs and
   exact arithmetic:
   [scale-projections.json](briefs/03-artifacts/scale-projections.json).

5. To survive a session ending, follow these four steps:

   - Interrupt the running cell while there is time to save. Archive **the entire
     work directory**, then download the archive from the notebook's Files pane:

     ```bash
     %%bash
     tar -czf /kaggle/working/blueprint_scale-resume.tar.gz \
       -C /kaggle/working/fog/runs blueprint_scale
     ```

   - Upload that archive as a private Kaggle dataset. In the next session enable
     Internet and the same GPU accelerator type/count, repeat cells 2–3 at the
     same commit, and attach the saved dataset with **Add Input**.
   - Restore it into the writable checkout with this Python cell:

     ```python
     from pathlib import Path
     import subprocess
     archive = next(Path("/kaggle/input").rglob("blueprint_scale-resume.tar.gz"))
     destination = Path("/kaggle/working/fog/runs")
     destination.mkdir(exist_ok=True)
     subprocess.run(["tar", "-xzf", str(archive), "-C", str(destination)], check=True)
     ```

   - Run cell 4 again. Completed stages are skipped; training prints
     `[bc] resume: completed epoch N/6`. It restores the last complete epoch and
     repeats an unfinished epoch. An unfinished conversion is rerun. Retain all of
     `raw/`, `cache/`, `decisions/`,
     `split/`, `training/`, `output/` and `run.json`; `results/` alone cannot resume.

6. When the runner prints `Results:`, download and hand back exactly
   **`/kaggle/working/fog/runs/blueprint_scale/results/`**, including `bc_best.pt`,
   `metrics.csv`, both holdout JSONs, `train-tail.txt` and `run-summary.txt`.
   Package that one folder with this cell, then download
   `blueprint_scale-results.tar.gz` from the Files pane. Keep the larger resume
   archive separately.

   ```bash
   %%bash
   tar -czf /kaggle/working/blueprint_scale-results.tar.gz \
     -C /kaggle/working/fog/runs/blueprint_scale results
   ```

For a CPU-only machine, clone the same commit, run `uv sync --locked --extra dev`,
build Showdown as in cell 3, and run from the repository root:

```sh
uv run --no-sync python -m fog.scripts.run_replay_blueprint \
  --games 22000 --work-dir runs/blueprint_cpu20k --device cpu --batch-size 1024 --epochs 6
```

The 22,000-game import retains the fixed holdout and leaves 20,000 training games.
Import/conversion project to `194 s + 1,467 s = 27.7 min` active; six training passes
are `6 × 1,023,525 / {194.04, 157.72} = 8.79–10.82 h` at measured batch-256
speeds. Adding the overhead above gives about **9.6–11.7 h total**. Actual
batch-1,024 throughput is unmeasured.
The recipe file keeps batch 256; only the runner command requests 1,024.
