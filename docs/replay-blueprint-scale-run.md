# Replay blueprint scale run

**Hosted GPU measured; scale training is in progress.** Two G3 epochs on Tesla T4 ran at
604–622 training rows/s (613 aggregate), with 150–154 s validation passes and 2.44 GiB peak
PyTorch allocation. Installation took 89–137 s. These are **smoke** measurements at batch
1,024, FP32, CUDA device 0; two T4s were visible. Full G3 does not fit the one-hour calibration
allowance, so its batch/device quality comparison is unverified. Current results and budget:
[brief 05 report](briefs/05-report.md).

1. **Use the CPU-prepared private inputs.** Both exist under the configured owner's account:

   | Private dataset slug | Training games / rows | Fixed holdout games / rows |
   | --- | --- | --- |
   | `fog-p2-gpu-22000-20261114` | 20,000 / 1,023,525 | 2,000 / 104,037 |
   | `fog-p2-gpu-112000-20261114` | 101,928 / 5,217,022 | 2,000 / 104,037 |

   Each contains `replay-work.bin`, a gzip tar archive whose opaque filename prevents archive
   expansion into thousands of mounted files. Sizes: 468,727,506 / 2,385,762,558 bytes. All
   holdout shards are byte-identical, and G20 is contained in the larger training set. Rating
   floor stays 1500; 8,072 newer non-holdout games were excluded from the larger import.
   Counts and archive hashes: `briefs/05-artifacts/prepare-*`.

2. **Rebuild only on CPU if needed.** Clone the public repository without credentials and
   follow `README.md` for locked dependencies and the pinned Showdown build. Application
   changes are at `bd6db4161cb3d6d563ae4a98f9415771601d4d8c`; the submitted G20 script below
   retains its exact clone pin.

   ```bash
   uv run --no-sync python -m fog.scripts.run_replay_blueprint \
     --games 22000 --work-dir runs/p2_gpu_22000 --device cuda \
     --batch-size 1024 --epochs 18 --prepare-only
   uv run --no-sync python -m fog.scripts.run_replay_blueprint \
     --games 112000 --work-dir runs/p2_gpu_112000 --device cuda \
     --batch-size 1024 --epochs 18 --prepare-only
   ```

   Preparation requires no GPU. Removing the flag from the identical command skips completed
   preparation and trains. Keep the fixed ID file. Actual import / conversion / split walls:
   92 / 1,000 / 3 s for 22k; 790 / 4,453 / 3 s for 112k. The latter reused initial raw/cache
   records; these walls include local contention and are not idle throughput benchmarks.
   Import/conversion/splitting run on CPU. Allow about 25 GB working-disk headroom for inputs,
   runtime and a resume copy.

3. **Submit one private GPU job at a time.** Wait for the active job before reproducing it.
   The exact source is [fog-p2-g20-20261114.py](briefs/05-artifacts/fog-p2-g20-20261114.py).
   It clones the recorded commit, preserves Kaggle's CUDA PyTorch, builds Showdown, restores
   the prepared data and trains from scratch with both selected checkpoints. Internet and T4
   are enabled. With the official client configured, copy that source to
   `runs/replay_launch/gpu_job.py` and create ignored metadata using the client's Python:

   ```python
   import json
   from pathlib import Path
   from kaggle.api.kaggle_api_extended import KaggleApi
   a = KaggleApi(); a.authenticate()
   owner = a.config_values["username"]
   Path("runs/replay_launch/kernel-metadata.json").write_text(json.dumps({
       "id": owner + "/fog-replay-blueprint-reproduction",
       "title": "fog replay blueprint reproduction", "code_file": "gpu_job.py",
       "language": "python", "kernel_type": "script", "is_private": True,
       "enable_gpu": True, "enable_internet": True,
       "dataset_sources": [owner + "/fog-p2-gpu-22000-20261114"],
       "kernel_sources": [], "competition_sources": []}))
   ```

   ```bash
   kaggle kernels push -p runs/replay_launch --accelerator NvidiaTeslaT4 --timeout 39600
   ```

   `uv tool install kaggle --with six` installs the client including its missing dependency.
   Keep account metadata under ignored `runs/`. The measured rate projects G20 to **9.6 h**
   including allowance; this is not a completed timing. Cap: 11 h; internal export deadline:
   10 h 40 min. Select G100 epochs/subset only after G20's actual rate and remaining 20-hour
   budget are known, with two hours spare.

4. **Continue from the last completed epoch.** `bc_state.pt` retains optimizer, schedule,
   RNG, earlier CSV rows and both best models. Jobs export the full work directory as
   `replay-resume.tar.gz`, plus selected weights and text results. Keep `run`, `games`,
   `epochs`, batch, seed and recipe unchanged in `SPEC`; set `resume_kernel_slug` to the
   preceding private job and attach it in metadata `kernel_sources` using the authenticated
   owner prefix. Set `stop_after_epoch` to `None` to complete the schedule. The controller
   requires the archive and epoch state. Real G3 continuation preserved raw CSV `[0,1]`
   exactly as the prefix of `[0,1,2]`.

5. **Download and verify the selected weights.** The single private dataset is
   `fog-p2-gpu-checkpoints-20261114`; its initial version contains partial G3 only. Later
   versions add G20/G100. This command was tested on both G3 files:

   ```bash
   python -c 'from kaggle.api.kaggle_api_extended import KaggleApi; a=KaggleApi(); a.authenticate(); a.dataset_download_files(a.config_values["username"]+"/fog-p2-gpu-checkpoints-20261114", path="checkpoints", unzip=True, quiet=True)'
   ```

   File sizes, SHA-256, selected epochs and partial/full status are in
   [checkpoints.json](briefs/05-artifacts/checkpoints.json). Check the inventory before
   evaluation. Earlier archive-layout and capped attempts are failed/partial evidence.
   The internal deadline reserves export time; platform cancellation can return no output.
