# Automated training locally or on Kaggle

The automated driver runs a sequence of complete training blocks. After every block it runs a
fresh held-out evaluation, records the result, and continues from the newest healthy model. It
never starts or connects to the local viewing server.

For a single PPO or behavior-cloning experiment, use `python -m fog.train --config CONFIG`. The
`fog.train.TrainingSession` object exposes the resolved config to a notebook before execution. The
automated driver below is specifically for multi-block PPO orchestration.

## Local use

```powershell
.\fog-train.cmd doctor --job configs/automated_gen9ou_typed_league.yaml
.\fog-train.cmd run --job configs/automated_gen9ou_typed_league.yaml
.\fog-train.cmd status --job configs/automated_gen9ou_typed_league.yaml
```

The `.cmd` wrapper is unaffected by PowerShell execution policy. It refuses to start unless the
project environment has PyTorch, PyYAML, and working CUDA. This avoids accidentally running
training with a different `python` that has only some dependencies installed. The `.ps1` wrapper
is also available on systems where PowerShell scripts are permitted.

Stopping the command is safe at a saved 24k checkpoint. Running the same command again skips every
completed block and evaluation. If a block was interrupted, it starts a recovery directory from
the newest saved model and trains only the remaining decision budget. That recovery restores model
weights, but the current trainer does not save Adam optimizer history; the status JSON records this
boundary instead of pretending it was an exact resume.

The human-readable summary is `runs/gen9ou_typed_autonomous_league/STATUS.txt`. The complete,
machine-readable record is `pipeline_state.json` beside it.

After each evaluation, the driver keeps only two numbered snapshots from that block while keeping
the final model and reports. One poor evaluation is treated as noise and training continues. Two
poor blocks in a row restore one of the two retained fallback models. If the same fallback is needed
more than twice, automatic training pauses and asks for a larger evaluation instead of repeating the
same cycle.

To preview or run conservative cleanup across older runs:

```powershell
.\.venv\Scripts\python.exe -m fog.scripts.checkpoint_cleanup
.\.venv\Scripts\python.exe -m fog.scripts.checkpoint_cleanup --execute
```

The cleanup command only considers old numbered snapshots. Named best/final models, referenced
models, and the newest two numbered snapshots in every run are protected.

## Kaggle setup

1. Build one uploadable archive. The command finds the starting model, opponent models, replay
   manifest and all referenced replay shards automatically:

```powershell
.\fog-train.cmd bundle-kaggle --job configs/automated_gen9ou_typed_league.yaml --out artifacts/fog-kaggle-input.zip
```

   Upload the extracted `fog` directory as a Kaggle Dataset.
2. Create a GPU notebook and add that Dataset. Internet access is not required if the dataset also
   contains all missing Python wheels; otherwise enable it briefly for the dependency install.
3. Copy the read-only input to Kaggle's writable disk and install the project:

```python
from pathlib import Path
import shutil, subprocess, sys

source = Path("/kaggle/input/fog-training-bundle/fog")
work = Path("/kaggle/working/fog")
if not work.exists():
    shutil.copytree(source, work)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", str(work), "--no-deps"])
%cd /kaggle/working/fog
```

Kaggle already supplies many scientific packages. If `doctor` or the import reports a missing
package, install the direct-simulator requirements (this workflow does not need `poke-engine`):

```python
!pip install poke-env polars pyyaml tensorboard aiohttp tqdm
```

4. Validate and run:

```python
!python -m fog.scripts.automated_training doctor --job configs/automated_gen9ou_typed_league.yaml
!python -m fog.scripts.automated_training run --job configs/automated_gen9ou_typed_league.yaml
```

5. Before ending the notebook, create one downloadable artifact:

```python
!python -m fog.scripts.automated_training export \
  --job configs/automated_gen9ou_typed_league.yaml \
  --out /kaggle/working/gen9ou-training-output.zip
```

Turn on Kaggle's GPU accelerator. A T4 should work but will be slower than the current local GPU.
The direct simulator uses local Node child processes in memory; it does not need a Pokémon
Showdown web server or port 8000. Kaggle's output is temporary until the notebook version is saved,
so export the job or save `/kaggle/working` as notebook output before the session ends.

### Behavior cloning on Kaggle or Colab

Behavior cloning does not need simulator workers during optimization. Upload the repository plus
the parsed replay manifest/shards referenced by the chosen config, install the project as above,
and run:

```python
from fog.train import TrainingSession

session = TrainingSession.from_file("configs/bc_ou.yaml")
session.config["device"] = "cuda"
metrics = session.run()
```

Public replays do not contain the acting player's submitted private sheet. Policy rows therefore
use provenance-tracked prior imputations; those inferred values are inputs, not belief truth. For a
run used for reported results, bundle all referenced shards, preserve the config and seed, and
compare multiple private-set imputations or seeds so the result is not tied to one guessed sheet.

## Editing a job

Copy `configs/automated_gen9ou_typed_league.yaml` and change only the new copy. The important
settings are:

- `initial_checkpoint`: model used for the first block.
- `training.blocks`: number of 240k blocks.
- `training.decisions_per_block`: decisions per block.
- `evaluation.games`: games per model/opponent row.
- `output_dir`: unique directory for this job.

Use `--dry-run` to print the first training command and generated resolved configuration without
starting a battle:

```powershell
.\fog-train.cmd run --job YOUR_JOB.yaml --dry-run
```
