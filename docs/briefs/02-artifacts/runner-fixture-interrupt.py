"""Run generated-data interruption check from repo root after runner-fixture-setup.py.

The work directory must be fresh; results do not measure real replay learning.
"""
import csv
import json
import signal
import subprocess
import time
from pathlib import Path

root = Path.cwd()
work = Path('runs/blueprint_runner_fixture_final')
command = [str(root/'.venv/bin/python'), '-u', '-m', 'fog.scripts.run_replay_blueprint', '--games', '600', '--work-dir', str(work), '--device', 'cpu', '--batch-size', '256', '--epochs', '2', '--chronological-holdout-games', '100', '--parquet', 'runs/blueprint_runner_fixture_source/synthetic-public-replays.parquet']
root.joinpath(work).mkdir(parents=True,exist_ok=False)
records = {'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), 'fixture': '600 generated two-turn protocol logs; not the pinned replay corpus', 'command': command}
started=time.perf_counter()
with (root/work/'first-attempt.txt').open('w') as output:
    child=subprocess.Popen(command,cwd=root,stdout=output,stderr=subprocess.STDOUT)
    deadline=time.monotonic()+600
    state=root/work/'training/checkpoints/bc_state.pt'
    while not state.is_file():
        if child.poll() is not None:
            raise RuntimeError(f'runner exited before epoch state: {child.returncode}')
        if time.monotonic()>deadline:
            child.terminate();child.wait();raise RuntimeError('timeout waiting for epoch state')
        time.sleep(.1)
    child.send_signal(signal.SIGTERM)
    records['interrupted_exit_code']=child.wait(timeout=30)
records['first_attempt_seconds']=time.perf_counter()-started
with (root/work/'training/metrics.csv').open() as stream:
    records['metrics_epochs_after_interrupt']=[int(row['epoch']) for row in csv.DictReader(stream)]
records['final_checkpoint_after_interrupt']=(root/work/'training/checkpoints/bc_final.pt').exists()
started=time.perf_counter()
with (root/work/'resumed-attempt.txt').open('w') as output:
    completed=subprocess.run(command,cwd=root,stdout=output,stderr=subprocess.STDOUT)
records['resumed_exit_code']=completed.returncode
records['resumed_seconds']=time.perf_counter()-started
with (root/work/'training/metrics.csv').open() as stream:
    records['metrics_epochs_after_resume']=[int(row['epoch']) for row in csv.DictReader(stream)]
(root/work/'interruption-test.json').write_text(json.dumps(records,indent=2))
print(json.dumps(records,indent=2),flush=True)
if completed.returncode:
    raise SystemExit(completed.returncode)
