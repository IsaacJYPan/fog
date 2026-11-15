"""Bounded single-T4 brief 08 job; inject FOG_COMMIT in the private submission.

Submit with platform timeout 3300 s and NvidiaTeslaT4, with the private converted
replay input attached. Reuses brief 06's bootstrap and native training command.
The internal deadline is 2900 s from job entry, including setup, reserving export.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import zipfile


def main():
    started = time.monotonic()
    root = Path('/kaggle/temp/fog')
    output = Path('/kaggle/working/08-artifacts')
    output.mkdir(parents=True, exist_ok=True)
    deadline = started + 2900
    commit = os.environ['FOG_COMMIT']
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               FOG08_OUTPUT=str(output))
    records = []
    monitor = None
    gpu_count = None

    def write(name, data):
        (output / name).write_text(json.dumps({'label': 'smoke', 'commit': commit, **data}, indent=2) + '\n')

    def run(command, name, cwd=root):
        begin = time.monotonic()
        env['FOG08_STARTED'] = str(time.time())
        with (output / (name + '.txt')).open('w') as stream:
            child = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = child.wait(timeout=max(1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                code = 124
        record = {'name': name, 'command': command, 'exit_code': code,
                  'seconds': time.monotonic() - begin}
        records.append(record)
        write(name + '-execution.json', record)
        print(name, code, flush=True)
        if code:
            print((output / (name + '.txt')).read_text()[-5000:], flush=True)
            raise RuntimeError(name + ' failed; stop without changing the recipe')

    status = 'failed'
    try:
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total',
                                       '--format=csv,noheader'], text=True).strip().splitlines()
        gpu_count = len(gpu)
        if gpu_count != 1:
            raise RuntimeError('Expected exactly one allocated GPU; release immediately')
        with (output / 'gpu-memory.csv').open('w') as stream:
            monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,name,memory.used,utilization.gpu',
                                        '--format=csv', '--loop=1'], stdout=stream)
        root.parent.mkdir(parents=True, exist_ok=True)
        run(['git', 'clone', 'https://github.com/IsaacJYPan/fog.git', str(root)], 'clone', root.parent)
        run(['git', 'checkout', '--detach', commit], 'checkout')
        run(['git', 'submodule', 'update', '--init', '--recursive', '--depth', '1'], 'submodule')
        run(['npm', '--prefix', 'server/pokemon-showdown', 'ci', '--ignore-scripts', '--no-audit', '--no-fund'], 'showdown-install')
        run(['npm', '--prefix', 'server/pokemon-showdown', 'run', 'build'], 'showdown-build')
        run([sys.executable, '-m', 'fog.scripts.kaggle_setup'], 'setup')
        python = str(root / '.venv/bin/python')
        run([python, '-m', 'pip', 'install', 'py-spy==0.4.3'], 'py-spy-install')
        cores = len(os.sched_getaffinity(0))
        quota = Path('/sys/fs/cgroup/cpu.max')
        quota = quota.read_text().strip() if quota.exists() else 'unavailable'
        if quota != 'unavailable' and not quota.startswith('max'):
            amount, period = map(int, quota.split())
            cores = min(cores, max(1, math_ceil(amount, period)))
        env['FOG08_WORKERS'] = str(cores)
        cpu = next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                   if line.startswith('model name'))
        write('hardware.json', {'cpu_model': cpu, 'logical_cpus': os.cpu_count(),
                                'affinity_cpus': len(os.sched_getaffinity(0)), 'cpu_quota': quota,
                                'rollout_workers': cores, 'gpu': gpu,
                                'started_utc': datetime.now(timezone.utc).isoformat()})
        run([python, '-c', 'import json,torch; print(json.dumps({"torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0)}))'], 'torch-version')
        # Resolve only shard basenames from the private attached input; no account
        # identifier belongs in the script or public evidence.
        candidates = list(Path('/kaggle/input').rglob('manifest.parquet'))
        if len(candidates) != 1:
            raise RuntimeError('Attach exactly one converted replay manifest')
        manifest = candidates[0]
        work = root / 'runs/update-speed'
        work.mkdir(parents=True, exist_ok=True)
        code = '''import polars as pl,sys
from pathlib import Path
source=Path(sys.argv[1]); root=source.parent
m=pl.read_parquet(source)
m=m.with_columns(pl.col("shard").map_elements(lambda s: str(root / "shards" / Path(s).name), return_dtype=pl.String))
assert all(Path(p).exists() for p in m["shard"])
m.write_parquet(sys.argv[2])
print(m.select(pl.len(),pl.col("n_rows").sum()))
'''
        target = work / 'manifest.parquet'
        run([python, '-c', code, str(manifest), str(target)], 'replay-manifest')
        env['FOG08_MANIFEST'] = str(target)
        run([python, 'docs/briefs/08-artifacts/measure.py', 'ppo'], 'ppo')
        shutil.copy2(root / 'runs/collect_gpu/metrics.csv', output / 'ppo-metrics.csv')
        summarize(output, commit)
        print('PPO_STAGE_COMPLETE', flush=True)
        run([python, 'docs/briefs/08-artifacts/measure.py', 'replay'], 'replay')
        print('REPLAY_STAGE_COMPLETE', flush=True)
        # Export the native update checkpoints for restart outside the public repo.
        with zipfile.ZipFile('/kaggle/working/checkpoints.zip', 'w', zipfile.ZIP_STORED) as archive:
            for folder in (root / 'runs/collect_gpu/checkpoints', work):
                for path in folder.glob('*.pt'):
                    archive.write(path, str(path.relative_to(root)))
        status = 'complete'
    finally:
        if monitor:
            monitor.terminate()
            monitor.wait(timeout=10)
        seconds = time.monotonic() - started
        write('budget.json', {'status': status, 'allocated_gpus': gpu_count, 'job_seconds': seconds,
                              'gpu_hours': gpu_count * seconds / 3600 if gpu_count else None,
                              'platform_timeout_seconds': 3300,
                              'single_job_allocation_upper_bound_gpu_hours': 3300 / 3600,
                              'records': records})


def math_ceil(amount, period):
    return (amount + period - 1) // period


def summarize(out, commit):
    stop = json.loads((out / 'ppo-stop.json').read_text())
    rows = list(csv.DictReader((out / 'ppo-metrics.csv').open()))
    events = [json.loads(line) for line in (out / 'training-timing.jsonl').read_text().splitlines()]
    updates = [e for e in events if e['kind'] == 'update']
    assert len(rows) == len(updates) == 2
    decisions = stop['decisions']
    assert decisions == sum(e['rows'] for e in updates)
    collection = sum(float(r['collection_seconds']) for r in rows)
    update = sum(float(r['update_seconds']) for r in rows)
    optimization = sum(float(r['optimization_seconds']) for r in rows)
    visits = sum(float(r['optimizer_rows']) for r in rows)
    steps = sum(float(r['optimizer_steps']) for r in rows)
    wall = stop['wall_seconds_through_checkpoint']
    result = {'label': 'smoke', 'commit': commit, 'completed_updates': 2,
              'decisions': decisions, 'optimizer_steps': steps, 'processed_rows': visits,
              'collection_seconds': collection, 'update_seconds': update,
              'optimization_seconds': optimization, 'wall_seconds': wall,
              'mean_optimizer_step_seconds': optimization / steps,
              'processed_rows_per_second': visits / optimization,
              'collection_decisions_per_second': decisions / collection,
              'end_to_end_decisions_per_second': decisions / wall,
              'collection_wall_share': collection / wall, 'update_wall_share': update / wall,
              'other_wall_share': (wall - collection - update) / wall,
              'peak_learner_allocated_bytes': max(e['peak_learner_allocated_bytes'] for e in updates),
              'timing_caveat': 'One minibatch in update 2 has synchronized part hooks and three concurrent py-spy dumps; includes their overhead. Native config and schedules unchanged except rollout_workers; stops after checkpoint 2.'}
    (out / 'ppo-summary.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
