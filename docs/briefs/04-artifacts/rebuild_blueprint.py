"""Recreate the brief-03 run-A initializer, retaining its 18-epoch schedule through epoch eight."""
import csv
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

import polars as pl
import torch

from run_scratch import ARTIFACTS, command

PREFIX = ['uv', 'run', '--no-sync', 'python', '-m']
BUILD = Path('runs/rl_check/blueprint_build')
TRAIN = Path('runs/blueprint/g1000e18')
RAW = Path('data/metamon_gen9ou_r1500')
DECISIONS = Path('data/metamon_gen9ou_r1500_decisions')
SPLIT = Path('runs/blueprint/split')


def copy_tail(source, destination):
    (ARTIFACTS/destination).write_text(''.join(Path(source).read_text().splitlines(True)[-60:]))


def rebuild():
    BUILD.mkdir(parents=True, exist_ok=True)
    access = subprocess.run([
        'curl', '-sS', '-o', '/dev/null', '-w', '%{http_code}\n', '-m', '30',
        'https://huggingface.co/api/datasets/jakegrigsby/metamon-raw-replays',
    ], capture_output=True, text=True)
    (ARTIFACTS/'blueprint-host-access.txt').write_text(access.stdout+access.stderr)
    if access.returncode or access.stdout.strip() != '200':
        result = dict(status='skipped', reason='replay-host precheck did not return 200')
        (ARTIFACTS/'blueprint-build-summary.json').write_text(json.dumps(result, indent=2)+'\n')
        return
    imported = RAW/'.metamon/import.json'
    if not imported.exists() or json.loads(imported.read_text()).get('kept_games') != 22000:
        command(PREFIX+[
            'fog.scripts.import_metamon_replays', '--out', str(RAW), '--revision',
            '792a86b5cdbcab9696032e947483b218453a0d4f', '--min-rating', '1500',
            '--after', '2024-02-01', '--max-games', '22000', '--max-shards', '4',
            '--max-download-mb', '600',
        ], BUILD/'import.txt')
    if not (DECISIONS/'manifest.parquet').exists():
        command(PREFIX+[
            'fog.scripts.convert_replay_decisions', '--raw', str(RAW), '--out', str(DECISIONS),
            '--imputations', '1', '--workers', '4',
        ], BUILD/'convert.txt')
    if not (SPLIT/'train_1000.parquet').exists():
        command(PREFIX+[
            'fog.scripts.split_replay_manifest', '--manifest', str(DECISIONS/'manifest.parquet'),
            '--out', str(SPLIT), '--chronological-holdout-games', '2000',
            '--train-sizes', '40', '1000', '3000', '10000', '20000',
        ], BUILD/'split.txt')
    holdout = pl.read_parquet(SPLIT/'holdout_manifest.parquet')
    fixed_ids = set(Path('data/replay_splits/p2_holdout_2000_ids.txt').read_text().split())
    counts = dict(holdout_games=len(holdout), holdout_rows=holdout['n_rows'].sum(),
                  fixed_ids_match=set(holdout['id'].cast(pl.String).to_list()) == fixed_ids,
                  training={})
    for size, expected_rows in [(1000, 52983), (3000, 156597), (20000, 1023525)]:
        train = pl.read_parquet(SPLIT/f'train_{size}.parquet')
        counts['training'][str(size)] = dict(games=len(train), rows=train['n_rows'].sum())
        assert len(train) == size and train['n_rows'].sum() == expected_rows, counts
    (ARTIFACTS/'blueprint-input-check.json').write_text(json.dumps(counts, indent=2)+'\n')
    assert (len(holdout), holdout['n_rows'].sum(), counts['fixed_ids_match']) == (2000, 104037, True), counts
    for name in ('import', 'convert', 'split'):
        if (BUILD/f'{name}.txt').exists():
            copy_tail(BUILD/f'{name}.txt', f'blueprint-{name}-tail.txt')

    state = TRAIN/'checkpoints/bc_state.pt'
    metrics = TRAIN/'metrics.csv'
    current_epoch = 0
    if state.exists():
        current_epoch = int(torch.load(state, map_location='cpu', weights_only=False)['epoch'])
    assert current_epoch <= 8, 'run A has already passed epoch eight; stop and inspect'
    if current_epoch < 8:
        args = PREFIX+[
            'fog.train', '--config', 'configs/replay_blueprint_cpu.yaml', '--trainer', 'bc',
            '--set', f'bc.manifest={SPLIT}/train_1000.parquet',
            f'bc.holdout_manifest={SPLIT}/holdout_manifest.parquet', 'bc.epochs=18',
            'bc.streaming=true', 'bc.early_stopping_patience=0', 'bc.selection_metric=action_ce',
            'bc.resume=true', f'log.dir={TRAIN}',
        ]
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        started = time.time()
        print(shlex.join(args), flush=True)
        with (BUILD/'bc.txt').open('a') as stream:
            process = subprocess.Popen(args, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            reached_limit = False
            while process.poll() is None:
                if metrics.exists() and state.exists():
                    rows = list(csv.DictReader(metrics.open()))
                    # The atomic state replacement follows the epoch's CSV and best weights.
                    if (rows and int(rows[-1]['epoch']) == 8
                            and state.stat().st_mtime_ns > metrics.stat().st_mtime_ns):
                        reached_limit = True
                        os.killpg(process.pid, signal.SIGTERM)
                        break
                time.sleep(2)
            code = process.wait()
        record = dict(command=shlex.join(args), commit=commit, start_unix=started,
                      seconds=time.time()-started, exit_code=code, log=str(BUILD/'bc.txt'),
                      resumed_from_epoch=current_epoch, intentional_stop_after_epoch_8=reached_limit)
        with (ARTIFACTS/'commands.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        assert reached_limit, f'blueprint training stopped unexpectedly: {code}; inspect bc.txt'
    payload = torch.load(state, map_location='cpu', weights_only=False)
    assert int(payload['epoch']) == 8
    (ARTIFACTS/'blueprint-build-metrics.csv').write_text(metrics.read_text())
    copy_tail(BUILD/'bc.txt', 'blueprint-build-tail.txt')
    result = dict(status='complete', epochs=8, schedule_epochs=18,
                  best_epoch=int(payload['best_epoch']),
                  checkpoint=str(TRAIN/'checkpoints/bc_best.pt'),
                  best_holdout_action_ce=float(payload['best_eval_loss']), inputs=counts)
    (ARTIFACTS/'blueprint-build-summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    rebuild()
