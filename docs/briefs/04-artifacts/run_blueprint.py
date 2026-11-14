"""Run the limited blueprint PPO arm and its fixed 400-game evaluation points."""
import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import threading
import time

import yaml

from run_scratch import ARTIFACTS, command

WORK = Path('runs/rl_check/blueprint')
CONFIG = WORK/'config.yaml'
PREFIX = ['uv', 'run', '--no-sync', 'python', '-m']


def evaluate(counter, weights):
    for pool in ('pool_a', 'pool_b'):
        out = ARTIFACTS/f'blueprint-{counter}-{pool}.json'
        if out.exists() and json.loads(out.read_text()).get('status') == 'complete':
            continue
        args = PREFIX+[
            'fog.eval.showdown_direct_selection', '--pool', f'data/teams/{pool}',
            '--config', 'configs/rl_check_blueprint.yaml', '--checkpoint', f'blueprint={weights}',
            '--checkpoint-config', 'blueprint=configs/rl_check_blueprint.yaml', '--device', 'cpu',
            '--temperature', '0', '--games', '400', '--seed', '0',
            '--opponent', 'simpleheuristics', '--opponent', 'maxdamage', '--resume', '--out', str(out),
        ]
        log = ARTIFACTS/f'blueprint-{counter}-{pool}-tail.txt'
        command(args, log, environment={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
        log.write_text(''.join(log.read_text().splitlines(True)[-60:]))


def snapshots():
    return sorted([p for p in (WORK/'checkpoints').glob('ppo_[0-9]*.pt')
                   if '.state.' not in p.name
                   and p.with_name(p.stem+'.state.pt').exists()],
                  key=lambda p: int(p.stem.split('_')[1]))


def evaluate_ready():
    points = {}
    for weights in snapshots():
        counter = int(weights.stem.split('_')[1])
        bucket = counter//200000
        if bucket:
            points.setdefault(bucket, (counter, weights))
    for counter, weights in points.values():
        evaluate(counter, weights)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budget', type=int, default=556748)
    args = parser.parse_args()
    build = json.loads((ARTIFACTS/'blueprint-build-summary.json').read_text())
    assert build['status'] == 'complete' and build['epochs'] == 8
    initial = Path(build['checkpoint'])
    assert initial.is_file()
    WORK.mkdir(parents=True, exist_ok=True)
    overrides = {'extends': str(Path('configs/rl_check_blueprint.yaml').resolve()),
                 'ppo': {'init_ckpt': str(initial), 'total_decisions': args.budget},
                 'league': {'past_ckpts': [str(initial)]}}
    CONFIG.write_text(yaml.safe_dump(overrides, sort_keys=False))
    (ARTIFACTS/'blueprint-runtime-overrides.json').write_text(json.dumps(overrides, indent=2)+'\n')
    evaluate(0, initial)
    final = WORK/'checkpoints/ppo_final.pt'
    capped = False
    if not final.exists():
        deadline_file = WORK/'deadline.json'
        if not deadline_file.exists():
            deadline_file.write_text(json.dumps({'started_unix': time.time(),
                                                'deadline_unix': time.time()+36000})+'\n')
        deadline = json.loads(deadline_file.read_text())['deadline_unix']
        states = [p for p in (WORK/'checkpoints').glob('ppo_[0-9]*.state.pt')]
        resume = max(states, key=lambda p: int(p.name.split('_')[1].split('.')[0])) if states else None
        train = PREFIX+['fog.train', '--config', str(CONFIG), '--trainer', 'ppo']
        if resume:
            train += ['--set', f'ppo.resume_ckpt={resume}']
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        if time.time() < deadline:
            started = time.time()
            print(shlex.join(train), flush=True)
            with (WORK/'output.txt').open('a') as stream:
                process = subprocess.Popen(train, stdout=stream, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                cap = [False]

                def stop_at_deadline():
                    if process.poll() is None:
                        cap[0] = True
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass

                timer = threading.Timer(max(0., deadline-time.time()), stop_at_deadline)
                timer.daemon = True
                timer.start()
                try:
                    while process.poll() is None:
                        evaluate_ready()
                        time.sleep(20)
                    code = process.wait()
                finally:
                    timer.cancel()
            capped = cap[0]
            record = dict(command=shlex.join(train), commit=commit, start_unix=started,
                          seconds=time.time()-started, exit_code=code, log=str(WORK/'output.txt'),
                          resumed_from=str(resume) if resume else None, ten_hour_cap=capped)
            with (ARTIFACTS/'commands.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            assert code == 0 or capped, f'PPO interrupted ({code}); inspect output and resume'
        else:
            capped = True
    evaluate_ready()
    metrics = WORK/'metrics.csv'
    rows = list(csv.DictReader(metrics.open()))
    last_logged = int(float(rows[-1]['decisions']))
    if final.exists():
        counter, endpoint = last_logged, final
    else:
        assert capped and snapshots(), 'no completed checkpoint at the time limit'
        endpoint = snapshots()[-1]
        counter = int(endpoint.stem.split('_')[1])
    evaluate(counter, endpoint)
    (ARTIFACTS/'blueprint-metrics.csv').write_text(metrics.read_text())
    (ARTIFACTS/'blueprint-tail.txt').write_text(''.join((WORK/'output.txt').read_text().splitlines(True)[-60:]))
    result = dict(smoke=True, status='capped' if capped else 'complete',
                  requested_decisions=args.budget, checkpoint_decisions=counter,
                  last_logged_decisions=last_logged, checkpoint=str(endpoint),
                  initializer=str(initial), ten_hour_cap=capped)
    (ARTIFACTS/'blueprint-summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
