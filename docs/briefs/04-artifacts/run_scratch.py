"""Run the brief's fixed-team scratch arm; checkpoints and resumable outputs stay in runs/."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

import torch
from fog.config import load_config
from fog.models.agent import FogAgent

CONFIG = 'configs/rl_check_scratch.yaml'
ARTIFACTS = Path('docs/briefs/04-artifacts')


def command(args, log, *, environment=None):
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    start = time.time()
    print(shlex.join(args), flush=True)
    with Path(log).open('a') as stream:
        result = subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT, env=environment)
    record = dict(command=shlex.join(args), commit=commit, start_unix=start,
                  seconds=time.time()-start, exit_code=result.returncode, log=str(log))
    with (ARTIFACTS/'commands.jsonl').open('a') as stream:
        stream.write(json.dumps(record)+'\n')
    if result.returncode:
        raise RuntimeError(f'command failed ({result.returncode}): {log}')
    return record


def evaluate(run, label, weights):
    for opponent in ('maxdamage', 'simpleheuristics'):
        out = ARTIFACTS/f'{run}-{label}-{opponent}.json'
        if out.exists() and json.loads(out.read_text()).get('status') == 'complete':
            continue
        args = ['uv', 'run', '--no-sync', 'python', '-m', 'fog.eval.showdown_direct_selection',
                '--pool', 'data/teams/pool_a/core_v1_offensive.txt', '--config', CONFIG,
                '--checkpoint', f'{run}={weights}', '--checkpoint-config', f'{run}={CONFIG}',
                '--device', 'cpu', '--temperature', '0', '--games', '200', '--seed', '0',
                '--opponent', opponent, '--out', str(out), '--resume']
        environment = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
        command(args, ARTIFACTS/f'{run}-{label}-{opponent}-tail.txt', environment=environment)
        log = ARTIFACTS/f'{run}-{label}-{opponent}-tail.txt'
        log.write_text(''.join(log.read_text().splitlines(True)[-60:]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='scratch')
    parser.add_argument('--budget', type=int, required=True)
    parser.add_argument('--lr', default='3.0e-4')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--end-only', action='store_true')
    args = parser.parse_args()
    work = Path('runs/rl_check')/args.run
    work.mkdir(parents=True, exist_ok=True)
    initial = work/'untrained.pt'
    if not initial.exists():
        cfg = load_config(CONFIG)
        torch.manual_seed(cfg['seed'])
        agent = FogAgent.from_tables(cfg).cpu().eval()
        torch.save(agent.state_dict(), initial)
        del agent
    if not args.end_only:
        evaluate(args.run, '0', initial)
    final = work/'checkpoints/ppo_final.pt'
    if not final.exists():
        train = ['uv', 'run', '--no-sync', 'python', '-m', 'fog.train', '--config', CONFIG,
                 '--trainer', 'ppo', '--set', f'ppo.total_decisions={args.budget}',
                 f'ppo.lr={args.lr}', f'ppo.epochs={args.epochs}', f'log.dir={work}']
        states = list((work/'checkpoints').glob('ppo_*.state.pt'))
        if states:
            state = max(states, key=lambda p: int(p.name.split('_')[1].split('.')[0]))
            train.append(f'ppo.resume_ckpt={state}')
        command(train, work/'output.txt')
    metrics = work/'metrics.csv'
    (ARTIFACTS/f'{args.run}-metrics.csv').write_text(metrics.read_text())
    (ARTIFACTS/f'{args.run}-tail.txt').write_text(''.join((work/'output.txt').read_text().splitlines(True)[-60:]))
    checkpoints = [] if args.end_only else sorted(
        [p for p in (work/'checkpoints').glob('ppo_[0-9]*.pt') if '.state.' not in p.name],
        key=lambda p: int(p.stem.split('_')[1]))
    import csv
    last = int(float(list(csv.DictReader(metrics.open()))[-1]['decisions']))
    if not checkpoints or int(checkpoints[-1].stem.split('_')[1]) != last:
        checkpoints.append(final)
    for ckpt in checkpoints:
        label = str(last) if ckpt == final else ckpt.stem.split('_')[1]
        evaluate(args.run, label, ckpt)


if __name__ == '__main__':
    main()
