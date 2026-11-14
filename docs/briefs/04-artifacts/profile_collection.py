"""Profile the existing PPO worker against MaxDamage (diagnostic, smoke)."""
import cProfile
import json
import os
import pstats
import random
import time
from pathlib import Path

from fog.config import load_config
from fog.env.teams import load_pool
from fog.eval.showdown_rollout_probe import build_worker
from fog.training.direct_parallel import WorkerGame


def main():
    cfg = load_config('configs/rl_check_scratch.yaml')
    worker, agent = build_worker(cfg, load_pool(cfg['env']['team_pool']), threads=2)
    belief_calls = {'belief': 0, 'self_belief': 0}

    def count(name):
        def hook(*args):
            belief_calls[name] += 1
        return hook

    for name in belief_calls:
        getattr(agent, name).register_forward_hook(count(name))
    rng = random.Random(0)
    assigned = [WorkerGame(0, n, rng.getrandbits(62), 0, 0) for n in range(64)]
    profile = cProfile.Profile()
    def node_cpu():
        fields = Path(f'/proc/{worker.client._process.pid}/stat').read_text().split()
        return (int(fields[13]) + int(fields[14])) / os.sysconf('SC_CLK_TCK')

    cpu_before = node_cpu()
    start = time.perf_counter()
    try:
        profile.enable()
        buf, wins = worker.run((assigned, 'MaxDamage', 0.))
        profile.disable()
        elapsed = time.perf_counter() - start
        decisions = buf.n_decisions
        node_cpu_seconds = node_cpu() - cpu_before
    finally:
        worker.close()
    stats = pstats.Stats(profile)
    def cumulative(file_suffix, function):
        return sum(v[3] for (f, line, name), v in stats.stats.items()
                   if f.endswith(file_suffix) and name == function)
    result = dict(smoke=True, games=64, learner_decisions=decisions, wins=wins,
                  profiled_seconds=elapsed, learner_decisions_per_sec=decisions/elapsed,
                  node_cpu_seconds=node_cpu_seconds,
                  node_roundtrip_seconds=cumulative('showdown_direct.py', '_request'),
                  observation_seconds=cumulative('observation.py', 'from_battle'),
                  forward_seconds=cumulative('agent.py', 'forward'), belief_calls=belief_calls)
    out = Path('docs/briefs/04-artifacts')
    (out/'stage0-profile.json').write_text(json.dumps(result, indent=2)+'\n')
    with (out/'stage0-profile.txt').open('w') as stream:
        stats.stream = stream
        stats.sort_stats('cumulative').print_stats(30)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
