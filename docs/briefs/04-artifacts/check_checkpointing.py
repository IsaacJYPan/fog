"""Compare complete PPO updates on recorded Showdown observations, including both belief heads."""
from copy import deepcopy
import json
from pathlib import Path
import torch
from fog.config import load_config
from fog.env.teams import load_pool
from fog.eval.showdown_rollout_probe import build_worker
from fog.training.direct_parallel import WorkerGame
from fog.training.ppo import _index_nested, ppo_update


def main():
    cfg = load_config('configs/rl_check_scratch.yaml')
    worker, agent = build_worker(cfg, load_pool(cfg['env']['team_pool']), threads=1)
    try:
        buf, _ = worker.run(([WorkerGame(0, n, n+1, 0, 0) for n in range(2)], 'MaxDamage', 0.))
    finally:
        worker.close()
    batch = buf.to_batch('cpu')
    batch.pop('preview', None)
    batch = _index_nested(batch, torch.arange(8))
    agent.requires_grad_(True)
    torch.set_grad_enabled(True)
    torch.set_num_threads(4)
    other = deepcopy(agent)
    cfg['ppo'].update(epochs=2, minibatch_size=4, activation_checkpointing=False)
    rng = torch.random.get_rng_state()
    before = ppo_update(agent, torch.optim.Adam(agent.parameters(), lr=cfg['ppo']['lr']), batch, cfg)
    final_rng = torch.random.get_rng_state()
    cfg['ppo']['activation_checkpointing'] = True
    torch.random.set_rng_state(rng)
    after = ppo_update(other, torch.optim.Adam(other.parameters(), lr=cfg['ppo']['lr']), batch, cfg)
    exact = all(torch.equal(value, other.state_dict()[name]) for name, value in agent.state_dict().items())
    result = dict(rows=8, epochs=2, parameters_bit_identical=exact, metrics_identical=before==after,
                  rng_identical=torch.equal(final_rng, torch.random.get_rng_state()))
    Path('docs/briefs/04-artifacts/checkpointing-equivalence.json').write_text(json.dumps(result, indent=2)+'\n')
    print(result)
    assert all(result[key] for key in ('parameters_bit_identical', 'metrics_identical', 'rng_identical'))


if __name__ == '__main__':
    main()
