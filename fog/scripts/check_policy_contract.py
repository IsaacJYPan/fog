"""Check a frozen learner against behavior probabilities from real CPU actor decisions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml

from fog.env.player import _behavior_log_probs
from fog.eval.learning_signal import probability_contract
from fog.models.agent import FogAgent
from fog.scripts.automated_training import _block_config
from fog.training.ppo import _index_nested


def check(agent, batch: dict, *, device: str, temperature: float, bonus: float,
          tolerance: float = 1e-3) -> dict:
    if not 0 < tolerance < .01:
        raise ValueError("Probability tolerance must be positive and below .01")
    agent.eval()
    original = {k: v.detach().cpu().clone() for k, v in agent.state_dict().items()}
    modes = []

    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {k: move(v) for k, v in value.items()}
        return value

    batch = {k: v for k, v in batch.items() if k != 'preview'}
    with torch.no_grad():
        for size in (32, 512):
            for chain in (False, True):
                predictions = []
                for start in range(0, len(batch['actions']), size):
                    indices = torch.arange(start, min(start+size, len(batch['actions'])))
                    part = move(_index_nested(batch, indices))
                    part['compute_belief_chain'] = chain
                    out = agent(part)
                    predictions.append(_behavior_log_probs(
                        out.log_probs, temperature=temperature, tera_logit_bonus=bonus).cpu())
                result = probability_contract(batch['behavior_log_probs'], torch.cat(predictions),
                                              batch['actions'], batch['old_log_probs'])
                result.update(minibatch_size=size, compute_belief_chain=chain)
                modes.append(result)
    unchanged = all(torch.equal(v.detach().cpu(), original[k])
                    for k, v in agent.state_dict().items())
    passed = unchanged and all(
        r['legal_log_prob_max_abs_error'] <= tolerance and r['clip_fraction'] == 0
        and r['recorded_vs_distribution_max_abs_error'] <= tolerance
        and r['chosen_log_prob_abs_error_quantiles'][-1] <= tolerance
        and r['categorical_kl_max'] <= 1e-6 for r in modes)
    return dict(passed=passed, weights_unchanged=unchanged, device=device,
                log_probability_tolerance=tolerance, modes=modes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    job = yaml.safe_load(args.job.read_text())
    settings = job['safety']['policy_contract']
    checkpoint = Path(job['initial_checkpoint'])
    batch_path = Path(settings['batch'])
    cfg = _block_config(job, Path.cwd(), args.out.parent/'contract', checkpoint,
                        int(job['training']['decisions_per_block']), int(job['training']['seed'])+1)
    device = cfg['device']
    torch.set_num_threads(1)
    agent = FogAgent.from_tables(cfg).to(device)
    agent.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    batch = torch.load(batch_path, map_location='cpu', weights_only=False)
    result = check(agent, batch, device=device, temperature=cfg['ppo']['rollout_temperature'],
                   bonus=cfg['ppo'].get('rollout_tera_logit_bonus', 0.),
                   tolerance=float(settings.get('tolerance', 1e-3)))
    result.update(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  batch_sha256=hashlib.sha256(batch_path.read_bytes()).hexdigest(),
                  torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  gpu_name=torch.cuda.get_device_name() if str(device).startswith('cuda') else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    if not result['passed']:
        raise RuntimeError("Zero-update policy contract failed; training must not start")
    print('Policy contract passed:', args.out, flush=True)


if __name__ == '__main__':
    main()
