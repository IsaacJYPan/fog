"""Fingerprint an update on eight recorded Showdown decisions on the same CPU and PyTorch."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from fog.config import load_config
from fog.env.teams import load_pool
from fog.eval.showdown_rollout_probe import _feed, build_worker
from fog.training.direct_parallel import WorkerGame
from fog.training.ppo import _index_nested, ppo_update


def digest(value):
    checksum = hashlib.sha256()
    _feed(checksum, value)
    return checksum.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--explicit-defaults", action="store_true")
    args = parser.parse_args()
    config = load_config("configs/rl_check_scratch.yaml")
    worker, agent = build_worker(config, load_pool(config["env"]["team_pool"]), threads=1)
    try:
        games = [WorkerGame(0, i, i + 1, 0, 0) for i in range(2)]
        buffer, _ = worker.run((games, "MaxDamage", 0.0))
    finally:
        worker.close()
    batch = buffer.to_batch("cpu")
    batch.pop("preview", None)
    batch = _index_nested(batch, torch.arange(8))
    agent.requires_grad_(True)
    torch.set_num_threads(1)
    config["ppo"].update(epochs=2, minibatch_size=4)
    if args.explicit_defaults:
        config["ppo"].update(belief_row_fraction=1.0, value_after_kl_stop=False,
                             mixed_precision=False)
    optimizer = torch.optim.Adam(agent.parameters(), lr=config["ppo"]["lr"])
    torch.manual_seed(20261201)
    metrics = ppo_update(agent, optimizer, batch, config)
    result = dict(torch=str(torch.__version__), recorded_rows=len(batch["actions"]),
                  batch=digest(batch), parameters=digest(agent.state_dict()),
                  optimizer=digest(optimizer.state_dict()), metrics=digest(metrics),
                  rng=digest(torch.random.get_rng_state()), optimizer_steps=metrics["optimizer_steps"])
    assert result["optimizer_steps"] > 0
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
