"""Compare replay policies by the labelled move's frequency in the training split."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from fog.config import load_config
from fog.constants.tables import TABLES
from fog.eval.replay_slot_order_check import game_resampled_intervals
from fog.models.agent import FogAgent
from fog.training.bc import _policy_supervision_mask
from fog.training.dataset import StreamingReplayDataset

GROUPS = ('never', '1_to_19', '20_or_more')


def labelled_move_ids(batch: dict, loss_cfg: dict) -> tuple[torch.Tensor, torch.Tensor]:
    selected = _policy_supervision_mask(batch, loss_cfg) & (batch['action'] < 8)
    row = torch.arange(len(batch['action']))
    mine = batch['active_idx'][:, 0]
    ids = batch['entity_ids'][row, mine, 3 + batch['action'].remainder(4)]
    return selected, ids


def frequency_bucket(frequency: torch.Tensor) -> torch.Tensor:
    return torch.where(frequency == 0, 0, torch.where(frequency < 20, 1, 2))


def _statistic(total: np.ndarray) -> dict:
    n = total[0]
    return {'nll': float(total[1]/n) if n else None,
            'top1': float(total[2]/n) if n else None}


def score(configs: dict[str, str], checkpoints: dict[str, str], train: Path, holdout: Path,
          *, batch_size: int = 256, samples: int = 2000, seed: int = 0) -> dict:
    import polars as pl
    if set(configs) != set(checkpoints):
        raise ValueError('config and checkpoint names must match')
    cfgs = {name: load_config(path) for name,path in configs.items()}
    loss_cfg = next(iter(cfgs.values())).get('loss', {})
    # All compared recipes must supervise the very same policy labels.
    for cfg in cfgs.values():
        for key in ('require_exact_chosen_action_features','require_exact_chosen_legality'):
            if cfg.get('loss', {}).get(key, False) != loss_cfg.get(key, False):
                raise ValueError('compared recipes have different policy supervision masks')
    frequencies = torch.zeros(TABLES.n_moves + 1, dtype=torch.long)
    for batch in StreamingReplayDataset(train).batches(batch_size, shuffle=False):
        selected, ids = labelled_move_ids(batch, loss_cfg)
        if bool(((ids[selected] < 0) | (ids[selected] >= TABLES.n_moves)).any()):
            raise ValueError('policy-valid training move label has an unknown table identity')
        frequencies += torch.bincount(ids[selected], minlength=len(frequencies))
    models = {}
    for name, cfg in cfgs.items():
        model = FogAgent.from_tables(cfg).eval()
        model.load_state_dict(torch.load(checkpoints[name], map_location='cpu'))
        models[name] = model
    manifest = pl.read_parquet(holdout)
    counts = manifest['n_rows'].to_numpy()
    game_of_row = np.repeat(np.arange(len(manifest)), counts)
    totals = {name: {group:np.zeros((len(manifest),3),dtype=np.float64)
                     for group in GROUPS} for name in models}
    offset = 0
    with torch.no_grad():
        for batch in StreamingReplayDataset(holdout).batches(batch_size, shuffle=False):
            games = game_of_row[offset:offset+len(batch['action'])]
            offset += len(games)
            selected, ids = labelled_move_ids(batch, loss_cfg)
            if bool(((ids[selected] < 0) | (ids[selected] >= TABLES.n_moves)).any()):
                raise ValueError('policy-valid holdout move label has an unknown table identity')
            bucket = frequency_bucket(frequencies[ids])
            for name, model in models.items():
                output = model(batch, need_belief=False)
                nll = -output.log_probs.gather(1, batch['action'][:,None])[:,0]
                correct = output.log_probs.argmax(1) == batch['action']
                for group_index, group in enumerate(GROUPS):
                    keep = selected & (bucket == group_index)
                    values = torch.stack([torch.ones_like(nll[keep]),nll[keep],correct[keep]],dim=1)
                    np.add.at(totals[name][group], games[keep.numpy()], values.double().numpy())
    if offset != len(game_of_row):
        raise ValueError('holdout manifest counts disagree with stored rows')
    result = {'smoke':True,'train_manifest':str(train),'holdout_manifest':str(holdout),
              'training_label_frequency':{
                  TABLES.moves['ids'][i]:int(n) for i,n in enumerate(frequencies[:TABLES.n_moves]) if n
              },'models':{},'interval_method':{
                  'unit':'game (both perspectives together)','samples':samples,'seed':seed,
                  'method':'percentile bootstrap of row-weighted statistics'},
              'group_definition':'training policy-valid labels; ordinary/Tera share a move id'}
    for name in models:
        result['models'][name] = {'config':configs[name],'checkpoint':checkpoints[name],'groups':{}}
        for group, per_game in totals[name].items():
            total = per_game.sum(0)
            result['models'][name]['groups'][group] = {
                'rows':int(total[0]),**_statistic(total),
                'interval95':game_resampled_intervals(per_game,_statistic,samples=samples,seed=seed)}
    return result


def _named(values: list[str]) -> dict[str,str]:
    result = {}
    for value in values:
        name, sep, path = value.partition('=')
        if not sep or not name or not path or name in result:
            raise ValueError('expected unique NAME=PATH arguments')
        result[name] = path
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',action='append',required=True,help='NAME=PATH')
    parser.add_argument('--checkpoint',action='append',required=True,help='NAME=PATH')
    parser.add_argument('--train-manifest',type=Path,required=True)
    parser.add_argument('--holdout-manifest',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=256)
    parser.add_argument('--bootstrap-samples',type=int,default=2000)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--threads',type=int,default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    result = score(_named(args.config),_named(args.checkpoint),args.train_manifest,
                   args.holdout_manifest,batch_size=args.batch_size,
                   samples=args.bootstrap_samples,seed=args.seed)
    result['command'] = shlex.join([sys.executable,'-m','fog.eval.replay_move_frequency_report',*sys.argv[1:]])
    result['commit'] = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
