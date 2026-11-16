"""Whole-game holdout diagnostics for supervised preview lead selection."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from fog.config import load_config
from fog.eval.replay_slot_order_check import game_resampled_intervals
from fog.models.agent import FogAgent
from fog.training.dataset import PreviewDataset


def frequency_predictions(species: np.ndarray, counts: dict) -> np.ndarray:
    """Choose the most-led training species present; alphabetical ties ignore slot order."""
    return np.asarray([min(range(6), key=lambda i: (-counts.get(team[i], 0), team[i]))
                       for team in species], dtype=np.int64)


def _metrics(total):
    n = total[0]
    return {'nll': float(total[1]/n), 'top1': float(total[2]/n),
            'frequency_top1': float(total[3]/n), 'chance_top1': 1/6,
            'top1_minus_frequency': float((total[2]-total[3])/n),
            'top1_minus_chance': float(total[2]/n-1/6)}


def score(config: str, checkpoints: dict, train: Path, holdout: Path, *, samples=2000, seed=0):
    train_ds = PreviewDataset(train)
    ds = PreviewDataset(holdout)
    counts = Counter(team[label] for team,label in zip(train_ds.data['species'],train_ds.data['label']))
    baseline = frequency_predictions(ds.data['species'],counts)
    cfg = load_config(config)
    results = {}
    for name, checkpoint in checkpoints.items():
        agent = FogAgent.from_tables(cfg).eval()
        agent.load_state_dict(torch.load(checkpoint,map_location='cpu'))
        per_game = np.zeros((len(ds.game_ids),4),dtype=np.float64)
        offset = 0
        with torch.no_grad():
            for batch in ds.batches(256,shuffle=False):
                logits = agent.preview_logits(batch)
                labels = batch['preview_action']
                nll = torch.nn.functional.cross_entropy(logits,labels,reduction='none').numpy()
                correct = (logits.argmax(-1)==labels).numpy()
                size = len(labels)
                values = np.stack([np.ones(size),nll,correct,
                                   baseline[offset:offset+size]==labels.numpy()],axis=1)
                np.add.at(per_game,ds.game_index[offset:offset+size],values)
                offset += size
        results[name] = {'checkpoint':checkpoint,**_metrics(per_game.sum(0)),
                         'interval95':game_resampled_intervals(per_game,_metrics,samples=samples,seed=seed)}
    return {'smoke':True,'config':config,'train_manifest':str(train),'holdout_manifest':str(holdout),
            'train_games':len(train_ds.game_ids),'holdout_games':len(ds.game_ids),
            'train_valid_preview_rows':len(train_ds),'holdout_valid_preview_rows':len(ds),
            'train_games_with_valid_labels':int(len(np.unique(train_ds.game_index))),
            'holdout_games_with_valid_labels':int(len(np.unique(ds.game_index))),
            'train_label_share':(np.bincount(train_ds.data['label'],minlength=6)/len(train_ds)).tolist(),
            'holdout_label_share':(np.bincount(ds.data['label'],minlength=6)/len(ds)).tolist(),
            'training_lead_species_counts':dict(sorted(counts.items())),
            'baseline':'Most often led training species present on this team; alphabetical species ties.',
            'models':results,'interval_method':{'unit':'game (both perspectives together)',
                                              'samples':samples,'seed':seed,'method':'percentile bootstrap'}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--checkpoint',action='append',required=True,help='NAME=PATH')
    parser.add_argument('--train-manifest',required=True,type=Path)
    parser.add_argument('--holdout-manifest',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--bootstrap-samples',type=int,default=2000)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--threads',type=int,default=2)
    args = parser.parse_args()
    checkpoints = {}
    for item in args.checkpoint:
        name,sep,path = item.partition('=')
        if not sep or not name or not path or name in checkpoints:
            parser.error('checkpoints must be unique NAME=PATH pairs')
        checkpoints[name] = path
    torch.set_num_threads(args.threads)
    result = score(args.config,checkpoints,args.train_manifest,args.holdout_manifest,
                   samples=args.bootstrap_samples,seed=args.seed)
    result['command'] = shlex.join([sys.executable,'-m','fog.eval.replay_preview_report',*sys.argv[1:]])
    result['commit'] = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
