"""Brief 01 stage 3a: how many replay rows may supervise the policy.

    uv run --no-sync python docs/briefs/01-artifacts/stage3a_check.py \
        runs/blueprint/split/train_1000.parquet > docs/briefs/01-artifacts/stage3a.json
"""
import json
import sys

import numpy as np
import polars as pl

SWITCH_START = 8

manifest = pl.read_parquet(sys.argv[1])
rows = valid = feat = legal = both = switch = 0
legal_sum = 0.0
chosen_in_mask = 0
switch_valid = switch_both = 0
for shard in manifest["shard"].to_list():
    z = np.load(shard)
    action = z["action"]
    n = len(action)
    index = np.arange(n)
    pv = z["policy_valid"].astype(bool)
    f = z["action_feature_provenance"][index, action] == 2
    g = z["legal_provenance"][index, action] == 2
    mask = z["legality_mask"].astype(bool)
    rows += n
    valid += int(pv.sum())
    feat += int((pv & f).sum())
    legal += int((pv & g).sum())
    both += int((pv & f & g).sum())
    is_switch = action >= SWITCH_START
    switch += int(is_switch.sum())
    switch_valid += int((pv & is_switch).sum())
    switch_both += int((pv & f & g & is_switch).sum())
    legal_sum += float(mask.sum(-1).sum())
    chosen_in_mask += int(mask[index, action].sum())

print(json.dumps({
    "manifest": sys.argv[1],
    "games": len(manifest),
    "rows": rows,
    "policy_valid_share": valid / rows,
    "exact_chosen_action_features_share_of_valid": feat / valid,
    "exact_chosen_legality_share_of_valid": legal / valid,
    "both_exact_share_of_valid": both / valid,
    "both_exact_share_of_switch_valid": switch_both / max(1, switch_valid),
    "both_exact_share_of_move_valid": (both - switch_both) / max(1, valid - switch_valid),
    "mean_legal_actions": legal_sum / rows,
    "switch_share": switch / rows,
    "chosen_action_in_legality_mask_share": chosen_in_mask / rows,
}, indent=2))
