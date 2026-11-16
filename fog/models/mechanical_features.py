"""Public move-table facts and the exact candidate-column contract for replay models."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

MECHANICAL_ACTION_FEATURE_NAMES = (
    'move_power', 'move_accuracy', 'expected_hits', 'physical', 'special', 'status_move',
    'same_type_bonus', 'type_effectiveness', 'physical_stat_ratio', 'special_stat_ratio',
    'estimated_damage',
)


class MechanicalMoveFeatures(nn.Module):
    """A fixed lookup over generated tables, including their full boolean/flag vocabulary."""

    def __init__(self, tables) -> None:
        super().__init__()
        moves = tables.moves
        features = moves['features']
        types = sorted({f['type'] for f in features.values()})
        categories = sorted({f['category'] for f in features.values()})
        booleans = sorted({key for f in features.values() for key, value in f.items()
                           if isinstance(value, bool)})
        flags = sorted({flag for f in features.values() for flag in f.get('flags', ())})
        self.names = tuple([f'type:{x}' for x in types] + [f'category:{x}' for x in categories]
                           + ['base_power', 'accuracy', 'priority']
                           + [f'boolean:{x}' for x in booleans] + [f'flag:{x}' for x in flags])
        values = np.zeros((len(moves['ids']) + 1, len(self.names)), dtype=np.float32)
        for index, move_id in enumerate(moves['ids']):
            f = features[move_id]
            values[index] = (
                [float(f['type'] == t) for t in types]
                + [float(f['category'] == c) for c in categories]
                + [float(f.get('base_power', 0)) / 250., float(f.get('accuracy', 100)) / 100.,
                   float(f.get('priority', 0)) / 7.]
                + [float(f.get(key, False)) for key in booleans]
                + [float(flag in f.get('flags', ())) for flag in flags]
            )
        self.register_buffer('values', torch.from_numpy(values), persistent=False)
        self.out_dim = len(self.names)

    def forward(self, move_ids: torch.Tensor) -> torch.Tensor:
        return self.values[move_ids]
