from collections import Counter

import numpy as np

from fog.constants.tables import TABLES
from fog.scripts.diagnose_belief_value_error import _compatible, _smoothed_probability


def test_smoothed_probability_is_finite_for_unseen_target():
    value = _smoothed_probability(Counter({2: 3}), target=7, classes=10, alpha=0.1)
    assert 0.0 < value < 1.0


def test_public_reveal_compatibility_checks_known_fields():
    candidate = np.arange(15, dtype=np.int64)
    view = np.full(15, -1, dtype=np.int64)
    view[0] = candidate[0]
    view[2] = candidate[2]
    assert _compatible(candidate, view)
    view[2] += 1
    assert not _compatible(candidate, view)


def test_no_item_does_not_constrain_submitted_item():
    candidate = np.arange(15, dtype=np.int64)
    view = np.full(15, -1, dtype=np.int64)
    view[0] = candidate[0]
    view[1] = TABLES.item_index("<NO_ITEM>")
    assert _compatible(candidate, view)
