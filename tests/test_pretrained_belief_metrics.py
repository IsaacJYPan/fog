import numpy as np

from fog.pretrained.belief_metrics import SampleMetrics
from fog.pretrained.labels import exact_team_tokens


def test_joint_accuracy_ignores_move_order_and_masks_unverified_sets():
    truth = np.tile(np.arange(15), (6, 1))
    known = np.ones((6, 15), dtype=bool)
    known[0, 3:7] = False
    known[1, 1] = False
    verified = np.ones_like(known)
    verified[1, 1] = False
    samples = np.stack([truth.copy(), truth.copy()])
    samples[0, 0, 3:7] = samples[0, 0, 3:7][::-1]
    samples[1, 0, 3] = 99
    metrics = SampleMetrics()
    metrics.update(samples, truth, verified, known)
    report = metrics.report()
    assert report["joint_set_accuracy"] == 0.5
    assert report["joint_sample_count"] == 2
    assert report["joint_recall_at_k"] == 1
    assert report["families"]["moveset"]["sample_accuracy"] == 0.5
    assert "item" not in report["families"]


def test_absent_submitted_item_is_exact_no_item():
    from fog.constants.tables import TABLES

    truth = exact_team_tokens("Great Tusk\nAbility: Protosynthesis\n- Headlong Rush\n", TABLES)
    assert truth[0, 1] == TABLES.item_index("<NO_ITEM>")
    assert (truth[1:, 1] == -1).all()
