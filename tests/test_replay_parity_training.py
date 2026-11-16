import numpy as np

from fog.eval.replay_parity_training import common_indices


def test_pairing_keeps_normal_struggle_and_separates_replacements():
    def arrays(actions, turns, phases):
        n = len(actions)
        mask = np.zeros((n, 14), dtype=bool)
        mask[:, [8, 13]] = True  # old model inference calls this forced; actual request is normal
        return {'action': np.asarray(actions), 'row_turn': np.asarray(turns),
                'row_role': np.ones(n, dtype=np.int8), 'decision_phase': np.asarray(phases),
                'legality_mask': mask, 'policy_valid': np.ones(n, bool),
                'legal_provenance': np.full((n, 14), 2),
                'action_feature_provenance': np.full((n, 14), 2)}

    old = arrays([8, 0, 2], [1, 2, 3], [1, 0, 0])
    new = arrays([8, 9, 0, 1], [1, 1, 2, 3], [0, 1, 0, 0])
    left, right, replacements, disagreements = common_indices(old, new, old)
    np.testing.assert_array_equal(left, [0, 1])
    np.testing.assert_array_equal(right, [0, 2])
    np.testing.assert_array_equal(replacements, [1])
    assert disagreements == 1
