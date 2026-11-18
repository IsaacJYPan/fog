import numpy as np
import pytest

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


def test_holdout_resume_matches_uninterrupted_run(tmp_path, monkeypatch):
    import json
    import sys

    import polars as pl

    from fog.eval import replay_parity_training as evaluation

    checkpoint = tmp_path / 'model.pt'
    checkpoint.write_bytes(b'fixed model')
    games = [{'id': f'{i:03}', 'shard': 'unused'} for i in range(101)]
    monkeypatch.setattr(pl, 'read_parquet', lambda _: pl.DataFrame(games))
    monkeypatch.setattr('fog.training.search_distill.build_agent', lambda *args: object())
    arrays = {'legality_mask': np.ones((1, 14), bool)}
    monkeypatch.setattr(evaluation, 'load_arrays', lambda _: arrays)
    monkeypatch.setattr(evaluation, 'common_indices', lambda *args: (
        np.array([0]), np.array([0]), np.array([0]), 0))
    calls = 0

    def score(*args):
        nonlocal calls
        calls += 1
        if calls == 801:
            raise InterruptedError('after the first checkpoint')
        return [1, 2., 1]

    monkeypatch.setattr(evaluation, 'score', score)
    output = tmp_path / 'results.json'
    argv = ['evaluate', '--old-manifest', 'old', '--new-manifest', 'new',
            '--old-metadata', 'metadata', '--old-checkpoint', str(checkpoint),
            '--new-checkpoint', str(checkpoint), '--out', str(output), '--resume']
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(InterruptedError):
        evaluation.main()
    assert len(json.loads(output.with_suffix('.progress.json').read_text())['per_game']) == 100
    evaluation.main()
    resumed = json.loads(output.read_text())
    assert calls == 809  # only the eight unfinished score calls are repeated
    assert not output.with_suffix('.progress.json').exists()
    evaluation.main()
    assert json.loads(output.read_text()) == resumed
