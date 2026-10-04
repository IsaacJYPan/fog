import subprocess

import pytest

from fog.scripts import kaggle_worker


class FakeProcess:
    pid = 123

    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.waits = []

    def wait(self, timeout=None):
        self.waits.append(timeout)
        result = next(self.outcomes)
        if result == 'timeout':
            raise subprocess.TimeoutExpired('training', timeout)
        return result


def test_completed_training_uses_an_isolated_process_group(monkeypatch):
    process = FakeProcess([0])
    calls = []
    monkeypatch.setattr(kaggle_worker.subprocess, 'Popen',
        lambda command, **kwargs: calls.append((command, kwargs)) or process)
    assert kaggle_worker.run_training(['python', 'train'], {'A': 'b'}, 12) == 0
    assert calls == [(['python', 'train'], {'env': {'A': 'b'}, 'start_new_session': True})]
    assert process.waits == [12]


@pytest.mark.parametrize('outcomes,expected_waits,signals', [
    (['timeout', -15], [12, 20], 1),
    (['timeout', 'timeout', -9], [12, 20, None], 2),
])
def test_deadline_stops_training_descendants_before_export(monkeypatch, outcomes, expected_waits, signals):
    process = FakeProcess(outcomes)
    killed = []
    monkeypatch.setattr(kaggle_worker.subprocess, 'Popen', lambda *_args, **_kwargs: process)
    monkeypatch.setattr(kaggle_worker, '_signal_training', lambda p, sig: killed.append((p, sig)))
    assert kaggle_worker.run_training(['python', 'train'], {}, 12) == 124
    assert process.waits == expected_waits
    assert len(killed) == signals and all(p is process for p, _ in killed)


def test_no_remaining_budget_refuses_to_start_a_process(monkeypatch):
    def unexpected(*_args, **_kwargs):
        pytest.fail('no process should start')
    monkeypatch.setattr(kaggle_worker.subprocess, 'Popen', unexpected)
    with pytest.raises(RuntimeError, match='No training time'):
        kaggle_worker.run_training(['python', 'train'], {}, 0)
