from fog.training.replay_overfit import _passes_identifiable_gate


def _evaluation(*, move_probability: float = 0.99) -> dict:
    def stats(probability: float = 0.99) -> dict:
        return {
            "count": 64,
            "mean_true_probability": probability,
            "confident98_rate": 0.97,
        }

    return {
        "identifiable_leave_one_out": stats(),
        "identifiable_by_family": {
            "item": stats(),
            "ability": stats(),
            "move": stats(move_probability),
            "tera": stats(),
        },
    }


def test_identifiable_gate_requires_every_observable_family():
    assert _passes_identifiable_gate(_evaluation())
    assert not _passes_identifiable_gate(_evaluation(move_probability=0.95))


def test_identifiable_gate_requires_family_coverage():
    evaluation = _evaluation()
    del evaluation["identifiable_by_family"]["tera"]
    assert not _passes_identifiable_gate(evaluation)
