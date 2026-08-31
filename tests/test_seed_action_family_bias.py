from collections import OrderedDict

import pytest
import torch

from fog.scripts.seed_action_family_bias import POLICY_BIAS_KEY, seed_tera_bias


def test_seed_tera_bias_changes_only_slots_four_through_seven() -> None:
    original = OrderedDict(
        [("other", torch.arange(3.0)), (POLICY_BIAS_KEY, torch.arange(14.0))]
    )
    seeded = seed_tera_bias(original, 6.0)

    assert torch.equal(seeded["other"], original["other"])
    assert torch.equal(seeded[POLICY_BIAS_KEY][:4], original[POLICY_BIAS_KEY][:4])
    assert torch.equal(seeded[POLICY_BIAS_KEY][4:8], original[POLICY_BIAS_KEY][4:8] + 6.0)
    assert torch.equal(seeded[POLICY_BIAS_KEY][8:], original[POLICY_BIAS_KEY][8:])
    assert seeded[POLICY_BIAS_KEY].data_ptr() != original[POLICY_BIAS_KEY].data_ptr()


@pytest.mark.parametrize("amount", [0.0, -1.0, float("nan"), float("inf")])
def test_seed_tera_bias_rejects_invalid_amount(amount: float) -> None:
    state = {POLICY_BIAS_KEY: torch.zeros(14)}
    with pytest.raises(ValueError):
        seed_tera_bias(state, amount)
