import torch

from fog.scripts.diagnose_initial_value_prior_direct import DirectInitialPrior


def test_direct_prior_is_permutation_symmetric_within_each_team() -> None:
    torch.manual_seed(1)
    model = DirectInitialPrior(4, 3, [7, 9])
    entities = torch.randn(2, 12, 4)
    identities = torch.stack(
        [torch.randint(0, 7, (2, 12)), torch.randint(0, 9, (2, 12))], dim=-1
    )
    field = torch.randn(2, 3)
    expected = model(entities, identities, field)
    permutation = torch.tensor([2, 0, 5, 1, 4, 3, 9, 7, 11, 8, 6, 10])
    actual = model(entities[:, permutation], identities[:, permutation], field)
    torch.testing.assert_close(actual, expected)
