from fog.training.simple_clone_control import _unique_parameters, action_family


def test_clone_action_family_partition() -> None:
    assert action_family(0) == "move"
    assert action_family(4) == "tera"
    assert action_family(8) == "switch"
    assert action_family(13) == "forced"


def test_unique_parameters_deduplicates_shared_modules() -> None:
    import torch

    layer = torch.nn.Linear(3, 2)
    assert _unique_parameters([layer, layer]) == list(layer.parameters())
