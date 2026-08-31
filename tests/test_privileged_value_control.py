import torch

from fog.scripts.diagnose_privileged_value_control import (
    privileged_value_batch,
    select_value_parameters,
)


def test_privileged_batch_replaces_reveals_without_mutating_actor_batch():
    revealed = torch.zeros((2, 6, 15), dtype=torch.long)
    labels = {
        "species": torch.ones((2, 6), dtype=torch.long),
        "item": torch.full((2, 6), 2, dtype=torch.long),
        "ability": torch.full((2, 6), 3, dtype=torch.long),
        "move": torch.full((2, 6, 4), 4, dtype=torch.long),
        "nature": torch.full((2, 6), 5, dtype=torch.long),
        "ev": torch.full((2, 6, 6), 6, dtype=torch.long),
        "tera": torch.full((2, 6), 7, dtype=torch.long),
    }
    entity_ids = torch.zeros((2, 12, 17), dtype=torch.long)
    batch = {
        "opp_revealed": {"tokens": revealed, "masks": {}},
        "belief_targets": labels,
        "entity_ids": entity_ids,
    }
    privileged = privileged_value_batch(batch)
    assert privileged is not batch
    assert torch.equal(batch["opp_revealed"]["tokens"], revealed)
    assert torch.all(privileged["opp_revealed"]["tokens"][..., 0] == 1)
    assert torch.all(privileged["opp_revealed"]["tokens"][..., 1] == 2)
    assert torch.equal(batch["entity_ids"], entity_ids)
    assert torch.all(privileged["entity_ids"][:, 6:12, 0] == 1)
    assert torch.all(privileged["entity_ids"][:, 6:12, 1] == 2)
    assert torch.all(privileged["entity_ids"][:, 6:12, 2] == 3)
    assert torch.all(privileged["entity_ids"][:, 6:12, 3:7] == 4)
    assert torch.all(privileged["entity_ids"][:, 6:12, 8] == 5)
    assert torch.all(privileged["entity_ids"][:, 6:12, 9:15] == 6)


def test_value_scope_freezes_everything_else():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.policy_value = torch.nn.Module()
            self.policy_value.value = torch.nn.Linear(3, 1)
            self.policy_value.policy = torch.nn.Linear(3, 2)
            self.shared_to_value = torch.nn.Linear(3, 3)

    model = Tiny()
    selected = select_value_parameters(model)  # type: ignore[arg-type]
    assert selected
    assert all(
        parameter.requires_grad == name.startswith(("policy_value.value.", "shared_to_value."))
        for name, parameter in model.named_parameters()
    )


def test_value_interface_scope_option_includes_shared_latent_only():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.policy_value = torch.nn.Module()
            self.policy_value.value = torch.nn.Linear(3, 1)
            self.policy_value.policy = torch.nn.Linear(3, 2)
            self.shared_to_value = torch.nn.Linear(3, 3)
            self.shared_latent = torch.nn.Linear(3, 3)
            self.belief = torch.nn.Linear(3, 3)

    model = Tiny()
    selected = select_value_parameters(  # type: ignore[arg-type]
        model, adapt_shared_latent=True
    )
    assert selected
    expected = ("policy_value.value.", "shared_to_value.", "shared_latent.")
    assert all(
        parameter.requires_grad == name.startswith(expected)
        for name, parameter in model.named_parameters()
    )
