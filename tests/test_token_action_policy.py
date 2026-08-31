import torch

from fog.models.agent import FogAgent


def test_token_action_context_maps_bench_in_team_order_minus_active() -> None:
    cfg = {
        "model": {"token_action_policy": True},
        "encoding": {"entity_hidden": 8, "state_dim": 16, "dyn_dim": 16},
    }
    agent = FogAgent(cfg, sizes={
        "n_species": 10, "n_items": 10, "n_abilities": 10, "n_moves": 20,
        "n_natures": 25, "n_ev_units": 64,
    })
    contexts = type("Contexts", (), {})()
    contexts.private_tokens = torch.arange(6 * 8).reshape(1, 6, 8).float()
    contexts.public_state_tokens = torch.arange(12 * 8).reshape(1, 12, 8).float()
    batch = {
        "active_idx": torch.tensor([[2, 8]]),
        "entity_ids": torch.zeros(1, 12, 17, dtype=torch.long),
    }
    action = agent.action_context(contexts, batch)
    assert action is not None
    assert torch.equal(action[0, 8, :8], contexts.private_tokens[0, 0])
    assert torch.equal(action[0, 9, :8], contexts.private_tokens[0, 1])
    assert torch.equal(action[0, 10, :8], contexts.private_tokens[0, 3])
    assert torch.equal(action[0, 12, :8], contexts.private_tokens[0, 5])
    assert torch.equal(action[0, 0, 8:16], contexts.public_state_tokens[0, 8])
