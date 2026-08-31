import torch

from fog.encoding.dims import DIMS
from fog.encoding.entities import ENTITY_RAW_DIM, N_ID_FIELDS
from fog.eval.contextual_roundtrip import pack_exact_current, unpack_exact_current
from fog.models.agent import FogAgent


def test_exact_contextual_control_roundtrips_every_required_input() -> None:
    agent = FogAgent(
        {"encoding": {"entity_hidden": 16}},
        sizes={
            "n_species": 32,
            "n_items": 24,
            "n_abilities": 24,
            "n_moves": 48,
            "n_natures": 25,
            "n_ev_units": 64,
        },
    )
    generator = torch.Generator().manual_seed(20261105)
    ids = torch.zeros(3, 12, N_ID_FIELDS, dtype=torch.long)
    limits = [32, 24, 24, *([48] * 5), 25, *([64] * 6), 24, 24]
    for index, limit in enumerate(limits):
        ids[..., index] = torch.randint(
            0, limit, ids.shape[:2], generator=generator
        )
    batch = {
        "entities": torch.randn(3, 12, ENTITY_RAW_DIM, generator=generator),
        "entity_ids": ids,
        "field": torch.randn(3, DIMS.field_dim, generator=generator),
        "active_idx": torch.tensor([[0, 6], [2, 9], [5, 11]]),
        "legality_mask": torch.randint(
            0, 2, (3, 14), generator=generator, dtype=torch.bool
        ),
    }
    latent = pack_exact_current(batch, agent.backbone.embeds)
    recovered = unpack_exact_current(latent, agent.backbone.embeds)

    assert latent.shape == (3, 2048)
    for key in ("entities", "entity_ids", "field", "active_idx", "legality_mask"):
        torch.testing.assert_close(recovered[key], batch[key], rtol=0, atol=0)
