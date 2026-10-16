import pytest
import torch
from torch import nn

from fog.pretrained.attention import WindowAttention
from fog.pretrained.latent_cache import tensor_digest, windows
from fog.pretrained.retention import configure_upper_training, upper_forward, upper_modules


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_layer = nn.Module()
        self.attention_layer.self_attention = WindowAttention()
        self.projection = nn.Linear(8, 24)
        self.register_buffer("training_calls", torch.tensor(0))

    def forward(self, value, pos_idxs=None):
        if self.training:
            self.training_calls.add_(1)
        qkv = self.projection(value).reshape(*value.shape[:2], 3, 2, 4)
        return value + self.attention_layer.self_attention(qkv).reshape_as(value)


def native():
    m = nn.Module()
    m.actor = nn.Linear(8, 3)
    m.traj_encoder = nn.Module()
    m.traj_encoder.tformer = nn.Module()
    m.traj_encoder.tformer.layers = nn.ModuleList([Block() for _ in range(10)])
    m.traj_encoder.tformer.norm = nn.LayerNorm(8)
    m.traj_encoder.tformer.use_rope = False
    return m


def test_last_block_windows_match_full_history_and_right_padding_is_causal():
    torch.manual_seed(1)
    m = native().eval()
    record = {"features": torch.randn(150, 8), "positions": torch.arange(150)}
    indices = [0, 4, 96, 97, 149]
    features, positions, lengths = windows(record, indices)
    actual = upper_forward(m, features, positions, lengths)
    block, norm = upper_modules(m)
    expected = norm(block(record["features"][None]))[0, indices]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    # Padded entries are future entries relative to the selected valid output.
    for i, length in enumerate(lengths.tolist()):
        features[i, length:] = 999999
    torch.testing.assert_close(upper_forward(m, features, positions, lengths), expected, atol=1e-6, rtol=1e-5)


def test_only_final_block_and_norm_can_change_and_eval_buffers_are_preserved():
    m = native()
    parameters = configure_upper_training(m, adapt=True)
    assert set(map(id, parameters)) == {id(p) for n, p in m.named_parameters() if n.startswith(("traj_encoder.tformer.layers.9.", "traj_encoder.tformer.norm."))}
    before = tensor_digest({n: v for n, v in m.state_dict().items() if "layers.9." not in n and "tformer.norm." not in n})
    x = torch.randn(2, 7, 8)
    loss = upper_forward(m, x, torch.arange(7).expand(2, -1), torch.tensor([7, 4])).square().mean()
    loss.backward()
    torch.optim.Adam(parameters, lr=0.001).step()
    assert not any(module.training for module in m.modules())
    assert m.traj_encoder.tformer.layers[-1].training_calls == 0
    assert before == tensor_digest({n: v for n, v in m.state_dict().items() if "layers.9." not in n and "tformer.norm." not in n})


def test_cache_boundary_and_indices_cannot_silently_change():
    m = native()
    m.traj_encoder.tformer.layers = m.traj_encoder.tformer.layers[:-1]
    with pytest.raises(ValueError, match="ten"):
        upper_modules(m)
    with pytest.raises(ValueError, match="actual"):
        windows({"features": torch.zeros(2, 8)}, [-1])
    m = native()
    m.traj_encoder.tformer.layers[-1].attention_layer.self_attention.window_size = (3, 0)
    with pytest.raises(ValueError, match="96-step"):
        upper_modules(m)


def test_digest_detects_frozen_buffer_changes():
    m = native()
    before = tensor_digest(m.state_dict())
    m.traj_encoder.tformer.layers[0].training_calls.add_(1)
    assert tensor_digest(m.state_dict()) != before
