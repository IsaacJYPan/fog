import math

import pytest
import torch

from fog.pretrained.attention import WindowAttention


def reference(qkv, left):
    q, k, v = qkv.unbind(2)
    scores = torch.einsum("blhd,bshd->bhls", q, k) / math.sqrt(q.shape[-1])
    positions = torch.arange(q.shape[1])
    legal = positions[:, None] >= positions[None, :]
    legal &= positions[:, None] - positions[None, :] <= left
    probabilities = scores.masked_fill(~legal, -torch.inf).softmax(-1)
    return torch.einsum("bhls,bshd->blhd", probabilities, v)


def test_window_matches_dense_reference_and_cannot_see_future():
    torch.manual_seed(3)
    qkv = torch.randn(2, 103, 3, 3, 8)
    attention = WindowAttention().eval()
    result = attention(qkv)
    torch.testing.assert_close(result, reference(qkv, 96), atol=1e-6, rtol=1e-5)
    changed = qkv.clone()
    changed[:, 102] = 100
    torch.testing.assert_close(attention(changed)[:, :102], result[:, :102])
    changed = qkv.clone()
    changed[:, :6] = 100
    torch.testing.assert_close(attention(changed)[:, 102], result[:, 102])


def test_cached_sequence_matches_full_sequence_and_resets_nan_cache():
    torch.manual_seed(4)
    qkv = torch.randn(2, 101, 3, 2, 7)
    attention = WindowAttention().eval()
    keys = torch.full((2, 128, 2, 7), torch.nan)
    values = torch.full_like(keys, torch.nan)
    cached = []
    for t in range(qkv.shape[1]):
        cached.append(attention(qkv[:, t : t + 1], keys, values, torch.full((2,), t)))
    torch.testing.assert_close(torch.cat(cached, 1), attention(qkv), atol=1e-6, rtol=1e-5)
    # The first batch has been reset; another retains a much longer history.
    lengths = torch.tensor([0, 101])
    result = attention(qkv[:, :1], keys, values, lengths)
    torch.testing.assert_close(result[:1], attention(qkv[:1, :1]))
    assert torch.isfinite(result).all()


def test_cache_partial_arguments_and_overflow_fail_explicitly():
    attention = WindowAttention().eval()
    qkv = torch.randn(1, 1, 3, 1, 2)
    with pytest.raises(ValueError, match="together"):
        attention(qkv, key_cache=torch.zeros(1, 1, 1, 2))
    with pytest.raises(ValueError, match="capacity"):
        attention(qkv, torch.zeros(1, 1, 1, 2), torch.zeros(1, 1, 1, 2), torch.tensor([1]))
