"""Parameter-free AMAGO attention with the pretrained causal window preserved."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class WindowAttention(nn.Module):
    """AMAGO QKV/cache interface implemented with PyTorch SDPA.

    Window bounds are inclusive, matching FlashAttention's (left, right) convention.
    Cache lengths count preceding tokens; AMAGO updates lengths after all layers.
    FP32 avoids unsupported BF16 operations on T4 and gives a common CPU/CUDA path.
    This changes the attention kernel, never the checkpoint's parameters or topology.
    """

    def __init__(
        self, causal: bool = True, dropout: float = 0.0, window_size: tuple[int, int] = (96, 0)
    ) -> None:
        super().__init__()
        self.causal = causal
        self.dropout = float(dropout)
        self.window_size = tuple(window_size)
        if len(self.window_size) != 2 or any(v < -1 for v in self.window_size):
            raise ValueError("attention window must contain two bounds >= -1")

    def forward(self, qkv, key_cache=None, val_cache=None, cache_seqlens=None):
        if qkv.ndim != 5 or qkv.shape[2] != 3:
            raise ValueError("QKV must have shape [batch, length, 3, heads, dimension]")
        supplied = (key_cache is not None, val_cache is not None, cache_seqlens is not None)
        if any(supplied) and not all(supplied):
            raise ValueError("key cache, value cache and lengths must be supplied together")
        query, key, value = qkv.float().unbind(2)
        batch, length, *_ = query.shape
        if all(supplied):
            if self.training:
                raise ValueError("cache writes are inference-only")
            lengths = cache_seqlens.long()
            if lengths.shape != (batch,) or bool((lengths < 0).any()):
                raise ValueError("invalid cache lengths")
            end = lengths + length
            if bool((end > key_cache.shape[1]).any()):
                raise ValueError("cache capacity exceeded; roll the AMAGO cache first")
            indices = lengths[:, None] + torch.arange(length, device=qkv.device)[None, :]
            rows = torch.arange(batch, device=qkv.device)[:, None]
            key_cache[rows, indices] = key.to(key_cache.dtype)
            val_cache[rows, indices] = value.to(val_cache.dtype)
            total = int(end.max())
            key = key_cache[:, :total].float()
            value = val_cache[:, :total].float()
            # Stale/NaN entries after a reset must not poison masked dot products.
            valid = torch.arange(total, device=qkv.device)[None, :] < end[:, None]
            key = torch.where(valid[:, :, None, None], key, 0.0)
            value = torch.where(valid[:, :, None, None], value, 0.0)
            query_pos = indices
        else:
            total = length
            end = torch.full((batch,), length, device=qkv.device)
            query_pos = torch.arange(length, device=qkv.device)[None, :].expand(batch, -1)
        key_pos = torch.arange(total, device=qkv.device)[None, None, :]
        query_pos = query_pos[:, :, None]
        allowed = key_pos < end[:, None, None]
        if self.causal:
            allowed = allowed & (key_pos <= query_pos)
        left, right = self.window_size
        if left >= 0:
            allowed = allowed & (key_pos >= query_pos - left)
        if right >= 0:
            allowed = allowed & (key_pos <= query_pos + right)
        out = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=allowed[:, None],
            dropout_p=self.dropout if self.training else 0.0,
        )
        return out.transpose(1, 2)
