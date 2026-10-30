from __future__ import annotations

import torch
from torch import nn

from .rms_norm import RMSNorm1d


class SelfAttention1d(nn.Module):
    def __init__(self, channels: int, heads: int, dim_head: int = 32) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        hidden_channels = heads * dim_head
        self.norm = RMSNorm1d(channels)
        self.qkv = nn.Conv1d(channels, hidden_channels * 3, kernel_size=1, bias=False)
        self.out = nn.Conv1d(hidden_channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        bsz, _, length = q.shape
        q = q.view(bsz, self.heads, self.dim_head, length) * self.scale
        k = k.view(bsz, self.heads, self.dim_head, length)
        v = v.view(bsz, self.heads, self.dim_head, length)

        similarity = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        weights = similarity.softmax(dim=-1)
        h = torch.einsum("b h i j, b h d j -> b h i d", weights, v)
        h = h.permute(0, 1, 3, 2).reshape(bsz, self.heads * self.dim_head, length)
        h = self.out(h)
        return x + h
