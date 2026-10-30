"""Residual cross-attention from sequence features to 3Di structural context."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ThreeDiCrossAttention1d(nn.Module):
    """Residual attention from denoising features to structural context."""

    def __init__(self, channels: int, context_channels: int, heads: int = 4,
                 dim_head: int = 32) -> None:
        super().__init__()
        if heads < 1 or dim_head < 1:
            raise ValueError("3Di cross-attention heads and head dimension must be positive")
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        self.query_norm = nn.LayerNorm(channels)
        self.context_norm = nn.LayerNorm(context_channels)
        self.query = nn.Linear(channels, inner_dim, bias=False)
        self.key_value = nn.Linear(context_channels, 2 * inner_dim, bias=False)
        self.output = nn.Linear(inner_dim, channels)
        # Start as an exact residual identity. The output projection learns
        # first; gradients reach the query/key/value projections afterward.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor,
                availability_mask: torch.Tensor) -> torch.Tensor:
        """Attend to the [B,D,S] context and [B,S] mask prepared by the U-Net."""
        available = availability_mask.bool()
        has_context = available.any(dim=-1)
        safe_available = available.clone()
        safe_available[:, 0] |= ~has_context
        tokens = context.transpose(1, 2).masked_fill(~available[..., None], 0)
        q = self.query(self.query_norm(x.transpose(1, 2)))
        k, v = self.key_value(self.context_norm(tokens)).chunk(2, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(tensor.shape[0], tensor.shape[1],
                                  self.heads, self.dim_head).transpose(1, 2)

        attended = F.scaled_dot_product_attention(
            split_heads(q), split_heads(k), split_heads(v),
            attn_mask=safe_available[:, None, None, :],
            dropout_p=0.0,
        )
        attended = attended.transpose(1, 2).reshape(x.shape[0], x.shape[-1], -1)
        update = self.output(attended).transpose(1, 2)
        # All-missing structure must remain an exact identity even after the
        # output bias learns; the temporary key only prevents all-masked NaNs.
        update = update.masked_fill(~has_context[:, None, None], 0)
        return x + update
