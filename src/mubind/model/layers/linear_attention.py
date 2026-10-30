from __future__ import annotations
import torch
from torch import nn
from .rms_norm import RMSNorm1d
from .utils import _resize_availability_mask

# "Efficient attention" variant from Shen et al. (2018).
class LinearAttention1d(nn.Module):
    def __init__(self, channels: int, heads: int, dim_head: int = 32) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        hidden_channels = heads * dim_head
        self.norm = RMSNorm1d(channels)
        self.qkv = nn.Conv1d(channels, hidden_channels * 3, kernel_size=1, bias=False)
        self.out = nn.Sequential(
            nn.Conv1d(hidden_channels, channels, kernel_size=1),
            RMSNorm1d(channels),
        )

    def forward(
        self,
        x: torch.Tensor,
        availability_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        availability_mask = (
            _resize_availability_mask(availability_mask, x.shape[-1])
            if availability_mask is not None
            else None
        )
        queries, keys, values = self.qkv(self.norm(x)).chunk(3, dim=1)
        batch_size, _, sequence_length = queries.shape
        queries = queries.view(batch_size, self.heads, self.dim_head, sequence_length)
        keys = keys.view(batch_size, self.heads, self.dim_head, sequence_length)
        values = values.view(batch_size, self.heads, self.dim_head, sequence_length)

        if availability_mask is not None:
            expanded_availability_mask = availability_mask[:, None, None, :].to(dtype=queries.dtype)
            keys = keys.masked_fill(expanded_availability_mask == 0, -torch.finfo(keys.dtype).max)
            values = values * expanded_availability_mask

        queries = queries.softmax(dim=-2)
        keys = keys.softmax(dim=-1)
        queries = queries * (self.dim_head**-0.5)
        context = torch.einsum("b h d n, b h e n -> b h d e", keys, values)
        attention_output = torch.einsum("b h d e, b h d n -> b h e n", context, queries)
        attention_output = attention_output.reshape(batch_size, self.heads * self.dim_head, sequence_length)
        attention_output = self.out(attention_output)
        if availability_mask is not None:
            attention_output = attention_output * availability_mask[:, None, :]
        return x + attention_output
