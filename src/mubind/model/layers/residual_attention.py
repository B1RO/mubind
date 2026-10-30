"""An additional residual layer followed by attention."""

import torch
from torch import nn

from .attention import make_attention
from .residual_block import ResidualBlock1d


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self, channels: int, condition_channels: int, *, time_dim: int,
        attention_mode: str, attention_heads: int,
        bottleneck: bool = False,
    ) -> None:
        super().__init__()
        # Keep initialization order stable for training with the same seed.
        attention = make_attention(
            attention_mode, channels, attention_heads, bottleneck=bottleneck,
        )
        self.residual = ResidualBlock1d(
            channels, channels, time_dim,
            spatial_cond_channels=condition_channels,
        )
        self.attention = attention

    def forward(
        self, features: torch.Tensor, time_condition: torch.Tensor,
        spatial_condition: torch.Tensor,
    ) -> torch.Tensor:
        features = self.residual(features, time_condition, spatial_condition)
        return self.attention(features)
