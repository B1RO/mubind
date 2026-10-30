"""The U-Net bottleneck, including optional attention to the supplied structure."""

import torch
from torch import nn

from ..layers import ThreeDiCrossAttention1d
from ..layers.attention import make_attention
from ..layers.residual_attention import ResidualAttentionBlock
from ..layers.residual_block import ResidualBlock1d


class MiddleBlock(nn.Module):
    def __init__(
        self, channels: int, condition_channels: int, *, time_dim: int,
        attention_mode: str, attention_heads: int, blocks_per_level: int,
    ) -> None:
        super().__init__()
        residual_options = dict(
            time_dim=time_dim, spatial_cond_channels=condition_channels,
        )
        self.residual1 = ResidualBlock1d(channels, channels, **residual_options)
        self.attention = make_attention(
            attention_mode, channels, attention_heads, bottleneck=True,
        )
        self.residual2 = ResidualBlock1d(channels, channels, **residual_options)
        self.extra_blocks = nn.ModuleList([
            ResidualAttentionBlock(
                channels, condition_channels, time_dim=time_dim,
                attention_mode=attention_mode, attention_heads=attention_heads,
                bottleneck=True,
            )
            for _ in range(blocks_per_level - 2)
        ])
        # Attached after backbone initialization to preserve the random seed stream.
        self.structure_attention: ThreeDiCrossAttention1d | None = None

    def forward(
        self, features: torch.Tensor, time_condition: torch.Tensor,
        spatial_condition: torch.Tensor, structure: torch.Tensor,
        structure_available: torch.Tensor,
    ) -> torch.Tensor:
        features = self.residual1(features, time_condition, spatial_condition)
        features = self.attention(features)
        if self.structure_attention is not None:
            # Sequence features are coarse; structure keys retain full resolution.
            features = self.structure_attention(features, structure, structure_available)
        features = self.residual2(features, time_condition, spatial_condition)
        for block in self.extra_blocks:
            features = block(features, time_condition, spatial_condition)
        return features
