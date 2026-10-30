"""One encoder stage: process features, save skips, then downsample."""

from typing import NamedTuple

import torch
from torch import nn

from ..layers.attention import make_attention
from ..layers.residual_attention import ResidualAttentionBlock
from ..layers.residual_block import ResidualBlock1d


class SkipFeatures(NamedTuple):
    """Features saved before downsampling for the matching decoder stage."""

    first_residual: torch.Tensor
    final_features: torch.Tensor


class DownBlock(nn.Module):
    def __init__(
        self, channels: int, next_channels: int, condition_channels: int, *,
        time_dim: int, attention_mode: str, attention_heads: int,
        blocks_per_level: int, downsample: bool,
    ) -> None:
        super().__init__()
        residual_options = dict(
            time_dim=time_dim, spatial_cond_channels=condition_channels,
        )
        self.residual1 = ResidualBlock1d(channels, channels, **residual_options)
        self.residual2 = ResidualBlock1d(channels, channels, **residual_options)
        self.attention = make_attention(attention_mode, channels, attention_heads)
        # The final encoder stage changes channels but keeps its sequence length.
        self.downsample = nn.Conv1d(
            channels, next_channels, kernel_size=4 if downsample else 3,
            stride=2 if downsample else 1, padding=1,
        )
        self.extra_blocks = nn.ModuleList([
            ResidualAttentionBlock(
                channels, condition_channels, time_dim=time_dim,
                attention_mode=attention_mode, attention_heads=attention_heads,
            )
            for _ in range(blocks_per_level - 2)
        ])

    def forward(
        self, features: torch.Tensor, time_condition: torch.Tensor,
        spatial_condition: torch.Tensor,
    ) -> tuple[torch.Tensor, SkipFeatures]:
        features = self.residual1(features, time_condition, spatial_condition)
        first_residual = features
        features = self.residual2(features, time_condition, spatial_condition)
        features = self.attention(features)
        for block in self.extra_blocks:
            features = block(features, time_condition, spatial_condition)
        skip = SkipFeatures(first_residual, features)
        return self.downsample(features), skip


class DownBlocks(nn.ModuleList):
    """The encoder: run each level in order and collect its skip features."""

    def __init__(
        self, dimensions: list[int], condition_multiplier: int, **block_options,
    ) -> None:
        super().__init__()
        levels = list(zip(dimensions[:-1], dimensions[1:]))
        for level, (channels, next_channels) in enumerate(levels):
            self.append(DownBlock(
                channels, next_channels, channels * condition_multiplier,
                downsample=level < len(levels) - 1, **block_options,
            ))

    def forward(
        self, features: torch.Tensor, time_condition: torch.Tensor,
        condition_maps: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[SkipFeatures]]:
        skips = []
        for block, spatial_condition in zip(self, condition_maps):
            features, skip = block(features, time_condition, spatial_condition)
            skips.append(skip)
        return features, skips
