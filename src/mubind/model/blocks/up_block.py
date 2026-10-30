"""One decoder stage: combine encoder skips, process features, then upsample."""

import torch
from torch import nn

from .down_block import SkipFeatures
from ..layers.attention import make_attention
from ..layers.residual_attention import ResidualAttentionBlock
from ..layers.residual_block import ResidualBlock1d
from ..layers.utils import _match_length


class UpBlock(nn.Module):
    def __init__(
        self, channels: int, skip_channels: int, condition_channels: int, *,
        time_dim: int, attention_mode: str, attention_heads: int,
        blocks_per_level: int,
        output_length: int | None,
    ) -> None:
        super().__init__()
        residual_options = dict(
            time_dim=time_dim, spatial_cond_channels=condition_channels,
        )
        self.residual1 = ResidualBlock1d(channels + skip_channels, channels, **residual_options)
        self.residual2 = ResidualBlock1d(channels + skip_channels, channels, **residual_options)
        self.attention = make_attention(attention_mode, channels, attention_heads)
        convolution = nn.Conv1d(channels, skip_channels, kernel_size=3, padding=1)
        self.upsample = (
            nn.Sequential(nn.Upsample(size=output_length, mode="nearest"), convolution)
            if output_length is not None else convolution
        )
        self.extra_blocks = nn.ModuleList([
            ResidualAttentionBlock(
                channels, condition_channels, time_dim=time_dim,
                attention_mode=attention_mode, attention_heads=attention_heads,
            )
            for _ in range(blocks_per_level - 2)
        ])

    def forward(
        self, features: torch.Tensor, skip: SkipFeatures,
        time_condition: torch.Tensor, spatial_condition: torch.Tensor,
    ) -> torch.Tensor:
        features = _match_length(features, skip.final_features.shape[-1])
        features = torch.cat([features, skip.final_features], dim=1)
        features = self.residual1(features, time_condition, spatial_condition)
        features = _match_length(features, skip.first_residual.shape[-1])
        features = torch.cat([features, skip.first_residual], dim=1)
        features = self.residual2(features, time_condition, spatial_condition)
        features = self.attention(features)
        for block in self.extra_blocks:
            features = block(features, time_condition, spatial_condition)
        return self.upsample(features)


class UpBlocks(nn.ModuleList):
    """The decoder: run each level, consuming encoder skips from deepest to shallowest."""

    def __init__(
        self, dimensions: list[int], condition_multiplier: int, max_length: int,
        **block_options,
    ) -> None:
        super().__init__()
        levels = list(zip(dimensions[:-1], dimensions[1:]))
        for level in reversed(range(len(levels))):
            skip_channels, channels = levels[level]
            # Encoder level i has length max_length // 2**i; upsample to the level above.
            output_length = max_length // 2 ** (level - 1) if level > 0 else None
            self.append(UpBlock(
                channels, skip_channels, skip_channels * condition_multiplier,
                output_length=output_length, **block_options,
            ))

    def forward(
        self, features: torch.Tensor, skips: list[SkipFeatures],
        time_condition: torch.Tensor, condition_maps: list[torch.Tensor],
    ) -> torch.Tensor:
        for block, skip, spatial_condition in zip(
            self, reversed(skips), reversed(condition_maps),
        ):
            features = block(features, skip, time_condition, spatial_condition)
        return features
