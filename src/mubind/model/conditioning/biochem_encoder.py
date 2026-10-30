"""Multi-layer biochemical property encoder for the diffusion U-Net.

PropertyConditioningEncoder stretches the global peptide properties (length,
charge, hydrophobicity) across the full 70-residue length and runs a stack of
downsampling layers (70 -> 35 -> 17 length, 32 -> 64 -> 128 channels) to
produce one feature map for each U-Net resolution. Before entering the encoder,
the raw values are standardized and any omitted properties are masked out.
"""

from __future__ import annotations

import torch
from torch import nn

from ._pyramid import _conditioning_downs


class PropertyConditioningEncoder(nn.Module):
    """Expand global biochemical properties into per-resolution feature maps.

      width = sequence length (70 -> 35 -> 17), height = channels (32 -> 64 -> 128)

      properties [B, 3 or 6]  ███
                               │
                               ▼  Linear + GELU + Unflatten
      maps[0]    [B,  32, 70]  ████████████████   -> level 0 (DownBlock 0, UpBlock 2)
                               │
                               ▼  stride-2 down (length 70 -> 35)
      maps[1]    [B,  32, 35]  ████████           -> level 1 (DownBlock 1, UpBlock 1)
                               │
                               ▼  stride-2 down (length 35 -> 17, 32 -> 64 channels)
      maps[2]    [B,  64, 17]  ████               -> level 2 (DownBlock 2, UpBlock 0)
                               ████
                               │
                               ▼  stride-1 conv (length 17 -> 17, 64 -> 128 channels)
      maps[3]    [B, 128, 17]  ████               -> MiddleBlock (bottleneck)
                               ████
                               ████
                               ████
    """

    def __init__(
        self,
        *,
        input_dim: int,
        max_length: int,
        level_channels: list[int],
        bottleneck_channels: int,
        attention_heads: int,
    ) -> None:
        super().__init__()
        if not level_channels:
            raise ValueError("property conditioning requires at least one U-Net level")

        self.initial = nn.Sequential(
            nn.Linear(input_dim, level_channels[0] * max_length),
            nn.GELU(),
            nn.Unflatten(1, (level_channels[0], max_length)),
        )
        self.downs = _conditioning_downs(
            level_channels, bottleneck_channels, attention_heads,
        )

    def forward(self, properties: torch.Tensor) -> list[torch.Tensor]:
        condition = self.initial(properties)
        maps = [condition]
        for block1, block2, attention, resize in self.downs:
            condition = block1(condition)
            condition = block2(condition)
            condition = attention(condition)
            condition = resize(condition)
            maps.append(condition)
        return maps


def _property_statistics(
    property_mean: torch.Tensor | list[float] | tuple[float, ...] | None,
    property_std: torch.Tensor | list[float] | tuple[float, ...] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate and convert [length, charge, hydrophobicity] mean and std to float32 tensors."""
    property_mean_tensor = torch.as_tensor(
        [0.0, 0.0, 0.0] if property_mean is None else property_mean,
        dtype=torch.float32,
    )
    property_std_tensor = torch.as_tensor(
        [1.0, 1.0, 1.0] if property_std is None else property_std,
        dtype=torch.float32,
    )
    if property_mean_tensor.shape != (3,) or property_std_tensor.shape != (3,):
        raise ValueError("property mean and std must each contain three values")
    if not torch.isfinite(property_mean_tensor).all():
        raise ValueError("property mean must be finite")
    if not torch.isfinite(property_std_tensor).all() or (property_std_tensor <= 0).any():
        raise ValueError("property std must be finite and positive")
    return property_mean_tensor, property_std_tensor


def _property_input(
    properties: torch.Tensor,
    property_mean: torch.Tensor,
    property_std: torch.Tensor,
    *,
    condition_mask_enabled: bool,
    condition_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Standardize [B, 3] properties and interleave presence flags into [B, 6] when masking is enabled."""
    standardized_properties = (properties - property_mean) / property_std
    if condition_mask_enabled:
        if condition_mask is None:
            condition_mask = torch.ones_like(properties)
        condition_mask = condition_mask.to(device=properties.device, dtype=properties.dtype)
        condition_present = condition_mask > 0.5
        masked_properties = torch.where(
            condition_present,
            standardized_properties,
            torch.zeros_like(standardized_properties),
        )
        mask_indicator = condition_mask - 0.5
        property_input = torch.stack(
            [masked_properties, mask_indicator],
            dim=-1,
        ).flatten(1)
    else:
        property_input = standardized_properties
    return property_input
