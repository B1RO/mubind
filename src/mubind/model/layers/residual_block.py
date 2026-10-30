from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .utils import _group_count, _match_length


class ResidualBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int | None,
        *,
        spatial_cond_channels: int = 0,
    ) -> None:
        super().__init__()
        conditioned_channels = in_channels + spatial_cond_channels
        self.spatial_cond_channels = spatial_cond_channels
        self.conv1 = nn.Conv1d(conditioned_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.time_condition = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_dim, out_channels * 2),
            )
            if time_dim is not None
            else None
        )
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(
        self,
        x: torch.Tensor,
        time_condition: torch.Tensor | None = None,
        spatial_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.spatial_cond_channels:
            spatial_condition = _match_length(spatial_condition, x.shape[-1])
            block_input = torch.cat([x, spatial_condition], dim=1)
        else:
            block_input = x

        h = self.norm1(self.conv1(block_input))
        if self.time_condition is not None:
            scale, shift = self.time_condition(time_condition).chunk(2, dim=1)
            h = h * (1 + scale[..., None]) + shift[..., None]
        h = F.silu(h)
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)
