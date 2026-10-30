from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def _match_length(x: torch.Tensor, length: int) -> torch.Tensor:
    if x.shape[-1] > length:
        return x[..., :length]
    if x.shape[-1] < length:
        return F.pad(x, (0, length - x.shape[-1]))
    return x


def _resize_availability_mask(
    availability_mask: torch.Tensor,
    length: int,
) -> torch.Tensor:
    """Resize 3Di-token availability without affecting main U-Net attention."""
    if availability_mask.shape[-1] == length:
        return availability_mask
    return F.interpolate(
        availability_mask[:, None, :].float(),
        size=length,
        mode="nearest",
    )[:, 0].bool()


def _apply_attention(
    attention: nn.Module,
    x: torch.Tensor,
) -> torch.Tensor:
    if isinstance(attention, nn.Identity):
        return x
    return attention(x)
