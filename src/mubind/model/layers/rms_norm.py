from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class RMSNorm1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1) * self.scale * (x.shape[1] ** 0.5)
