"""Exponential moving average (`ModelEMA`) weights for diffusion training.

Raw UNet weights jitter from batch to batch during stochastic gradient descent. `ModelEMA`
keeps a non-trainable shadow copy (`self.module`) updated after every optimizer step via
`ema = decay * ema + (1 - decay) * current` (`decay = 0.999` by default). Validation and
reverse sampling always use this smoothed `ema.module` copy.
"""

from __future__ import annotations

import copy

import torch
from torch import nn


class ModelEMA:
    """Maintain a non-trainable, smoothed moving-average copy of a model's parameters."""

    def __init__(
        self,
        model: nn.Module,
        *,
        decay: float = 0.999,
        warmup: bool = True,
    ) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = float(decay)
        self.warmup = bool(warmup)
        self.num_updates = 0
        self.module = copy.deepcopy(model).eval()
        self.module.requires_grad_(False)

    @property
    def effective_decay(self) -> float:
        """Return the current EMA decay factor (using a faster warmup ramp at early steps when `warmup=True`)."""
        if not self.warmup:
            return self.decay
        # Track rapidly at the beginning of training, then approach the configured decay (0.999).
        adaptive_decay = (1.0 + self.num_updates) / (10.0 + self.num_updates)
        return min(self.decay, adaptive_decay)

    @torch.no_grad()
    def update(self, model: nn.Module) -> float:
        """Blend `self.module` parameters toward `model` parameters using `effective_decay` and copy buffers."""
        self.num_updates += 1
        decay = self.effective_decay

        ema_parameters = list(self.module.parameters())
        current_parameters = list(model.parameters())
        if len(ema_parameters) != len(current_parameters):
            raise ValueError("EMA and training models have different parameter counts")
        torch._foreach_lerp_(
            ema_parameters,
            [parameter.detach() for parameter in current_parameters],
            1.0 - decay,
        )

        ema_buffers = dict(self.module.named_buffers())
        current_buffers = dict(model.named_buffers())
        if ema_buffers.keys() != current_buffers.keys():
            raise ValueError("EMA and training models have different buffers")
        for name, ema_buffer in ema_buffers.items():
            ema_buffer.copy_(current_buffers[name].detach())
        return decay

    def metadata(self) -> dict[str, int | float | bool]:
        """Return serializable EMA metadata (`decay`, `warmup`, `num_updates`, `effective_decay`) for checkpoints."""
        return {
            "decay": self.decay,
            "warmup": self.warmup,
            "num_updates": self.num_updates,
            "effective_decay": self.effective_decay,
        }

    def load_checkpoint_state(
        self,
        model_state: dict[str, torch.Tensor],
        metadata: dict | None = None,
    ) -> None:
        """Restore `self.module` weights and `num_updates` counter from a saved checkpoint."""
        self.module.load_state_dict(model_state)
        if metadata is not None:
            self.num_updates = int(metadata.get("num_updates", 0))
        self.module.eval()
