"""Wrap the training UNet in `torch.compile` on CUDA while keeping the uncompiled `model` for checkpoints."""

from __future__ import annotations

import torch


def compile_training_model(model, config, device):
    """Return a `torch.compile`-wrapped view of `model` when `config.compile` is enabled on CUDA."""
    if not config.compile or device.type != "cuda":
        return model
    from torch._inductor import list_mode_options

    # Preserve eager BF16 rounding boundaries so kernel fusion does not alter
    # gradients in the 3Di structural conditioning network.
    options = dict(list_mode_options(config.compile_mode))
    options["emulate_precision_casts"] = True
    return torch.compile(model, options=options)
