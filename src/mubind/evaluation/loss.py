"""Repeatable denoising validation with isolated random state."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from itertools import islice

import torch


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    batches: Iterable,
    loss_fn: Callable,
    *,
    seed: int,
    max_batches: int,
) -> dict[str, float | int]:
    """loss_fn(batch, task_generator) returns a mean loss and reduction weight."""
    if max_batches < 1:
        raise ValueError("max_batches must be positive")
    devices = (
        list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    )
    was_training = model.training
    total = weight_sum = 0.0
    count = 0
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            generator = torch.Generator().manual_seed(seed)
            model.eval()
            for batch in islice(batches, max_batches):
                loss, weight = loss_fn(batch, generator)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non-finite validation loss")
                total += float(loss) * float(weight)
                weight_sum += float(weight)
                count += 1
    finally:
        model.train(was_training)
    if weight_sum <= 0:
        raise ValueError("Validation has no examples")
    return {"loss": total / weight_sum, "batches": count}
