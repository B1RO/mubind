"""Single diffusion training step and reconstruction loss (`compute_loss`).

Given a batch of proteins already prepared by `training/tasks/`, `compute_loss` runs one
forward pass and returns the scalar MSE loss

1. Sample a random timestep `times` in `0..999` for each protein in the batch (optionally
   replacing a fraction of them with extra-noisy steps when `high_noise_mixture` is enabled).
2. Corrupt the clean protein embeddings `clean` with Gaussian noise using `diffusion.add_noise`.
3. Optionally run a detached `torch.no_grad()` pass (`self_condition`) on a subset of the
   batch so the UNet gets an initial clean guess `hint` as an extra input.
4. Run the trainable UNet forward pass `model(noisy, times, self_cond_x=hint, ...)` and
   score its prediction with `residue_pad_balanced_mse_loss`.
"""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch

from ..diffusion import (
    model_output_to_x0,
    residue_pad_balanced_mse_loss,
    training_target,
)
from ..model import PropertyInput, SeedInput, StructureInput
from .config import TrainingConfig


def compute_loss(model, diffusion, batch, config: TrainingConfig, *, training: bool):
    """Run one diffusion forward pass on `batch` and return the scalar reconstruction loss."""
    clean = batch["x"]

    # 1. Pick a random timestep in 0..999 for each protein in the batch.
    times = diffusion.sample_timesteps(len(clean), clean.device)
    sampling = config.timestep_sampling
    if (
        training
        and sampling.mode == "high_noise_mixture"
        and sampling.high_noise_probability > 0
    ):
        # Optionally replace some timesteps with extra-noisy steps near 999 so the
        # network practices harder, high-noise steps more often during training.
        first = min(
            math.floor(sampling.high_noise_start * diffusion.timesteps),
            diffusion.timesteps - 1,
        )
        emphasized = torch.randint(
            first, diffusion.timesteps, times.shape, device=times.device
        )
        choose = (
            torch.rand(len(clean), device=times.device)
            < sampling.high_noise_probability
        )
        times = torch.where(choose, emphasized, times)

    # 2. Mix the clean protein embeddings with Gaussian noise at the chosen timesteps.
    noisy, noise = diffusion.add_noise(clean, times)

    # 3. Run the UNet (with an optional no-grad self-conditioning pass first).
    with autocast_context(config, clean.device):
        hint = None
        if config.self_condition.enabled:
            hint = self_condition(
                model,
                diffusion,
                noisy,
                times,
                batch,
                probability=config.self_condition.probability if training else 1.0,
            )
        prediction = model(noisy, times, self_cond_x=hint, **model_inputs(batch))

    # 4. Score the prediction against the clean target in FP32.
    target = training_target(diffusion, clean, noise, times)
    loss = residue_pad_balanced_mse_loss(
        diffusion,
        prediction.float(),
        target.float(),
        batch["edit_mask"],
        batch["residue_mask"],
        timesteps=times,
        pad_weighting=config.reconstruction.pad_weighting,
        pad_weight=config.reconstruction.pad_weight,
        example_weights=batch["target_length"]
        if config.reconstruction.example_weighting == "target_length"
        else None,
    )
    return loss


@torch.no_grad()
def self_condition(model, diffusion, noisy, timesteps, batch, *, probability):
    """Run a no-grad pass on a random subset of batch rows to produce a clean `x0` hint.

    During reverse sampling (`sampling.py`), each step passes its clean guess `pred_x0`
    as `self_cond_x` into the next step. To teach the UNet how to use that hint during
    single-step training, we pick a fraction of rows (`probability = 0.5` in training,
    `1.0` in validation), run the UNet once without gradients to get a preliminary clean
    guess, and place that guess into a zero-initialized tensor for the second, trainable pass.
    """
    rows = sample_self_condition_rows(len(noisy), probability, noisy.device)
    if rows.numel() == 0:
        return None
    # Disable autocast weight caching so this no-grad pass does not cache detached
    # fp16/bf16 weights that the subsequent trainable pass needs gradients through.
    with torch.autocast(
        device_type=noisy.device.type,
        dtype=torch.get_autocast_dtype(noisy.device.type),
        enabled=torch.is_autocast_enabled(noisy.device.type),
        cache_enabled=False,
    ):
        prediction = model(
            noisy[rows], timesteps[rows], **model_inputs(batch, rows)
        )
    clean = model_output_to_x0(diffusion, noisy[rows], timesteps[rows], prediction)
    return clean.new_zeros(noisy.shape).index_copy_(0, rows, clean.detach())


def sample_self_condition_rows(batch_size, probability, device):
    """Pick a random subset of row indices of size `batch_size * probability` (rounded stochastically)."""
    if not 0 <= probability <= 1:
        raise ValueError("self-conditioning probability must be in [0, 1]")
    expected = batch_size * probability
    count = math.floor(expected)
    if expected > count and torch.rand(()).item() < expected - count:
        count += 1
    if count == batch_size:
        return torch.arange(batch_size, device=device)
    if count == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.randperm(batch_size, device=device)[:count]


def model_inputs(batch, rows=slice(None)):
    """Pack batch dictionary tensors into `StructureInput`, `PropertyInput`, and `SeedInput`."""
    def optional(name):
        return None if batch.get(name) is None else batch[name][rows]

    return {
        "structure": StructureInput(
            tokens=batch["seed_3di"][rows],
            hidden_mask=optional("threedi_conditioning_hidden_mask"),
        ),
        "properties": PropertyInput(batch["c"][rows], mask=optional("condition_mask")),
        "seed": SeedInput(batch["seed_x"][rows], batch["edit_mask"][rows]),
    }


def autocast_context(config: TrainingConfig, device: torch.device):
    """Return a `torch.autocast` context for `bf16` or `fp16` precision, or `nullcontext` for `fp32`."""
    if config.precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if config.precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def batch_to_device(batch, device):
    """Move all tensor values in `batch` to `device`."""
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }
