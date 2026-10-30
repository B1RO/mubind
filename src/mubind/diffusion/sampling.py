"""Reverse diffusion sampling loop (DDPM and DDIM) with optional property guidance.

While `schedule.py` defines how to add noise to a clean protein during training, this
module runs the process in reverse to generate new proteins starting from pure noise at
step `999` and walking down to a clean protein at step `0`.

Both samplers use the exact same trained UNet and `DiffusionSchedule`, trading off
generation speed against full 1,000-step stochasticity

- **DDPM** (`sampler="ddpm"`, implemented in `ddpm.py`) walks through all 1,000 steps one
  by one (`999 -> 998 -> ... -> 0`) and injects a tiny pinch of fresh random noise at every
  step. We use it when maximum sample quality and diversity matter more than speed, such as
  in the binder design pipeline (`inference/design.py`) and full checkpoint benchmarks.
- **DDIM** (`sampler="ddim"`, implemented in `ddim.py`) is a shortcut sampler that
  jumps across a smaller number of evenly spaced steps (for example 50 steps instead of
  1,000) and is deterministic by default (`ddim_eta=0.0`). We use it for fast validation
  checks during training and quick CLI generation (`inference/generate.py`).

`sample` is the shared entry point for both samplers. At each step in the reverse loop it

1. Calls the UNet directly, or uses `property_guidance` when `property_guidance_scale != 1.0`,
   and converts the output to a clean guess `pred_x0`.
2. Optionally saves `pred_x0` as `self_cond_x` so the next step can use the previous
   step's guess as an extra hint.
3. Advances `x` toward `pred_x0` using either `ddpm_sample_step` (DDPM) or
   `ddim_sample_step` (DDIM).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace

import torch
from torch import nn

from ..model.inputs import PropertyInput, SeedInput, StructureInput
from .ddim import ddim_sample_step, ddim_timesteps
from .ddpm import ddpm_sample_step
from .objective import model_output_to_x0
from .schedule import DiffusionSchedule


@torch.no_grad()
def sample(
    diffusion: DiffusionSchedule,
    model: nn.Module,
    *,
    structure: StructureInput,
    properties: PropertyInput,
    shape: tuple[int, int, int],
    seed: SeedInput | None = None,
    self_condition: bool = False,
    property_guidance_scale: float = 1.0,
    noise_group_ids: torch.Tensor | None = None,
    sampler: str = "ddpm",
    sampling_steps: int | None = None,
    ddim_eta: float = 0.0,
    on_step: Callable[[int, int], None] | None = None,
) -> torch.Tensor:
    """Generate a batch of clean protein embeddings by walking backward from pure noise.

    Starts from random Gaussian noise `x` of shape `[batch, channels, canvas_length]` at
    the highest timestep (`999`) and repeatedly nudges `x` toward the UNet's clean guess
    `pred_x0` until reaching step `0`.
    """
    _validate_sample_inputs(
        model, shape, structure, properties, seed, property_guidance_scale
    )
    sampler, ddim_eta, sampling_schedule = _resolve_sampling_schedule(
        diffusion, sampler, sampling_steps, ddim_eta
    )
    # Pair each step with the next step we are jumping to (ending at -1 when finished).
    # For 1,000-step DDPM this is (999, 998), (998, 997), ..., (0, -1).
    step_pairs = zip(sampling_schedule, sampling_schedule[1:] + (-1,))
    total_steps = len(sampling_schedule)
    device = structure.tokens.device
    draw_noise = _make_noise_drawer(shape, noise_group_ids, device)

    # 1. Start at step 999 with pure Gaussian noise (and no self-conditioning hint yet).
    self_cond_x = None
    x = draw_noise(torch.get_default_dtype())

    for step_index, (step, next_step) in enumerate(step_pairs, start=1):
        t = torch.full((shape[0],), step, device=device, dtype=torch.long)

        # 2. Ask the UNet to guess the clean protein (pred_x0) from the current noisy x.
        if property_guidance_scale == 1.0:
            prediction = model(
                x, t, structure, properties, seed=seed, self_cond_x=self_cond_x
            )
        else:
            prediction = property_guidance(
                model,
                x,
                t,
                structure=structure,
                properties=properties,
                seed=seed,
                self_cond_x=self_cond_x,
                property_guidance_scale=property_guidance_scale,
            )
        pred_x0 = model_output_to_x0(diffusion, x, t, prediction)

        # 3. If self-conditioning is enabled, pass this step's clean guess as a hint to the next step.
        if self_condition:
            self_cond_x = pred_x0.detach()

        # 4. Nudge x one step cleaner toward pred_x0 (using either DDPM or DDIM).
        if sampler == "ddpm":
            step_noise = None if noise_group_ids is None else draw_noise(x.dtype)
            x = ddpm_sample_step(diffusion, x, t, pred_x0, noise=step_noise)
        else:
            step_noise = (draw_noise(x.dtype) if ddim_eta != 0 and next_step >= 0 else None)
            x = ddim_sample_step(
                diffusion,
                x,
                pred_x0,
                step=step,
                next_step=next_step,
                eta=ddim_eta,
                noise=step_noise,
            )
        if on_step is not None:
            on_step(step_index, total_steps)
    return x


def property_guidance(
    model: nn.Module,
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    *,
    structure: StructureInput,
    properties: PropertyInput,
    seed: SeedInput | None = None,
    self_cond_x: torch.Tensor | None = None,
    property_guidance_scale: float = 1.0,
) -> torch.Tensor:
    """Apply classifier-free property guidance to the UNet's prediction.

    When `property_guidance_scale == 1.0` (the unguided default), this simply calls the
    UNet once. When `property_guidance_scale != 1.0`, it calls the UNet a second time with
    all biochemical properties hidden (`properties.mask` zeroed out, while keeping 3Di
    structure and sequence seed fixed) and pushes the prediction further toward the
    properties using `unconditional + scale * (conditional - unconditional)`.
    """
    _validate_property_guidance(model, properties, property_guidance_scale)
    guided = property_guidance_scale != 1.0

    if not guided:
        # Normal unguided pass (single UNet call).
        return model(
            x_t, timesteps, structure, properties, seed=seed, self_cond_x=self_cond_x
        )
    else:
        # Snapshot RNG state before the conditional pass so that if guidance runs a
        # second pass, any internal dropout in the network uses the exact same mask.
        cpu_rng_state = torch.get_rng_state()
        cuda_devices = [x_t.device.index] if x_t.is_cuda else []
        cuda_rng_state = torch.cuda.get_rng_state(x_t.device) if x_t.is_cuda else None

        # 1. Conditional pass with the caller's property values and mask.
        conditional = model(
            x_t, timesteps, structure, properties, seed=seed, self_cond_x=self_cond_x
        )

        # Skip the second pass if no row in the batch actually has visible properties.
        # in which case property guidance is useless
        guided_rows = properties.mask.ne(0).any(dim=1)
        if not bool(guided_rows.any()):
            return conditional

        # 2. Unconditional pass with properties.mask zeroed out (structure and seed stay fixed).
        with torch.random.fork_rng(devices=cuda_devices):
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, x_t.device)
            unconditional = model(
                x_t,
                timesteps,
                structure,
                replace(properties, mask=torch.zeros_like(properties.mask)),
                seed=seed,
                self_cond_x=self_cond_x,
            )

        # 3. Blend conditional and unconditional predictions on rows with active properties.
        prediction = unconditional + property_guidance_scale * (
            conditional - unconditional
        )
        return torch.where(guided_rows[:, None, None], prediction, conditional)


def _validate_property_guidance(model, properties, scale):
    """Check that `property_guidance_scale` is valid and the model was trained with a property mask."""
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("property guidance scale must be finite and non-negative")
    if scale != 1.0 and (
        properties.mask is None
        or not getattr(getattr(model, "property_config", None), "mask_enabled", False)
    ):
        raise ValueError(
            "property guidance requires a model trained with property dropout and a property mask"
        )


def _validate_sample_inputs(
    model: nn.Module,
    shape: tuple[int, int, int],
    structure: StructureInput,
    properties: PropertyInput,
    seed: SeedInput | None,
    property_guidance_scale: float,
) -> None:
    """Check that batch size and canvas length match across `shape`, `structure`, `properties`, and `seed`."""
    if len(shape) != 3 or any(not isinstance(size, int) or size < 1 for size in shape):
        raise ValueError("shape must contain positive batch, channel, and canvas sizes")
    if structure.tokens.shape != (shape[0], shape[2]):
        raise ValueError("structure tokens must match the requested batch and canvas")
    if properties.values.shape != (shape[0], 3):
        raise ValueError("properties must have shape [batch, 3]")
    if seed is not None and (
        seed.x.shape != shape or seed.edit_mask.shape != structure.tokens.shape
    ):
        raise ValueError("seed tensors must match the requested batch and canvas")
    _validate_property_guidance(model, properties, property_guidance_scale)


def _resolve_sampling_schedule(
    diffusion: DiffusionSchedule,
    sampler: str,
    sampling_steps: int | None,
    ddim_eta: float,
) -> tuple[str, float, tuple[int, ...]]:
    """Validate sampler options and return the descending tuple of timesteps to visit."""
    ddim_eta = float(ddim_eta)
    sampler = str(sampler).lower()
    if sampler not in {"ddpm", "ddim"}:
        raise ValueError("sampler must be ddpm or ddim")
    if sampler == "ddpm":
        if sampling_steps not in {None, diffusion.timesteps}:
            raise ValueError(
                "DDPM uses every diffusion timestep; choose sampler='ddim' "
                "to reduce sampling_steps"
            )
        if ddim_eta != 0.0:
            raise ValueError("ddim_eta only applies when sampler='ddim'")
        return sampler, ddim_eta, tuple(reversed(range(diffusion.timesteps)))
    schedule = ddim_timesteps(
        diffusion, diffusion.timesteps if sampling_steps is None else sampling_steps
    )
    if not 0 <= ddim_eta <= 1:
        raise ValueError("DDIM eta must be between 0 and 1")
    return sampler, ddim_eta, schedule


def _make_noise_drawer(
    shape: tuple[int, int, int],
    noise_group_ids: torch.Tensor | None,
    device: torch.device,
) -> Callable[[torch.dtype], torch.Tensor]:
    """Return a function that draws random Gaussian noise of `shape`.

    When `noise_group_ids` is provided, batch rows with the same group ID receive the
    exact same random noise tensor (used in evaluation to compare different conditioning
    inputs from identical starting noise).
    """
    inverse = None
    noise_shape = shape
    if noise_group_ids is not None:
        if noise_group_ids.shape != (shape[0],):
            raise ValueError("noise_group_ids must contain one value per sample")
        groups, inverse = torch.unique(
            noise_group_ids.to(device=device, dtype=torch.long),
            sorted=True,
            return_inverse=True,
        )
        noise_shape = (groups.numel(), *shape[1:])

    def draw_noise(dtype: torch.dtype) -> torch.Tensor:
        noise = torch.randn(noise_shape, device=device, dtype=dtype)
        return noise if inverse is None else noise.index_select(0, inverse)

    return draw_noise
