"""DDIM reverse steps and timestep selection for fast diffusion sampling.

While DDPM (`ddpm.py`) must walk through all 1,000 steps one by one (`999 -> 998 -> ... -> 0`),
DDIM can jump across a much smaller list of steps (for example 50 steps `999 -> 979 -> ... -> 0`
built by `ddim_timesteps`), making generation roughly 20x faster.
"""

from __future__ import annotations

import torch

from .schedule import DiffusionSchedule


@torch.no_grad()
def ddim_sample_step(
    diffusion: DiffusionSchedule,
    x_t: torch.Tensor,
    pred_x0: torch.Tensor,
    *,
    step: int,
    next_step: int,
    eta: float = 0.0,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Jump from `step` directly to a cleaner `next_step` by unmixing and remixing `x_t`.

    Suppose at step 500, the schedule says a noisy protein is 0.6 clean protein and 0.8 noise

        x_500 = 0.6 * pred_x0 + 0.8 * noise

    Given `x_500` and the UNet's clean estimate `pred_x0`, we solve that equation to isolate `noise`

        noise = (x_500 - 0.6 * pred_x0) / 0.8

    To jump straight to step 400, where the schedule wants more clean protein (0.8) and less
    noise (0.6), we remix the exact same `pred_x0` and `noise` with step 400's weights

        x_400 = 0.8 * pred_x0 + 0.6 * noise

    That unmix-and-remix is essentially all the code below does (with optional `eta > 0`
    support to swap part of `pred_noise` for fresh random noise during the jump).
    """
    step = int(step)
    next_step = int(next_step)
    eta = float(eta)
    if not 0 <= step < diffusion.timesteps:
        raise ValueError(f"DDIM step must be in [0, {diffusion.timesteps - 1}]")
    if next_step >= step or next_step < -1:
        raise ValueError("DDIM next_step must be smaller than step and at least -1")
    if not 0 <= eta <= 1:
        raise ValueError("DDIM eta must be between 0 and 1")
    if pred_x0.shape != x_t.shape:
        raise ValueError("predicted x0 must match diffusion state shape")
    if eta != 0 and next_step >= 0 and noise is not None and noise.shape != x_t.shape:
        raise ValueError("sampling noise must match diffusion state shape")

    # Once we step past 0 (to -1), generation is finished and pred_x0 is the final clean protein.
    if next_step == -1:
        return pred_x0

    # 1. Grab signal_variance at the current step (alpha) and the target step (alpha_next).
    # Note that alpha.sqrt() is signal_weight and (1 - alpha).sqrt() is noise_weight.
    alpha = diffusion.signal_variance[step].to(
        device=x_t.device, dtype=x_t.dtype
    )
    alpha_next = diffusion.signal_variance[next_step].to(
        device=x_t.device,
        dtype=x_t.dtype,
    )
    alpha = alpha.reshape(*((1,) * x_t.ndim))
    alpha_next = alpha_next.reshape(*((1,) * x_t.ndim))

    # 2. Rearrange x_t = signal_weight * pred_x0 + noise_weight * pred_noise to solve for pred_noise.
    pred_noise = (x_t - alpha.sqrt() * pred_x0) / (1 - alpha).sqrt().clamp_min(1e-8)

    # 3. Remix pred_x0 and pred_noise at next_step's weights.
    # When eta == 0 (default), sigma is 0 and direction is simply (1 - alpha_next).sqrt(),
    # which is noise_weight[next_step]. When eta > 0, sigma carves out part of the noise
    # budget for fresh random noise while keeping direction^2 + sigma^2 == 1 - alpha_next.
    sigma = (
        eta
        * (
            ((1 - alpha_next) / (1 - alpha)).clamp_min(0)
            * (1 - alpha / alpha_next).clamp_min(0)
        ).sqrt()
    )
    direction = (1 - alpha_next - sigma.square()).clamp_min(0).sqrt()
    x_next = alpha_next.sqrt() * pred_x0 + direction * pred_noise
    if eta == 0:
        return x_next
    if noise is None:
        noise = torch.randn_like(x_t)
    return x_next + sigma * noise


def ddim_timesteps(
    diffusion: DiffusionSchedule, sampling_steps: int
) -> tuple[int, ...]:
    """Pick `sampling_steps` evenly spaced timesteps from `999` down to `0`.

    For example, with 1,000 total timesteps and `sampling_steps=50`, this returns 50
    descending steps `(999, 979, 958, ..., 20, 0)` for `ddim_sample_step` to jump through.
    """
    sampling_steps = int(sampling_steps)
    if not 1 <= sampling_steps <= diffusion.timesteps:
        raise ValueError(
            "DDIM sampling_steps must be between 1 and the configured "
            f"{diffusion.timesteps} diffusion timesteps"
        )
    if sampling_steps == 1:
        return (diffusion.timesteps - 1,)
    steps = (
        torch.linspace(
            diffusion.timesteps - 1,
            0,
            sampling_steps,
            dtype=torch.float64,
        )
        .round()
        .to(dtype=torch.long)
    )
    return tuple(steps.tolist())
