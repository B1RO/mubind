"""DDPM reverse diffusion step."""

from __future__ import annotations

import torch

from .schedule import DiffusionSchedule


@torch.no_grad()
def ddpm_sample_step(
    diffusion: DiffusionSchedule,
    x_t: torch.Tensor,
    t: torch.Tensor,
    pred_x0: torch.Tensor,
    *,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Take one single reverse DDPM step from `t` to `t - 1`.

    Uses `diffusion.posterior_step` from `schedule.py` to blend `x_t` with `pred_x0`
    using timestep-dependent weights (`mean`), then adds fresh Gaussian noise
    (`sqrt(variance) * noise`) on every step except the final step `t == 0`.
    """
    mean, variance = diffusion.posterior_step(x_t, t, pred_x0)
    if noise is None:
        noise = torch.randn_like(x_t)
    elif noise.shape != x_t.shape:
        raise ValueError("sampling noise must match diffusion state shape")
    # Do not add fresh noise on the final step (t == 0) when outputting the finished protein.
    nonzero = (t > 0).float().reshape(t.shape[0], *((1,) * (x_t.ndim - 1)))
    return mean + nonzero * torch.sqrt(variance) * noise
