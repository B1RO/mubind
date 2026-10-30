"""Noise timetable for blurring clean proteins and stepping backward from static.

Diffusion runs across a discrete sequence of timesteps t in [0, timesteps - 1],
where t = 0 is almost clean and t = timesteps - 1 is almost pure Gaussian static.
This module precomputes the 1D lookup tables for every timestep t once at startup
and registers them as PyTorch buffers so they move to the GPU with the schedule.

At timestep t, a clean protein is mixed with random Gaussian noise as
    noisy = signal_weight[t] * clean + noise_weight[t] * noise
Both buffers come from the cumulative clean signal fraction signal_variance[t] in (0, 1)
    signal_weight[t] = sqrt(signal_variance[t])
    noise_weight[t]  = sqrt(1 - signal_variance[t])
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class DiffusionConfig:
    timesteps: int = 1000
    schedule: str = "cosine"
    objective: str = "pred_x0"
    loss_weighting: str = "uniform"

    def build(self) -> DiffusionSchedule:
        return DiffusionSchedule(**asdict(self))


class DiffusionSchedule(nn.Module):

    def __init__(
        self,
        *,
        timesteps: int = 1000,
        schedule: str = "cosine",
        objective: str = "pred_x0",
        loss_weighting: str = "uniform",
    ) -> None:
        super().__init__()
        _validate_schedule_config(timesteps, objective, loss_weighting)
        self.timesteps = timesteps
        self.objective = objective
        self.loss_weighting = loss_weighting

        # 1. Per-step noise fraction (betas) and cumulative variances (signal_variance, noise_variance).
        # In the step-by-step chain step_n = a_n * step_{n-1} + fresh_noise
        #   betas           per-step loss [b_1, b_2, ..., b_n] (how much one step throws away)
        #   alphas          per-step survival [a_1, a_2, ..., a_n] where a_n = 1 - b_n
        #   signal_variance total surviving signal [a_1, a_1 * a_2, a_1 * a_2 * a_3, ...]
        #   noise_variance  total accumulated noise 1 - signal_variance
        betas = _make_beta_schedule(schedule, timesteps)
        alphas = 1.0 - betas
        signal_variance = torch.cumprod(alphas, dim=0)
        noise_variance = 1.0 - signal_variance

        # 2. Forward weights for mixing clean protein and Gaussian noise at step t.
        self.register_buffer("signal_variance", signal_variance)
        self.register_buffer("signal_weight", torch.sqrt(signal_variance))
        self.register_buffer("noise_weight", torch.sqrt(noise_variance))

        # 3. Per-timestep training loss weights.
        # Conceptually, how much to weight each timestep's training error belongs with the
        # loss functions in objective.py rather than in the noise schedule. Keeping it here
        # is a slightly messy shortcut so this nn.Module precomputes the 1D buffer once and
        # moves it to the GPU automatically alongside the schedule tensors.
        signal_to_noise_ratio = signal_variance / noise_variance
        self.register_buffer(
            "loss_weight",
            _compute_loss_weight(signal_to_noise_ratio, objective, loss_weighting),
        )

        # 4. Reverse DDPM weights for stepping from noisy step t to t - 1 given predicted_clean.
        # During generation, the UNet takes noisy (at step t) and guesses step 0 (predicted_clean).
        # To step one notch cleaner to t - 1 instead of jumping all the way to 0, posterior_step
        # blends ~99% of noisy (posterior_noisy_weight) with ~1% of predicted_clean
        # (posterior_clean_weight), plus a tiny pinch of fresh noise (posterior_variance).
        # The formulas below are just the Gaussian algebra solving for step t - 1 between 0 and t.
        previous_signal_variance = torch.cat([torch.ones(1), signal_variance[:-1]])
        previous_noise_variance = 1.0 - previous_signal_variance
        self.register_buffer(
            "posterior_variance",
            betas * previous_noise_variance / noise_variance,
        )
        self.register_buffer(
            "posterior_clean_weight",
            betas * torch.sqrt(previous_signal_variance) / noise_variance,
        )
        self.register_buffer(
            "posterior_noisy_weight",
            previous_noise_variance * torch.sqrt(alphas) / noise_variance,
        )

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Pick one random timestep t in [0, timesteps - 1] per batch example."""
        return torch.randint(
            0, self.timesteps, (batch_size,), device=device, dtype=torch.long
        )

    def noise_weights(
        self, timesteps: torch.Tensor, shape: torch.Size
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Look up (signal_weight, noise_weight) at timesteps for a batch tensor."""
        return (
            _at_timesteps(self.signal_weight, timesteps, shape),
            _at_timesteps(self.noise_weight, timesteps, shape),
        )

    def add_noise(
        self,
        clean: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mix clean protein with random Gaussian noise at each example's timestep."""
        if noise is None:
            noise = torch.randn_like(clean)
        signal_weight, noise_weight = self.noise_weights(timesteps, clean.shape)
        noisy = signal_weight * clean + noise_weight * noise
        return noisy, noise

    def posterior_step(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        predicted_clean: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the reverse-step (mean, variance) for step t - 1 from noisy and predicted_clean."""
        clean_weight = _at_timesteps(
            self.posterior_clean_weight, timesteps, noisy.shape
        )
        noisy_weight = _at_timesteps(
            self.posterior_noisy_weight, timesteps, noisy.shape
        )
        mean = clean_weight * predicted_clean + noisy_weight * noisy
        variance = _at_timesteps(self.posterior_variance, timesteps, noisy.shape)
        return mean, variance


def _validate_schedule_config(
    timesteps: int, objective: str, loss_weighting: str
) -> None:
    if not isinstance(timesteps, int) or timesteps < 1:
        raise ValueError("timesteps must be a positive integer")
    if objective not in {"pred_noise", "pred_x0", "pred_v"}:
        raise ValueError("objective must be pred_noise, pred_x0, or pred_v")
    if loss_weighting not in {"objective", "uniform"}:
        raise ValueError("loss_weighting must be objective or uniform")


def _make_beta_schedule(schedule: str, timesteps: int) -> torch.Tensor:
    if schedule == "cosine":
        return cosine_beta_schedule(timesteps)
    if schedule == "linear":
        return linear_beta_schedule(timesteps)
    raise ValueError(f"unknown schedule: {schedule}")


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    """Build linearly spaced per-step noise fractions (betas) from 1e-4 to 0.02."""
    # diffusion is confusing
    return torch.linspace(1e-4, 0.02, timesteps)


def cosine_beta_schedule(timesteps: int, offset: float = 0.008) -> torch.Tensor:
    """Build per-step noise fractions (betas) whose cumulative signal follows a cosine curve."""
    if not isinstance(timesteps, int) or timesteps < 1:
        raise ValueError("timesteps must be a positive integer")
    # Follow a squared cosine curve from ~1 down to ~0 across timesteps + 1 boundaries,
    # with a small offset so the first few steps near t = 0 do not add zero noise.
    steps = torch.linspace(0, timesteps, timesteps + 1)
    angles = ((steps / timesteps) + offset) / (1 + offset) * (math.pi * 0.5)
    signal_variance = torch.cos(angles) ** 2
    signal_variance = signal_variance / signal_variance[0]
    # Convert cumulative signal survival into single-step noise fractions beta_t.
    betas = 1 - (signal_variance[1:] / signal_variance[:-1])
    return betas.clamp(0.0001, 0.9999)


# Decide how much training errors at each timestep t should count in the total loss.
# With "uniform", every timestep gets weight 1.0 so all noise levels matter equally.
# With "objective", weights are scaled by signal_to_noise_ratio so predicting clean
# protein (pred_x0) or velocity (pred_v) matches the effective loss of predicting noise.
def _compute_loss_weight(
    signal_to_noise_ratio: torch.Tensor, objective: str, loss_weighting: str
) -> torch.Tensor:
    """Return per-timestep loss weights (either uniform 1s or SNR-scaled weights)."""
    if loss_weighting == "uniform" or objective == "pred_noise":
        return torch.ones_like(signal_to_noise_ratio)
    if objective == "pred_x0":
        return signal_to_noise_ratio
    return signal_to_noise_ratio / (signal_to_noise_ratio + 1)


def _at_timesteps(
    schedule_values: torch.Tensor, timesteps: torch.Tensor, target_shape: torch.Size
) -> torch.Tensor:
    """Gather 1D schedule values at each example's timestep and reshape to [B, 1, ...]."""
    gathered = schedule_values.gather(0, timesteps)
    return gathered.reshape(timesteps.shape[0], *((1,) * (len(target_shape) - 1)))
