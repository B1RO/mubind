"""Prediction targets and MSE loss functions for the UNet denoiser.

In line with OmegAMP, we train the UNet to predict the clean protein directly
(`"pred_x0"`, default), as it fits this representation best. We never use `"pred_noise"`
or `"pred_v"` in production, but we tried them during earlier experiments and kept them
here for completeness. This module handles both sides of that objective choice

- During training, `training_target` builds the ground-truth tensor for the chosen
  objective, and `masked_mse_loss` or `residue_pad_balanced_mse_loss` scores the UNet's
  prediction against it.
- During generation and self-conditioning, `model_output_to_x0` converts the UNet's raw
  output back into a clean protein `x0` estimate regardless of which objective was used.
"""

from __future__ import annotations

import torch

from .schedule import DiffusionSchedule


def training_target(
    diffusion: DiffusionSchedule,
    x0: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """Return the ground-truth "answer key" tensor the UNet is graded against in the loss.
    
    With the default `"pred_x0"` objective, this simply returns the clean protein `x0`
    unchanged. It only exists as a switch so training code does not need its own if/else
    when experimenting with `"pred_noise"` or `"pred_v"`.
    """
    if diffusion.objective == "pred_x0":
        return x0
    if diffusion.objective == "pred_noise":
        return noise
    signal_weight, noise_weight = diffusion.noise_weights(timesteps, x0.shape)
    return signal_weight * noise - noise_weight * x0


def model_output_to_x0(
    diffusion: DiffusionSchedule,
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    model_output: torch.Tensor,
) -> torch.Tensor:
    """Convert the UNet's raw prediction into a clean protein `x0` estimate.

    With the default `"pred_x0"` objective, the UNet already outputs `x0`, so this
    returns `model_output` unchanged. For `"pred_noise"` or `"pred_v"`, it solves the
    forward mixing equation `x_t = signal_weight * x0 + noise_weight * noise` for `x0`.
    """
    if diffusion.objective == "pred_x0":
        return model_output
    signal_weight, noise_weight = diffusion.noise_weights(timesteps, x_t.shape)
    if diffusion.objective == "pred_noise":
        return (x_t - noise_weight * model_output) / signal_weight.clamp_min(1e-8)
    return signal_weight * x_t - noise_weight * model_output


def masked_mse_loss(
    diffusion: DiffusionSchedule,
    prediction: torch.Tensor,
    target: torch.Tensor,
    edit_mask: torch.Tensor | None = None,
    *,
    timesteps: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute plain MSE between `prediction` and `target`, optionally restricted to `edit_mask`."""
    sq = (prediction - target) ** 2
    if edit_mask is None:
        per_example = sq.flatten(1).mean(dim=1)
    else:
        mask_f = edit_mask[:, None, :].to(dtype=sq.dtype)
        numerator = (sq * mask_f).sum(dim=(1, 2))
        denominator = (mask_f.sum(dim=(1, 2)) * sq.shape[1]).clamp_min(1.0)
        per_example = numerator / denominator

    return _loss_mean(diffusion, per_example, timesteps)


def residue_pad_balanced_mse_loss(
    diffusion: DiffusionSchedule,
    prediction: torch.Tensor,
    target: torch.Tensor,
    edit_mask: torch.Tensor,
    residue_mask: torch.Tensor,
    *,
    timesteps: torch.Tensor | None = None,
    pad_weighting: str = "class_equal",
    pad_weight: float = 1.0,
    example_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute MSE across the fixed-length canvas while balancing real residues against PAD.

    Because the UNet diffuses on a fixed-size canvas (e.g. 128 positions), a short
    20-residue peptide has 20 real amino acid positions (`residue_mask == True`) and
    108 `PAD` positions (`residue_mask == False`). The UNet must predict both so it
    learns where the generated protein ends, but a naive average across all 128 slots
    would let the 108 `PAD` slots drown out the 20 real amino acids.

    Two `pad_weighting` modes control how those two regions are combined per protein

    - `"class_equal"` (default) computes the mean error over the real residues and the
      mean error over the `PAD` slots separately, then averages those two group means
      50/50 so real residues always count for half the loss regardless of protein length.
    - `"weighted"` takes a single flat average across all canvas positions after scaling
      each `PAD` slot's error by `pad_weight`.
    """
    if edit_mask.shape != residue_mask.shape:
        raise ValueError("edit_mask and residue_mask must have the same shape")
    if pad_weighting not in {"class_equal", "weighted"}:
        raise ValueError("pad_weighting must be class_equal or weighted")
    if pad_weight < 0:
        raise ValueError("pad_weight must be non-negative")

    # 1. Compute MSE at each canvas slot [batch, length] by averaging across embedding channels (dim=1),
    # and split the canvas into real amino acid slots vs PAD slots.
    position_error = (prediction - target).square().mean(dim=1)
    residue_positions = residue_mask.bool()
    pad_positions = ~residue_positions

    if pad_weighting == "weighted":
        # 2a. Single flat average across all canvas slots (real residues get weight 1.0, PAD gets pad_weight).
        weights = residue_positions.to(dtype=position_error.dtype)
        weights = weights + pad_weight * pad_positions.to(dtype=position_error.dtype)
        per_example = (position_error * weights).sum(dim=1)
        per_example = per_example / weights.sum(dim=1).clamp_min(1e-8)
    else:
        # 2b. "class_equal" (default) computes the mean error on real residues and the mean error
        # on PAD slots separately, then averages those two group means 50/50 per protein.
        # (class_present tracks whether a group has at least 1 slot, in case a protein has 0 PAD slots.)
        class_means = []
        class_present = []
        for positions in (residue_positions, pad_positions):
            weights = positions.to(dtype=position_error.dtype)
            counts = weights.sum(dim=1)
            class_means.append(
                (position_error * weights).sum(dim=1) / counts.clamp_min(1.0)
            )
            class_present.append(counts > 0)
        present = torch.stack(class_present, dim=1)
        means = torch.stack(class_means, dim=1)
        per_example = (means * present.to(means.dtype)).sum(dim=1)
        per_example = per_example / present.sum(dim=1).clamp_min(1).to(means.dtype)

    return _loss_mean(diffusion, per_example, timesteps, example_weights)


def _loss_mean(diffusion, per_example, timesteps, example_weights=None):
    """Combine a batch of per-protein errors into one single loss number for PyTorch.

    The loss functions above calculate one error number for each protein in the batch
    (say 32 numbers for a batch of 32 proteins, passed in as `per_example`). PyTorch's
    `loss.backward()` needs a single number for the whole batch, so this helper averages
    those 32 numbers together after applying two optional weights

    1. `timesteps` multiplies each protein's error by `diffusion.loss_weight[t]` from
       `schedule.py` (with our default `"uniform"` loss weighting, this is just 1.0).
    2. `example_weights` lets the caller give some proteins in the batch more weight
       than others when averaging. When `None`, all proteins are averaged equally.
    """
    # 1. Multiply each protein's error by the weight for its timestep t.
    if timesteps is not None:
        if timesteps.shape != per_example.shape:
            raise ValueError("timesteps must contain one value per batch example")
        per_example = per_example * diffusion.loss_weight[timesteps].to(
            dtype=per_example.dtype
        )

    # 2. If no per-protein weights were passed, return the plain average across the batch.
    if example_weights is None:
        return per_example.mean()

    # 3. Otherwise, return the weighted average (per_example * example_weights).sum() / weight_sum.
    if example_weights.shape != per_example.shape:
        raise ValueError("example_weights must contain one value per batch example")
    example_weights = example_weights.to(
        device=per_example.device,
        dtype=per_example.dtype,
    )
    # Sanity-check on CPU that weights are non-negative and not all zero (skipped on GPU
    # because `.item()` would force the CPU to wait for the GPU).
    if not example_weights.is_cuda and bool((example_weights < 0).any().item()):
        raise ValueError("example_weights must be non-negative")
    weight_sum = example_weights.sum()
    if not weight_sum.is_cuda and float(weight_sum.detach()) <= 0:
        raise ValueError("example_weights must contain positive total weight")
    return (per_example * example_weights).sum() / weight_sum
