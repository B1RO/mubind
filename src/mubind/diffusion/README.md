# Diffusion

This folder contains the noise schedule, training objectives, and reverse samplers used to corrupt and reconstruct continuous ProstT5 protein embeddings.

## Structure

```text
diffusion/
├── schedule.py        Precomputes 1,000-step noise tables, adds forward noise, and computes reverse DDPM weights
├── objective.py       Defines prediction targets (pred_x0 default) and residue/PAD-balanced MSE losses
├── sampling.py        Runs the shared reverse sampling loop and classifier-free property guidance
├── ddpm.py            Single-step reverse DDPM update (1,000 stochastic steps from 999 down to 0)
└── ddim.py            Multi-step reverse DDIM jump and timestep selection for fast generation
```

## Overview 

1. **Forward noise schedule (`schedule.py`).** `DiffusionSchedule` precomputes 1,000-element lookup buffers for `"cosine"` (default) or `"linear"` schedules. At any timestep `t`, `add_noise` mixes a clean protein embedding with Gaussian noise in one shot using `signal_weight[t] * clean + noise_weight[t] * noise`.
2. **Training target and loss (`objective.py`).** In line with OmegAMP, MUbind trains the UNet with `"pred_x0"` so the network predicts the clean protein directly. `training_target` builds the ground-truth tensor, `model_output_to_x0` converts raw network outputs back into clean estimates, and `residue_pad_balanced_mse_loss` scores the prediction across the fixed-length canvas while giving equal weight to real amino acid positions and `PAD` positions.
3. **Reverse generation (`sampling.py`, `ddpm.py`, and `ddim.py`).** `sample` starts from pure Gaussian noise at step `999` and walks backward to step `0`. At each step it calls the UNet directly (or through `property_guidance` when `property_guidance_scale != 1.0`), optionally feeds `pred_x0` forward as a self-conditioning hint, and steps toward `pred_x0` using either
   - **`ddpm_sample_step` (`ddpm.py`)** for full 1,000-step stochastic generation via `diffusion.posterior_step` (used in binder design and full benchmarks).
   - **`ddim_sample_step` (`ddim.py`)** for fast, deterministic multi-step jumps across an evenly spaced schedule from `ddim_timesteps` (used in validation and quick CLI generation).
