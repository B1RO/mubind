# Training

Runs the diffusion training loop on prepared `ProstT5Dataset` proteins.

```text
training/
├── step.py            Runs one training batch (add noise, self-conditioning, UNet forward pass, MSE loss)
├── tasks/             Chooses which residues, 3Di tokens, and properties to hide or show (see tasks/README.md)
├── trainer.py         Outer loop over training steps, validation, and checkpoint saves
├── config.py          Parses configs/train.json into TrainingConfig dataclasses
└── helpers/
    ├── setup.py       Pre-flight validation, dataset loading, model/optimizer setup, and validation hooks
    ├── ema.py         Moving average of the UNet weights used for evaluation and generation
    ├── loader.py      DataLoader setup with a reproducible random seed per batch
    ├── checkpoint.py  Saves and loads model, EMA, optimizer, and RNG states
    └── compilation.py torch.compile wrapper for the UNet
```

## How training runs

1. **`tasks/`** (documented in `tasks/README.md`) prepares each batch before it hits the GPU. For each protein it chooses whether to corrupt the whole sequence or only an inpainting span, and randomly masks out 3Di structure tokens or biochemical properties.
2. **`step.py` (`compute_loss`)** runs one forward step. It picks random timesteps `t` with `diffusion.sample_timesteps`, adds noise with `diffusion.add_noise`, optionally runs a no-grad pass (`self_condition`) so the UNet gets an initial guess as a hint, calls `model(...)`, and computes the loss with `residue_pad_balanced_mse_loss`.
3. **`trainer.py` (`train`)** runs the outer loop (`compute_loss` -> `loss.backward()` -> `optimizer.step()`), updates the smoothed weights in `helpers/ema.py`, runs periodic validation, and writes checkpoints through `helpers/checkpoint.py`.
