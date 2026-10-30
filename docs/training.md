# Training

![mubind training setup and a 5,000-step run](assets/mubind-training-real-5000-steps.gif)

`mubind train` trains the mubind diffusion model on monomeric protein sequences paired with Foldseek **3Di** structural tokens from the ProstT5 dataset. The model learns to generate and edit sequences up to 70 residues long, conditioned on 3Di structure and biochemical properties (length, net charge, and hydrophobicity).

## Prepare the data

Default setup prepares the models and tools for binder design without the training dataset. Before your first training run, explicitly download and filter ProstT5 to sequences of 50–70 residues by running

```bash
mubind setup --only prostt5 --min-length 50 --max-length 70
```

In the setup menu, choose **Install missing** to prepare the training (1,505,527 proteins), validation (100), and test (92) splits, or add `--yes` to the command to start without the menu. Matching prepared data is reused. If you plan to use the default `full` validation mode during training and have not already installed ESMFold2 and Foldseek, run

```bash
mubind setup --only esmfold2 --only foldseek
```

## Start or resume a run

To open the interactive training menu, run

```bash
mubind train
```

The main screen shows the active dataset, total step count, and output path (`artifacts/checkpoints/experiment-001.pt` by default). Select **Configure training** to change the batch size, device, precision, validation mode, or JSON configuration. To start a new run, choose **Start training**. To resume, select **Run** and choose a listed checkpoint or **Resume another checkpoint…**, enter the new total step count, and choose **Resume training**.

To run directly from the command line, pass your options to `mubind train`

```bash
mubind train --out artifacts/checkpoints/my-run.pt \
  --steps 300000 --batch-size 1024 --evaluation full
```

To continue a saved run to a higher total step count, pass `--resume`

```bash
mubind train --resume artifacts/checkpoints/my-run.pt --steps 350000
```

Resuming restores the model, EMA weights, optimizer, scheduler, and dataset position from the checkpoint. To customize model architecture, task probabilities, or optimization settings beyond the CLI flags, copy `configs/train.json` and pass your modified file with `--config`.

## Validation and outputs

During training the trainer evaluates an exponential moving average (EMA) of the model weights on fixed validation tasks. To keep sampling checks fast, it uses DDIM to traverse the 1,000-step noise schedule in 50 updates.

Every 1,000 steps it measures denoising loss on validation proteins and generates a small set of conditioned and inpainting sequences. These samples test whether each output forms one contiguous block of residues without internal padding, preserves supplied residues during inpainting, and matches the requested length, charge, and hydrophobicity.

Every 5,000 steps, as well as after resuming and at the end of training, the trainer runs a broader evaluation across different property combinations (such as length alone or length with charge) paired with both full and partly masked 3Di. In `full` mode, it also tests whether generated sequences fold into the requested structures across 10 fixed validation proteins bundled in `artifacts/validation/`. ESMFold2 predicts each generated sequence's monomer structure, Foldseek re-encodes the prediction into 3Di for comparison with the requested tokens, and the Cα TM-score measures similarity to the reference structure.

Use `--evaluation` or **Configure training** to choose how much of this suite runs.

| Mode | What it checks | Requirements |
| --- | --- | --- |
| `full` (default) | Denoising loss and sequence checks every 1,000 steps, plus property and monomer folding checks every 5,000 steps. | ESMFold2, Foldseek, and the 50–70 residue ProstT5 split |
| `sequence` | Denoising loss and sequence checks every 1,000 steps, plus larger sequence checks every 5,000 steps. These include property and inpainting checks, without structure prediction. | No additional tools |
| `loss` | Validation denoising loss every 1,000 steps without sequence sampling. | No additional tools |
| `off` | Training loop only, without validation. | No additional tools |

For an output path `artifacts/checkpoints/my-run.pt`, training writes the following files in `artifacts/checkpoints/`.

| Path | Contents |
| --- | --- |
| `my-run.pt` | Latest model, EMA, optimizer, and RNG state, saved every 1,000 steps and at completion. |
| `my-run.best.pt` | Checkpoint with the lowest validation denoising loss. |
| `my-run.best-3di.pt` | Checkpoint with the highest failure-adjusted 3Di token recovery on the 10-protein folding panel (`full` mode), with ties broken by mean TM-score and monomer pLDDT. |
| `my-run.pt.config.json` | Resolved training configuration and dataset property normalization. |
| `my-run.pt.metrics.jsonl` | Training loss at the configured logging interval, learning rate, and validation metrics. |
| `my-run.pt.evaluation/` | Per-step validation JSON summaries, generated sequences, and cached ESMFold2 structures. |

## Model and training tasks

Each protein is encoded as an **8 × 70** array combining five normalized physicochemical amino-acid features, a padding mask channel, and two continuous 3Di channels. Global conditioning provides length, net charge at pH 7, and mean Eisenberg hydrophobicity, standardized using training-split statistics stored in the checkpoint.

By default, a 300,000-step run follows a three-stage curriculum (`tasks.stages` in `configs/train.json`) that gradually introduces property conditioning and sequence inpainting while keeping the same model weights, EMA, optimizer, and scheduler state.

| Updates | Task mixture |
| --- | --- |
| 1–150,000 | Whole-sequence generation conditioned on full or partially masked 3Di tokens, without property conditioning. |
| 150,001–200,000 | Whole-sequence generation from varying combinations of 3Di and global properties (60% both, 20% 3Di only, 15% properties only, and 5% unconditional) with random property subset dropout. |
| 200,001–300,000 | Joint training on 50% whole-sequence generation and 50% span inpainting. Inpainting reconstructs the full protein from a partially masked sequence. Of all tasks, 35% mask spans across the full protein, and 5% each also mask the sequence outside a prefix, suffix, or middle region. |

At each step, the trainer samples a uniform diffusion timestep on a 1,000-step cosine schedule, adds Gaussian noise, and trains a 1D U-Net to predict the clean array (`pred_x0`). The loss is continuous MSE across all eight channels, weighted equally between active residue positions and padding positions.

## Customizing the configuration

Edit a copy of `configs/train.json` and pass it with `--config`. Explicit CLI options override matching fields. Keep the `evaluation` section, since omitting it selects loss-only validation.

- `tasks.layout_probabilities` uses `[complete, prefix, suffix, middle, redesign]`; `tasks.redesign_conditioning_probabilities` uses `[unconditional, properties, 3di, properties_3di]`.
- `properties.mean`, `properties.std` and `evaluation.suites.<name>.property_condition_mask` follow `[length, charge, hydrophobicity]`. Supply both `mean` and `std`, or leave both `null` to use training-split statistics.
- `backbone.x_channels` must be `8` when `data.joint_3di` is `true`, otherwise `6`.
- Supplying either base task probability array without `tasks.stages` disables the default curriculum. Include `stages` to retain a curriculum, or set it to `[]` to use the base mixture from the beginning.
- Set `structure.context` or `structure.cross_attention` to `null` to disable the corresponding module.
- `batched_tasks` selects the batched sampler by default; set it to `false` to use the scalar reference sampler. The retired `fast_tasks` option is ignored when reading older configs and checkpoints. Resuming a custom run that used those retired optimizations can change its task random-number sequence.

On `--resume`, only `steps`, `out`, `device`, `log_every`, `save_every`, `val_every`, `evaluation`, `compile`, `compile_mode` and `data.num_workers` may change. Other resolved settings and the prepared dataset manifest must match the checkpoint.

In `full` mode, put these overrides inside `evaluation.suites` to run the main structural panel every 10,000 steps and disable the partial-3Di panel. Other preset settings are inherited.

```json
{
  "threedi_structure": {"interval_steps": 10000},
  "partial_threedi": {"enabled": false}
}
```

Full configuration definitions are in `src/mubind/training/config.py` and `src/mubind/evaluation/config.py`, with task and model settings in `src/mubind/training/tasks/config.py` and `src/mubind/model/config.py`.
