# Repository overview

This repository contains a pipeline for designing minibinders for a target protein, the mubind sequence model used by that pipeline, and a command-line interface for generation, training and evaluation.

| Folder | Contents |
| --- | --- |
| [`src/`](../src/) | mubind's Python code. |
| [`configs/`](../configs/) | Training configuration. |
| [`artifacts/`](../artifacts/) | Models, datasets, reference inputs and run outputs. |
| [`docs/`](.) | Documentation. |

Within `artifacts/`, the small reference inputs in `benchmarks/`, `validation/` and `examples/` are tracked in Git. The remaining subfolders hold downloaded models and datasets, local targets and outputs from training, generation and evaluation.

```text
artifacts/
├── models/          Model weights, tools and model environments
├── datasets/        Downloaded and prepared training data
├── benchmarks/
│   ├── ppiref/
│   │   ├── complexes/       45 binder–target complexes and their manifest
│   │   └── sequences.json  81 binder sequences for ESMC comparison
│   ├── prostt5.json        92 test proteins with 3Di and training-corpus identity
│   ├── 3di_matrix.txt      Substitution scores for structural agreement
│   └── manifest.json      Benchmark input checksums
├── validation/      10 training reference proteins and panel metadata
├── examples/        Bundled target examples
├── targets/         Working target files
├── checkpoints/     Checkpoints from your training runs
├── generations/     Generated designs
└── evaluations/     Benchmark results
```

## CLI

The terminal interface and commands described in [docs/cli.md](cli.md) live in [`src/mubind/cli/`](../src/mubind/cli/). [`src/mubind/cli/app.py`](../src/mubind/cli/app.py) parses arguments with Typer and routes each command to its screen and workflow:

| Command | Code called by the CLI |
| --- | --- |
| `setup` | [`src/mubind/setup/workflow.py`](../src/mubind/setup/workflow.py) |
| `train` | [`src/mubind/cli/train.py`](../src/mubind/cli/train.py) and [`src/mubind/training/trainer.py`](../src/mubind/training/trainer.py) |
| `evaluate` | [`src/mubind/cli/evaluate_ui.py`](../src/mubind/cli/evaluate_ui.py) for menus; [`src/mubind/cli/evaluate.py`](../src/mubind/cli/evaluate.py) for subcommands |
| `generate` | [`src/mubind/cli/design.py`](../src/mubind/cli/design.py) and [`src/mubind/inference/design.py`](../src/mubind/inference/design.py) |
| `sample` | [`src/mubind/cli/sample.py`](../src/mubind/cli/sample.py) and [`src/mubind/inference/generate.py`](../src/mubind/inference/generate.py) |
| `provider` | [`src/mubind/inference/services/provider.py`](../src/mubind/inference/services/provider.py) and [`src/mubind/cli/provider_display.py`](../src/mubind/cli/provider_display.py) |
| `view` | [`src/mubind/cli/viewer.py`](../src/mubind/cli/viewer.py) |

The interfaces for `generate`, `sample`, `train` and `evaluate` collect settings and display progress before calling the code that performs each operation. They use Questionary for prompts and Rich for terminal output.

## Setup

On a new installation, the first CLI operation is usually `setup` (see [README.md](../README.md)). [`src/mubind/setup/`](../src/mubind/setup/) prepares model weights, installs tools such as Foldseek, and creates separate Python environments for the external models Complexa and ESMFold2/ESMC. Training data preparation is optional. mubind's own Python environment is prepared by the [`mubind`](../mubind) launcher using uv, with dependencies declared in [`pyproject.toml`](../pyproject.toml).

By default, datasets are stored in `artifacts/datasets/` and downloaded weights, tools and external model environments are stored in `artifacts/models/`. You can choose other locations during setup, and [`src/mubind/setup/config.py`](../src/mubind/setup/config.py) records them so subsequent commands can find them.

## Data

[`src/mubind/preprocessing/`](../src/mubind/preprocessing/) implements the dataset preparation described in [docs/training.md](training.md). Selecting ProstT5 explicitly, for example with `mubind setup --only prostt5`, calls this code to download and filter the dataset, read its Parquet files with PyArrow, and write NumPy arrays.

[`src/mubind/data/`](../src/mubind/data/) reads those arrays and supplies encoded protein examples through PyTorch data loaders. It also defines sequence encodings, biochemical property calculations and decoding routines used by the model and generation code.

## Model

mubind's model and sampling code use three parts of the package:

- [`src/mubind/model/`](../src/mubind/model/) defines the PyTorch U-Net, conditioning encoders and attention layers, together with their configuration and input types.
- [`src/mubind/diffusion/`](../src/mubind/diffusion/) implements diffusion schedules, training objectives and DDPM/DDIM sampling.
- [`src/mubind/data/`](../src/mubind/data/) provides the sequence representations and decoding routines described above.

The `predict()` function in [`src/mubind/inference/predict.py`](../src/mubind/inference/predict.py) combines the network with a sampler and decodes the resulting sequences. Generation and evaluation reuse this function.

## Training

The `train` command (see [docs/training.md](training.md)) passes settings from [`src/mubind/cli/train.py`](../src/mubind/cli/train.py) to the main loop in [`src/mubind/training/trainer.py`](../src/mubind/training/trainer.py). The loop loads the prepared dataset, creates or restores the model, then uses two parts of the training code:

- [`src/mubind/training/tasks/`](../src/mubind/training/tasks/) constructs training examples, choosing which residues and conditions are visible to the model.
- [`src/mubind/training/step.py`](../src/mubind/training/step.py) adds diffusion noise, runs the network and computes the reconstruction loss for a batch.

The trainer uses that loss to update the model's weights. It also logs measurements and saves checkpoints containing the state needed to resume training.

To monitor progress and select checkpoints, the trainer calls [`src/mubind/evaluation/`](../src/mubind/evaluation/) with the model already in memory. By default, full evaluation checks denoising loss and routine generation every 1,000 steps. Larger panels check property control, structural recovery and inpainting every 5,000 steps.

The main structural panel generates 10 candidates for each of 10 fixed reference proteins in `artifacts/validation/` and folds each sequence alone with ESMFold2. TM-score and 3Di agreement measure how closely the predicted structure matches the reference, while pLDDT records confidence in the fold. These checks do not include a target protein or produce ipTM.

Configuration types are defined in [`src/mubind/training/config.py`](../src/mubind/training/config.py); [`configs/train.json`](../configs/train.json) supplies the training recipe.

## Evaluation

The `evaluate` command (see [docs/evaluation.md](evaluation.md)) tests a saved checkpoint on the benchmarks in [`src/mubind/benchmarks/`](../src/mubind/benchmarks/), covering unconditional sequence quality, property control, structural steering, full binder redesign and inpainting. Whereas [`src/mubind/evaluation/`](../src/mubind/evaluation/) runs validation checks during training, `benchmarks/` runs evaluation protocols derived from the paper and reuses its folding and structure-analysis tools. [`src/mubind/benchmarks/workflow.py`](../src/mubind/benchmarks/workflow.py) runs each experiment in three stages: candidate generation, structure prediction where needed, and metric calculation and reporting.

- [`src/mubind/benchmarks/generation.py`](../src/mubind/benchmarks/generation.py) and [`src/mubind/benchmarks/checkpoint_generation.py`](../src/mubind/benchmarks/checkpoint_generation.py) sample candidate sequences through `predict()`. Property control draws fresh requests from the selected training dataset. Structural steering loads the 3Di strings of all 92 proteins in the original ProstT5 test split that are 50–70 amino acids long from [`artifacts/benchmarks/prostt5.json`](../artifacts/benchmarks/prostt5.json). Redesign and inpainting instead read the seed binder and target from each PDB in the chosen folder, using the binder’s structure and sequence to obtain its 3Di and property conditions.
- For the structural experiments, [`src/mubind/benchmarks/folding.py`](../src/mubind/benchmarks/folding.py) predicts structures with ESMFold2 and compares them against reference structures. Structural steering folds each candidate alone, while redesign and inpainting fold the binder together with its target chain.
- [`src/mubind/benchmarks/metrics.py`](../src/mubind/benchmarks/metrics.py) and [`src/mubind/benchmarks/reports.py`](../src/mubind/benchmarks/reports.py) compute the benchmark metrics, select the best candidate per target or complex, and write `results.md` and `summary.json`.

## Generation

A saved checkpoint can also be used to generate candidates. `generate` runs the binder design pipeline described in [docs/generation.md](generation.md). [`src/mubind/cli/design.py`](../src/mubind/cli/design.py) handles settings and terminal interaction, then calls [`src/mubind/inference/design.py`](../src/mubind/inference/design.py) to coordinate the run, including structure prediction, candidate ranking and saving results.

`sample` generates sequences directly from a mubind checkpoint. Its interface in [`src/mubind/cli/sample.py`](../src/mubind/cli/sample.py) calls [`src/mubind/inference/generate.py`](../src/mubind/inference/generate.py). Both commands use `predict()` in [`src/mubind/inference/predict.py`](../src/mubind/inference/predict.py) for sequence sampling and decoding.

Complexa and ESMFold2 run in separate Python processes using the environments prepared during setup. [`src/mubind/runtime.py`](../src/mubind/runtime.py) locates their interpreters and model files. The `provider` command uses [`src/mubind/inference/services/provider.py`](../src/mubind/inference/services/provider.py) to keep those processes loaded between design runs.

## Viewer

`view` uses [`src/mubind/cli/viewer.py`](../src/mubind/cli/viewer.py) to read saved generation results and serve the browser viewer. [`src/mubind/cli/viewer_assets/`](../src/mubind/cli/viewer_assets/) contains its HTML and bundled 3Dmol.js.

## Configuration

Settings are held in Python dataclasses defined alongside the code that uses them. The CLI updates these objects from arguments and menu choices, then passes them to the implementation.

For training, [`src/mubind/training/config.py`](../src/mubind/training/config.py) defines TrainingConfig, its defaults and the JSON loader. A new run starts from those defaults. --config loads a recipe such as [`configs/train.json`](../configs/train.json); explicit CLI flags then override supported values. Settings beyond the CLI options, including the task mixture and masking parameters, can be changed in the JSON recipe. Resuming starts from the configuration saved in the checkpoint.

Generation and sampling settings are defined in [`src/mubind/inference/config.py`](../src/mubind/inference/config.py). Setup settings are in [`src/mubind/setup/config.py`](../src/mubind/setup/config.py).

Early experiments used Hydra/OmegaConf. When the CLI was introduced, we chose Python dataclasses for mubind's configuration. Hydra/OmegaConf remain in [`src/mubind/inference/services/complexa_worker.py`](../src/mubind/inference/services/complexa_worker.py) for the external Complexa integration.
