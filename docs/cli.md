# mubind CLI

The CLI brings mubind's workflows into one interface. Menus help you explore the available settings; command arguments let you specify the work directly and repeat it from a script.

## Using the CLI

Run `mubind` (or `./mubind` from a repository checkout) to open the main menu for setup, generation, training and evaluation. You can also open a workflow directly, for example with `mubind generate`. Its overview shows the active target and checkpoint. Change them or choose **Configure generation** before starting, where **All settings and paths** reveals the detailed configuration. Use `mubind --help` for the command list and `mubind generate --help` for that command's options.


## Commands

Run commands as `mubind <command>` (or `./mubind <command>` from a repository checkout).

| Command | Purpose |
| --- | --- |
| `setup` | Prepare model weights, tools and their environments for binder design, with optional training data. |
| `generate` | Design binders for a target PDB, including scaffold selection, sequence redesign and folding. |
| `view` | Browse structures and scores from saved design runs. |
| `provider` | Keep Complexa and ESMFold2 loaded between design runs. |
| `sample` | Generate sequences from a mubind checkpoint without a target, scaffolds or folding. |
| `train` | Train or resume a mubind model, with validation during training. |
| `evaluate` | Evaluate a checkpoint on fixed benchmarks. |

### Setup for design or training

Default setup installs the mubind checkpoint, Proteina-Complexa, ESMFold2/ESMC and Foldseek. Binder design needs no training dataset. To download and prepare ProstT5 for training, explicitly run `mubind setup --only prostt5` and choose **Install missing**, or select **ProstT5 training data** in the setup menu. Add `--yes` to start without the menu. Setup reuses matching prepared data and remembers explicit component selections, data paths and length filters.

### Sampling sequences without a target

Use `mubind sample` to generate sequences directly from a mubind checkpoint with DDIM sampling, without a target PDB, Complexa scaffolds or ESMFold2 folding. Run `mubind sample` in a terminal to open the interactive sampling menu, or pass a checkpoint path and flags to sample directly from the command line.

```bash
mubind sample artifacts/models/mubind/prostt5-300k.pt \
  --count 64 --length 60 \
  --charge -2.0 --hydrophobicity -0.1 \
  --steps 50 --out artifacts/generations/sample-run
```

- `--count` sets the number of candidates to generate (`16` by default).
- `--length` sets the requested sequence length (`60` by default, up to the checkpoint's 70-residue canvas). Length is a soft condition, and the model predicts its own padding mask.
- `--charge` and `--hydrophobicity` optionally condition on net charge and mean Eisenberg hydrophobicity in training units (`omegamp_modlamp_v1`); omit either flag to leave that property unconditioned.
- `--steps` sets the number of DDIM reverse-diffusion steps (`50` by default, up to `1000`).

Additional options include `--batch-size`, `--seed`, `--device` (`auto`, `cpu` or `cuda:0`) and `--precision` (`auto`, `fp32`, `bf16` or `fp16`). Results are saved in `artifacts/generations/` (or `--out`), with valid contiguous sequences in `sequences.fasta`, all decoded canvases and validity checks in `candidates.jsonl`, and run metadata in `generation.json`.

## How it works

The top-level `mubind` file is a launcher. It uses **uv** to prepare or reuse mubind's Python environment, then starts the `mubind` Python module inside it. This is why you do not need to activate an environment yourself, and why environment preparation can happen even before `--help`. The launcher remembers its folder choices separately from setup's data and model locations. Run `./mubind --configure-bootstrap` to change those choices without moving existing files.

Inside the environment, `src/mubind/cli/app.py` uses **Typer** to read the command and its arguments and route them to the relevant workflow. **Questionary** collects menu choices, while **Rich** displays overviews, progress and results across the modules in `src/mubind/cli/`. Arguments and menu choices feed the same workflow code.

For example, both ways of starting generation build a `DesignConfig`. The design code checks the settings, target, checkpoint and required tools before running the pipeline in `inference/design.py`. During the run, it reports progress and calls back to the interface when a scaffold choice or candidate review is needed. This keeps the generation logic separate from the terminal presentation. Training and paper evaluation have their own workflow modules behind the same command entry point.

Complexa and ESMFold2 run in separate Python environments so their dependencies can coexist with mubind's. Worker processes load these models and handle requests from the design workflow. To avoid loading them for every run, start `mubind provider` in another terminal and wait for **Provider ready**. Generation can then reuse them with matching model directory and GPU settings. Keep the provider running between designs; each design still loads its own mubind checkpoint.
