# Setup

This folder backs the `mubind setup` command and resolves installed assets at runtime. It downloads external model weights, installs tools, builds isolated Python environments, and optionally prepares training data so generation, evaluation, and training can run without conflicting dependencies.

## Structure

```text
setup/
├── workflow.py     Orchestrates setup and readiness checks across all components
├── config.py       Defines SetupConfig and persists paths and choices in ~/.config/mubind/setup.json
├── checkpoint.py   Downloads, verifies, and resolves the default MUbind checkpoint
├── models.py       Downloads and checksums Complexa, ESMFold2, ESMC-6B, and CCD assets
├── runtimes.py     Builds and verifies isolated uv environments for Complexa and ESMFold2
└── tools.py        Installs and locates the pinned Foldseek binary
```

## Why it lives outside cli

The interactive menus and progress displays for `mubind setup` live in `src/mubind/cli/app.py`, while `src/mubind/setup/` holds the UI-free backend logic. This matches how `cli/` delegates other commands to `training/`, `inference/`, and `benchmarks/`.

Keeping `setup/` outside `cli/` also prevents inverted dependencies across the codebase. Several runtime modules import from `setup/` to locate installed assets without pulling in Typer, Questionary, or Rich

- `runtime.py` uses `setup.config`, `setup.runtimes`, and `setup.workflow.ready_model` to resolve verified worker environments and model paths for Complexa and ESMFold2.
- `inference/design.py`, `inference/config.py`, and `inference/services/folding_service.py` use `setup` to find Foldseek, the default checkpoint, and pinned Hugging Face model IDs.
- `benchmarks/` and `evaluation/structure.py` use `find_foldseek` and `load_config` from `setup` to locate Foldseek and prepared datasets.

## Modules

1. **`workflow.py`** (`component_status`, `ready_model`, `setup_component`) checks readiness and runs setup for the five selectable components (`mubind`, `complexa`, `esmfold2`, `foldseek`, and `prostt5`). It uses file locks to prevent concurrent writes, delegates ProstT5 preparation to `preprocessing/`, and records completion state and file fingerprints in `.mubind/<component>.json` manifests.
2. **`config.py`** (`SetupConfig`, `load_config`, `save_config`) persists user settings in `~/.config/mubind/setup.json`, including the dataset and model directories (`artifacts/datasets` and `artifacts/models` by default), selected components, and sequence length filters.
3. **`checkpoint.py`** (`setup_checkpoint`, `resolve_default_checkpoint`) downloads, checksums, and locates the default pretrained MUbind checkpoint (`prostt5-300k.pt`) from Hugging Face without importing PyTorch or initializing a GPU.
4. **`models.py`** (`download_models`) handles resumable downloads and SHA-256 verification for external model weights, fetching Proteina-Complexa files from NVIDIA NGC and ESMFold2, ESMC-6B, and CCD files from Hugging Face.
5. **`runtimes.py`** (`ensure_runtime`, `runtime_ready`) builds and probes separate `uv` virtual environments for `complexa` and `esmfold2` under `.mubind/runtimes/` so their PyTorch, Transformers, and biology library versions do not clash with MUbind or each other.
6. **`tools.py`** (`ensure_foldseek`, `find_foldseek`) downloads, verifies, and locates the pinned Foldseek executable for Linux or macOS when it is not already available on `PATH`.

## Usage

Run interactive or unattended setup from the CLI

```bash
mubind setup
mubind setup --yes --only mubind,foldseek
```

Or query saved settings and run a component setup from Python

```python
from mubind.setup.config import load_config
from mubind.setup.workflow import component_status, setup_component

config = load_config()
status = component_status("mubind", config)
result = setup_component("mubind", config, emit=lambda event: None)
```
