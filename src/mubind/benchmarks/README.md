# Benchmarks

This package backs `mubind evaluate`. It takes a trained checkpoint, runs one of the five paper evaluation protocols, and writes metrics, plots, and top-ranked PDB structures to `artifacts/evaluations/`.

```text
benchmarks/
├── workflow.py               Plans and runs the evaluation stages and writes results.md
├── common.py                 Protocol definitions, JSON/gzip I/O, and reference bundle verification
├── unconditional.py          Selectable sequence, corpus-novelty, and ESMC metric definitions
├── property_inputs.py        Samples 4,000 property-steering targets from the ProstT5 dataset
├── complex_inputs.py         Validates binder-target PDBs and extracts reference binder 3Di tokens
├── generation.py             Builds conditioning batches, guidance scales, and inpainting masks
├── checkpoint_generation.py  Checks checkpoint compatibility and runs 1,000-step DDPM sampling
├── folding.py                Folds monomer and complex candidates and scores geometry with TM-align
├── esmc.py                   Builds sequence cohorts and launches the ESMC-6B scoring subprocess
├── esmc_worker.py            Isolated worker computing leave-one-out ESMC-6B log-likelihood and perplexity
├── metrics.py                Computes paper metrics, distribution distances, and candidate rankings
├── reports.py                Loads generated samples and folding scores to write summary.json
└── plots.py                  Renders the 2x3 requested-versus-generated property steering figure
```

## The five modes and their shared pipeline

The five benchmark modes test two kinds of behavior, but `workflow.py` runs all of them through the same staged pipeline (`inputs -> generation -> folding -> metrics -> esmc -> figures`), automatically skipping stages that a mode does not need.

- **Sequence-level modes (`unconditional` and `properties`)** never fold proteins. `unconditional` generates 2,000 unconditioned sequences and tests whether they look like natural proteins in composition, diversity, novelty against the training set, and language-model plausibility. `properties` generates 4,000 sequences conditioned on target length, net charge, and hydrophobicity (one property at a time or all three together) and measures how closely the decoded sequences match the requested values.
- **Structure-conditioned modes (`structure`, `full-design`, and `inpainting`)** test whether generated sequences fold back into the target 3D geometry. `structure` conditions on monomer 3Di tokens from 92 held-out ProstT5 proteins and folds each candidate as a monomer. `full-design` and `inpainting` start from 45 PPIRef binder-target complexes, either redesigning the entire binder from its 3Di scaffold (`full-design`) or refilling 20 masked half-sequence regions while keeping the rest of the binder fixed (`inpainting`), and fold each candidate binder together with its target partner.

## How a benchmark run flows through the code

1. **Pre-flight checks and conditioning inputs (`workflow.py`, `common.py`, `property_inputs.py`, `complex_inputs.py`).** Before any GPU sampling begins, `workflow.py` verifies that required datasets, Foldseek binaries, and model environments exist, and asks `checkpoint_generation.py` whether the checkpoint was trained with the conditioning modules required by the chosen mode. It then prepares the conditioning targets. `unconditional` needs none. `properties` (`property_inputs.py`) draws 4,000 target property vectors from the ProstT5 training distribution. `structure` loads the 92 held-out ProstT5 proteins from the checksummed reference bundle in `artifacts/benchmarks/` (`common.py`). `full-design` and `inpainting` (`complex_inputs.py`) validate a folder of two-chain binder-target PDBs, snapshot the structures, and run Foldseek on chain A to extract the reference binder 3Di tokens.
2. **Batch construction and DDPM sampling (`generation.py`, `checkpoint_generation.py`).** `generation.py` turns those inputs into task tensors (`seed_x`, `seed_3di`, `edit_mask`, `c`, and `condition_mask`) paired with deterministic seeds and classifier-free guidance scales (`1.0`, `1.5`, `2.0`, and `3.0`). For `inpainting`, it cuts out 20 distinct half-sequence masks per complex using up to three contiguous spans so every target is tested across multiple edit layouts. `checkpoint_generation.py` then runs 1,000-step DDPM reverse sampling through `inference.predict`, wrapping the U-Net in `StructuralGuidance` whenever the guidance scale is not `1.0`, and writes the decoded sequences and their biochemical properties to `samples.json`.
3. **External verification via folding or ESMC-6B (`folding.py`, `esmc.py`, `esmc_worker.py`).** Once `samples.json` is on disk, the structural modes call `folding.py` to fold every candidate with ESMFold2. Monomer refolds in `structure` mode are re-encoded into 3Di strings with Foldseek and aligned to native refolds with TM-align to measure 3Di recovery, TM-score, and RMSD. Complex refolds in `full-design` and `inpainting` fold the candidate binder alongside the target chain, extract predicted confidence metrics (`pLDDT`, `ipTM`, interface `pAE`), and align the predicted binder to the native binder. In `unconditional` mode, `esmc.py` instead packages the generated sequences alongside held-out test sets (`ProstT5` and `PPIRef`) and shuffled-interior negative controls, then spawns `esmc_worker.py` inside the isolated `esmfold2` virtual environment to compute leave-one-residue-out `ESMC-6B` pseudo-perplexity.
4. **Scoring, candidate selection, and reporting (`metrics.py`, `reports.py`, `unconditional.py`, `plots.py`).** `reports.py` reads the saved samples and folding scores and delegates the math to `metrics.py`. Depending on the mode, `metrics.py` computes training-corpus novelty and distribution distances (Kolmogorov-Smirnov and Jensen-Shannon divergence), property steering errors (`MAE`, `RMSE`, Pearson `r`), or structural recovery rates. For complex modes it also ranks all candidates for each target by a composite score combining `pLDDT`, `ipTM`, `pAE`, `RMSD`, and 3Di agreement so `workflow.py` can export the winning complex PDBs into `candidates/` alongside `summary.json`, `results.md`, and (for `properties`) the scatter plots from `plots.py`.
