# Evaluation

![Unconditional evaluation of 1,024 generated sequences](assets/mubind-unconditional-first-four-real.gif)

`mubind evaluate` runs evaluation protocols on a mubind checkpoint and saves the generated candidates, folded structures, and summary metrics in `artifacts/evaluations/`.

To open the interactive menu, run

```bash
mubind evaluate
```

Select **Benchmark** and **Checkpoint**, then choose **Start evaluation**. For full binder redesign or inpainting, **Complexes folder** selects the input PDBs. When you pick **Unconditional evaluation**, a submenu lets you toggle which sequence, novelty, distribution, and ESMC measurements to run. Use **Configure evaluation** to change the output folder, device, or reference paths.

## Benchmarks

All benchmarks generate sequences with 1,000 DDPM steps.

Property control samples fresh requests from a prepared training dataset. Structural steering and sequence-reference comparisons use the 92 test proteins bundled in `artifacts/benchmarks/prostt5.json`. These 92 proteins are all the entries in the original `Rostlab/ProstT5Dataset` test split that passed the 50–70-residue length filter (474 test entries before filtering and 92 afterward), with no truncation or further random selection. This is the test set used in the mubind project report, referred to here as the paper.

Full binder redesign and inpainting use the **45 bundled PPIRef complexes** in `artifacts/benchmarks/ppiref/complexes/` that passed the original ESMFold2 screening. Each contains an experimentally observed binder and its target. mubind obtains the 3Di and property conditions from the binder and measures structural recovery against its coordinates. These benchmarks begin with an existing binder, after the scaffold-generation stage.

ESMC sequence comparisons retain a separate set of 81 PPIRef binder sequences in `artifacts/benchmarks/ppiref/sequences.json`, alongside the 92 ProstT5 test proteins. These sequence references do not require refolding or screening the original complexes.

You can also provide your own folder of binder–target PDBs, using experimentally determined complexes or starting complexes generated with Proteina-Complexa or another tool.

| Mode | What it runs | What it measures |
| --- | --- | --- |
| `unconditional` | Samples 1,024 sequences with no structure, property, or sequence inputs, then filters out invalid canvases with gaps or empty sequences. | Uniqueness and pairwise Levenshtein diversity, exact novelty and nearest-sequence distance against the 1.5M ProstT5 training proteins, property (KS) and amino-acid composition (JS) distances against the 92 ProstT5 test proteins, and optional ESMC-6B pseudo-perplexity against ProstT5, PPIRef, and shuffled controls. |
| `properties` | Randomly draws 4,000 training rows with replacement and generates a sequence for each row’s requested properties, with no structural input, split into 1,000 each for length, net charge, hydrophobicity, and all three jointly. | Exact match rate for requested length, plus raw and training-standardized mean absolute error (MAE) between requested and realized properties. |
| `structure` | Given only the 3Di tokens of each of the 92 ProstT5 test proteins, samples 10 sequences with standard sampling and 10 with structural guidance (scale 5). Folds all 1,840 candidates and 92 native sequences as monomers with ESMFold2, encodes them into 3Di with Foldseek, and selects the highest-pLDDT candidate per target in each setting. | Cα TM-score of the selected candidate against the native sequence's ESMFold2 refold, plus exact and substitution-matrix soft 3Di token recovery against the input 3Di. |
| `full-design` | For each input complex, conditions on the binder's 3Di tokens and properties (length, charge, hydrophobicity) with no sequence context and samples 50 candidates with structural guidance (scale 5). Folds each candidate with its target using ESMFold2 and selects the top candidate per complex by binder pLDDT, ipTM, and property error. | Cα TM-score of the predicted binder against the supplied binder structure, complex interface confidence (ipTM), and property MAE. |
| `inpainting` | For each input binder, masks half its residues, rounded up, in contiguous spans across 20 distinct masks while keeping the unmasked residues, full 3Di, and properties fixed. Samples 1 completion per mask (scale 5), folds each completion with its target using ESMFold2, and selects the top candidate per complex among those preserving all unmasked residues. | Exact preservation rate of supplied residues, fraction of masked residues changed, binder TM-score, complex ipTM, and property MAE. |

## Input complexes

For your own inputs, provide a folder of PDB files, each containing exactly two protein chains in file order. **The first is the binder and the second is the target**, regardless of chain labels or lengths. mubind uses the first model and extracts the sequences from its resolved residues, computes the binder properties, and converts the isolated binder structure to 3Di with Foldseek. The binder must contain canonical amino acids and N, Cα and C backbone coordinates, with at most 70 residues. Inpainting requires at least 6 residues to draw 20 distinct masks.

The bundled PPIRef set runs 2,250 full-design candidates or 900 inpainting completions. It retains the complete target sequences used in the original screening, including residues absent from the experimental coordinates. A custom folder can contain any number of complexes; the number of candidates per complex stays the same. Input copies and derived conditioning data are saved in the run's `inputs/` folder.

## Run from the command line

Pass `run` and a mode name to run a benchmark directly without the interactive menu. The examples use the released checkpoint installed by setup at `artifacts/models/mubind/prostt5-300k.pt`; replace `artifacts/models` if you chose another models folder. You can also select a checkpoint from your own training run.

```bash
mubind evaluate run properties \
  --checkpoint artifacts/models/mubind/prostt5-300k.pt \
  --out artifacts/evaluations/properties \
  --figures
```

Default setup does not prepare training data. For `properties`, explicitly prepare it with `mubind setup --only prostt5`, or pass an existing dataset with `--dataset /path/to/prepared-dataset`. Optionally add `--seed 42` for repeatable sampling.

For `unconditional`, pass `--metrics` to select specific measurements or `--esmc` to include ESMC scoring. Novelty measurements compare against the complete 50–70-residue ProstT5 training split. Prepare it with `mubind setup --only prostt5 --min-length 50 --max-length 70`, then pass `--dataset artifacts/datasets/ProstT5Dataset_len50_70_diffusion`, adjusting the path to the dataset shown by setup. The interactive menu finds this dataset in your configured data directory. Its checksum is verified before generation. Measurements without novelty need no training corpus.

Each run writes a readable summary to `results.md`, full metrics to `summary.json`, and generated records to `candidates/`. The bundled data supplies benchmark inputs and scoring references. Historical candidate predictions, scores and figures are not distributed. The report's headline redesign and inpainting values came from a selected ten-complex panel; a new run over the default 45 complexes measures a different panel.

For full-design or inpainting evaluation, omit `--complexes` to use the 45 bundled PPIRef complexes, or pass `--complexes /path/to/complexes` to use your own folder.

```bash
mubind evaluate run full-design \
  --checkpoint artifacts/models/mubind/prostt5-300k.pt \
  --out artifacts/evaluations/full-design
```

Replace `full-design` with `inpainting` to evaluate masked completions from the same PDBs.


