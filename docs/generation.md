# Generating minibinders

## How to

Run

```bash
mubind generate
```

Or launch the interactive menu with

```bash
mubind
```

Then choose **Generate** (from a repository checkout without `mubind` on `PATH`, use `./mubind`).

![PD-L1 generation through the interactive CLI](assets/mubind-pdl1-real-1000-candidates-3x-compute.gif)

PD-L1 generation through the CLI, with computation sped up.

If needed, complete the [setup instructions](../README.md) first.

1. Choose **Target**. Select a bundled example, place your own PDB in `artifacts/targets/`, or choose another file path.
2. Check **Checkpoint** and choose **Start design**. Use **Configure generation** to change the defaults before starting.
3. Review the initial binders (**scaffolds**) and their rankings. Proteina-Complexa may not produce promising scaffolds with the chosen sampling settings on its first attempt, so you can enter `more` to generate another batch. The new scaffolds are ranked together with the previous ones, and you can repeat until you are satisfied. Then choose which to redesign: enter `3` for the top three, or `1,2,5` for specific ranks.
4. Choose how many sequences to generate **per selected scaffold**. Three scaffolds and 20 sequences each request 60 redesigns.
5. Review the requested **length, charge and hydrophobicity** for each scaffold. These default to values computed from its Proteina-Complexa sequence. Press Enter to keep a value or type a replacement.
6. After generation and folding, review the candidates. Choose **Finish and save**, or generate more sequences for the selected scaffolds. You can change the requested properties for the next batch; its results are added to the same run.

Results are saved throughout the run in a new folder under `artifacts/generations/`. The terminal prints its path when finished.

## Stages of the design run

The following stages give an overview of what happens during a design run. For more details, see the paper.

| Stage | What happens |
| --- | --- |
| Generate scaffolds | Proteina-Complexa proposes initial binder sequences and structures for the selected target. |
| Select scaffolds | ESMFold2 predicts each scaffold sequence together with the target sequence. The resulting confidence scores help you choose which scaffolds to redesign. |
| Convert to descriptor | Foldseek encodes each binder backbone as **3Di**, while length, charge and hydrophobicity are computed from its sequence. Together these form the descriptor for redesign; the property values can be changed before generation. |
| Redesign sequences | mubind generates new sequences from each selected scaffold's 3Di and requested length, charge and hydrophobicity. |
| Fold and rank | ESMFold2 predicts the redesigned complexes. mubind ranks the candidates and saves their sequences, structures, confidence scores and property errors. |

By default, mubind redesigns the complete sequence using the scaffold's 3Di and requested properties, or inpaints selected binder spans when `--edit-residues` is set. It receives no target information directly; the target is used by Complexa and the complex predictions.

## Interpreting results

The results report the following measures for each redesigned candidate:

| Measure | Interpretation |
| --- | --- |
| Binder pLDDT | Confidence in the binder's predicted local structure, on a 0–1 scale; higher is better. |
| ipTM | Confidence in the relative arrangement of target and binder, on a 0–1 scale; higher is better. |
| Interface PAE | Mean predicted error between chains, in ångströms; lower is better. |
| Property errors | Absolute differences between realized and requested length, charge and hydrophobicity; lower is better. |

Both scaffolds and redesigned candidates are sorted from highest to lowest score:

`score = 0.5 × binder_pLDDT + 0.5 × ipTM`

## Configuring a run

Open **Configure generation** before **Start design** to change how many scaffolds to generate, their length, the number of redesigns, or the output folder. Target region and hotspots are optional Proteina-Complexa settings; bundled targets may provide their own length and target presets.

Choose the mubind model through **Checkpoint**. Scaffold selection and requested properties can be set later, when the run prompts you.

### Run from the command line

Pass a target path and options to skip the initial settings menu:

```bash
mubind generate artifacts/targets/protein.pdb \
  --top-scaffolds 3 --num-diffusion 20 \
  --seed 42 --out artifacts/generations/my-design
```

This selects the three highest-ranked scaffolds and requests 20 sequences from each. In a terminal, the run still asks for properties and offers the final candidate review.

The main options are:

| Option | Default | What it changes |
| --- | --- | --- |
| `--num-complexa` | `100` | How many initial scaffolds Proteina-Complexa generates. |
| `--binder-length` | `64` | Length of those initial binders, up to 70 residues. |
| `--top-scaffolds` | Choose after comparison | How many highest-ranked scaffolds continue to redesign, without prompting for selection. |
| `--num-diffusion` | `5` | How many new sequences mubind generates per selected scaffold. |
| `--edit-residues` | Full-sequence redesign | 1-based inclusive binder positions or ranges to inpaint, such as `12-20,35-40`. |
| `--checkpoint` | Automatic selection | mubind checkpoint used for redesign. |

To aim Complexa at a specific domain or binding site instead of the whole protein, pass `--target-spec A10-120` to restrict the target to chain A, residues 10–120, and `--hotspots A35,A47` to steer scaffolds toward those residues. Without `--target-spec`, mubind uses the longest protein chain in the first PDB model.

Because `--binder-length` only sets the initial Complexa scaffold length, a terminal run still pauses after ranking so you can adjust each selected scaffold's target length, charge and hydrophobicity before redesign. To run end to end without prompts, such as in a batch job or with `</dev/null`, mubind keeps each scaffold's own properties, defaults to the top-ranked scaffold if none was preselected, and finishes after one redesign round. You can also pass `--scaffold-index` or `--all-scaffolds` to skip scaffold folding and redesign those scaffolds directly.

Each run writes to a new output folder and keeps any stages completed before an interruption. Run `mubind generate --help` for the remaining sampling, batch-size and device options, or use `mubind sample` to sample sequences from a checkpoint without a target or structure folding.

#### Inpaint selected residues

To redesign only selected binder spans while keeping all other scaffold residues and the binder length fixed, pass `--edit-residues` with 1-based inclusive positions or ranges

```bash
mubind generate artifacts/targets/protein.pdb \
  --edit-residues 12-20,35-40 \
  --top-scaffolds 3 --num-diffusion 20
```

mubind keeps the full scaffold 3Di tokens and masks the sequence seed channels at the specified positions for every selected scaffold. During property selection, each scaffold's length stays fixed to its original residue count while target charge and hydrophobicity can still be changed. Only completions that preserve all unmasked residues and the original length without internal gaps continue to ESMFold2 folding and `candidates.fasta`; candidates that change fixed positions or padding are marked `rejected` in `scores.csv` and `results.json`.

#### Reuse scaffolds

To generate another set of sequences from a previous run's scaffolds:

```bash
mubind generate artifacts/targets/protein.pdb \
  --reuse-scaffolds artifacts/generations/my-design/complexa_scaffolds \
  --top-scaffolds 3 --num-diffusion 50 \
  --out artifacts/generations/redesign-2
```

Use the same target and region, and pass the **Complexa output directory containing `rewards_*.csv` and its referenced PDBs**. This skips Complexa generation; scaffold comparison, sequence redesign and candidate folding still run.

## Pipeline outputs

Each run saves its results in a separate folder. By default, this is
`artifacts/generations/run-YYYY-MM-DD_HH-MM-SS/`, relative to the directory where you launched
mubind, such as the repository root in the walkthrough above. For example:

```text
artifacts/generations/run-2026-10-24_14-30-25/
├── scores.csv             # Candidate ranking, confidence scores and property errors
├── candidates.fasta       # Generated amino-acid sequences
├── results.json           # Full candidate records, structure paths and errors
├── folds/                 # Predicted complexes for redesigned sequences
├── complexa_scaffolds/    # Initial Proteina-Complexa outputs
├── scaffold_folds/        # Predicted complexes used to compare scaffolds
└── run.json               # Run settings and status
```
