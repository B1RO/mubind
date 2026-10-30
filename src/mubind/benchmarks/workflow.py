"""Checkpoint-first evaluation with preparation, stages and one final report."""

from __future__ import annotations

from dataclasses import dataclass, replace
from importlib import import_module
from pathlib import Path

from .common import (
    training_corpus,
    INPUTS,
    MODES,
    ROOT,
    new_run,
    read,
    require,
    save,
    sha256,
    verify_bundle,
)
from .unconditional import COMPUTATIONS, CORPUS_METRICS, ESMC_METRICS, selected_metrics
from ..paths import existing_artifact


EXPERIMENTS = {
    "unconditional": (
        "Sequence quality",
        "Generate 1,024 sequences; measure validity, diversity and novelty against the paper corpus.",
    ),
    "properties": (
        "Property control",
        "Test length, charge and hydrophobicity on 4,000 freshly sampled training requests.",
    ),
    "structure": (
        "Structural steering",
        "Recover 92 supplied structures; compare ordinary sampling with structural guidance.",
    ),
    "full-design": (
        "Full binder redesign",
        "Redesign each binder in the selected PDB folder; fold and rank 50 candidates per complex.",
    ),
    "inpainting": (
        "Binder inpainting",
        "Fill masked residues in the selected PDB complexes; test structure and preservation of supplied residues.",
    ),
}
STRUCTURAL = {"structure", "full-design", "inpainting"}


@dataclass
class EvaluationConfig:
    checkpoint: Path
    mode: str
    out: Path
    dataset: Path | None = None
    device: str = "auto"
    endpoint: str = "auto"
    models_dir: Path | None = None
    foldseek: str = "foldseek"
    figures: bool = False
    esmc: bool = False
    metrics: tuple[str, ...] | None = None
    complexes: Path | None = None
    seed: int | None = None

    @property
    def complexes_dir(self):
        from .complex_inputs import default_complexes

        return existing_artifact(self.complexes) if self.complexes is not None else default_complexes()

    @property
    def computations(self):
        return selected_metrics(self.metrics, self.esmc) if self.mode == "unconditional" else ()

    @property
    def needs_corpus(self):
        return bool(set(self.computations) & CORPUS_METRICS)

    @property
    def needs_esmc(self):
        return bool(set(self.computations) & ESMC_METRICS)


def prepare_evaluation(settings):
    """Check all known requirements before sampling; never start model workers."""
    import torch
    from .checkpoint_generation import inspect_checkpoint

    require(settings.mode in MODES, f"Unknown evaluation {settings.mode}")
    require(settings.seed is None or settings.mode == "properties", "--seed applies to property-control evaluation")
    require(settings.seed is None or (isinstance(settings.seed, int) and 0 <= settings.seed < 2**63),
            "Property sampling seed must be an integer from 0 to 2^63 - 1")
    require(settings.metrics is None or settings.mode == "unconditional", "Metric selection applies to unconditional evaluation")
    if settings.mode == "unconditional":
        require(settings.computations, "Choose at least one unconditional computation")
    settings = replace(
        settings,
        checkpoint=settings.checkpoint.expanduser().resolve(),
        out=settings.out.expanduser().absolute(),
        dataset=existing_artifact(settings.dataset).resolve() if settings.dataset else None,
    )
    complex_inputs = None
    if settings.mode in {'full-design', 'inpainting'}:
        from .complex_inputs import inspect_complexes

        settings = replace(settings, complexes=existing_artifact(settings.complexes_dir).resolve())
        complex_inputs = inspect_complexes(settings.complexes, settings.mode)
    require(
        not settings.out.resolve().is_relative_to(ROOT) and not settings.out.resolve().is_relative_to(INPUTS),
        "Results must be outside the evaluation source bundle",
    )
    recipe, problems = inspect_checkpoint(settings.checkpoint)
    require(problems[settings.mode] is None, problems[settings.mode])
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if settings.device == "auto"
        else torch.device(settings.device)
    )
    require(device.type in {"cpu", "cuda"}, "Evaluation supports cpu or cuda")
    require(
        device.type != "cuda"
        or (
            torch.cuda.is_available()
            and (device.index is None or device.index < torch.cuda.device_count())
        ),
        f"Requested device is unavailable: {device}",
    )
    settings = replace(settings, device=str(device))
    require(
        not settings.esmc or settings.mode == "unconditional",
        "ESMC comparison applies to Sequence quality",
    )
    require(
        not settings.figures or settings.mode == "properties",
        "Fresh figures are available for Property control",
    )
    dataset_hash = None
    property_inputs = None
    if settings.mode == "properties":
        import secrets
        from .property_inputs import default_property_dataset, inspect_property_dataset

        dataset = settings.dataset or default_property_dataset()
        property_inputs = inspect_property_dataset(dataset)
        seed = settings.seed if settings.seed is not None else secrets.randbits(63)
        property_inputs["seed"] = seed
        settings = replace(settings, dataset=Path(property_inputs["dataset"]), seed=seed)
    if settings.needs_corpus:
        require(
            settings.dataset is not None,
            "Selected novelty metrics need the original paper corpus (folder containing train/aa.npy). "
            "Prepare it with mubind setup --only prostt5 and pass --dataset.",
        )
        array = settings.dataset / "train/aa.npy"
        require(array.is_file(), f"Paper corpus not found: {array}")
        expected = training_corpus()["sha256"]
        dataset_hash = sha256(array)
        require(
            dataset_hash == expected,
            "This is not the paper's complete training corpus. Select the 50–70-residue "
            "ProstT5 dataset prepared by mubind setup --only prostt5.",
        )
    if set(settings.computations) & {"diversity", "nearest-training"}:
        import_module("rapidfuzz")
    if settings.mode in STRUCTURAL:
        from ..evaluation.config import FoldingConfig
        from ..evaluation.folding import FoldingClient
        from .folding import resolve_foldseek, validate_service

        client = FoldingClient(
            FoldingConfig(
                endpoint=settings.endpoint,
                models_dir=str(settings.models_dir) if settings.models_dir else None,
                device=settings.device,
            )
        )
        client.check_complex_ready() if settings.mode != "structure" else client.check_ready()
        if settings.endpoint != "auto":
            validate_service(client, settings.endpoint)
        executable = resolve_foldseek(settings.foldseek, settings.models_dir)
        if complex_inputs is not None:
            settings = replace(settings, foldseek=str(executable))
        import_module("tmtools")
    if settings.needs_esmc:
        from ..runtime import get_runtime
        from .esmc import ESMC_REVISION

        runtime = get_runtime("esmfold2", models_dir=settings.models_dir)
        require(
            Path(runtime.paths["esmc_model"]).name == ESMC_REVISION,
            "ESMC setup must use the paper's pinned snapshot",
        )
    if settings.figures:
        import_module("matplotlib")

    verified = verify_bundle()
    identity = {
        "operation": "checkpoint_evaluation",
        "mode": settings.mode,
        "checkpoint": str(settings.checkpoint),
        "checkpoint_sha256": sha256(settings.checkpoint),
        "checkpoint_step": recipe.step,
        "weights": recipe.weights,
        "dataset_sha256": dataset_hash,
        "device": settings.device,
        "endpoint": settings.endpoint,
        "models_dir": str(settings.models_dir) if settings.models_dir else None,
        "foldseek": settings.foldseek,
        "figures": settings.figures,
        "esmc": settings.needs_esmc,
        "bundle": verified,
    }
    if settings.mode == "unconditional":
        identity["metrics"] = list(settings.computations)
    if property_inputs is not None:
        identity["property_inputs"] = property_inputs
    if complex_inputs is not None:
        identity['complexes_dir'] = str(settings.complexes)
        identity['complexes'] = complex_inputs
    return settings, identity


def metric_lines(mode, result):
    """A short human-readable view of the unchanged metric definitions."""
    if mode == "unconditional":
        lines = []
        if "valid_fraction" in result:
            lines.append(f"Valid sequences {result['valid']:,}/{result['attempts']:,} · {result['valid_fraction']:.2%}")
        if "unique" in result:
            lines.append(f"Distinct valid sequences {result['unique']:,}/{result['valid']:,} · {result['unique_fraction']:.2%}")
        if "exact_novel" in result:
            lines.append(f"Exact novelty against the paper corpus {result['exact_novel']:,}/{result['valid']:,} · {result['exact_novel_fraction']:.2%}")
        if "pairwise_distance" in result:
            lines.append(f"Mean pairwise distance {result['pairwise_distance']:.4f} · held-out {result['test_pairwise_distance']:.4f}")
        if "nearest_training_distance" in result:
            lines.append(f"Mean distance to nearest paper training sequence {result['nearest_training_distance']:.4f}")
        for name, values in result.get("properties", {}).items():
            lines.append(f"{name.capitalize()} distribution · KS {values['ks']:.4f} · generated mean {values['generated_mean']:.4f} · held-out mean {values['test_mean']:.4f}")
        if "composition_js_bits" in result:
            lines.append(f"Amino-acid composition · Jensen–Shannon divergence {result['composition_js_bits']:.6f} bits")
        return lines
    if mode == "properties":
        lines = [
            f"Exact requested length {result['length_exact']:,}/{result['length_requested_valid']:,}"
        ]
        for name, group in result["settings"].items():
            errors = ", ".join(
                f"{prop} MAE {values['mae']:.4f}"
                for prop, values in group["metrics"].items()
            )
            lines.append(
                f"{name.replace('_', ' ').capitalize()} · {group['valid']:,}/{group['attempts']:,} valid · {errors}"
            )
        return lines
    if mode == "structure":
        return [
            f"{'Ordinary sampling' if arm == 'baseline' else 'Structural guidance'} · {values['selected']}/92 selected · mean TM-score {values['mean']['tm_to_native_refold']:.4f} · 3Di identity {values['mean']['identity']:.4f}"
            for arm, values in result.items()
        ]
    lines = [
        f"Selected candidates {result['selected']}/{result['cases']} cases",
        f"Mean binder TM-score {result['mean_binder_tm']:.4f}",
        f"Mean interface confidence (ipTM) {result['mean_iptm']:.4f}",
    ]
    if mode == "inpainting":
        lines += [
            f"Candidates preserving supplied residues {result['context_preserving_candidates']}/{result['candidates']}",
            f"Changed masked residues in selected candidates {result['changed_masked_residues']}/{result['masked_residues']}",
        ]
    return lines


def evaluation_stages(settings):
    stages = [("generation", "Generate candidates · mubind")]
    if settings.mode in {'full-design', 'inpainting'}:
        stages.insert(0, ('inputs', 'Prepare PDB complexes · Foldseek'))
    if settings.mode in STRUCTURAL:
        stages.append(("folding", "Fold and measure structures · ESMFold2"))
    if settings.mode != "unconditional" or set(settings.computations) - ESMC_METRICS:
        stages.append(("metrics", "Compute selected metrics"))
    if settings.needs_esmc:
        stages.append(("esmc", "Score sequences · ESMC"))
    if settings.figures:
        stages.append(("figures", "Save property figures"))
    return stages


def run_evaluation(settings, *, on_status=None, on_progress=None, on_result=None):
    from .checkpoint_generation import generate_checkpoint
    from .reports import score_run

    status = on_status or (lambda message: print(message, flush=True))
    result_message = on_result or status
    status("Checking checkpoint and evaluation requirements…")
    settings, identity = prepare_evaluation(settings)
    out = new_run(settings.out, identity)
    settings = replace(settings, out=out)
    status(f"Starting fresh evaluation in {out}")
    sequence_metrics = tuple(key for key in settings.computations if key not in ESMC_METRICS)
    needs_summary = settings.mode != "unconditional" or bool(sequence_metrics)
    stages = evaluation_stages(settings)

    def begin(key, message):
        if on_progress is not None:
            on_progress({"stage": key, "description": message})
        else:
            number = next(i for i, (name, _) in enumerate(stages, 1) if name == key)
            status(f"Stage {number}/{len(stages)} · {message}")

    callbacks = {"on_progress": on_progress, "on_status": status} if on_progress is not None else {}
    complexes = None
    if 'complexes' in identity:
        from .complex_inputs import prepare_complexes

        begin('inputs', 'Extracting binder and target inputs from PDB complexes…')
        complexes = prepare_complexes(settings.complexes, out / 'inputs', identity['complexes'], settings.foldseek, status)
    begin("generation", "Loading checkpoint and preparing candidates…")
    provenance = generate_checkpoint(
        settings, identity["bundle"], identity["checkpoint_sha256"], status,
        **({"on_progress": on_progress} if on_progress is not None else {}),
        **({'complexes': complexes} if complexes is not None else {}),
        **({'property_inputs': identity['property_inputs']} if 'property_inputs' in identity else {}),
    )
    if settings.mode in STRUCTURAL or settings.needs_esmc:
        import torch

        if settings.device.startswith("cuda"):
            with torch.cuda.device(settings.device):
                torch.cuda.empty_cache()
    folds = None
    if settings.mode in STRUCTURAL:
        from .folding import fold_run

        begin("folding", "Preparing structure prediction…")
        folded = fold_run(
            out / "candidates",
            settings.endpoint,
            settings.models_dir,
            settings.foldseek,
            settings.device,
            **callbacks,
        )
        folds = Path(folded["directory"])
    result = {}
    if needs_summary:
        begin("metrics", "Computing selected metrics…")
        if settings.mode == "unconditional":
            result = score_run(out / "candidates", out / "metrics", settings.dataset, sequence_metrics, **callbacks)
        else:
            result = score_run(out / "candidates", out / "metrics", settings.dataset, folds=folds, **callbacks)
    lines = metric_lines(settings.mode, result)
    for line in lines:
        result_message(line)
    notes = [provenance["evidence"]]
    if settings.needs_corpus:
        notes.append(
            "Novelty is relative to the paper's training corpus, which may differ from this checkpoint's training data."
        )
    if settings.mode in {"full-design", "inpainting"}:
        notes.append(
            "TM-score measures recovery of the supplied binder structure. Predicted structure and confidence do not establish experimental binding."
        )
    report = {
        "directory": str(out),
        "checkpoint": identity,
        "generation": provenance,
        "metrics": result,
        "notes": notes,
    }
    if settings.mode == "unconditional":
        report["computations"] = list(settings.computations)
        notes.append("All selected sequence analyses use the same 1,024 attempts; invalid canvases are excluded after the shared validity check.")
    if settings.needs_esmc:
        from .esmc import run_esmc

        begin("esmc", "Preparing ESMC sequence comparison…")
        run_esmc(
            out / "esmc",
            settings.models_dir,
            out / "candidates",
            settings.device,
            baselines="esmc" in settings.computations,
            shuffles="esmc-shuffle" in settings.computations,
            **callbacks,
        )
        comparison = read(out / "esmc/summary.json")
        report["esmc"] = comparison
        if "esmc" in settings.computations:
            esmc_line = (
                f"ESMC median pseudo-perplexity · generated {comparison['generated']['median']:.4f}"
                f" · ProstT5 {comparison['test']['median']:.4f} · PPIRef {comparison['ppiref']['median']:.4f}"
            )
            lines.append(esmc_line)
            result_message(esmc_line)
        if "esmc-shuffle" in settings.computations:
            for name in ("generated", "prostt5", "ppiref"):
                control = comparison[name + "_shuffled"]
                parents = comparison[name + "_shuffled_parents"]
                line = f"ESMC {name} shuffle controls · {control['count']} pairs · median pseudo-perplexity {parents['median']:.4f} → {control['median']:.4f}"
                lines.append(line)
                result_message(line)
            notes.append("Shuffle controls use 128 fixed generated sequences and all reference proteins, with termini preserved. Their unshuffled parent scores are included as prerequisites.")
    if settings.figures:
        from .plots import make_figures
        begin("figures", "Saving property figures…")
        make_figures(
            out / "metrics",
            {"properties": result},
            records=read(out / "candidates/samples.json")["records"],
        )
    status("Saving evaluation report…")
    save(out / "summary.json", report)
    text = [
        f"# {EXPERIMENTS[settings.mode][0]}",
        "",
        f"Checkpoint `{identity['checkpoint']}`",
        "",
        f"Step {identity['checkpoint_step']:,} · {identity['weights'].upper()} weights",
        "",
        *(["Selected computations · " + ", ".join(COMPUTATIONS[key][0] for key in settings.computations), ""] if settings.mode == "unconditional" else []),
        *[f"- {line}" for line in lines],
        "",
        *notes,
        "",
        "Full metrics are in `summary.json`; candidate records and folds are in `candidates/`.",
    ]
    (out / "results.md").write_text("\n".join(text) + "\n")
    result_message(f"Results saved to {out / 'results.md'}")
    return report
