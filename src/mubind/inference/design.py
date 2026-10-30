"""Target-directed 5-stage minibinder design workflow (`design`, backing `mubind generate`).

Coordinates Complexa backbone generation, Foldseek 3Di encoding, MUbind guided
diffusion redesign, and ESMFold2 complex ranking using helpers in `helpers/` and `services/`

1. **Stage 1/5 (Scaffolds)** Extract the target chain (`target.pdb`), generate backbone
   scaffolds with Complexa (or reuse existing final PDBs), and read binder sequences
   and biochemical properties (`describe_scaffolds`).
2. **Stage 2/5 (Scaffold ranking & selection)** Start persistent ESMFold2, fold and rank
   scaffolds by `0.5 * binder_plddt + 0.5 * iptm`, and select scaffolds and target
   properties (`select_scaffolds` or `interactive_select_scaffolds`).
3. **Stage 3/5 (3Di encoding)** Extract chain B of each selected scaffold and encode its
   backbone into a 3Di token string with Foldseek (`encode_scaffolds`).
4. **Stage 4/5 (Diffusion sequence redesign)** Sample redesigned sequences across bounded
   batches with `StructuralGuidance` and DDPM (`run_redesign_round`), checking optional
   inpainting constraints and offloading the U-Net to CPU before folding.
5. **Stage 5/5 (Candidate complex folding & ranking)** Fold valid candidates with the
   target chain in ESMFold2, rank by `0.5 * binder_plddt + 0.5 * iptm`, and write
   `results.json`, `scores.csv`, and `candidates.fasta` (`fold_and_rank_candidates`).
"""

from __future__ import annotations

import hashlib
import math
import secrets
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch

from ..common.properties import OMEGAMP_PROPERTY_DEFINITION
from ..evaluation.config import FoldingConfig
from ..evaluation.folding import FoldingClient
from ..model import ProteinDiffusionUNet
from ..runtime import get_runtime
from ..setup.config import load_config as load_setup
from ..setup.tools import find_foldseek
from .config import DesignConfig, MoreCandidates, next_output
from .helpers.checkpoint import (
    InferenceCheckpoint,
    load_inference_checkpoint,
    validate_design_checkpoint,
)
from .helpers.inpainting import inpainting_spec, parse_edit_residues
from .helpers.redesign import (
    export_results,
    fold_and_rank_candidates,
    redesign_tasks,
    run_redesign_round,
)
from .helpers.scaffolds import (
    confidence,
    describe_scaffolds,
    encode_scaffolds,
    fold_pair,
    interactive_select_scaffolds,
    rank_scaffolds,
    ranking_score,
    requested_properties,
    select_scaffolds,
    stage_progress,
    write_json,
)
from .helpers.targets import Target, inspect_target, write_chain
from .services.complexa import final_complexa_pdbs, run_complexa


@dataclass
class PreparedDesign:
    """Validated design paths, target, checkpoint, U-Net model, device, and external tool handles."""

    out: Path
    target: Target
    checkpoint: InferenceCheckpoint
    model: ProteinDiffusionUNet
    device: torch.device
    runtime: Any
    pdbs: list[Path] | None
    foldseek: str
    folding: FoldingClient

    def __iter__(self):
        return iter(
            (
                self.out,
                self.target,
                self.checkpoint,
                self.model,
                self.device,
                self.runtime,
                self.pdbs,
                self.foldseek,
                self.folding,
            )
        )


@dataclass(frozen=True)
class DesignResult:
    """Summary returned when `design` completes (`directory`, `candidates`, `ranked`, `best`)."""

    directory: Path
    candidates: int
    ranked: int
    best: dict | None


def prepare_design(settings: DesignConfig) -> PreparedDesign:
    """Validate all stages before starting a GPU job or creating output files on disk."""
    if settings.target is None:
        raise ValueError(f"Choose a target PDB from {settings.targets_dir} first")
    if settings.checkpoint is None:
        raise ValueError("Choose the 300k checkpoint or a compatible mubind checkpoint")
    for name in (
        "num_complexa",
        "binder_length",
        "complexa_steps",
        "replicas",
        "num_diffusion",
        "redesign_batch_size",
        "complexa_batch_size",
    ):
        if type(getattr(settings, name)) is not int or getattr(settings, name) < 1:
            raise ValueError(f"{name} must be a positive integer")
    if settings.binder_length > 70:
        raise ValueError("The learned-inpainting canvas supports at most 70 residues")
    if settings.edit_residues is not None:
        parse_edit_residues(
            settings.edit_residues,
            length=settings.binder_length if settings.reuse_scaffolds is None else 70,
        )
    if settings.seed is not None and (
        type(settings.seed) is not int or not 0 <= settings.seed < 2**32
    ):
        raise ValueError("Seed must be between 0 and 2**32 - 1")
    if (
        not math.isfinite(settings.structural_guidance)
        or settings.structural_guidance < 0
    ):
        raise ValueError("Structural guidance must be finite and nonnegative")
    sampler = str(settings.sampler).lower()
    if sampler not in {"ddpm", "ddim"}:
        raise ValueError("sampler must be ddpm or ddim")
    if settings.sampling_steps is not None and (
        type(settings.sampling_steps) is not int or settings.sampling_steps < 1
    ):
        raise ValueError("sampling_steps must be a positive integer")
    if not math.isfinite(settings.ddim_eta) or not 0 <= settings.ddim_eta <= 1:
        raise ValueError("DDIM eta must be between 0 and 1")
    if sampler == "ddpm" and settings.ddim_eta != 0.0:
        raise ValueError("ddim_eta only applies when sampler='ddim'")
    if settings.scaffold_index is not None:
        if type(settings.scaffold_index) is not int or settings.scaffold_index < 0:
            raise ValueError("Scaffold index must be a nonnegative integer")
        if settings.all_scaffolds:
            raise ValueError("Choose a scaffold index or all scaffolds, not both")
    if settings.top_scaffolds is not None:
        if type(settings.top_scaffolds) is not int or settings.top_scaffolds < 1:
            raise ValueError("Top scaffolds must be a positive integer")
        if settings.scaffold_index is not None or settings.all_scaffolds:
            raise ValueError(
                "Top scaffolds cannot be combined with an index or all scaffolds"
            )

    out = (settings.out or next_output()).expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"Output already exists: {out}. Choose a new folder.")

    target = inspect_target(settings.target, settings.target_spec, settings.hotspots)
    checkpoint = load_inference_checkpoint(settings.checkpoint)
    validate_design_checkpoint(checkpoint.config)
    timesteps = checkpoint.config.diffusion.timesteps
    if sampler == "ddpm" and settings.sampling_steps not in {None, timesteps}:
        raise ValueError(
            "DDPM uses every diffusion timestep; choose sampler='ddim' to reduce sampling_steps"
        )
    if settings.sampling_steps is not None and settings.sampling_steps > timesteps:
        raise ValueError(
            f"Sampling steps cannot exceed {timesteps} for this checkpoint."
        )
    model = checkpoint.build_model()
    checkpoint.state.clear()

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if settings.device == "auto"
        else torch.device(settings.device)
    )
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Design supports CPU and CUDA devices")
    if device.type == "cuda" and (
        not torch.cuda.is_available()
        or (device.index is not None and device.index >= torch.cuda.device_count())
    ):
        raise ValueError(f"Requested device is unavailable: {device}")

    root = settings.models_dir or load_setup().models_dir
    runtime = None
    pdbs = None
    if settings.reuse_scaffolds is None:
        if device.type != "cuda":
            raise ValueError(
                "Complexa scaffold generation requires CUDA; existing scaffolds can be reused on CPU"
            )
        runtime = get_runtime("complexa", models_dir=root)
    else:
        pdbs = final_complexa_pdbs(settings.reuse_scaffolds)
        if settings.scaffold_index is not None and settings.scaffold_index >= len(pdbs):
            raise ValueError("Scaffold index is outside the available final scaffolds")

    executable = find_foldseek(settings.foldseek, models_dir=root)
    if executable is None:
        raise ValueError("Foldseek is not ready. Run mubind setup --only foldseek")

    folding = FoldingClient(
        FoldingConfig(
            endpoint=settings.folding_endpoint,
            models_dir=str(root),
            device=str(device),
        )
    )
    folding.check_complex_ready()
    return PreparedDesign(
        out=out,
        target=target,
        checkpoint=checkpoint,
        model=model,
        device=device,
        runtime=runtime,
        pdbs=pdbs,
        foldseek=executable,
        folding=folding,
    )


def build_run_manifest(
    settings: DesignConfig,
    seed: int,
    target: Target,
    checkpoint: InferenceCheckpoint,
) -> dict[str, Any]:
    """Build the initial `run.json` manifest dictionary for a binder design run."""
    return {
        "settings": asdict(settings),
        "seed": seed,
        "target": asdict(target),
        "target_sha256": hashlib.sha256(target.path.read_bytes()).hexdigest(),
        "checkpoint": str(settings.checkpoint.expanduser().resolve()),
        "checkpoint_step": checkpoint.step,
        "weights": checkpoint.weights,
        "sampler": settings.sampler.lower(),
        "precision": "fp32",
        "status": "running",
        "property_definition": OMEGAMP_PROPERTY_DEFINITION,
        "property_source": "Complexa scaffold chain B",
        "checkpoint_config": checkpoint.config.to_dict(),
    }


def design(
    settings: DesignConfig,
    *,
    on_status=None,
    on_progress=None,
    on_complexa_progress=None,
    on_stage_progress=None,
    on_scaffolds=None,
    on_candidates=None,
    on_properties=None,
) -> DesignResult:
    """Run the 5-stage target-directed minibinder design workflow."""
    status = on_status or (lambda message: None)

    # Pre-flight validation of target, checkpoint, device, Complexa, Foldseek, and ESMFold2.
    status("Checking target, checkpoint, and model runtimes…")
    out, target, checkpoint, model, device, runtime, pdbs, foldseek, folding = (
        prepare_design(settings)
    )
    settings = replace(settings, out=out)
    seed = settings.seed if settings.seed is not None else secrets.randbelow(2**32)
    out.mkdir(parents=True, exist_ok=False)
    manifest = build_run_manifest(settings, seed, target, checkpoint)
    write_json(out / "run.json", manifest)

    rows: list[dict] = []
    try:
        # 1. Stage 1/5: Write target chain and generate (or reuse) Complexa scaffolds.
        target_path = out / "target.pdb"
        write_chain(target.path, target_path, target.chain, target.residue_ids)
        status(
            "1/5 · Starting Complexa…"
            if pdbs is None
            else "1/5 · Reusing final Complexa scaffolds…"
        )
        if pdbs is None:
            pdbs = run_complexa(
                settings,
                target,
                target_path,
                out,
                runtime,
                seed,
                device,
                status,
                on_progress=on_complexa_progress,
            )
        scaffolds = describe_scaffolds(pdbs, target, out, status)

        # 2. Stage 2/5: Start persistent ESMFold2, rank/select scaffolds, and resolve property targets.
        status("2/5 · Preparing persistent ESMFold2…")
        folding.start(status)
        manifest["folding_protocol"] = folding.protocol()
        if on_scaffolds is not None:
            selected, property_targets, settings = interactive_select_scaffolds(
                scaffolds,
                settings,
                target,
                target_path,
                out,
                runtime,
                seed,
                device,
                folding,
                manifest,
                status,
                on_scaffolds,
                on_complexa_progress=on_complexa_progress,
                on_stage_progress=on_stage_progress,
            )
        else:
            selected = select_scaffolds(
                scaffolds, settings, folding, out, status, on_stage_progress
            )
            property_targets = requested_properties(
                selected, edit_residues=settings.edit_residues
            )
        if on_properties is not None:
            overrides = on_properties(selected)
            if overrides is None:
                raise KeyboardInterrupt
            property_targets = requested_properties(
                selected, overrides, edit_residues=settings.edit_residues
            )
        inpainting_specs = {}
        if settings.edit_residues is not None:
            inpainting_specs = {
                s["id"]: inpainting_spec(s["sequence"], settings.edit_residues)
                for s in selected
            }
            manifest["inpainting"] = inpainting_specs
        manifest.update(
            status="running",
            settings=asdict(settings),
            selected_scaffolds=[s["id"] for s in selected],
            requested_properties=property_targets,
        )
        write_json(out / "run.json", manifest)

        # 3. Stage 3/5: Encode selected scaffold backbones into 3Di strings with Foldseek.
        encode_scaffolds(selected, out, foldseek, status)

        # 4. Stages 4/5 & 5/5: Guided DDPM sequence redesign and complex folding/ranking.
        generation_round = 0
        while True:
            run_redesign_round(
                selected,
                settings,
                checkpoint,
                model,
                device,
                seed,
                generation_round,
                property_targets,
                inpainting_specs,
                rows,
                manifest,
                out,
                status,
                on_progress=on_progress,
                on_stage_progress=on_stage_progress,
            )
            ranked = fold_and_rank_candidates(
                rows,
                folding,
                inpainting_specs,
                manifest,
                out,
                status,
                on_stage_progress=on_stage_progress,
            )
            manifest["candidate_count"] = len(rows)
            manifest["status"] = "awaiting_review" if on_candidates else "running"
            write_json(out / "run.json", manifest)

            extra = on_candidates(ranked) if on_candidates is not None else None
            if extra is None:
                break
            additional = (
                extra if isinstance(extra, MoreCandidates) else MoreCandidates(extra)
            )
            if type(additional.count) is not int or additional.count < 1:
                raise ValueError(
                    "Additional sequences per scaffold must be a positive integer"
                )
            if additional.properties is not None:
                property_targets = requested_properties(
                    selected,
                    additional.properties,
                    edit_residues=settings.edit_residues,
                )
            settings = replace(settings, num_diffusion=additional.count)
            generation_round += 1
            manifest.setdefault("additional_redesign_batches", []).append(
                {
                    "round": generation_round + 1,
                    "sequences_per_scaffold": additional.count,
                    "scaffolds": [scaffold["id"] for scaffold in selected],
                    "requested_properties": property_targets,
                }
            )
            manifest.update(
                status="running",
                settings=asdict(settings),
                requested_properties=property_targets,
            )
            write_json(out / "run.json", manifest)

        successes = [r for r in ranked if r["fold_status"] == "complete"]
        manifest["status"] = (
            "complete"
            if len(successes) == len(rows)
            else "partial"
            if successes
            else "failed"
        )
        return DesignResult(
            out, len(rows), len(successes), successes[0] if successes else None
        )
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        raise
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        if rows:
            export_results(out, rows)
        write_json(out / "run.json", manifest)
        folding.close()  # Leaves a shared persistent worker running.


__all__ = [
    "DesignResult",
    "PreparedDesign",
    "confidence",
    "describe_scaffolds",
    "design",
    "encode_scaffolds",
    "export_results",
    "fold_pair",
    "prepare_design",
    "rank_scaffolds",
    "ranking_score",
    "redesign_tasks",
    "requested_properties",
    "select_scaffolds",
    "stage_progress",
    "write_json",
]
