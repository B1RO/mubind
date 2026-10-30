"""Run evaluation tasks through the supported mubind inference API."""

from __future__ import annotations

import platform
from pathlib import Path

import torch

from ..inference.helpers.checkpoint import (
    load_inference_checkpoint,
    validate_design_checkpoint,
)
from ..inference.helpers.structural_guidance import StructuralGuidance
from ..inference.predict import inference_context, predict
from .common import (
    MODES,
    new_run,
    require,
    save,
    sha256,
)
from .generation import build_jobs


def incompatibility(config, mode):
    """Explain an unsupported protocol instead of changing its conditions."""
    require(mode in MODES, f"Unknown evaluation {mode}")
    try:
        validate_design_checkpoint(config)
    except ValueError as exc:
        return (
            str(exc)
            .replace("Binder design", "Paper evaluations")
            .replace("This workflow", "Paper evaluations")
        )
    if config.diffusion.timesteps != 1000:
        return "Paper evaluations require a checkpoint trained with 1,000 diffusion timesteps."
    if (
        mode in {"unconditional", "properties", "structure"}
        and not config.properties.mask_enabled
    ):
        return "This evaluation requires a checkpoint trained with property masks."
    return None


def inspect_checkpoint(path):
    recipe = load_inference_checkpoint(Path(path), metadata_only=True)
    problems = {mode: incompatibility(recipe.config, mode) for mode in MODES}
    if all(problems.values()):
        raise ValueError(next(iter(problems.values())))
    # Inspect only tensor shapes on the menu. Strict loading happens before sampling.
    return recipe, problems


def inpainting_canvas(
    source_seed_x, source_seed_3di, *, seed_length, internal_edit_mask, terminal_pad=0
):
    """The paper's zero-extension, zero-offset learned inpainting canvas."""
    require(terminal_pad == 0, "Paper inpainting has no terminal extensions")
    require(
        internal_edit_mask.shape == (seed_length,), "Inpainting mask length mismatch"
    )
    seed_x = source_seed_x.clone()
    mask = torch.zeros(source_seed_x.shape[-1], dtype=torch.bool)
    mask[:seed_length] = internal_edit_mask
    seed_x[:, mask] = 0
    return {"seed_x": seed_x, "seed_3di": source_seed_3di.clone(), "edit_mask": mask}


def candidate_records(mode, tasks, decoded, normalized, metadata, scale, seed, baselines=None):
    from ..evaluation.metrics import run_checks
    from .metrics import annotate_pipeline, properties

    if metadata is not None:
        require(baselines, "Candidate records require prepared complex inputs")
        records = []
        for spec, output in zip(metadata, decoded, strict=True):
            row = {**output, **spec, "batch_seed": seed}
            annotate_pipeline(row, baselines[row["ppi_id"]])
            records.append(row)
        return records
    records = run_checks(
        tasks, decoded, ["sequence_validity"], normalized_embeddings=normalized
    )
    for i, row in enumerate(records):
        if mode == "structure":
            row["arm"] = "guidance5" if scale == 5 else "baseline"
        if mode == "properties":
            if "dataset_index" in tasks:
                row["dataset_index"] = int(tasks["dataset_index"][i])
            observed = properties(row["generated"]) if row["sequence_valid"] else {}
            for j, prop in enumerate(("length", "charge", "hydrophobicity")):
                row[f"requested_{prop}"] = float(tasks["c"][i, j])
                row[f"observed_{prop}"] = observed.get(prop)
    return records


def generate_checkpoint(settings, verified, checkpoint_hash, on_status, *, on_progress=None, complexes=None, property_inputs=None):
    """Generate candidates with the selected checkpoint and recorded inputs."""
    recipe = load_inference_checkpoint(settings.checkpoint)
    problem = incompatibility(recipe.config, settings.mode)
    require(problem is None, problem)
    # Strict architecture/weight validation happens before output or generation.
    model = recipe.build_model()
    require(
        sha256(settings.checkpoint) == checkpoint_hash,
        "Checkpoint changed while loading. Select a stable checkpoint and retry.",
    )
    config = recipe.config
    protocol = {
        "schema": 1,
        "operation": "evaluate_checkpoint",
        "mode": settings.mode,
        "checkpoint": str(settings.checkpoint.expanduser().resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_step": recipe.step,
        "weights": recipe.weights,
        "checkpoint_config": config.to_dict(),
        "implementation": "mubind.inference.predict",
        "evidence": "Fresh evaluation of the selected checkpoint on the fixed paper cases using the current inference API.",
        "precision": "fp32",
        "device": settings.device,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "bundle": verified,
        "sampler": "ddpm",
        "sampling_steps": 1000,
    }
    baselines = None
    property_requests = None
    if settings.mode == "properties":
        from .property_inputs import sample_property_requests

        require(property_inputs is not None, "Property control needs prepared training inputs")
        snapshot = settings.out / 'inputs/property_requests.json'
        on_status(f"Sampling training property requests · seed {property_inputs['seed']}")
        property_requests = sample_property_requests(property_inputs)
        protocol['property_inputs'] = property_inputs
        protocol['evidence'] = (
            f"Fresh evaluation on 4,000 property requests sampled uniformly with replacement from the training split of {property_inputs['dataset']}. "
            f"Sampling and generation seed {property_inputs['seed']}; standardized errors use this dataset's training standard deviations."
        )
    if complexes is not None:
        protocol['complexes'] = complexes
        protocol['evidence'] = (
            f'Fresh evaluation of the selected checkpoint on {len(complexes)} supplied PDB complexes. '
            'Binder references and conditioning inputs come from the first protein chain; '
            'the second protein chain supplies the target sequence for folding.'
        )
        baselines = {case['ppi_id']: case for case in complexes}
    jobs = build_jobs(
        settings.mode, torch, config.data.normalize_embeddings,
        **({'complexes': complexes} if complexes is not None else {}),
        **({'property_requests': property_requests, 'seed': property_inputs['seed']} if property_requests is not None else {}),
    )
    protocol["jobs"] = [
        {
            "name": name,
            "seed": seed,
            "structural_guidance": scale,
            "batch_sizes": [len(b["seed_x"]) for b in batches],
        }
        for name, seed, scale, batches, _ in jobs
    ]
    out = new_run(settings.out / "candidates", protocol)
    if property_requests is not None:
        save(snapshot, {'provenance': property_inputs, 'records': property_requests})
    # Release optimizer-free CPU checkpoint tensors before folding needs the GPU.
    recipe.state.clear()
    device = torch.device(settings.device)
    model = model.to(device)
    diffusion = config.diffusion.build().to(device)
    records = []
    batch_count = sum(len(batches) for _, _, _, batches, _ in jobs)
    total_steps = batch_count * protocol["sampling_steps"]
    completed_batches = 0

    def report(completed, *, batch=None, step=None, steps=None):
        if on_progress is not None:
            on_progress({
                "stage": "generation", "completed": completed, "total": total_steps,
                "description": f"Sampling batch {batch}/{batch_count}",
                "detail": f"{completed:,}/{total_steps:,} steps",
                "step_detail": f"batch step {step}/{steps}" if step is not None else "",
            })

    for name, seed, scale, batches, metadata in jobs:
        batch_records = []
        with inference_context(model, device, seed=seed):
            for batch_index, tasks in enumerate(batches, 1):
                report(completed_batches * protocol["sampling_steps"], batch=completed_batches + 1)
                selected = (
                    StructuralGuidance(
                        model, scale, self_condition=config.self_condition.enabled
                    )
                    if scale != 1
                    else model
                )

                def progress(done, total):
                    if on_progress is not None:
                        report(completed_batches * protocol["sampling_steps"] + done,
                               batch=completed_batches + 1, step=done, steps=total)
                    elif done == 1 or done == total or done % 100 == 0:
                        on_status(
                            f"Generating {name} · batch {batch_index}/{len(batches)} · step {done}/{total}"
                        )

                prediction = predict(
                    selected,
                    diffusion,
                    tasks,
                    normalized_embeddings=config.data.normalize_embeddings,
                    device=device,
                    seed=None,
                    self_condition=config.self_condition.enabled,
                    precision="fp32",
                    sampler=protocol["sampler"],
                    sampling_steps=protocol["sampling_steps"],
                    property_guidance_scale=1.0,
                    on_step=progress,
                )
                batch_records.extend(
                    candidate_records(
                        settings.mode,
                        tasks,
                        prediction.decoded,
                        config.data.normalize_embeddings,
                        metadata,
                        scale,
                        seed,
                        baselines,
                    )
                )
                completed_batches += 1
        records.extend(batch_records)
    on_status("Saving generated candidates…")
    samples = {"mode": settings.mode, "records": records}
    if property_inputs is not None:
        samples['property_inputs'] = property_inputs
    if complexes is not None:
        samples['complexes'] = complexes
    save(out / "samples.json", samples)
    return protocol
