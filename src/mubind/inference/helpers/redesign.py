"""Redesign batch packing, task tensor construction, candidate complex folding, and result export."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import torch

from ...common.properties import (
    OMEGAMP_PROPERTY_DEFINITION,
    condition_tensor,
    sequence_properties,
)
from ...dataset.embed import encode_3di
from ..config import DesignConfig
from ..predict import predict
from .checkpoint import InferenceCheckpoint
from .inpainting import check_inpainting, inpainting_seed, inpainting_spec
from .scaffolds import fold_pair, ranking_score, stage_progress, write_json
from .structural_guidance import StructuralGuidance


def redesign_batches(scaffolds: list[dict], count: int, batch_size: int):
    """Yield `(scaffold, offset_within_round, slice_count)` slices packed up to `batch_size`."""
    if (
        type(count) is not int
        or count < 1
        or type(batch_size) is not int
        or batch_size < 1
    ):
        raise ValueError("Redesign count and batch size must be positive integers")
    batch, size = [], 0
    for scaffold in scaffolds:
        offset = 0
        while offset < count:
            take = min(count - offset, batch_size - size)
            batch.append((scaffold, offset, take))
            offset += take
            size += take
            if size == batch_size:
                yield batch
                batch, size = [], 0
    if batch:
        yield batch


def redesign_seed(batch: list[tuple], seed: int, generation_round: int, stride: int) -> int:
    """Derive a deterministic 32-bit batch seed from `(seed, generation_round, batch)`."""
    # Retain the single-scaffold stream when one whole scaffold is sampled alone.
    if len(batch) == 1 and batch[0][1] == 0:
        return (seed + batch[0][0]["idx"] + generation_round * stride) % 2**32
    recipe = [
        seed,
        generation_round,
        [(s["id"], offset, n) for s, offset, n in batch],
    ]
    digest = hashlib.sha256(
        json.dumps(recipe, separators=(",", ":")).encode()
    ).digest()
    return int.from_bytes(digest[:4], "little")


def redesign_tasks(
    scaffold: dict,
    count: int,
    properties: dict[str, float] | None = None,
    *,
    edit_residues: str | None = None,
    normalized_embeddings: bool = False,
) -> dict[str, torch.Tensor]:
    """Build `predict()` conditioning tensors (`seed_x`, `seed_3di`, `edit_mask`, `c`, `condition_mask`)."""
    condition = (
        condition_tensor(
            scaffold["sequence"], property_definition=OMEGAMP_PROPERTY_DEFINITION
        )
        if properties is None
        else torch.tensor(
            [
                properties["length"],
                properties["charge"],
                properties["hydrophobicity"],
            ],
            dtype=torch.float32,
        )
    )
    tasks = {
        "seed_x": torch.zeros(count, 8, 70),
        "seed_3di": encode_3di(scaffold["input_3di"], pad_to=70)
        .expand(count, -1)
        .clone(),
        "edit_mask": torch.ones(count, 70, dtype=torch.bool),
        "c": condition.expand(count, -1).clone(),
        "condition_mask": torch.ones(count, 3),
    }
    if edit_residues is not None:
        tasks.update(
            inpainting_seed(
                inpainting_spec(scaffold["sequence"], edit_residues),
                scaffold["input_3di"],
                count,
                normalized_embeddings=normalized_embeddings,
            )
        )
    return tasks


def run_redesign_round(
    selected: list[dict],
    settings: DesignConfig,
    checkpoint: InferenceCheckpoint,
    model: torch.nn.Module,
    device: torch.device,
    seed: int,
    generation_round: int,
    property_targets: dict[str, dict[str, float]],
    inpainting_specs: dict[str, dict],
    rows: list[dict],
    manifest: dict,
    out: Path,
    on_status,
    on_progress=None,
    on_stage_progress=None,
) -> None:
    """Run Stage 4 guided DDPM sequence redesign across bounded batches and offload the U-Net afterward."""
    model.to(device)
    diffusion = checkpoint.config.diffusion.build().to(device)
    sampler = settings.sampler.lower()
    steps_per_batch = (
        diffusion.timesteps
        if sampler == "ddpm"
        else (settings.sampling_steps or 50)
    )
    batches = list(
        redesign_batches(selected, settings.num_diffusion, settings.redesign_batch_size)
    )
    total_steps = len(batches) * steps_per_batch
    output_total = len(selected) * settings.num_diffusion
    completed_outputs = 0
    candidate_offsets = {
        s["id"]: sum(row["scaffold"] == s["id"] for row in rows) for s in selected
    }
    try:
        for index, batch in enumerate(batches):
            batch_count = sum(n for _, _, n in batch)
            on_status(
                f"4/5 · Redesign batch {index + 1}/{len(batches)} · "
                f"{batch_count} sequences across {len(batch)} scaffolds…"
            )
            guided = StructuralGuidance(
                model,
                settings.structural_guidance,
                self_condition=checkpoint.config.self_condition.enabled,
                cache_conditioning=True,
            ).eval()

            def progress(
                step,
                total,
                offset=index * steps_per_batch,
                overall_total=total_steps,
            ):
                if on_progress:
                    on_progress(offset + step, overall_total)
                stage_progress(
                    on_stage_progress,
                    "redesign",
                    offset + step,
                    overall_total,
                    outputs=completed_outputs,
                    output_total=output_total,
                    step=step,
                    steps=total,
                )

            stride = max(s["idx"] for s in selected) + 1
            sample_seed = redesign_seed(batch, seed, generation_round, stride)
            edit_options = (
                dict(
                    edit_residues=settings.edit_residues,
                    normalized_embeddings=checkpoint.config.data.normalize_embeddings,
                )
                if inpainting_specs
                else {}
            )
            parts = [
                redesign_tasks(s, n, property_targets[s["id"]], **edit_options)
                for s, _, n in batch
            ]
            tasks = (
                parts[0]
                if len(parts) == 1
                else {
                    key: torch.cat([part[key] for part in parts]) for key in parts[0]
                }
            )
            manifest.setdefault("redesign_batches", []).append(
                {
                    "round": generation_round + 1,
                    "batch": index + 1,
                    "seed": sample_seed,
                    "count": batch_count,
                    "scaffolds": [
                        {"id": s["id"], "offset": offset, "count": n}
                        for s, offset, n in batch
                    ],
                    "conditioning_cache": True,
                }
            )
            write_json(out / "run.json", manifest)
            progress(0, steps_per_batch)
            prediction = predict(
                guided,
                diffusion,
                tasks,
                normalized_embeddings=checkpoint.config.data.normalize_embeddings,
                device=device,
                seed=sample_seed,
                self_condition=checkpoint.config.self_condition.enabled,
                precision="fp32",
                sampler=sampler,
                sampling_steps=None if sampler == "ddpm" else steps_per_batch,
                ddim_eta=settings.ddim_eta if sampler == "ddim" else 0.0,
                on_step=progress,
            )
            attribution = [
                (s, candidate_offsets[s["id"]] + offset + j + 1)
                for s, offset, n in batch
                for j in range(n)
            ]
            for (scaffold, candidate), decoded in zip(
                attribution, prediction.decoded, strict=True
            ):
                checks = (
                    check_inpainting(decoded, inpainting_specs[scaffold["id"]])
                    if inpainting_specs
                    else {}
                )
                fold_status = "pending" if decoded["sequence_valid"] else "invalid"
                if checks and not checks["completion_specification_preserved"]:
                    fold_status = "rejected"
                properties = sequence_properties(
                    decoded["generated"],
                    property_definition=OMEGAMP_PROPERTY_DEFINITION,
                )
                errors = {
                    key: abs(properties[key] - value)
                    for key, value in property_targets[scaffold["id"]].items()
                }
                rows.append(
                    {
                        **decoded,
                        **checks,
                        "scaffold": scaffold["id"],
                        "scaffold_idx": scaffold["idx"],
                        "candidate_id": f"candidate_{candidate:02d}",
                        "generation_round": generation_round + 1,
                        "seed": sample_seed,
                        "properties": properties,
                        "property_errors": errors,
                        "target_properties": property_targets[scaffold["id"]],
                        "scaffold_properties": scaffold["properties"],
                        "target_seq": scaffold["target_seq"],
                        "fold_status": fold_status,
                    }
                )
            write_json(out / "generated.json", rows)
            export_results(out, rows)
            completed_outputs += batch_count
            stage_progress(
                on_stage_progress,
                "redesign",
                (index + 1) * steps_per_batch,
                total_steps,
                outputs=completed_outputs,
                output_total=output_total,
                step=steps_per_batch,
                steps=steps_per_batch,
            )
            del guided, prediction, tasks, parts
    finally:
        model.cpu()
        diffusion.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def fold_and_rank_candidates(
    rows: list[dict],
    folding,
    inpainting_specs: dict[str, dict],
    manifest: dict,
    out: Path,
    on_status,
    on_stage_progress=None,
) -> list[dict]:
    """Run Stage 5 complex folding with ESMFold2 on all pending candidates and export ranked results."""
    pending = [row for row in rows if row["fold_status"] == "pending"]
    if inpainting_specs:
        accepted = sum(
            row["sequence_valid"] and row["completion_specification_preserved"]
            for row in rows
        )
        manifest.update(
            inpainting_accepted_count=accepted,
            inpainting_rejected_count=len(rows) - accepted,
        )
        write_json(out / "run.json", manifest)
        on_status(
            f"4/5 · Inpainting checks · {accepted}/{len(rows)} candidates preserve "
            "the supplied residues and length; only these continue to folding."
        )
    for index, row in enumerate(pending):
        on_status(f"5/5 · Folding and ranking candidate {index + 1}/{len(pending)}…")
        stage_progress(on_stage_progress, "candidate_folding", index, len(pending))
        failed = False
        try:
            metrics, path = fold_pair(
                folding,
                row["target_seq"],
                row["generated"],
                out / "folds" / row["scaffold"] / row["candidate_id"],
            )
            row.update(
                metrics=metrics,
                structure=path,
                selection_score=ranking_score(metrics),
                fold_status="complete",
            )
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            row.update(fold_status="failed", fold_error=str(exc))
            failed = True
        export_results(out, rows)
        stage_progress(
            on_stage_progress,
            "candidate_folding",
            index + 1,
            len(pending),
            failed=failed,
        )
    return export_results(out, rows)


def export_results(out: Path, results: list[dict]) -> list[dict]:
    """Rank candidates by `selection_score` and write `results.json`, `scores.csv`, and `candidates.fasta`."""
    ranked = sorted(
        results,
        key=lambda r: (
            r.get("selection_score") is not None,
            r.get("selection_score", -math.inf),
        ),
        reverse=True,
    )
    write_json(out / "results.json", ranked)
    inpainting_columns = (
        [
            "edit_mask",
            "fixed_residues_preserved",
            "fixed_layout_preserved",
            "inpainting_error",
        ]
        if any("edit_mask" in row for row in ranked)
        else []
    )
    with (out / "scores.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "scaffold",
                "candidate",
                "length",
                "charge",
                "hydro",
                "charge_err",
                "hydro_err",
                "binder_plddt",
                "iptm",
                "pae",
                "score",
                "status",
                "length_err",
                "target_length",
                "target_charge",
                "target_hydro",
                *inpainting_columns,
            ]
        )
        for row in ranked:
            props, errors, metrics = (
                row["properties"],
                row["property_errors"],
                row.get("metrics", {}),
            )
            writer.writerow(
                [
                    row["scaffold"],
                    row["candidate_id"],
                    props["length"],
                    props["charge"],
                    props["hydrophobicity"],
                    errors["charge"],
                    errors["hydrophobicity"],
                    metrics.get("binder_plddt", ""),
                    metrics.get("iptm", ""),
                    metrics.get("interface_pae", ""),
                    row.get("selection_score", ""),
                    row.get("fold_status", "pending"),
                    errors["length"],
                    row["target_properties"]["length"],
                    row["target_properties"]["charge"],
                    row["target_properties"]["hydrophobicity"],
                    *(row.get(key, "") for key in inpainting_columns),
                ]
            )
    with (out / "candidates.fasta").open("w") as stream:
        for row in ranked:
            if row["sequence_valid"] and row.get(
                "completion_specification_preserved", True
            ):
                stream.write(
                    f">{row['scaffold']}_{row['candidate_id']}\n{row['generated']}\n"
                )
    return ranked
