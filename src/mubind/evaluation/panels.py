"""Fixed validation problems, preserving the archived 300k sampling order."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Any

import torch
from torch.utils.data import default_collate

from ..training.tasks import TaskCollator as ConditioningCollator
from ..training.tasks import ThreeDiMaskingConfig
from ..training.tasks.masking import mask_training_threedi_conditioning

PROPERTY_STEERING_CONDITIONS = (
    ("no_properties", (0.0, 0.0, 0.0)),
    ("length_charge", (1.0, 1.0, 0.0)),
    ("length_hydrophobicity", (1.0, 0.0, 1.0)),
    ("charge_hydrophobicity", (0.0, 1.0, 1.0)),
    ("length_only", (1.0, 0.0, 0.0)),
    ("charge_only", (0.0, 1.0, 0.0)),
    ("hydrophobicity_only", (0.0, 0.0, 1.0)),
    ("all_properties", (1.0, 1.0, 1.0)),
)


def materialize_task_batches(
    dataset: Any,
    collator: ConditioningCollator,
    *,
    sample_count: int,
    batch_size: int,
    cohort_seed: int,
    task_seed: int,
) -> list[dict[str, Any]]:
    """Create fixed held-out conditioning problems for repeated validation."""
    if sample_count < 1:
        raise ValueError("validation sample_count must be positive")
    if batch_size < 1:
        raise ValueError("validation batch_size must be positive")
    if sample_count > len(dataset):
        raise ValueError("validation sample_count exceeds dataset size")

    indices = random.Random(cohort_seed).sample(range(len(dataset)), sample_count)
    generator = torch.Generator(device="cpu").manual_seed(task_seed)
    batches: list[dict[str, Any]] = []
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        rows = [dataset[index] for index in chunk]
        batch = collator.collate(rows, generator=generator)
        batch["validation_case_id"] = [
            f"row_{dataset_index}" for dataset_index in chunk
        ]
        batch["dataset_index"] = torch.tensor(chunk, dtype=torch.long)
        batches.append(batch)
    return batches


def materialize_property_steering_task_batches(
    dataset: Any,
    *,
    sample_count: int,
    batch_size: int,
    cohort_seed: int,
) -> list[dict[str, Any]]:
    """Create balanced de-novo tasks conditioned on subsets of training properties."""
    condition_count = len(PROPERTY_STEERING_CONDITIONS)
    if sample_count < condition_count:
        raise ValueError(
            f"property steering sample_count must be at least {condition_count}"
        )
    if sample_count % condition_count != 0:
        raise ValueError(
            "property steering sample_count must be divisible by "
            f"{condition_count} for equal cohorts"
        )
    if batch_size < 1:
        raise ValueError("validation batch_size must be positive")
    if sample_count > len(dataset):
        raise ValueError("validation sample_count exceeds dataset size")

    indices = random.Random(cohort_seed).sample(range(len(dataset)), sample_count)
    per_condition = sample_count // condition_count
    assignments = [
        (dataset_index, condition_name, condition_mask)
        for (condition_name, condition_mask), condition_indices in zip(
            PROPERTY_STEERING_CONDITIONS,
            (
                indices[start : start + per_condition]
                for start in range(0, sample_count, per_condition)
            ),
            strict=True,
        )
        for dataset_index in condition_indices
    ]

    batches: list[dict[str, Any]] = []
    for start in range(0, len(assignments), batch_size):
        chunk = assignments[start : start + batch_size]
        batch = default_collate([dataset[index] for index, _, _ in chunk])
        batch["seed_x"] = torch.zeros_like(batch["x"])
        batch["seed_3di"] = torch.zeros_like(batch["seed_3di"])
        batch["edit_mask"] = torch.ones_like(batch["residue_mask"], dtype=torch.bool)
        batch["condition_mask"] = torch.tensor(
            [mask for _, _, mask in chunk], dtype=batch["c"].dtype
        )
        batch["target_length"] = batch["c"][:, 0].round().to(torch.long)
        batch["property_conditioning"] = [name for _, name, _ in chunk]
        batch["validation_case_id"] = [
            f"{name}_row_{dataset_index}" for dataset_index, name, _ in chunk
        ]
        batch["dataset_index"] = torch.tensor(
            [index for index, _, _ in chunk], dtype=torch.long
        )
        batches.append(batch)
    return batches


def materialize_threedi_task_batches(
    dataset: Any,
    *,
    sample_count: int,
    batch_size: int,
    cohort_seed: int,
    candidates_per_task: int = 1,
    dataset_indices: Sequence[int] | None = None,
    partial_mask_seed: int | None = None,
    visible_min_fraction: float = 0.35,
    visible_max_fraction: float = 0.80,
    property_condition_mask: Sequence[float] | None = None,
    property_offsets: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Create structural tasks, optionally with paired property interventions."""
    if property_condition_mask is not None and (
        len(property_condition_mask) != 3
        or any(v not in (0, 1) for v in property_condition_mask)
    ):
        raise ValueError("property_condition_mask must contain three binary values")
    if property_offsets is not None:
        if len(property_offsets) != 3 or not all(
            math.isfinite(v) for v in property_offsets
        ):
            raise ValueError("property_offsets must contain three finite values")
        if property_condition_mask is None or any(
            offset != 0 and not mask
            for offset, mask in zip(property_offsets, property_condition_mask)
        ):
            raise ValueError("property_offsets require enabled property conditions")
    if sample_count < 1:
        raise ValueError("validation sample_count must be positive")
    if batch_size < 1:
        raise ValueError("validation batch_size must be positive")
    if candidates_per_task < 1:
        raise ValueError("candidates_per_task must be positive")
    if dataset_indices is None:
        if sample_count > len(dataset):
            raise ValueError("validation sample_count exceeds dataset size")
        indices = random.Random(cohort_seed).sample(range(len(dataset)), sample_count)
    else:
        indices = [int(index) for index in dataset_indices]
        if len(indices) != sample_count:
            raise ValueError("explicit dataset_indices must match sample_count")
        if len(set(indices)) != len(indices):
            raise ValueError("explicit dataset_indices must be unique")
        if any(index < 0 or index >= len(dataset) for index in indices):
            raise ValueError("explicit dataset_indices contains an invalid index")
    assignments = [
        (dataset_index, replicate)
        for dataset_index in indices
        for replicate in range(1, candidates_per_task + 1)
    ]
    batches: list[dict[str, Any]] = []
    for start in range(0, len(assignments), batch_size):
        chunk = assignments[start : start + batch_size]
        batch = default_collate([dataset[index] for index, _ in chunk])
        residue_mask = batch["residue_mask"].bool()
        batch["seed_x"] = torch.zeros_like(batch["x"])
        batch["seed_3di"] = torch.where(
            residue_mask,
            batch["seed_3di"],
            torch.zeros_like(batch["seed_3di"]),
        )
        # Full redesign means the model must generate the amino-acid sequence
        # and its PAD placement.  The requested 3Di is the only visible input;
        # the known source length is retained solely as evaluation metadata.
        batch["edit_mask"] = torch.ones_like(residue_mask)
        batch["condition_mask"] = torch.zeros_like(batch["c"])
        batch["target_length"] = residue_mask.sum(dim=1).to(torch.long)
        if property_condition_mask is not None:
            batch["condition_mask"][:] = torch.tensor(
                property_condition_mask, dtype=batch["c"].dtype
            )
        if property_offsets is not None:
            batch["c"] = batch["c"] + torch.tensor(
                property_offsets, dtype=batch["c"].dtype
            )
        batch["validation_case_id"] = [
            f"row_{dataset_index}_rep_{replicate:02d}"
            for dataset_index, replicate in chunk
        ]
        batch["threedi_target_id"] = [
            f"row_{dataset_index}" for dataset_index, _ in chunk
        ]
        batch["candidate_replicate"] = torch.tensor(
            [replicate for _, replicate in chunk], dtype=torch.long
        )
        batch["threedi_conditioning"] = ["full_3di"] * len(chunk)
        batch["dataset_index"] = torch.tensor(
            [dataset_index for dataset_index, _ in chunk], dtype=torch.long
        )
        if partial_mask_seed is not None:
            batch["threedi_conditioning_hidden_mask"] = torch.zeros_like(residue_mask)
            masking = ThreeDiMaskingConfig(
                enabled=True,
                full_probability=0.0,
                visible_min_fraction=visible_min_fraction,
                visible_max_fraction=visible_max_fraction,
            )
            # A target keeps exactly the same mask across candidates, batch
            # sizes, checkpoints, and runs; only generation noise varies.
            for row, (dataset_index, _) in enumerate(chunk):
                single = {
                    key: batch[key][row : row + 1]
                    for key in ("seed_3di", "residue_mask", "edit_mask")
                }
                masked = mask_training_threedi_conditioning(
                    single,
                    masking,
                    generator=torch.Generator().manual_seed(
                        partial_mask_seed + 1009 * dataset_index
                    ),
                )
                batch["seed_3di"][row] = masked["seed_3di"][0]
                hidden = masked["threedi_conditioning_hidden_mask"][0]
                batch["threedi_conditioning_hidden_mask"][row] = hidden
            batch["threedi_conditioning"] = ["partial_3di"] * len(chunk)
        batches.append(batch)
    return batches


def materialize_inpainting_task_batches(dataset, config):
    batches = materialize_threedi_task_batches(
        dataset,
        sample_count=int(config.sample_count),
        batch_size=int(config.batch_size),
        cohort_seed=int(config.cohort_seed),
        dataset_indices=config.dataset_indices,
        candidates_per_task=int(config.candidates_per_task),
    )
    width = int(config.gap_length)
    for batch in batches:
        batch["seed_x"] = batch["x"].clone()
        batch["edit_mask"].zero_()
        hidden = torch.zeros_like(batch["residue_mask"], dtype=torch.bool)
        for row in range(len(batch["x"])):
            positions = batch["residue_mask"][row].nonzero().flatten()
            if width < 1 or len(positions) < width + 2:
                raise ValueError("The gap must leave a retained residue on both sides")
            gap = positions[
                (len(positions) - width) // 2 : (len(positions) - width) // 2 + width
            ]
            batch["edit_mask"][row, gap] = True
            batch["seed_x"][row, :, gap] = 0
            batch["seed_3di"][row, gap] = 0
            hidden[row, gap] = True
            batch["validation_case_id"][row] = (
                f"gap{width}_" + batch["validation_case_id"][row]
            )
        batch["threedi_conditioning_hidden_mask"] = hidden
        batch["threedi_conditioning"] = ["partial_3di"] * len(batch["x"])
    return batches
