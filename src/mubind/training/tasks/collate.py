"""Turn a list of clean proteins from the dataset into a training batch.

ProstT5Dataset yields individual, complete proteins (each with its sequence x,
3Di structure tokens seed_3di, and biochemical properties c). Before the
diffusion model sees them, we need to stack a list of those proteins into a
batch and turn each one into a fill-in-the-blank task.

TaskCollator is the callable object that does this in three steps

1. collate_task_rows stacks the list of proteins into batch tensors and calls
   sampling.py to pick a task for each protein (which sequence positions to keep
   as context in seed_x, which positions to generate in edit_mask, and which 3Di
   tokens to keep in seed_3di).
2. During training, mask_training_threedi_conditioning (from masking.py)
   randomly hides chunks of the remaining 3Di tokens on editable residues so the
   model learns to design sequences from partial structure hints.
3. property_mask (from masking.py) builds a 0/1 mask (condition_mask) that hides
   some or all of the biochemical properties so the model learns to work with
   any subset of properties.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import default_collate

from .config import PropertyDropoutConfig, TaskConfig, ThreeDiMaskingConfig
from .masking import mask_training_threedi_conditioning, property_mask
from .sampling import sample_task_batch


@dataclass(frozen=True)
class TaskCollator:
    """Turn clean dataset rows into one model-ready conditioning batch."""

    config: TaskConfig
    batched_mixed: bool = False
    training_threedi_masking: ThreeDiMaskingConfig | None = None
    property_dropout: PropertyDropoutConfig = field(default_factory=PropertyDropoutConfig)
    training: bool = True

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        return self.collate(rows)

    def collate(
        self,
        rows: list[dict[str, Any]],
        *,
        generator: torch.Generator | None = None,
        step: int | None = None,
    ) -> dict[str, Any]:
        """Run task sampling, 3Di masking, and property masking on a batch of rows."""
        # Step 1. Stack the clean proteins and sample a fill-in-the-blank task for each row.
        batch = collate_task_rows(
            rows,
            self.config.for_step(step),
            generator=generator,
            batched_mixed=self.batched_mixed,
        )

        # Step 2. During training, optionally hide spans of 3Di tokens on editable residues.
        if self.training and self.training_threedi_masking is not None:
            batch = mask_training_threedi_conditioning(
                batch,
                self.training_threedi_masking,
                generator=generator,
            )

        # Step 3. Choose which biochemical properties in batch["c"] remain visible.
        batch["condition_mask"] = property_mask(
            batch["c"],
            self.property_dropout,
            training=self.training,
            redesign_conditioning_ids=batch["redesign_conditioning_id"],
            generator=generator,
        )
        return batch


def collate_task_rows(
    rows: list[dict[str, Any]],
    config: TaskConfig,
    *,
    generator: torch.Generator | None = None,
    batched_mixed: bool = False,
) -> dict[str, Any]:
    """Stack a list of clean dataset dicts into tensors and sample a task per row."""
    if not rows:
        raise ValueError("cannot collate an empty row list")
    clean_batch = default_collate(rows)
    return sample_task_batch(
        clean_batch,
        config,
        generator=generator,
        batched_mixed=batched_mixed,
    )
