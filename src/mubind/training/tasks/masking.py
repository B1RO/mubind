"""Hide subsets of biochemical properties and 3Di structure tokens on sampled tasks.

After sampling.py picks a training task for each protein in a batch, a protein
still carries its full property vector c (length, charge, hydrophobicity) and any
3Di structure tokens retained by the task in seed_3di.

This module runs next inside TaskCollator.collate and randomly masks out parts
of those two hints so the model learns to generate sequences when only partial
conditioning is provided at inference time

1. property_mask builds batch["condition_mask"] (shape [B, P], 1 for visible and
   0 for hidden). It hides all properties on redesign tasks that are
   unconditional or 3Di-only, and during training randomly hides a subset of the
   remaining properties.
2. mask_training_threedi_conditioning optionally blanks out one or more spans of
   visible 3Di tokens on editable residues during training, setting those tokens to
   0 in seed_3di and marking them in threedi_conditioning_hidden_mask.
"""

from __future__ import annotations

from typing import Any

import torch

from .config import (
    NOT_REDESIGN_CONDITIONING_ID,
    REDESIGN_CONDITIONING_NAMES,
    REDESIGN_MODES_WITH_PROPERTIES,
    PropertyDropoutConfig,
    ThreeDiMaskingConfig,
)
from .spans import sample_edit_masks_for_budgets


def _redesign_property_flags(
    redesign_conditioning_ids: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Identify full-redesign rows and which of them are allowed to see properties."""
    if redesign_conditioning_ids.shape != (batch_size,):
        raise ValueError("redesign_conditioning_ids must have shape [B]")
    ids = redesign_conditioning_ids.to(device=device)
    valid = (ids == NOT_REDESIGN_CONDITIONING_ID) | (
        (ids >= 0) & (ids < len(REDESIGN_CONDITIONING_NAMES))
    )
    if not bool(valid.all()):
        raise ValueError("unknown redesign conditioning mode")
    redesign = ids >= 0
    with_properties = torch.zeros_like(redesign)
    for mode in REDESIGN_MODES_WITH_PROPERTIES:
        with_properties |= ids == mode
    return redesign, with_properties


def _sample_uniform_property_subset(
    properties: torch.Tensor,
    with_properties: torch.Tensor,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Pick how many properties K to keep uniformly, then keep K random columns."""
    count = properties.shape[1]
    # Inpainting rows can keep anywhere from 0 to all P properties.
    keep = torch.randint(
        0,
        count + 1,
        (len(properties),),
        device=properties.device,
        generator=generator,
    )
    # Redesign rows whose mode explicitly includes properties keep at least 1.
    redesign_keep = torch.randint(
        1,
        count + 1,
        (len(properties),),
        device=properties.device,
        generator=generator,
    )
    keep = torch.where(with_properties, redesign_keep, keep)

    # Shuffle column ranks per row and keep the first K columns.
    ranks = (
        torch.rand(properties.shape, device=properties.device, generator=generator)
        .argsort(dim=1)
        .argsort(dim=1)
    )
    return (ranks < keep[:, None]).to(properties.dtype)


def _sample_independent_property_subset(
    properties: torch.Tensor,
    config: PropertyDropoutConfig,
    redesign: torch.Tensor,
    with_properties: torch.Tensor,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Drop each property independently with probability p, restoring 1 if needed."""
    mask = (
        torch.rand(properties.shape, device=properties.device, generator=generator)
        >= config.probability
    ).to(properties.dtype)

    # If a row requires at least one property and all were dropped, restore one column.
    required = with_properties if config.allow_empty else torch.ones_like(redesign)
    empty = required & (mask.sum(dim=1) == 0)
    columns = torch.randint(
        properties.shape[1],
        (len(properties),),
        device=properties.device,
        generator=generator,
    )
    fallback = torch.nn.functional.one_hot(
        columns, num_classes=properties.shape[1]
    ).to(properties.dtype)
    return torch.where(empty[:, None], fallback, mask)


def property_mask(
    properties: torch.Tensor,
    config: PropertyDropoutConfig,
    *,
    training: bool,
    redesign_conditioning_ids: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return a 0/1 mask of shape [B, P] selecting which properties are visible.

    Redesign rows sampled as unconditional or 3Di-only always have all properties
    zeroed out, even during validation or when property dropout is disabled.
    """
    redesign, with_properties = _redesign_property_flags(
        redesign_conditioning_ids,
        properties.shape[0],
        properties.device,
    )
    if not training or not config.enabled:
        mask = torch.ones_like(properties)
    elif config.mode == "uniform_subset":
        mask = _sample_uniform_property_subset(properties, with_properties, generator)
    else:
        mask = _sample_independent_property_subset(
            properties, config, redesign, with_properties, generator
        )
    # Hide all properties on unconditional and 3Di-only redesign rows.
    return mask.masked_fill((redesign & ~with_properties)[:, None], 0)


def _hide_multiple_threedi_spans(
    eligible: torch.Tensor,
    counts: torch.Tensor,
    config: ThreeDiMaskingConfig,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Keep a random budget of eligible 3Di tokens across multiple spans and hide the rest."""
    lengths = counts.to(torch.float64)
    lower = (lengths * config.visible_min_fraction).ceil().long()
    upper = (lengths * config.visible_max_fraction).floor().long()
    nearest = (lengths * config.visible_min_fraction).round().long()
    empty_range = lower > upper
    lower = torch.where(empty_range, nearest, lower)
    upper = torch.where(empty_range, nearest, upper)

    draws = torch.rand((2, len(counts)), device=eligible.device, generator=generator)
    budgets = lower + (draws[0] * (upper - lower + 1)).floor().long()
    budgets = torch.where(draws[1] < config.full_probability, counts, budgets)

    # Partition the visible budget across the compact [0, counts[i]) index space.
    compact_visible = sample_edit_masks_for_budgets(
        counts,
        budgets,
        eligible.shape[1],
        span_count_geometric_probability=config.span_count_geometric_probability,
        dirichlet_concentration=config.dirichlet_concentration,
        generator=generator,
    )
    # Map compact indices back to full canvas positions and invert to get hidden tokens.
    positions = (eligible.long().cumsum(dim=1) - 1).clamp_min(0)
    return eligible & ~compact_visible.gather(1, positions)


def _hide_single_threedi_span(
    eligible: torch.Tensor,
    counts: torch.Tensor,
    config: ThreeDiMaskingConfig,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Hide one contiguous interval of eligible 3Di tokens on rows not kept in full."""
    random_values = torch.rand(
        (3, eligible.shape[0]),
        device=eligible.device,
        generator=generator,
    )
    partial_rows = (random_values[0] >= config.full_probability) & counts.gt(1)
    visible_fraction = config.visible_min_fraction + random_values[1] * (
        config.visible_max_fraction - config.visible_min_fraction
    )
    hidden_counts = ((1 - visible_fraction) * counts).round().long().clamp_min(1)
    hidden_counts = torch.minimum(hidden_counts, (counts - 1).clamp_min(0))
    hidden_counts = torch.where(partial_rows, hidden_counts, 0)

    starts = (random_values[2] * (counts - hidden_counts + 1)).floor().long()
    positions = eligible.long().cumsum(dim=1) - 1
    return (
        eligible
        & (positions >= starts[:, None])
        & (positions < (starts + hidden_counts)[:, None])
    )


def mask_training_threedi_conditioning(
    batch: dict[str, Any],
    config: ThreeDiMaskingConfig,
    *,
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    """Randomly blank out parts of visible 3Di on editable residues during training.

    Ground-truth targets (x), sequence seeds (seed_x), and edit_mask stay
    untouched. Hidden positions are zeroed in seed_3di and recorded in
    threedi_conditioning_hidden_mask.
    """
    if not config.enabled:
        return batch
    tokens = batch["seed_3di"]
    # Only mask 3Di tokens on real, editable residues that currently have a 3Di token.
    eligible = tokens.ne(0) & batch["residue_mask"].bool() & batch["edit_mask"].bool()
    counts = eligible.sum(dim=1)
    if config.mode == "multiple_spans":
        hidden = _hide_multiple_threedi_spans(eligible, counts, config, generator)
    else:
        hidden = _hide_single_threedi_span(eligible, counts, config, generator)
    return _apply_threedi_hidden_mask(batch, hidden)


def _apply_threedi_hidden_mask(
    batch: dict[str, Any], hidden: torch.Tensor
) -> dict[str, Any]:
    return {
        **batch,
        "seed_3di": batch["seed_3di"].masked_fill(hidden, 0),
        "threedi_conditioning_hidden_mask": hidden,
    }
