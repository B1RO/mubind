"""Sample sequence editing, extension, and redesign tasks on a fixed canvas.

Given clean left-aligned proteins from ProstT5Dataset (x, seed_3di, c,
residue_mask), this module turns each row into a fill-in-the-blank diffusion
task by building three main tensors
- seed_x ([B, 8, L]), the visible amino acid and 3Di features kept as context
  (with all edited positions zeroed out)
- edit_mask ([B, L]), True where the diffusion model must generate residues or
  padding and False where seed_x is fixed context
- seed_3di ([B, L]), the initial 3Di structure tokens exposed for this task
  (before training-time 3Di dropout in masking.py)

Every task is sampled in four steps
1. Choose a layout (complete, prefix, suffix, middle, or redesign) and, when
   layout is redesign, choose which cues are active (unconditional, properties,
   3di, or properties_3di).
2. Choose the visible slice [visible_start, visible_end) of the protein that
   starts as retained context.
3. Punch internal edit gaps inside that visible slice (via spans.py) and mark
   editable prefix/suffix ends to form seed_x and edit_mask.
4. Choose which 3Di structure tokens remain visible across the edited regions
   (always kept on fixed context, kept with probability
   editable_3di_retention_probability per contiguous editable run on inpainting
   rows, and kept on 3di/properties_3di redesign rows).

This file contains two implementations of those four steps
- sample_task_batch runs a per-row Python loop (helpers in Section 1 below) that
  also supports shifting proteins on the canvas and preserves the RNG stream
  used by fixed validation panels.
- _batched_aligned_mixed_batch runs the same four steps across the whole batch
  with vectorized tensor ops (helpers in Section 2 below) when proteins stay
  left-aligned at position 0, which is the default training path.

This code is cleaner than most research repos, so please don't blame us for 
having this slight code duplication.:)
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import torch

from .config import (
    LAYOUT_NAMES,
    LAYOUT_TO_ID,
    NOT_REDESIGN_CONDITIONING_ID,
    REDESIGN_CONDITIONING_TO_ID,
    REDESIGN_MODES_WITH_3DI,
    TaskConfig,
)
from .spans import (
    _bernoulli,
    _continuous_true_runs,
    _randint,
    sample_edit_masks_for_budgets,
    sample_internal_edit_spans,
)


class _VisibleWindow(NamedTuple):
    """Protein placement and visible bounds, all in canvas coordinates."""

    target_offset: int
    visible_start: int
    visible_end: int
    window_start: int
    window_end: int


def sample_task_batch(
    batch: dict[str, Any],
    config: TaskConfig,
    *,
    generator: torch.Generator | None = None,
    batched_mixed: bool = False,
) -> dict[str, Any]:
    """Choose layout, visible region, sequence edits, then visible structure."""
    source_x = batch["x"]
    source_3di = batch["seed_3di"]
    source_residue_mask = batch["residue_mask"]
    if source_x.ndim != 3 or source_3di.ndim != 2 or source_residue_mask.ndim != 2:
        raise ValueError("expected x [B,C,L], seed_3di [B,L], and residue_mask [B,L]")
    if (
        source_x.shape[0] != source_residue_mask.shape[0]
        or source_x.shape[2] != source_residue_mask.shape[1]
    ):
        raise ValueError("source batch shapes do not share one canvas")
    if source_3di.shape != source_residue_mask.shape:
        raise ValueError("seed_3di and residue_mask must have the same shape")

    batch_size, channels, canvas_length = source_x.shape
    device = source_x.device
    source_lengths = source_residue_mask.sum(dim=1).to(dtype=torch.long)
    expected_source_mask = (
        torch.arange(canvas_length, device=device)[None, :] < source_lengths[:, None]
    )
    if not torch.equal(source_residue_mask.bool(), expected_source_mask):
        raise ValueError("source proteins must be contiguous and left-aligned")
    if (
        batched_mixed
        and batch_size > 0
        and channels > 0
        and canvas_length > 0
        and config.terminal_pad_max == 0
        and config.global_offset_probability == 0
    ):
        return _batched_aligned_mixed_batch(
            batch, source_lengths, config, generator=generator
        )
    x0 = torch.zeros_like(source_x)
    seed_x = torch.zeros_like(source_x)
    seed_3di = torch.zeros_like(source_3di)
    residue_mask = torch.zeros_like(source_residue_mask, dtype=torch.bool)
    edit_mask = torch.zeros_like(source_residue_mask, dtype=torch.bool)

    layout_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
    redesign_conditioning_ids = torch.full(
        (batch_size,),
        NOT_REDESIGN_CONDITIONING_ID,
        dtype=torch.long,
        device=device,
    )
    target_offsets = torch.zeros(batch_size, dtype=torch.long, device=device)
    target_lengths = source_lengths
    visible_starts = torch.zeros(batch_size, dtype=torch.long, device=device)
    visible_ends = torch.zeros(batch_size, dtype=torch.long, device=device)
    window_starts = torch.zeros(batch_size, dtype=torch.long, device=device)
    window_ends = torch.zeros(batch_size, dtype=torch.long, device=device)
    internal_budgets = torch.zeros(batch_size, dtype=torch.long, device=device)
    span_counts = torch.zeros(batch_size, dtype=torch.long, device=device)

    for row_idx in range(batch_size):
        protein_length = int(target_lengths[row_idx].item())
        if protein_length <= 0:
            continue

        # 1. Choose the layout and redesign mode.
        layout, redesign_conditioning_id = _sample_layout_and_mode(
            protein_length,
            config,
            device=device,
            generator=generator,
        )

        # 2. Choose the visible slice and its placement on the canvas.
        window = _sample_visible_window(
            layout,
            redesign_conditioning_id,
            protein_length,
            canvas_length,
            config,
            device=device,
            generator=generator,
        )

        target_offset = window.target_offset
        target_end = target_offset + protein_length
        x0[row_idx, :, target_offset:target_end] = source_x[row_idx, :, :protein_length]
        residue_mask[row_idx, target_offset:target_end] = True

        # 3. Cut internal edit spans and mark editable ends.
        internal_budgets[row_idx], span_counts[row_idx] = _build_sequence_seed(
            source_x[row_idx], seed_x[row_idx], edit_mask[row_idx], layout, window,
            config, generator=generator,
        )

        # 4. Choose which structure tokens remain visible.
        _build_structure_seed(
            source_3di[row_idx, :protein_length],
            seed_3di[row_idx, target_offset:target_end],
            edit_mask[row_idx, target_offset:target_end],
            layout, redesign_conditioning_id, config, generator=generator,
        )

        layout_ids[row_idx] = LAYOUT_TO_ID[layout]
        redesign_conditioning_ids[row_idx] = redesign_conditioning_id
        target_offsets[row_idx] = target_offset
        visible_starts[row_idx] = window.visible_start
        visible_ends[row_idx] = window.visible_end
        window_starts[row_idx] = window.window_start
        window_ends[row_idx] = window.window_end

    conditioned: dict[str, Any] = {
        "x": x0,
        "seed_x": seed_x,
        "seed_3di": seed_3di,
        "c": batch["c"],
        "residue_mask": residue_mask,
        "edit_mask": edit_mask,
        "layout_id": layout_ids,
        "redesign_conditioning_id": redesign_conditioning_ids,
        "target_offset": target_offsets,
        "target_length": target_lengths,
        "visible_start": visible_starts,
        "visible_end": visible_ends,
        "window_start": window_starts,
        "window_end": window_ends,
        "internal_budget": internal_budgets,
        "span_count": span_counts,
    }
    conditioned.update(
        {
            key: value
            for key, value in batch.items()
            if key
            not in {
                "x",
                "seed_3di",
                "c",
                "residue_mask",
            }
        }
    )
    return conditioned


def _batched_aligned_mixed_batch(
    batch: dict[str, Any],
    lengths: torch.Tensor,
    config: TaskConfig,
    *,
    generator: torch.Generator | None,
) -> dict[str, Any]:
    """Build left-aligned tasks in four batch-wide steps.

    Training uses this path by default. Draw order differs from the per-row
    sampler used for validation and unaligned tasks.
    """
    source_x = batch["x"]
    positions = torch.arange(source_x.shape[2], device=source_x.device)[None, :]
    residue_mask = positions < lengths[:, None]

    # 1. Choose layouts and redesign modes.
    layouts, modes = _sample_layouts_and_modes(lengths, config, generator=generator)

    # 2. Choose visible slices and their internal edit budgets.
    starts, ends, budgets = _sample_visible_intervals(
        lengths, layouts, config, generator=generator
    )

    # 3. Cut internal edit spans and mark editable ends.
    seed_x, edit_mask, span_counts = _build_sequence_seeds(
        source_x, residue_mask, layouts, starts, ends, budgets, config,
        generator=generator,
    )

    # 4. Choose which structure tokens remain visible.
    seed_3di = _build_structure_seeds(
        batch["seed_3di"], residue_mask, edit_mask, layouts, modes, config,
        generator=generator,
    )

    zeros = torch.zeros_like(lengths)
    result = {
        "x": source_x.masked_fill(~residue_mask[:, None, :], 0),
        "seed_x": seed_x,
        "seed_3di": seed_3di,
        "c": batch["c"],
        "residue_mask": residue_mask,
        "edit_mask": edit_mask,
        "layout_id": layouts,
        "redesign_conditioning_id": modes,
        "target_offset": zeros.clone(),
        "target_length": lengths,
        "visible_start": starts,
        "visible_end": ends,
        "window_start": zeros.clone(),
        "window_end": lengths.clone(),
        "internal_budget": budgets,
        "span_count": span_counts,
    }
    result.update(
        {
            key: value
            for key, value in batch.items()
            if key
            not in {
                "x",
                "seed_3di",
                "c",
                "residue_mask",
            }
        }
    )
    return result


# ---------------------------------------------------------------------------
# Section 1. Per-row helpers (Steps 1 to 4 for a single protein)
# ---------------------------------------------------------------------------


def _sample_layout_and_mode(
    protein_length: int,
    config: TaskConfig,
    *,
    device: torch.device,
    generator: torch.Generator | None,
) -> tuple[str, int]:
    """Choose a valid layout and, for redesign, its conditioning mode."""
    weights = torch.tensor(
        config.layout_probabilities, device=device, dtype=torch.float32
    )
    if protein_length < 2:
        weights[LAYOUT_TO_ID["prefix"]] = 0
        weights[LAYOUT_TO_ID["suffix"]] = 0
    if protein_length < 3:
        weights[LAYOUT_TO_ID["middle"]] = 0
    if float(weights.sum().item()) <= 0:
        weights[LAYOUT_TO_ID["complete"]] = 1
    layout_idx = int(torch.multinomial(weights, 1, generator=generator).item())
    layout = LAYOUT_NAMES[layout_idx]
    mode = NOT_REDESIGN_CONDITIONING_ID
    if layout == "redesign":
        weights = torch.tensor(
            config.redesign_conditioning_probabilities,
            device=device,
            dtype=torch.float32,
        )
        mode = int(torch.multinomial(weights, 1, generator=generator).item())
    return layout, mode


def _sample_visible_interval(
    layout: str,
    protein_length: int,
    config: TaskConfig,
    *,
    device: torch.device,
    generator: torch.Generator | None,
) -> tuple[int, int]:
    if layout == "complete":
        return 0, protein_length
    if layout == "redesign":
        return 0, 0

    maximum = protein_length - 1
    if layout == "middle":
        maximum = protein_length - 2
    minimum = max(1, math.ceil(protein_length * config.visible_min_fraction))
    preferred_maximum = math.floor(protein_length * config.visible_max_fraction)
    maximum = min(maximum, max(minimum, preferred_maximum))
    minimum = min(minimum, maximum)
    visible_length = _randint(
        minimum,
        maximum + 1,
        device=device,
        generator=generator,
    )

    if layout == "prefix":
        return 0, visible_length
    if layout == "suffix":
        return protein_length - visible_length, protein_length
    start = _randint(
        1,
        protein_length - visible_length,
        device=device,
        generator=generator,
    )
    return start, start + visible_length


def _sample_terminal_padding(
    spare: int,
    *,
    left_active: bool,
    right_active: bool,
    config: TaskConfig,
    device: torch.device,
    generator: torch.Generator | None,
) -> tuple[int, int]:
    pads = [0, 0]
    active = [idx for idx, enabled in enumerate((left_active, right_active)) if enabled]
    for idx in active:
        pads[idx] = _randint(
            config.terminal_pad_min,
            config.terminal_pad_max + 1,
            device=device,
            generator=generator,
        )
    while sum(pads) > spare:
        reducible = [idx for idx in active if pads[idx] > 0]
        if not reducible:
            break
        selected = reducible[
            _randint(0, len(reducible), device=device, generator=generator)
        ]
        pads[selected] -= 1
    return pads[0], pads[1]


def _sample_visible_window(
    layout: str,
    mode: int,
    protein_length: int,
    canvas_length: int,
    config: TaskConfig,
    *,
    device: torch.device,
    generator: torch.Generator | None,
) -> _VisibleWindow:
    """Choose the visible slice, then place it with terminal padding and offset."""
    visible_start, visible_end = _sample_visible_interval(
        layout, protein_length, config, device=device, generator=generator
    )
    left_pad, right_pad = _sample_terminal_padding(
        canvas_length - protein_length,
        left_active=layout in {"suffix", "middle", "redesign"},
        right_active=layout in {"prefix", "middle", "redesign"},
        config=config,
        device=device,
        generator=generator,
    )
    minimum_offset = left_pad
    maximum_offset = canvas_length - protein_length - right_pad
    target_offset = minimum_offset
    if maximum_offset > minimum_offset and _bernoulli(
        config.global_offset_probability, device=device, generator=generator
    ):
        target_offset = _randint(
            minimum_offset, maximum_offset + 1, device=device, generator=generator
        )
    if (
        config.left_align_threedi_redesign
        and layout == "redesign"
        and mode in REDESIGN_MODES_WITH_3DI
    ):
        # Keep the draws above so alignment does not resample other tasks.
        left_pad = right_pad = target_offset = 0
    return _VisibleWindow(
        target_offset=target_offset,
        visible_start=target_offset + visible_start,
        visible_end=target_offset + visible_end,
        window_start=target_offset - left_pad,
        window_end=target_offset + protein_length + right_pad,
    )


def _build_sequence_seed(
    source_x: torch.Tensor,
    seed_x: torch.Tensor,
    edit_mask: torch.Tensor,
    layout: str,
    window: _VisibleWindow,
    config: TaskConfig,
    *,
    generator: torch.Generator | None,
) -> tuple[int, int]:
    """Fill one row's seed and edit mask; return its edit budget and span count."""
    start, end = window.visible_start, window.visible_end
    budget, span_count = 0, 0
    if end > start:
        seed_x[:, start:end] = source_x[
            :, start - window.target_offset : end - window.target_offset
        ]
        spans, budget = sample_internal_edit_spans(
            end - start, config, device=source_x.device, generator=generator
        )
        span_count = len(spans)
        for span_start, span_end in spans:
            edit_mask[start + span_start : start + span_end] = True
            seed_x[:, start + span_start : start + span_end] = 0
    if layout in {"suffix", "middle", "redesign"}:
        edit_mask[window.window_start:start] = True
    if layout in {"prefix", "middle", "redesign"}:
        edit_mask[end:window.window_end] = True
    if layout == "redesign":
        # Redesign retains no sequence and makes the entire canvas editable.
        edit_mask[:] = True
    return budget, span_count


def _build_structure_seed(
    source_3di: torch.Tensor,
    seed_3di: torch.Tensor,
    edit_mask: torch.Tensor,
    layout: str,
    mode: int,
    config: TaskConfig,
    *,
    generator: torch.Generator | None,
) -> None:
    """Fill one protein's 3Di seed, sampling retention once per editable run."""
    if layout == "redesign":
        if mode in REDESIGN_MODES_WITH_3DI:
            seed_3di[:] = source_3di
    else:
        fixed_target = ~edit_mask
        seed_3di[fixed_target] = source_3di[fixed_target]
        for run_start, run_end in _continuous_true_runs(edit_mask):
            if _bernoulli(
                config.editable_3di_retention_probability,
                device=source_3di.device,
                generator=generator,
            ):
                seed_3di[run_start:run_end] = source_3di[run_start:run_end]


# ---------------------------------------------------------------------------
# Section 2. Batched tensor helpers (Steps 1 to 4 across all rows at once)
# ---------------------------------------------------------------------------


def _sample_layouts_and_modes(
    lengths: torch.Tensor,
    config: TaskConfig,
    *,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Step 1 (batched) choose valid layouts and redesign modes for all rows."""
    count, device = len(lengths), lengths.device
    # Disable prefix/suffix on 1-residue proteins and middle on <3-residue proteins.
    weights = (
        torch.tensor(config.layout_probabilities, device=device)
        .expand(count, -1)
        .clone()
    )
    weights[lengths < 2, LAYOUT_TO_ID["prefix"]] = 0
    weights[lengths < 2, LAYOUT_TO_ID["suffix"]] = 0
    weights[lengths < 3, LAYOUT_TO_ID["middle"]] = 0
    weights[weights.sum(dim=1) <= 0, LAYOUT_TO_ID["complete"]] = 1
    layouts = torch.multinomial(weights, 1, generator=generator).squeeze(1)
    layouts = layouts.masked_fill(lengths == 0, LAYOUT_TO_ID["complete"])

    # Sample a redesign conditioning mode for every row, keeping it only on redesign rows.
    redesign = layouts == LAYOUT_TO_ID["redesign"]
    mode_weights = torch.tensor(
        config.redesign_conditioning_probabilities, device=device
    )
    modes = torch.multinomial(
        mode_weights, count, replacement=True, generator=generator
    )
    modes = modes.masked_fill(~redesign, NOT_REDESIGN_CONDITIONING_ID)
    return layouts, modes


def _sample_visible_intervals(
    lengths: torch.Tensor,
    layouts: torch.Tensor,
    config: TaskConfig,
    *,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Step 2 (batched) choose visible slices [starts, ends) and internal edit budgets."""
    redesign = layouts == LAYOUT_TO_ID["redesign"]
    middle = layouts == LAYOUT_TO_ID["middle"]
    suffix = layouts == LAYOUT_TO_ID["suffix"]
    complete = layouts == LAYOUT_TO_ID["complete"]

    # Draw 3 uniforms per row up front for (visible_length, middle_start, internal_budget).
    # Keep the budget draw here to preserve the batched sampler's RNG stream.
    draws = torch.rand(
        (3, len(lengths)), dtype=torch.float64, device=lengths.device, generator=generator
    )
    minimum = (lengths * config.visible_min_fraction).ceil().long().clamp_min(1)
    preferred_maximum = (lengths * config.visible_max_fraction).floor().long()
    maximum = torch.minimum(
        lengths - 1 - middle.long(), torch.maximum(minimum, preferred_maximum)
    )
    minimum = torch.minimum(minimum, maximum)

    # Complete keeps the whole protein; redesign keeps 0 residues; prefix/suffix/middle
    # keep a random slice of length in [minimum, maximum].
    visible_lengths = minimum + (draws[0] * (maximum - minimum + 1)).floor().long()
    visible_lengths = torch.where(complete, lengths, visible_lengths)
    visible_lengths = visible_lengths.masked_fill(redesign, 0)

    # Anchor the visible slice at 0 for prefix/complete, at the right end for suffix,
    # or at an interior offset for middle.
    starts = torch.where(suffix, lengths - visible_lengths, 0)
    middle_starts = (
        1 + (draws[1] * (lengths - visible_lengths - 1).clamp_min(1)).floor().long()
    )
    starts = torch.where(middle, middle_starts, starts)
    ends = starts + visible_lengths

    # Pick how many residues inside the visible slice will be masked out (1..visible_length).
    budgets = (1 + (draws[2] * visible_lengths).floor().long()).masked_fill(
        visible_lengths == 0, 0
    )
    return starts, ends, budgets


def _build_sequence_seeds(
    source_x: torch.Tensor,
    residue_mask: torch.Tensor,
    layouts: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    budgets: torch.Tensor,
    config: TaskConfig,
    *,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Step 3 (batched) punch internal edit spans and build seed_x and edit_mask."""
    width = source_x.shape[2]
    positions = torch.arange(width, device=source_x.device)[None, :]
    redesign = layouts == LAYOUT_TO_ID["redesign"]
    middle = layouts == LAYOUT_TO_ID["middle"]
    suffix = layouts == LAYOUT_TO_ID["suffix"]
    prefix = layouts == LAYOUT_TO_ID["prefix"]

    # Sample internal edit spans within each row's [0, visible_length) local coordinates.
    compact_edits = sample_edit_masks_for_budgets(
        ends - starts,
        budgets,
        width,
        span_count_geometric_probability=config.span_count_geometric_probability,
        dirichlet_concentration=config.dirichlet_concentration,
        generator=generator,
    )
    compact_previous = torch.nn.functional.pad(
        compact_edits[:, :-1], (1, 0), value=False
    )
    span_counts = (compact_edits & ~compact_previous).sum(dim=1)

    # Shift the local edit mask to [starts, ends) on the canvas.
    visible = (positions >= starts[:, None]) & (positions < ends[:, None])
    internal_edits = (
        compact_edits.gather(1, (positions - starts[:, None]).clamp(0, width - 1))
        & visible
    )

    # Combine internal edit holes with editable prefix/suffix tails (or the full canvas on redesign).
    left_active = suffix | middle | redesign
    right_active = prefix | middle | redesign
    edit_mask = (
        internal_edits
        | (left_active[:, None] & (positions < starts[:, None]))
        | (right_active[:, None] & (positions >= ends[:, None]) & residue_mask)
        | redesign[:, None]
    )

    # Keep source features only on visible positions that were not punched out by internal_edits.
    retained = visible & ~internal_edits
    seed_x = source_x.masked_fill(~retained[:, None, :], 0)
    return seed_x, edit_mask, span_counts


def _build_structure_seeds(
    source_3di: torch.Tensor,
    residue_mask: torch.Tensor,
    edit_mask: torch.Tensor,
    layouts: torch.Tensor,
    modes: torch.Tensor,
    config: TaskConfig,
    *,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Step 4 (batched) choose which 3Di structure tokens stay visible in seed_3di."""
    count, width = source_3di.shape
    retained = residue_mask & ~edit_mask
    redesign = layouts == LAYOUT_TO_ID["redesign"]

    # Number each contiguous editable run (0, 1, 2, ...) along each protein and flip
    # one coin per run to decide whether its 3Di tokens stay visible.
    target_edits = edit_mask & residue_mask
    previous = torch.nn.functional.pad(target_edits[:, :-1], (1, 0), value=False)
    run_ids = (target_edits & ~previous).long().cumsum(dim=1) - 1
    keep_runs = (
        torch.rand((count, width), device=source_3di.device, generator=generator)
        < config.editable_3di_retention_probability
    )
    available_3di = retained | (
        target_edits & keep_runs.gather(1, run_ids.clamp_min(0))
    )

    # On full redesign rows, expose all 3Di tokens only when the mode includes 3Di.
    redesign_3di = (modes == REDESIGN_CONDITIONING_TO_ID["3di"]) | (
        modes == REDESIGN_CONDITIONING_TO_ID["properties_3di"]
    )
    available_3di = torch.where(
        redesign[:, None], redesign_3di[:, None] & residue_mask, available_3di
    )
    return source_3di.masked_fill(~available_3di, 0)
