"""Dirichlet-multinomial span allocation for sequence editing and 3Di masking.

Given a sequence of length L and a budget of B positions to mask (for example,
masking 20 residues inside a 60-residue protein), this module groups those B
masked positions into contiguous spans rather than isolated single residues

1. Draw how many spans K to create from a truncated geometric distribution in
   [1, B], so 1 contiguous span is most common, 2 is half as likely, and so on.
2. Split the B masked positions across the K spans using a symmetric
   Dirichlet-multinomial draw (giving each span at least 1 position).
3. Split the remaining L - B unmasked positions into the K + 1 surrounding gaps
   using a second Dirichlet-multinomial draw, and interleave gaps and spans.

Two places in this folder use these random spans
- sampling.py uses them to punch edit holes into a protein sequence during inpainting.
- masking.py uses them to hide chunks of 3Di structure tokens during training.

The file provides two versions of the same algorithm, sample_edit_spans_for_budget
for a single protein and sample_edit_masks_for_budgets for an entire batch of
proteins at once.
"""

from __future__ import annotations

import math

import torch

from .config import TaskConfig


def _randint(
    low: int,
    high: int,
    *,
    device: torch.device,
    generator: torch.Generator | None,
) -> int:
    """Return an integer from [low, high)."""
    if high <= low:
        return low
    return int(torch.randint(low, high, (), device=device, generator=generator).item())


def _bernoulli(
    probability: float,
    *,
    device: torch.device,
    generator: torch.Generator | None,
) -> bool:
    return bool(torch.rand((), device=device, generator=generator).item() < probability)


def _multinomial_counts(
    total: int,
    probabilities: torch.Tensor,
    *,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if total <= 0:
        return torch.zeros_like(probabilities, dtype=torch.long)
    draws = torch.multinomial(
        probabilities,
        total,
        replacement=True,
        generator=generator,
    )
    return torch.bincount(draws, minlength=probabilities.numel())


def _dirichlet_multinomial_counts(
    total: int,
    group_count: int,
    concentration: float,
    *,
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Allocate integer counts using symmetric Dirichlet-distributed weights."""
    if group_count <= 0:
        raise ValueError("group_count must be positive")
    if total <= 0:
        return torch.zeros(group_count, dtype=torch.long, device=device)
    concentrations = torch.full(
        (group_count,),
        concentration,
        device=device,
    )
    weights = torch._standard_gamma(concentrations, generator=generator)
    if float(weights.sum().item()) <= 0:
        weights = torch.ones_like(weights)
    weights = weights / weights.sum()
    return _multinomial_counts(total, weights, generator=generator)


def _sample_truncated_geometric(
    maximum: int,
    probability: float,
    *,
    device: torch.device,
    generator: torch.Generator | None,
) -> int:
    """Draw a geometric variate conditioned to lie in ``[1, maximum]``."""
    if maximum < 1:
        raise ValueError("maximum must be positive")
    if probability == 1:
        return 1
    exponents = torch.arange(maximum, device=device, dtype=torch.float32)
    weights = torch.exp(exponents * math.log1p(-probability))
    return int(torch.multinomial(weights, 1, generator=generator).item()) + 1


def _continuous_true_runs(mask: torch.Tensor) -> list[tuple[int, int]]:
    """Return half-open intervals for maximal contiguous true regions."""
    if mask.ndim != 1:
        raise ValueError("mask must be one-dimensional")
    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for position, active in enumerate(mask.tolist()):
        if active and run_start is None:
            run_start = position
        elif not active and run_start is not None:
            runs.append((run_start, position))
            run_start = None
    if run_start is not None:
        runs.append((run_start, mask.numel()))
    return runs


def sample_internal_edit_spans(
    seed_length: int,
    config: TaskConfig,
    *,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> tuple[list[tuple[int, int]], int]:
    """Sample hierarchical internal edits within a nonempty retained seed."""
    if seed_length <= 0:
        return [], 0

    budget = _randint(
        1,
        seed_length + 1,
        device=device,
        generator=generator,
    )

    spans = sample_edit_spans_for_budget(
        seed_length,
        budget,
        config,
        device=device,
        generator=generator,
    )
    return spans, budget


def sample_edit_spans_for_budget(
    seed_length: int,
    budget: int,
    config: TaskConfig,
    *,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> list[tuple[int, int]]:
    """Partition a budget into Dirichlet-weighted candidate spans and gaps."""
    if not 0 <= budget <= seed_length:
        raise ValueError("edit budget must be between zero and seed_length")
    if budget == 0:
        return []

    # 1. Pick how many masked spans K to create (geometric, favoring fewer spans).
    candidate_count = _sample_truncated_geometric(
        budget,
        config.span_count_geometric_probability,
        device=device,
        generator=generator,
    )
    # 2. Split the B masked positions across the K spans (at least 1 per span).
    span_lengths = torch.ones(candidate_count, dtype=torch.long, device=device)
    span_lengths += _dirichlet_multinomial_counts(
        budget - candidate_count,
        candidate_count,
        config.dirichlet_concentration,
        device=device,
        generator=generator,
    )
    # 3. Split the remaining L - B visible positions into the K + 1 surrounding gaps.
    gaps = _dirichlet_multinomial_counts(
        seed_length - budget,
        candidate_count + 1,
        config.dirichlet_concentration,
        device=device,
        generator=generator,
    )

    # Interleave [gap_0, span_0, gap_1, span_1, ..., gap_K] and merge touching spans.
    spans: list[tuple[int, int]] = []
    cursor = int(gaps[0].item())
    for span_idx, span_length in enumerate(span_lengths.tolist()):
        end = cursor + int(span_length)
        if spans and spans[-1][1] == cursor:
            spans[-1] = (spans[-1][0], end)
        else:
            spans.append((cursor, end))
        cursor = end + int(gaps[span_idx + 1].item())
    return spans


def _batched_dirichlet_multinomial_counts(
    totals: torch.Tensor,
    groups: torch.Tensor,
    width: int,
    concentration: float,
    *,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """The scalar allocation above, with independent distributions per row."""
    result = torch.zeros((len(totals), width), dtype=torch.long, device=totals.device)
    draws_per_row = int(totals.max().item())
    if draws_per_row == 0:
        return result
    active = (
        torch.arange(width, device=totals.device)[None, :]
        < groups.clamp_min(1)[:, None]
    )
    weights = torch._standard_gamma(
        torch.full(result.shape, concentration, device=totals.device),
        generator=generator,
    ).masked_fill(~active, 0)
    weights = torch.where(weights.sum(dim=1, keepdim=True) > 0, weights, active.float())
    draws = torch.multinomial(
        weights, draws_per_row, replacement=True, generator=generator
    )
    retained = (
        torch.arange(draws_per_row, device=totals.device)[None, :] < totals[:, None]
    )
    return result.scatter_add_(1, draws, retained.long())


def sample_edit_masks_for_budgets(
    lengths: torch.Tensor,
    budgets: torch.Tensor,
    canvas_length: int,
    *,
    span_count_geometric_probability: float,
    dirichlet_concentration: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Batched equivalent in distribution of ``sample_edit_spans_for_budget``.

    Random draws are reordered, so seeds are reproducible within this method
    but do not reproduce the scalar sampler's exact masks.
    """
    if lengths.numel() == 0:
        return torch.zeros((0, canvas_length), dtype=torch.bool, device=lengths.device)
    if span_count_geometric_probability == 1:
        candidates = torch.ones_like(budgets)
    else:
        # Invert the geometric CDF conditioned on 1 <= K <= budget.
        log_q = math.log1p(-span_count_geometric_probability)
        u = torch.rand(
            budgets.shape,
            dtype=torch.float64,
            device=budgets.device,
            generator=generator,
        )
        mass = -torch.expm1(budgets.to(torch.float64) * log_q)
        candidates = (torch.log1p(-u * mass) / log_q).floor().long() + 1
    candidates = torch.minimum(candidates.clamp_min(1), budgets)
    width = max(1, int(candidates.max().item()))
    active = torch.arange(width, device=lengths.device)[None, :] < candidates[:, None]
    spans = active.long() + _batched_dirichlet_multinomial_counts(
        budgets - candidates,
        candidates,
        width,
        dirichlet_concentration,
        generator=generator,
    )
    gaps = _batched_dirichlet_multinomial_counts(
        lengths - budgets,
        candidates + 1,
        width + 1,
        dirichlet_concentration,
        generator=generator,
    )
    ends = spans.cumsum(dim=1) + gaps[:, :width].cumsum(dim=1)
    starts = ends - spans
    edges = torch.zeros(
        (len(lengths), canvas_length + 1), dtype=torch.long, device=lengths.device
    )
    edges.scatter_add_(1, starts, active.long())
    edges.scatter_add_(1, ends, -active.long())
    return edges[:, :canvas_length].cumsum(dim=1).bool()
