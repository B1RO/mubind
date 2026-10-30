"""DataLoader and epoch sampler helpers with deterministic per-batch random seeds.

To make training runs deterministically resumable even when the number of DataLoader
workers changes, each batch gets its own random seed (`next_task_seed`) drawn from
`task_rng`. Background workers clone `task_rng` inside `_SeededBatches` so prefetching
ahead never advances the trainer's saved `task_rng` state until the batch is actually
consumed by a training step.
"""

from __future__ import annotations

import math

import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler

from ..tasks import TaskCollator

TASK_RNG_SCHEME = "batch_seed_v1"


def make_training_loader(
    dataset, config, sampler, task_rng, *, device, epoch, start_step: int = 0
):
    """Build a `DataLoader` that collates, masks, and pins complete training batches in worker processes."""
    return DataLoader(
        _PreparedBatches(dataset, config),
        # Each sampler item requests a whole batch. Default conversion leaves
        # its tensors intact, and DataLoader pins the final prepared tensors.
        batch_size=None,
        sampler=_SeededBatches(
            sampler, config, task_rng, start_step=start_step
        ),
        num_workers=config.data.num_workers,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(config.seed + epoch + 2),
    )


class EpochSampler(Sampler[int]):
    """Deterministically shuffle dataset indices for an epoch and skip `offset` examples when resuming."""

    def __init__(self, length: int, seed: int, offset: int):
        self.length, self.seed, self.offset = length, seed, offset

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        order = torch.randperm(self.length, generator=generator)
        return iter(order[self.offset :].tolist())

    def __len__(self):
        return self.length - self.offset


def training_batches_per_epoch(length, data_config):
    """Return the number of batches in one full epoch for `length` dataset examples."""
    if data_config.drop_last:
        return length // data_config.batch_size
    return math.ceil(length / data_config.batch_size)


def next_task_seed(generator: torch.Generator) -> int:
    """Draw one 63-bit integer seed per batch from `generator`."""
    return int(torch.randint(0, 2**63 - 1, (), generator=generator).item())


class _SeededBatches(Sampler):
    """Wrap `BatchSampler` to pair each batch of row indices with its own deterministic task seed and step number."""

    def __init__(self, sampler, config, task_rng, *, start_step: int = 0):
        self.batches = BatchSampler(
            sampler, config.data.batch_size, config.data.drop_last
        )
        self.state = task_rng.get_state().clone()
        self.start_step = start_step

    def __iter__(self):
        # Prefetch can run ahead on a cloned generator without touching the trainer's saved stream.
        generator = torch.Generator().set_state(self.state)
        for offset, indices in enumerate(self.batches, start=1):
            yield indices, next_task_seed(generator), self.start_step + offset

    def __len__(self):
        return len(self.batches)


class _PreparedBatches(Dataset):
    """Dataset adapter that fetches a list of rows at once and runs `TaskCollator.collate` inside the worker."""

    def __init__(self, dataset, config):
        self.dataset = dataset
        self.collator = TaskCollator(
            config.tasks,
            batched_mixed=config.batched_tasks,
            training_threedi_masking=config.threedi_masking,
            property_dropout=config.property_dropout,
            training=True,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, request):
        if len(request) == 3:
            indices, seed, step = request
        else:
            indices, seed = request
            step = None
        rows = self.dataset.__getitems__(indices)
        return self.collator.collate(
            rows,
            generator=torch.Generator().manual_seed(seed),
            step=step,
        )
