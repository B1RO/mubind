"""Prepared ProstT5 data, before task-specific masking or diffusion noise.

Download and preparation live separately in ``mubind.preprocessing``.

Configure the data representation separately from batching::

    from mubind.dataset import ProstT5Dataset, make_dataloader

    dataset = ProstT5Dataset(
        "/path/to/ProstT5Dataset_len50_70_diffusion",
        max_length=70, normalize_embeddings=True, joint_3di=True,
    )
    loader = make_dataloader(dataset, batch_size=64)
    batch = next(iter(loader))

Each row contains ``x`` [C,L] (six AA/PAD channels, optionally two 3Di
coordinates), ``seed_3di`` [L] (token IDs; zero is PAD), ``c`` [3]
(length, charge, hydrophobicity), ``residue_mask`` [L], and ``dataset_index``.
``seed_3di`` contains the complete structure here; task preparation may hide
it later. ``residue_mask`` describes the clean target, not a model input.
"""

from .prostt5_dataset import ProstT5Dataset, make_dataloader

__all__ = [
    "ProstT5Dataset",
    "make_dataloader",
]
