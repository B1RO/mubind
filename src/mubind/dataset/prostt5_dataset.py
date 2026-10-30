"""Read the prepared, memory-mapped ProstT5 format."""

from __future__ import annotations

from collections.abc import Callable
import json
from operator import index as integer_index
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..common.properties import (
    DEFAULT_PROPERTY_DEFINITION,
    canonical_property_definition,
)
from ..model.conditioning.threedi_2d_representation import foldseek_3di_codebook
from ..paths import existing_artifact
from .embed import (
    EMBEDDING_DIM,
    THREEDI_TO_ID,
    embed_sequence,
)

# IDs in the stored ProstT5 arrays differ from the model's 3Di IDs.
# AA occupies 3..22, 3Di occupies 128..147, and zero is padding.
AA = "ALGVSREDTIPKFQNYMHWC"
THREEDI = AA.lower()


class _Rows:
    """Decode original IDs and strings on demand for evaluation callers."""

    def __init__(self, dataset: ProstT5Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[int, str, str]:
        dataset = self.dataset
        index = dataset._normalize_index(index)
        sequence = "".join(AA[int(token) - 3] for token in dataset.aa[index] if token)
        structure = "".join(
            THREEDI[int(token) - 128].upper()
            for token in dataset.threedi[index]
            if token
        )
        return index, sequence, structure


class ProstT5Dataset(Dataset):
    """Load clean AA/3Di examples from a prepared dataset directory.

    ``path`` contains ``prostt5_diffusion.json`` and split directories with
    ``aa.npy``, ``threedi.npy``, and ``properties.npy``. Arrays
    stay memory-mapped; only requested examples are converted to tensors.

    ``max_length`` can remove trailing padding, but must not truncate any AA
    or 3Di tokens. ``joint_3di`` appends two structural coordinates to the
    six residue channels. Normalization affects only the five AA properties.

    ``dataset_index`` is the row index, and ``seed_3di`` is the complete,
    unmasked structure. Optional raw strings and reference paths are loaded
    for evaluation when ``include_raw`` is true.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        split: str = "train",
        max_length: int = 128,
        include_raw: bool = False,
        normalize_embeddings: bool = False,
        joint_3di: bool = False,
        property_definition: str = DEFAULT_PROPERTY_DEFINITION,
    ) -> None:
        self.path = existing_artifact(Path(path))
        report = json.loads((self.path / "prostt5_diffusion.json").read_text())
        if report["state"] != "complete":
            raise ValueError("Incomplete prepared data")
        self.property_definition = canonical_property_definition(property_definition)
        if self.property_definition != report["property_definition"]:
            raise ValueError("Property definition differs from prepared data")

        self.split = "valid" if split == "val" else split
        if self.split not in report["splits"]:
            raise ValueError(f"Prepared data has no split {self.split!r}")
        self.info = report["splits"][self.split]
        self.max_length = integer_index(max_length)
        self.include_raw = include_raw
        self.joint_3di = bool(joint_3di)
        self.normalize_embeddings = normalize_embeddings
        self.threedi_coordinates = foldseek_3di_codebook()

        directory = self.path / self.split
        row_count = integer_index(self.info["rows"])
        arrays = {
            name: np.load(directory / filename, mmap_mode="r")
            for name, filename in (
                ("aa", "aa.npy"),
                ("threedi", "threedi.npy"),
                ("properties", "properties.npy"),
            )
        }
        if row_count < 0 or any(len(array) < row_count for array in arrays.values()):
            raise ValueError("Prepared arrays contain fewer rows than the manifest")
        if (
            arrays["aa"].ndim != 2
            or arrays["threedi"].shape != arrays["aa"].shape
            or arrays["properties"].ndim != 2
            or arrays["properties"].shape[1] != 3
        ):
            raise ValueError("Prepared AA, 3Di, and property arrays are misaligned")

        # Expose only the manifest's valid rows, including for
        # negative indexing and bulk reads.
        self.aa = arrays["aa"][:row_count]
        self.threedi = arrays["threedi"][:row_count]
        self.properties = arrays["properties"][:row_count]
        stored_length = self.aa.shape[1]
        if not 1 <= self.max_length <= stored_length:
            raise ValueError(f"max_length must be between 1 and {stored_length}")
        if self.max_length < stored_length:
            # Scan once in bounded chunks; shorter canvases only remove PAD.
            for start in range(0, row_count, 65536):
                stop = min(start + 65536, row_count)
                if np.any(self.aa[start:stop, self.max_length :] != 0) or np.any(
                    self.threedi[start:stop, self.max_length :] != 0
                ):
                    raise ValueError(
                        f"max_length={self.max_length} would truncate non-padding "
                        f"AA or 3Di tokens in split {self.split!r}"
                    )
            self.aa = self.aa[:, : self.max_length]
            self.threedi = self.threedi[:, : self.max_length]

        self.rows = _Rows(self)
        reference_name = {
            "valid": "validation_references.json",
            "test": "test_references.json",
        }.get(self.split)
        reference_path = self.path / reference_name if reference_name else None
        self.references = (
            json.loads(reference_path.read_text())
            if include_raw and reference_path is not None and reference_path.exists()
            else None
        )
        self.embedding = torch.zeros((256, EMBEDDING_DIM))
        self.di_lookup = torch.zeros(256, dtype=torch.long)
        for token, amino_acid in enumerate(AA, 3):
            self.embedding[token] = embed_sequence(
                amino_acid,
                normalized=normalize_embeddings,
            )[0]
        for token, letter in enumerate(THREEDI, 128):
            self.di_lookup[token] = THREEDI_TO_ID[letter.upper()]

    def __len__(self) -> int:
        return int(self.info["rows"])

    def _normalize_index(self, index: int) -> int:
        index = integer_index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("ProstT5 index out of bounds")
        return index

    def property_statistics(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the population mean/std recorded during data preparation."""
        return (
            torch.tensor(self.info["property_mean"], dtype=torch.float32),
            torch.tensor(self.info["property_std"], dtype=torch.float32),
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.__getitems__([self._normalize_index(index)])[0]

    def __getitems__(self, indices) -> list[dict[str, Any]]:
        """Fetch a batch with vectorized lookups, preserving order and repeats."""
        if len(indices) == 0:
            return []
        indices = np.asarray(indices)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise IndexError(
                "ProstT5 indices must be a one-dimensional integer sequence"
            )
        indices = np.where(indices < 0, indices + len(self), indices)
        if np.any(indices < 0) or np.any(indices >= len(self)):
            raise IndexError("ProstT5 index out of bounds")

        amino_acids = torch.tensor(self.aa[indices], dtype=torch.long)
        structure_tokens = torch.tensor(self.threedi[indices], dtype=torch.long)
        seed_3di = self.di_lookup[structure_tokens]
        x = self.embedding[amino_acids]
        if self.joint_3di:
            x = torch.cat((x, self.threedi_coordinates[seed_3di]), dim=-1)
        columns = {
            "x": x.transpose(1, 2).contiguous(),
            "seed_3di": seed_3di,
            "c": torch.tensor(self.properties[indices]),
            "residue_mask": amino_acids != 0,
            "dataset_index": torch.tensor(indices, dtype=torch.long),
        }
        rows = [
            dict(zip(columns, values))
            for values in zip(*(column.unbind(0) for column in columns.values()))
        ]
        if self.include_raw:
            for index, item in zip(indices, rows):
                _, item["binder_aa"], item["binder_3di"] = self.rows[int(index)]
                if self.references is not None:
                    reference = self.references.get(str(index), {})
                    item["reference_pdb"] = reference.get("path", "")
                    item["reference_id"] = reference.get("id", "")
        return rows


def make_dataloader(
    dataset: ProstT5Dataset,
    *,
    batch_size: int = 64,
    shuffle: bool | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    drop_last: bool = False,
    generator: torch.Generator | None = None,
    collate_fn: Callable[[list[dict[str, Any]]], Any] | None = None,
) -> DataLoader:
    """Stack clean examples; shuffle training data by default."""
    if shuffle is None:
        shuffle = dataset.split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        drop_last=drop_last,
        generator=generator,
        collate_fn=collate_fn,
    )
