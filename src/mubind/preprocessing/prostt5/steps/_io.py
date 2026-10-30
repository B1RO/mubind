"""Manifest, shard, and Parquet batch helpers shared by preprocessing steps."""

from __future__ import annotations

import json
from pathlib import Path

SPLITS = ("train", "valid", "test")


def write_manifest(path: Path, report: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def source_shards(source: Path) -> dict[str, list[Path]]:
    for name in ("download_manifest.json", "subset_manifest.json"):
        manifest = source / name
        if (
            manifest.exists()
            and json.loads(manifest.read_text())["state"] != "complete"
        ):
            raise ValueError(f"Incomplete source: {manifest}")
    files = {
        split: sorted((source / "data").glob(f"{split}-*.parquet")) for split in SPLITS
    }
    for split, paths in files.items():
        if not paths:
            raise ValueError(f"Missing split {split!r} in {source / 'data'}")
    return files


def _parquet():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Install preprocessing dependencies: pip install -e '.[preprocessing]'"
        ) from exc
    return pq


def _token_columns(batch):
    """Validate paired integer token lists and their terminal EOS tokens."""
    import numpy as np
    import pyarrow as pa

    columns = {}
    lengths = None
    for name in ("input_id_y", "input_id_x"):
        index = batch.schema.get_field_index(name)
        if index < 0:
            raise ValueError(f"Missing token column {name!r}")
        column = batch.column(index)
        if not (pa.types.is_list(column.type) or pa.types.is_large_list(column.type)):
            raise ValueError(f"Expected integer token lists in {name!r}")
        if (
            column.null_count
            or column.values.null_count
            or not pa.types.is_integer(column.type.value_type)
        ):
            raise ValueError(f"Expected non-null integer tokens in {name!r}")
        offsets = column.offsets.to_numpy()
        values = column.values.to_numpy()
        current = np.diff(offsets) - 1
        if np.any(current < 1) or np.any(values[offsets[1:] - 1] != 1):
            raise ValueError("Invalid length or terminal EOS")
        if lengths is not None and not np.array_equal(lengths, current):
            raise ValueError("AA/3Di length mismatch")
        lengths = current
        columns[name] = (offsets, values)
    return columns, lengths
