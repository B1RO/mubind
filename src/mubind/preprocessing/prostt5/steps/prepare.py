"""Convert length-filtered ProstT5 Parquet shards into memory-mapped arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ....common.progress import ProgressCallback, ProgressEvent
from ....common.properties import DEFAULT_PROPERTY_DEFINITION, residue_property_scales
from ....dataset.prostt5_dataset import AA
from ._io import _parquet, _token_columns, source_shards, write_manifest


def _padded_tokens(column, lengths: np.ndarray, width: int) -> np.ndarray:
    offsets, values = column
    matrix = np.zeros((len(lengths), width), dtype=np.int64)
    rows = np.repeat(np.arange(len(lengths)), lengths)
    starts = np.repeat(offsets[:-1], lengths)
    positions = np.arange(lengths.sum()) - np.repeat(
        np.cumsum(lengths) - lengths, lengths
    )
    matrix[rows, positions] = values[starts + positions]
    return matrix


def prepare_prostt5(
    source: Path | str,
    out: Path | str,
    *,
    max_length: int = 128,
    batch_size: int = 8192,
    on_progress: ProgressCallback | None = None,
) -> Path:
    """Convert length-filtered Parquet to the existing memory-mapped format.

    Use ``subset_prostt5`` first if any sequence exceeds ``max_length``.
    Unsupported AA or 3Di tokens are errors.
    Properties and population statistics match the original OmegAMP-based
    preparation. The output directory must not already exist.
    """
    if max_length < 1 or batch_size < 1:
        raise ValueError("max_length and batch_size must be positive")
    pq = _parquet()
    source, out = Path(source), Path(out)
    files = source_shards(source)
    counts = {
        split: sum(pq.ParquetFile(path).metadata.num_rows for path in paths)
        for split, paths in files.items()
    }
    total_rows = sum(counts.values())
    processed = retained = 0
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "state": "preparing",
        "source": str(source),
        "max_length": max_length,
        "property_definition": DEFAULT_PROPERTY_DEFINITION,
        "splits": {},
    }
    subset_manifest = source / "subset_manifest.json"
    if subset_manifest.exists():
        subset = json.loads(subset_manifest.read_text())
        report["length_filter"] = {
            "min_amino_acids": subset["min_amino_acids"],
            "max_amino_acids": subset["max_amino_acids"],
            "inclusive": True,
        }
    manifest = out / "prostt5_diffusion.json"
    write_manifest(manifest, report)
    if on_progress:
        on_progress(ProgressEvent("prepare", total=total_rows))
    charges, hydros, terminal = residue_property_scales(DEFAULT_PROPERTY_DEFINITION)
    charge_lut, hydro_lut = np.zeros(256), np.zeros(256)
    charge_lut[3:23] = [charges[aa] for aa in AA]
    hydro_lut[3:23] = [hydros[aa] for aa in AA]
    try:
        for split, paths in files.items():
            count = counts[split]
            if count == 0:
                raise ValueError(f"No supported sequences in {split}")
            directory = out / split
            directory.mkdir()
            arrays = {
                name: np.lib.format.open_memmap(
                    directory / f"{name}.npy", mode="w+", dtype=dtype, shape=shape
                )
                for name, dtype, shape in (
                    ("aa", "uint8", (count, max_length)),
                    ("threedi", "uint8", (count, max_length)),
                    ("properties", "float32", (count, 3)),
                )
            }
            cursor = 0
            sums, squares = np.zeros(3), np.zeros(3)
            for path in paths:
                for batch in pq.ParquetFile(path).iter_batches(
                    batch_size=batch_size, use_threads=False
                ):
                    columns, lengths = _token_columns(batch)
                    if np.any(lengths > max_length):
                        raise ValueError(
                            "Sequence exceeds max_length; use subset_prostt5 first"
                        )
                    a = _padded_tokens(columns["input_id_y"], lengths, max_length)
                    d = _padded_tokens(columns["input_id_x"], lengths, max_length)
                    mask = np.arange(max_length)[None, :] < lengths[:, None]
                    if np.any(mask & ((a < 3) | (a > 22))):
                        raise ValueError("Unsupported AA token")
                    if np.any(mask & ((d < 128) | (d > 147))):
                        raise ValueError("Unsupported 3Di token")
                    stop = cursor + len(a)
                    properties = np.stack(
                        [
                            lengths,
                            np.round(charge_lut[a].sum(axis=1) + terminal, 3),
                            hydro_lut[a].sum(axis=1) / lengths,
                        ],
                        axis=1,
                    )
                    arrays["aa"][cursor:stop] = a
                    arrays["threedi"][cursor:stop] = d
                    arrays["properties"][cursor:stop] = properties
                    sums += properties.sum(axis=0)
                    squares += (properties**2).sum(axis=0)
                    cursor = stop
                    processed += len(batch)
                    retained += len(a)
                    if on_progress:
                        on_progress(
                            ProgressEvent(
                                "prepare",
                                processed,
                                total_rows,
                                retained,
                                message=split,
                            )
                        )
                if not on_progress:
                    print(
                        f"Prepared {split}: {cursor:,} rows after {path.name}",
                        flush=True,
                    )
            for array in arrays.values():
                array.flush()
            if cursor == 0:
                raise ValueError(f"No supported sequences in {split}")
            mean = sums / cursor
            std = np.sqrt(np.maximum(0, squares / cursor - mean**2))
            std[std == 0] = 1
            report["splits"][split] = {
                "rows": cursor,
                "property_mean": mean.tolist(),
                "property_std": std.tolist(),
            }
            write_manifest(manifest, report)
        report["state"] = "complete"
        write_manifest(manifest, report)
    except Exception as exc:
        report.update(state="failed", error=str(exc))
        write_manifest(manifest, report)
        raise
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8192)
    args = vars(parser.parse_args())
    print(prepare_prostt5(**args))


if __name__ == "__main__":
    main()
