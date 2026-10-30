"""Filter ProstT5 Parquet shards by sequence length without truncation."""

from __future__ import annotations

import argparse
from pathlib import Path

from ....common.progress import ProgressCallback, ProgressEvent
from ._io import _parquet, _token_columns, source_shards, write_manifest


def subset_prostt5(
    source: Path | str,
    out: Path | str,
    *,
    min_length: int = 1,
    max_length: int = 128,
    batch_size: int = 8192,
    on_progress: ProgressCallback | None = None,
) -> Path:
    """Stream an inclusive length subset, preserving split, shard, and row order.

    The output contains Parquet shards under ``data/``. Out-of-range sequences
    are excluded, never truncated. Noncanonical AA filtering happens later,
    during ``prepare_prostt5``. The output directory must not already exist.
    """
    if not 1 <= min_length <= max_length or batch_size < 1:
        raise ValueError("Require 1 <= min_length <= max_length and batch_size > 0")
    pq = _parquet()
    source, out = Path(source), Path(out)
    files = source_shards(source)
    total_rows = sum(
        pq.ParquetFile(path).metadata.num_rows
        for paths in files.values()
        for path in paths
    )
    processed = retained = 0
    out.mkdir(parents=True, exist_ok=False)
    (out / "data").mkdir()
    report = {
        "state": "preparing",
        "source": str(source),
        "min_amino_acids": min_length,
        "max_amino_acids": max_length,
        "splits": {},
        "files": [],
    }
    manifest = out / "subset_manifest.json"
    write_manifest(manifest, report)
    if on_progress:
        on_progress(ProgressEvent("filter", total=total_rows))
    try:
        for split, paths in files.items():
            stats = {"source_rows": 0, "kept_rows": 0, "max_length": 0}
            report["splits"][split] = stats
            for path in paths:
                parquet = pq.ParquetFile(path)
                target = out / "data" / path.name
                total = kept = 0
                with pq.ParquetWriter(
                    target,
                    parquet.schema_arrow.remove_metadata(),
                    compression="zstd",
                ) as writer:
                    for batch in parquet.iter_batches(
                        batch_size=batch_size, use_threads=False
                    ):
                        _, lengths = _token_columns(batch)
                        keep = (lengths >= min_length) & (lengths <= max_length)
                        selected = batch.filter(keep)
                        if len(selected):
                            writer.write_batch(selected.replace_schema_metadata(None))
                            stats["max_length"] = max(
                                stats["max_length"], int(lengths[keep].max())
                            )
                        total += len(batch)
                        kept += len(selected)
                        processed += len(batch)
                        retained += len(selected)
                        if on_progress:
                            on_progress(
                                ProgressEvent(
                                    "filter",
                                    processed,
                                    total_rows,
                                    retained,
                                    message=split,
                                )
                            )
                stats["source_rows"] += total
                stats["kept_rows"] += kept
                report["files"].append(
                    {"file": path.name, "source_rows": total, "kept_rows": kept}
                )
                write_manifest(manifest, report)
                if not on_progress:
                    print(f"{path.name}: retained {kept:,}/{total:,}", flush=True)
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
    parser.add_argument("--min-length", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8192)
    args = vars(parser.parse_args())
    print(subset_prostt5(**args))


if __name__ == "__main__":
    main()
