"""Find and verify existing dataset folders on disk so they can be reused.

Preparing the dataset takes three slow steps: downloading the raw protein data,
filtering it by sequence length, and converting it into training-ready arrays.
This module checks a data directory to see which of these steps have already
been completed and verifies that the files are intact and not half-finished,
allowing the cli to reuse existing data instead of rebuilding it from scratch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ._io import SPLITS, source_shards


@dataclass
class DatasetInfo:
    """Summary of a single dataset folder found on disk."""

    path: Path
    kind: str  # "Source", "Filtered", or "Prepared"
    state: str  # "ready", "incomplete", or "invalid"
    minimum: int | None
    maximum: int | None
    counts: dict[str, int | None]
    report: dict
    problem: str = ""


def discover_datasets(directory: Path | str) -> list[DatasetInfo]:
    """List every dataset stage folder inside a data directory.

    Everything comes from the same ProstT5 dataset, but each processing step
    saves a separate subfolder on disk: one for the raw download
    (`ProstT5Dataset`), one for the length-filtered subset
    (e.g. `ProstT5Dataset_len50_70`), and one for the final training arrays
    (e.g. `ProstT5Dataset_len50_70_diffusion`), plus extra folders if you try
    other length ranges. Treating each folder as a separate entry lets the
    pipeline know whether it can jump straight to training arrays, filter from
    an existing download, or download from scratch.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    paths = {
        directory,
        *(
            path
            for path in directory.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ),
    }
    return [
        info for path in sorted(paths) if (info := inspect_dataset(path)) is not None
    ]


def is_ready_for_training(
    info: DatasetInfo, minimum: int, maximum: int, source: Path | None = None
) -> bool:
    """Return True if this dataset folder has finished all preparation steps for the given length range.

    Specifically, it checks that:
    1. The folder is in the final "Prepared" stage, finished without errors, and matches `minimum`/`maximum`.
    2. It was built from the requested `source` folder (if one was specified).
    3. Every protein inside the saved `.npy` files actually falls between `minimum` and `maximum`.
    """
    if (
        info.kind != "Prepared"
        or info.state != "ready"
        or (info.minimum, info.maximum) != (minimum, maximum)
        or info.report.get("property_definition") != "omegamp_modlamp_v1"
    ):
        return False
    if source is not None and info.report.get("mubind", {}).get("source") != str(
        source
    ):
        return False
    import numpy as np

    for split in SPLITS:
        properties = np.load(info.path / split / "properties.npy", mmap_mode="r")
        lengths = properties[: info.counts[split], 0]
        if not np.all((lengths >= minimum) & (lengths <= maximum)):
            return False
    return True


def inspect_dataset(path: Path) -> DatasetInfo | None:
    """Check one folder to see which dataset stage it holds and whether its files are intact."""
    for name, kind in (
        ("prostt5_diffusion.json", "Prepared"),
        ("subset_manifest.json", "Filtered"),
        ("download_manifest.json", "Source"),
    ):
        manifest = path / name
        if not manifest.is_file():
            continue
        info = DatasetInfo(path, kind, "invalid", None, None, {}, {})
        try:
            report = json.loads(manifest.read_text())
            info.report = report
            bounds = report.get("length_filter", report)
            info.minimum = bounds.get("min_amino_acids")
            info.maximum = bounds.get("max_amino_acids")
            if kind == "Prepared" and info.maximum is None:
                info.minimum, info.maximum = 1, report.get("max_length")
            info.counts = {
                split: report.get("splits", {})
                .get(split, {})
                .get("rows", report.get("splits", {}).get(split, {}).get("kept_rows"))
                for split in SPLITS
            }
            if report.get("state") != "complete":
                info.state = "incomplete"
                info.problem = report.get("error", "Preparation did not finish")
                return info
            if kind == "Prepared":
                _validate_arrays(path, report)
            else:
                source_shards(path)
                # Manifests list every shard: don't mistake a partial copy for a complete source.
                for entry in report.get("files", []):
                    relative = (
                        entry if isinstance(entry, str) else "data/" + entry["file"]
                    )
                    if not (path / relative).is_file():
                        raise ValueError(f"Missing shard: {relative}")
            info.state = "ready"
        except (
            OSError,
            EOFError,
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
        ) as exc:
            info.problem = str(exc)
        return info
    return None


def _validate_arrays(path: Path, report: dict) -> None:
    """Verify that the saved `.npy` number arrays exist and match the manifest shapes."""
    import numpy as np

    width = report["max_length"]
    if not isinstance(width, int) or width < 1:
        raise ValueError("Invalid padding width")
    for split in SPLITS:
        stats = report["splits"][split]
        rows = stats["rows"]
        if not isinstance(rows, int) or rows < 1:
            raise ValueError(f"{split}: no usable rows")
        for name, dtype, tail in (
            ("aa", "uint8", (width,)),
            ("threedi", "uint8", (width,)),
            ("properties", "float32", (3,)),
        ):
            array = np.load(
                path / split / f"{name}.npy", mmap_mode="r", allow_pickle=False
            )
            if (
                array.ndim != len(tail) + 1
                or len(array) < rows
                or array.shape[1:] != tail
                or array.dtype != dtype
            ):
                raise ValueError(f"{split}/{name}.npy does not match its manifest")
        for field in ("property_mean", "property_std"):
            values = np.asarray(stats[field], dtype=float)
            if values.shape != (3,) or not np.isfinite(values).all():
                raise ValueError(f"{split}: invalid {field}")
