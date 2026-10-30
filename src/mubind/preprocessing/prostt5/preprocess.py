"""Restartable download → filter → array preparation, independent of the UI."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from filelock import FileLock, Timeout

from ...common.progress import ProgressCallback, ProgressEvent
from .steps._io import source_shards, write_manifest
from .steps.download import download_prostt5
from .steps.validate import discover_datasets, inspect_dataset, is_ready_for_training


def prepare_prostt5(
    directory: Path | str,
    *,
    min_length: int = 50,
    max_length: int = 70,
    source: Path | str | None = None,
    on_progress: ProgressCallback | None = None,
) -> Path:
    """Reuse completed data and retain the full download for future filters.

    Length bounds filter whole sequences; they are not the model's canvas size.
    Arrays keep at least 128 padding columns, which the loader can trim.
    Failed transformations restart only inside a marked, private work directory.
    """
    if not 1 <= min_length <= max_length:
        raise ValueError("Require 1 <= min_length <= max_length")
    directory = Path(directory).expanduser().resolve()
    explicit_source = (
        Path(source).expanduser().resolve() if source is not None else None
    )
    emit = on_progress or (lambda event: None)
    directory.mkdir(parents=True, exist_ok=True)
    cache = directory / ".mubind"
    if cache.is_symlink():
        raise ValueError(f"Working directory must not be a symlink: {cache}")
    cache.mkdir(exist_ok=True)
    try:
        with FileLock(cache / "prepare.lock", timeout=0):
            emit(ProgressEvent("inspect", message="Looking for reusable data"))
            datasets = discover_datasets(directory)
            for info in datasets:
                if is_ready_for_training(info, min_length, max_length, explicit_source):
                    emit(ProgressEvent("reuse", message=str(info.path)))
                    return info.path

            source = explicit_source
            if source is None:
                # Prefer the original source to keep the identity stable across retries.
                candidates = [
                    i
                    for i in datasets
                    if i.state == "ready"
                    and (
                        i.kind == "Source"
                        or (
                            i.kind == "Filtered"
                            and i.minimum is not None
                            and i.maximum is not None
                            and i.minimum <= min_length <= max_length <= i.maximum
                        )
                    )
                ]
                candidates.sort(key=lambda i: (i.kind != "Source", str(i.path)))
                source = (
                    candidates[0].path if candidates else directory / "ProstT5Dataset"
                )
                if not candidates:
                    download_prostt5(source, on_progress=emit)
            source_shards(source)
            info = inspect_dataset(source)
            if info is not None:
                if info.state != "ready":
                    raise ValueError(f"Source is {info.state}: {info.problem}")
                if info.kind == "Filtered" and not (
                    info.minimum is not None
                    and info.maximum is not None
                    and info.minimum <= min_length <= max_length <= info.maximum
                ):
                    raise ValueError(
                        "The source does not cover the requested length range"
                    )
            emit(ProgressEvent("download", 1, 1, unit="files", message="Source ready"))

            signature = {
                "source": str(source),
                "min_length": min_length,
                "max_length": max_length,
                "format": 1,
            }
            key = hashlib.sha256(
                json.dumps(signature, sort_keys=True).encode()
            ).hexdigest()[:12]
            job = cache / key
            if job.is_symlink():
                raise ValueError(f"Working directory must not be a symlink: {job}")
            marker = job / "request.json"
            if job.exists():
                if not marker.is_file() or json.loads(marker.read_text()) != signature:
                    raise ValueError(f"Unrecognised working directory: {job}")
            else:
                job.mkdir()
                write_manifest(marker, signature)

            def build(stage, name, manifest_name, operation):
                for destination in (directory / name, directory / f"{name}_{key}"):
                    if not destination.exists() and not destination.is_symlink():
                        break
                    existing = inspect_dataset(destination)
                    if (
                        existing
                        and existing.state == "ready"
                        and existing.report.get("mubind") == signature
                    ):
                        emit(
                            ProgressEvent(
                                stage, 1, 1, message="Reusing completed stage"
                            )
                        )
                        return destination
                else:
                    raise FileExistsError(f"Output already exists: {destination}")
                working = job / stage
                if working.is_symlink():
                    raise ValueError(
                        f"Working directory must not be a symlink: {working}"
                    )
                if working.exists():
                    shutil.rmtree(working)
                operation(working)
                manifest = working / manifest_name
                report = json.loads(manifest.read_text())
                report["mubind"] = signature
                write_manifest(manifest, report)
                working.rename(destination)
                return destination

            from .steps.prepare import prepare_prostt5 as prepare_arrays
            from .steps.subset import subset_prostt5

            base = f"ProstT5Dataset_len{min_length}_{max_length}"
            if (
                info
                and info.kind == "Filtered"
                and (info.minimum, info.maximum) == (min_length, max_length)
            ):
                subset = source
                emit(
                    ProgressEvent(
                        "filter", 1, 1, message="Using existing length subset"
                    )
                )
            else:
                subset = build(
                    "filter",
                    base,
                    "subset_manifest.json",
                    lambda out: subset_prostt5(
                        source,
                        out,
                        min_length=min_length,
                        max_length=max_length,
                        on_progress=emit,
                    ),
                )
            return build(
                "prepare",
                base + "_diffusion",
                "prostt5_diffusion.json",
                lambda out: prepare_arrays(
                    subset,
                    out,
                    max_length=max(128, max_length),
                    on_progress=emit,
                ),
            )
    except Timeout as exc:
        raise RuntimeError(f"Another preparation is using {directory}.") from exc
