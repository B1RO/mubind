"""Benchmark inputs, JSON output and fresh run directories."""

from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import json
import os
import tempfile
from pathlib import Path
from ..resources import resource_directory

ROOT = Path(__file__).resolve().parent
INPUTS = resource_directory("benchmarks")
DATA = INPUTS
PROPERTIES = ("length", "charge", "hydrophobicity")
STD = (5.895077705383301, 4.508426666259766, 0.20465117692947388)
MODES = ("unconditional", "properties", "structure", "full-design", "inpainting")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        if ".jsonl" in path.name:
            return [json.loads(line) for line in handle if line.strip()]
        return json.load(handle)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".paper-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prostt5_targets():
    return read(DATA / "prostt5.json")["test"]


def training_corpus():
    return read(DATA / "prostt5.json")["training"]


def verify_bundle():
    manifest = read(INPUTS / "manifest.json")
    for name, record in manifest["files"].items():
        path = INPUTS / name
        require(path.is_file(), f"Missing paper artifact: {name}")
        require(sha256(path) == record["sha256"], f"Changed paper artifact: {name}")
    return {
        "files_verified": len(manifest["files"]),
        "manifest_sha256": sha256(INPUTS / "manifest.json"),
    }


def runtime_versions(*packages):
    return {name: importlib.metadata.version(name) for name in packages}


def unused_output(path):
    path = Path(path)
    candidate, number = path, 1
    while candidate.exists() or candidate.is_symlink():
        candidate = path.with_name(f"{path.name}-{number:03d}")
        number += 1
    return candidate


def new_run(out, settings):
    """Claim a new output folder, numbering collisions without replacing files."""
    out = Path(out).expanduser().absolute()
    require(
        not out.resolve().is_relative_to(ROOT) and not out.resolve().is_relative_to(INPUTS),
        "Outputs must be outside the benchmark code and input folders"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    while True:
        destination = unused_output(out)
        try:
            destination.mkdir()
        except FileExistsError:
            continue
        save(destination / "run.json", settings)
        return destination
