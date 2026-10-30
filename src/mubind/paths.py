"""Default working directories and lookup of files moved into artifacts/."""

import os
from pathlib import Path

ARTIFACTS = Path(os.environ.get("MUBIND_ARTIFACTS_DIR", "artifacts")).expanduser()
MODELS = ARTIFACTS / "models"
DATASETS = ARTIFACTS / "datasets"
TARGETS = ARTIFACTS / "targets"
CHECKPOINTS = ARTIFACTS / "checkpoints"
GENERATIONS = ARTIFACTS / "generations"
EVALUATIONS = ARTIFACTS / "evaluations"


def _resource_location(root: Path, relative: Path) -> Path:
    for old, new in (
        ("benchmarks/data/ppiref45", "benchmarks/ppiref/complexes"),
        ("evaluation/references", "validation"),
        ("evaluation", "validation"),
    ):
        if relative.is_relative_to(old):
            return root / new / relative.relative_to(old)
    return root / relative


def existing_artifact(path: Path) -> Path:
    """Find a moved input while respecting any file still at the supplied path."""
    path = Path(path).expanduser()
    try:
        exists = path.exists()
    except OSError:
        # Setup must remain editable when a saved directory is inaccessible.
        return path
    if exists:
        return path
    for directory in (path, *path.parents):
        if directory.name == "artifacts":
            relative = path.relative_to(directory)
            for root in (directory, ARTIFACTS):
                moved = _resource_location(root, relative)
                if moved.exists():
                    return moved
        if directory.parent.name == "mubind" and directory.parent.parent.name == "src":
            group = {"benchmarks": "benchmarks", "evaluation": "evaluation", "example_targets": "examples/targets"}.get(directory.name)
            if group:
                moved = _resource_location(
                    directory.parent.parent.parent / "artifacts",
                    Path(group) / path.relative_to(directory),
                )
                if moved.exists():
                    return moved
        if directory.name in {"targets", "checkpoints", "generations", "evaluations", "models", "datasets"}:
            if directory.parent.name == "artifacts":
                continue
            moved = directory.parent / "artifacts" / path.relative_to(directory.parent)
            if moved.exists():
                return moved
            configured = ARTIFACTS / path.relative_to(directory.parent)
            if configured.exists():
                return configured
    return path
