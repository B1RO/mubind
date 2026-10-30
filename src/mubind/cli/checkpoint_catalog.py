"""Discover local redesign checkpoints without loading their tensor data."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from stat import S_ISREG

from ..paths import CHECKPOINTS, existing_artifact
from ..setup.config import load_config

def load_inference_checkpoint(*args, **kwargs):
    from ..inference.helpers.checkpoint import load_inference_checkpoint as load

    return load(*args, **kwargs)


def validate_design_checkpoint(config):
    from ..inference.helpers.checkpoint import validate_design_checkpoint as validate

    return validate(config)



@dataclass(frozen=True)
class CheckpointOption:
    path: Path
    step: int | None = None
    problem: str | None = None
    weights: str | None = None
    checked: bool = True
    fingerprint: tuple[int, ...] | None = None

    @property
    def compatible(self) -> bool:
        return self.checked and self.problem is None

    @property
    def label(self) -> str:
        step = f" · step {self.step:,}" if self.step is not None else ""
        return f"{self.path.name}{step}"


def checkpoint_folders(selected: Path | None = None, models_dir: Path | None = None) -> list[Path]:
    folders = [CHECKPOINTS, (models_dir or load_config().models_dir) / "mubind"]
    if selected is not None:
        folders.insert(0, existing_artifact(selected).parent)
    return list(dict.fromkeys(folder.expanduser().resolve() for folder in folders))


def _fingerprint(stat) -> tuple[int, ...]:
    # Catch replacement and same-size rewrites, including preserved mtimes.
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


@lru_cache(maxsize=128)
def _inspect(path: Path, fingerprint: tuple[int, ...]) -> CheckpointOption:
    try:
        checkpoint = load_inference_checkpoint(path, metadata_only=True)
        validate_design_checkpoint(checkpoint.config)
        if _fingerprint(path.stat()) != fingerprint:
            raise ValueError("Checkpoint changed while being checked. Select it again when writing finishes.")
        return CheckpointOption(
            path, checkpoint.step, weights=checkpoint.weights, fingerprint=fingerprint
        )
    except Exception as exc:
        return CheckpointOption(
            path,
            problem=(str(exc).splitlines() or [type(exc).__name__])[0],
            fingerprint=fingerprint,
        )


_checked_options: dict[Path, CheckpointOption] = {}


def inspect_design_checkpoint(
    path: Path, *, refresh: bool = False, check: bool = True
) -> CheckpointOption:
    if refresh:
        _inspect.cache_clear()
        _checked_options.clear()
    path = existing_artifact(path).resolve()
    try:
        stat = path.stat()
        if not S_ISREG(stat.st_mode):
            return CheckpointOption(path, problem="Not a checkpoint file")
    except OSError:
        return CheckpointOption(path, problem="File not found or unreadable")
    fingerprint = _fingerprint(stat)
    if not check:
        previous = _checked_options.get(path)
        if previous is not None and previous.fingerprint == fingerprint:
            return previous
        return CheckpointOption(path, checked=False, fingerprint=fingerprint)
    option = _inspect(path, fingerprint)
    if len(_checked_options) >= 128:
        _checked_options.clear()
    _checked_options[path] = option
    return option


def discover_design_checkpoints(
    selected: Path | None = None, models_dir: Path | None = None, *, check: bool = True
):
    """List files; menus defer compatibility checks until a file is selected."""
    folders = checkpoint_folders(selected, models_dir)
    paths = set()
    for folder in folders:
        if folder.is_dir():
            paths.update(
                path.resolve() for path in folder.iterdir()
                if path.suffix.lower() in (".pt", ".pth", ".ckpt") and path.is_file()
            )
    if selected is not None:
        paths.add(existing_artifact(selected).resolve())
    return [
        inspect_design_checkpoint(path) if check else CheckpointOption(path, checked=False)
        for path in sorted(paths)
    ]
