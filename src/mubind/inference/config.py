"""Configuration dataclasses and menu choices for sequence sampling and binder design.

Kept free of PyTorch and model imports so CLI menus and argument parsers can load
and validate user settings without initializing CUDA or importing the U-Net.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..paths import GENERATIONS, TARGETS
from ..setup.checkpoint import default_checkpoint


# -----------------------------------------------------------------------------
# Direct sequence sampling (`mubind sample` -> `inference/generate.py`)
# -----------------------------------------------------------------------------


@dataclass
class GenerationConfig:
    """Settings for direct checkpoint sampling (`mubind sample`, defaults to 50-step DDIM)."""

    checkpoint: Path | None = None
    out: Path | None = None
    count: int = 16
    length: int = 60
    charge: float | None = None
    hydrophobicity: float | None = None
    batch_size: int = 16
    sampler: str = "ddim"
    sampling_steps: int | None = 50
    ddim_eta: float = 0.0
    seed: int = 0
    device: str = "auto"
    precision: str = "auto"


def next_output(directory: Path = GENERATIONS) -> Path:
    """Choose a fresh timestamped run folder without creating anything on disk."""
    stem = f"run-{datetime.now():%Y-%m-%d_%H-%M-%S}"
    path = directory / stem
    index = 1
    while path.exists() or path.is_symlink():
        path = directory / f"{stem}-{index:03d}"
        index += 1
    return path


# -----------------------------------------------------------------------------
# Target-directed minibinder design (`mubind generate` -> `inference/design.py`)
# -----------------------------------------------------------------------------


@dataclass
class DesignConfig:
    """Settings for the 5-stage target-directed binder design pipeline (`mubind generate`)."""

    target: Path | None = None
    checkpoint: Path | None = None
    targets_dir: Path = TARGETS
    out: Path | None = None
    target_spec: str | None = None
    hotspots: tuple[str, ...] = ()
    num_complexa: int = 100
    binder_length: int = 64
    complexa_steps: int = 400
    replicas: int = 4
    num_diffusion: int = 5
    redesign_batch_size: int = 2048
    complexa_batch_size: int = 16
    structural_guidance: float = 5.0
    sampler: str = "ddpm"
    sampling_steps: int | None = None
    ddim_eta: float = 0.0
    scaffold_index: int | None = None
    all_scaffolds: bool = False
    top_scaffolds: int | None = None
    reuse_scaffolds: Path | None = None
    edit_residues: str | None = None
    seed: int | None = None
    device: str = "auto"
    folding_endpoint: str = "auto"
    models_dir: Path | None = None
    foldseek: str = "foldseek"


@dataclass(frozen=True)
class RedesignChoice:
    """Interactive selection of ranked Complexa scaffolds and optional property overrides."""

    top_scaffolds: int
    sequences_per_scaffold: int
    ranks: tuple[int, ...] | None = None
    properties: dict[str, dict[str, float]] | None = None

    def validate(self, available: int) -> None:
        if (
            type(self.top_scaffolds) is not int
            or not 1 <= self.top_scaffolds <= available
        ):
            raise ValueError(f"Choose between 1 and {available} ranked scaffolds")
        if (
            type(self.sequences_per_scaffold) is not int
            or self.sequences_per_scaffold < 1
        ):
            raise ValueError("Sequences per scaffold must be a positive integer")
        if self.ranks is not None:
            if len(self.ranks) != self.top_scaffolds or len(set(self.ranks)) != len(
                self.ranks
            ):
                raise ValueError("Choose distinct scaffold ranks")
            if any(
                type(rank) is not int or not 1 <= rank <= available
                for rank in self.ranks
            ):
                raise ValueError(f"Scaffold ranks must be between 1 and {available}")

    def select(self, ranked: list[dict]) -> list[dict]:
        self.validate(len(ranked))
        return (
            [ranked[rank - 1] for rank in self.ranks]
            if self.ranks is not None
            else ranked[: self.top_scaffolds]
        )


@dataclass(frozen=True)
class MoreScaffolds:
    """Interactive request to sample an additional batch of Complexa scaffolds."""

    count: int


@dataclass(frozen=True)
class MoreCandidates:
    """Interactive request to sample another round of redesigned sequences."""

    count: int
    properties: dict[str, dict[str, float]] | None = None


def parse_scaffold_ranks(value: str, available: int) -> tuple[int, ...]:
    """Parse a single count (`'3'` -> top 3) or comma-separated 1-based ranks (`'1,2,5'`)."""
    parts = value.strip().split(",")
    if not all(part.strip().isdecimal() for part in parts):
        raise ValueError("Enter a top count such as 3, or ranks such as 1,2,5")
    numbers = tuple(int(part.strip()) for part in parts)
    if any(not 1 <= number <= available for number in numbers):
        raise ValueError(f"Choose ranks between 1 and {available}")
    if len(set(numbers)) != len(numbers):
        raise ValueError("Choose each scaffold rank only once")
    return tuple(range(1, numbers[0] + 1)) if len(parts) == 1 else numbers


def default_design_checkpoint(models_dir: Path | None = None) -> Path | None:
    """Resolve the saved checkpoint preference or local default 300k weights."""
    return default_checkpoint(models_dir)


__all__ = [
    "DesignConfig",
    "GenerationConfig",
    "MoreCandidates",
    "MoreScaffolds",
    "RedesignChoice",
    "default_design_checkpoint",
    "next_output",
    "parse_scaffold_ranks",
]
