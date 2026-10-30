"""Task mixture and conditioning augmentation settings."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

LAYOUT_NAMES = ("complete", "prefix", "suffix", "middle", "redesign")
LAYOUT_TO_ID = {name: idx for idx, name in enumerate(LAYOUT_NAMES)}
REDESIGN_CONDITIONING_NAMES = (
    "unconditional",
    "properties",
    "3di",
    "properties_3di",
)
REDESIGN_CONDITIONING_TO_ID = {
    name: idx for idx, name in enumerate(REDESIGN_CONDITIONING_NAMES)
}
REDESIGN_MODES_WITH_PROPERTIES = frozenset(
    {
        REDESIGN_CONDITIONING_TO_ID["properties"],
        REDESIGN_CONDITIONING_TO_ID["properties_3di"],
    }
)
REDESIGN_MODES_WITH_3DI = frozenset(
    {
        REDESIGN_CONDITIONING_TO_ID["3di"],
        REDESIGN_CONDITIONING_TO_ID["properties_3di"],
    }
)
NOT_REDESIGN_CONDITIONING_ID = -1


def _validate_task_probabilities(
    layout_probabilities: tuple[float, ...],
    redesign_conditioning_probabilities: tuple[float, ...],
) -> None:
    if len(layout_probabilities) != len(LAYOUT_NAMES):
        raise ValueError(
            f"layout_probabilities must contain {len(LAYOUT_NAMES)} values"
        )
    if any(
        not math.isfinite(probability) or probability < 0
        for probability in layout_probabilities
    ):
        raise ValueError("layout probabilities must be non-negative")
    if sum(layout_probabilities) <= 0:
        raise ValueError("at least one layout probability must be positive")
    if len(redesign_conditioning_probabilities) != len(
        REDESIGN_CONDITIONING_NAMES
    ):
        raise ValueError(
            "redesign_conditioning_probabilities must contain "
            f"{len(REDESIGN_CONDITIONING_NAMES)} values"
        )
    if any(
        not math.isfinite(probability) or probability < 0
        for probability in redesign_conditioning_probabilities
    ):
        raise ValueError("redesign conditioning probabilities must be non-negative")
    if sum(redesign_conditioning_probabilities) <= 0:
        raise ValueError(
            "at least one redesign conditioning probability must be positive"
        )


@dataclass(frozen=True)
class TaskStageConfig:
    until_step: int
    layout_probabilities: tuple[float, float, float, float, float]
    redesign_conditioning_probabilities: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if type(self.until_step) is not int or self.until_step < 1:
            raise ValueError("until_step must be a positive integer")
        _validate_task_probabilities(
            self.layout_probabilities,
            self.redesign_conditioning_probabilities,
        )


DEFAULT_TASK_STAGES: tuple[TaskStageConfig, ...] = (
    TaskStageConfig(
        until_step=150000,
        layout_probabilities=(0.0, 0.0, 0.0, 0.0, 1.0),
        redesign_conditioning_probabilities=(0.0, 0.0, 1.0, 0.0),
    ),
    TaskStageConfig(
        until_step=200000,
        layout_probabilities=(0.0, 0.0, 0.0, 0.0, 1.0),
        redesign_conditioning_probabilities=(0.05, 0.15, 0.20, 0.60),
    ),
)


@dataclass(frozen=True)
class TaskConfig:
    layout_probabilities: tuple[float, float, float, float, float] = (
        0.35,
        0.05,
        0.05,
        0.05,
        0.50,
    )
    redesign_conditioning_probabilities: tuple[float, float, float, float] = (
        0.10,
        0.40,
        0.30,
        0.20,
    )
    visible_min_fraction: float = 0.35
    visible_max_fraction: float = 0.80
    span_count_geometric_probability: float = 0.50
    dirichlet_concentration: float = 0.50
    editable_3di_retention_probability: float = 0.50
    terminal_pad_min: int = 1
    terminal_pad_max: int = 8
    global_offset_probability: float = 0.50
    left_align_threedi_redesign: bool = False
    stages: tuple[TaskStageConfig, ...] = ()

    def __post_init__(self) -> None:
        _validate_task_probabilities(
            self.layout_probabilities,
            self.redesign_conditioning_probabilities,
        )
        if not 0 < self.visible_min_fraction <= self.visible_max_fraction < 1:
            raise ValueError("visible fractions must satisfy 0 < min <= max < 1")
        if not 0 < self.span_count_geometric_probability <= 1:
            raise ValueError("span_count_geometric_probability must be in (0, 1]")
        if self.dirichlet_concentration <= 0:
            raise ValueError("dirichlet_concentration must be positive")
        if not 0 <= self.editable_3di_retention_probability <= 1:
            raise ValueError("editable_3di_retention_probability must be in [0, 1]")
        if self.terminal_pad_min < 0 or self.terminal_pad_max < self.terminal_pad_min:
            raise ValueError("terminal PAD bounds must satisfy 0 <= min <= max")
        if not 0 <= self.global_offset_probability <= 1:
            raise ValueError("global_offset_probability must be in [0, 1]")
        previous_step = 0
        for stage in self.stages:
            if not isinstance(stage, TaskStageConfig):
                raise TypeError("task stages must be TaskStageConfig instances")
            if stage.until_step <= previous_step:
                raise ValueError(
                    "task stage until_step values must be strictly increasing"
                )
            previous_step = stage.until_step

    def for_step(self, step: int | None = None) -> TaskConfig:
        if step is not None:
            for stage in self.stages:
                if step <= stage.until_step:
                    return replace(
                        self,
                        layout_probabilities=stage.layout_probabilities,
                        redesign_conditioning_probabilities=stage.redesign_conditioning_probabilities,
                        stages=(),
                    )
        return self


@dataclass(frozen=True)
class ThreeDiMaskingConfig:
    """Training-only structural conditioning augmentation; targets stay intact."""

    enabled: bool = False
    full_probability: float = 0.20
    visible_min_fraction: float = 0.35
    visible_max_fraction: float = 0.80
    mode: str = "single_span"
    span_count_geometric_probability: float = 0.50
    dirichlet_concentration: float = 0.50

    def __post_init__(self) -> None:
        if not 0 <= self.full_probability <= 1:
            raise ValueError("3Di masking full_probability must be in [0, 1]")
        if self.mode not in {"single_span", "multiple_spans"}:
            raise ValueError("unknown 3Di masking mode")
        if self.mode == "multiple_spans":
            if not 0 <= self.visible_min_fraction <= self.visible_max_fraction <= 1:
                raise ValueError(
                    "multiple-span visible fractions require 0 <= min <= max <= 1"
                )
        elif not 0 < self.visible_min_fraction <= self.visible_max_fraction < 1:
            raise ValueError("3Di masking visible fractions require 0 < min <= max < 1")
        if not 0 < self.span_count_geometric_probability <= 1:
            raise ValueError("span count probability must be in (0, 1]")
        if self.dirichlet_concentration <= 0:
            raise ValueError("Dirichlet concentration must be positive")


@dataclass(frozen=True)
class PropertyDropoutConfig:
    """Drop property subsets while honoring each sampled redesign mode."""

    enabled: bool = True
    mode: str = "uniform_subset"
    probability: float = 0.25
    allow_empty: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"uniform_subset", "independent"}:
            raise ValueError(
                "property dropout mode must be uniform_subset or independent"
            )
        if not 0 <= self.probability <= 1:
            raise ValueError("property dropout probability must be in [0, 1]")
