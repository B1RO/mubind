"""Serializable settings for the standalone ProstT5 trainer.

JSON files use the same field names as these dataclasses. Unknown options are
rejected rather than silently ignored; the legacy Hydra files are not this API.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import torch

from ..paths import CHECKPOINTS
from ..common.properties import DEFAULT_PROPERTY_DEFINITION
from ..diffusion import DiffusionConfig
from ..evaluation.config import EvaluationConfig, evaluation_from_dict
from ..model import (
    BackboneConfig,
    PropertyConfig,
    ThreeDiConfig,
    ThreeDiContextConfig,
    ThreeDiCrossAttentionConfig,
)
from .tasks.config import (
    DEFAULT_TASK_STAGES,
    PropertyDropoutConfig,
    TaskConfig,
    TaskStageConfig,
    ThreeDiMaskingConfig,
)


@dataclass
class DataConfig:
    path: str = ""
    batch_size: int = 1024
    num_workers: int = 2
    normalize_embeddings: bool = True
    joint_3di: bool = True
    property_definition: str = DEFAULT_PROPERTY_DEFINITION
    drop_last: bool = True


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 0.01
    scheduler_gamma: float = 0.9999994759735578
    grad_clip: float | None = None


@dataclass
class EMAConfig:
    enabled: bool = True
    decay: float = 0.999
    warmup: bool = True


@dataclass
class ReconstructionConfig:
    pad_weighting: str = "class_equal"
    pad_weight: float = 1.0
    example_weighting: str = "uniform"


@dataclass
class SelfConditionConfig:
    enabled: bool = True
    probability: float = 0.5


@dataclass
class TimestepSamplingConfig:
    mode: str = "uniform"
    high_noise_start: float = 0.75
    high_noise_probability: float = 0.5


@dataclass
class TrainingConfig:
    """New-run defaults from the frozen TUM 250k-to-300k training recipe."""

    data: DataConfig = field(default_factory=DataConfig)
    backbone: BackboneConfig = field(
        default_factory=lambda: BackboneConfig(
            max_length=70,
            x_channels=8,
            hidden_channels=32,
            blocks_per_level=3,
            attention_mode="unet_linear",
        )
    )
    structure: ThreeDiConfig = field(
        default_factory=lambda: ThreeDiConfig(
            dim=128,
            representation="foldseek_2d",
            multiscale=True,
            context=ThreeDiContextConfig(
                layers=4, heads=4, feedforward_dim=512, dropout=0.0
            ),
            cross_attention=ThreeDiCrossAttentionConfig(
                heads=4, dim_head=32, bottleneck=True
            ),
        )
    )
    properties: PropertyConfig = field(
        default_factory=lambda: PropertyConfig(mask_enabled=True)
    )
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    tasks: TaskConfig = field(
        default_factory=lambda: TaskConfig(
            redesign_conditioning_probabilities=(0.05, 0.15, 0.2, 0.6),
            terminal_pad_min=0,
            terminal_pad_max=0,
            global_offset_probability=0.0,
            left_align_threedi_redesign=True,
            stages=DEFAULT_TASK_STAGES,
        )
    )
    threedi_masking: ThreeDiMaskingConfig = field(
        default_factory=lambda: ThreeDiMaskingConfig(
            enabled=True,
            mode="multiple_spans",
            full_probability=0.0,
            visible_min_fraction=0.0,
            visible_max_fraction=1.0,
        )
    )
    property_dropout: PropertyDropoutConfig = field(
        default_factory=PropertyDropoutConfig
    )
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    self_condition: SelfConditionConfig = field(default_factory=SelfConditionConfig)
    timestep_sampling: TimestepSamplingConfig = field(
        default_factory=TimestepSamplingConfig
    )
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    seed: int = 20260910
    steps: int = 300000
    device: str = "auto"
    precision: str = "bf16"
    log_every: int = 100
    save_every: int = 1000
    val_every: int = 1000
    val_batches: int = 3
    validation_seed: int = 20260910
    out: str = str(CHECKPOINTS / "diffusion.pt")
    batched_tasks: bool = True
    compile: bool = True
    compile_mode: str = "reduce-overhead"

    def validate(self) -> None:
        if not self.data.path:
            raise ValueError("Provide a prepared dataset directory")
        for name, value in (
            ("steps", self.steps),
            ("batch_size", self.data.batch_size),
            ("log_every", self.log_every),
            ("val_batches", self.val_batches),
            ("max_length", self.backbone.max_length),
        ):
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (
            ("save_every", self.save_every),
            ("val_every", self.val_every),
            ("num_workers", self.data.num_workers),
        ):
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("precision must be fp32, bf16, or fp16")
        if type(self.compile) is not bool:
            raise ValueError("compile must be true or false")
        if self.compile_mode not in {
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        }:
            raise ValueError("Invalid compile_mode")
        if type(self.data.drop_last) is not bool:
            raise ValueError("data.drop_last must be true or false")
        if self.optimizer.name not in {"radam", "adamw"}:
            raise ValueError("optimizer.name must be radam or adamw")
        if not math.isfinite(self.optimizer.lr) or self.optimizer.lr <= 0:
            raise ValueError("optimizer.lr must be finite and positive")
        if (
            not math.isfinite(self.optimizer.weight_decay)
            or self.optimizer.weight_decay < 0
        ):
            raise ValueError("optimizer.weight_decay must be finite and non-negative")
        if not 0 < self.optimizer.scheduler_gamma <= 1:
            raise ValueError("optimizer.scheduler_gamma must be in (0, 1]")
        if self.optimizer.grad_clip is not None and (
            not math.isfinite(self.optimizer.grad_clip) or self.optimizer.grad_clip <= 0
        ):
            raise ValueError("optimizer.grad_clip must be finite and positive")
        if not 0 <= self.ema.decay < 1:
            raise ValueError("ema.decay must be in [0, 1)")
        if not 0 <= self.self_condition.probability <= 1:
            raise ValueError("self_condition.probability must be in [0, 1]")
        if self.timestep_sampling.mode not in {"uniform", "high_noise_mixture"}:
            raise ValueError(
                "timestep_sampling.mode must be uniform or high_noise_mixture"
            )
        if (
            not 0 <= self.timestep_sampling.high_noise_start < 1
            or not 0 <= self.timestep_sampling.high_noise_probability <= 1
        ):
            raise ValueError("invalid high-noise timestep sampling bounds")
        if self.reconstruction.pad_weighting not in {"class_equal", "weighted"}:
            raise ValueError(
                "reconstruction.pad_weighting must be class_equal or weighted"
            )
        if (
            not math.isfinite(self.reconstruction.pad_weight)
            or self.reconstruction.pad_weight < 0
        ):
            raise ValueError(
                "reconstruction.pad_weight must be finite and non-negative"
            )
        if self.reconstruction.example_weighting not in {"uniform", "target_length"}:
            raise ValueError(
                "reconstruction.example_weighting must be uniform or target_length"
            )
        if self.backbone.x_channels != (8 if self.data.joint_3di else 6):
            raise ValueError(
                "backbone.x_channels must match the data representation (6, or 8 for joint_3di)"
            )
        if not self.properties.mask_enabled:
            raise ValueError(
                "Task training requires properties.mask_enabled to represent absent properties"
            )
        if (self.properties.mean is None) != (self.properties.std is None):
            raise ValueError(
                "Provide both properties.mean and properties.std, or neither"
            )
        if not self.out:
            raise ValueError("out must be a checkpoint path")
        self.diffusion.build()  # Validate the mathematical settings before loading data.
        self.evaluation.validate(self.diffusion.timesteps)

    def to_dict(self) -> dict[str, Any]:
        def plain(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().tolist()
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {key: plain(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [plain(item) for item in value]
            return value

        return plain(asdict(self))


def _construct(cls, values):
    if not isinstance(values, dict):
        raise TypeError(f"{cls.__name__} settings must be an object")
    unknown = values.keys() - {item.name for item in fields(cls)}
    if unknown:
        raise ValueError(
            f"Unknown {cls.__name__} options: {', '.join(sorted(unknown))}"
        )
    return cls(**values)


def config_from_dict(values: dict[str, Any]) -> TrainingConfig:
    values = dict(values)
    # Older recipes/checkpoints included retired optimization and ablation keys.
    values.pop("fast_tasks", None)
    retired = {
        "backbone": ("spatial_conditioning_mode",),
        "structure": ("partner_offset", "partner_offset_max_abs"),
        "diffusion": (
            "cads_enabled", "cads_noise_strength", "cads_tau1", "cads_tau2",
            "cads_rescaled_phi",
        ),
    }
    defaults = TrainingConfig().to_dict()
    # Checkpoints and JSON recipes written before evaluation was introduced
    # retain their original loss-only monitoring until explicitly changed.
    values["evaluation"] = evaluation_from_dict(
        values.get("evaluation", {"regime": "loss"})
    )
    classes = {
        "data": DataConfig,
        "backbone": BackboneConfig,
        "structure": ThreeDiConfig,
        "properties": PropertyConfig,
        "diffusion": DiffusionConfig,
        "tasks": TaskConfig,
        "threedi_masking": ThreeDiMaskingConfig,
        "property_dropout": PropertyDropoutConfig,
        "optimizer": OptimizerConfig,
        "ema": EMAConfig,
        "reconstruction": ReconstructionConfig,
        "self_condition": SelfConditionConfig,
        "timestep_sampling": TimestepSamplingConfig,
    }
    for name, cls in classes.items():
        if name not in values:
            continue
        raw = values[name]
        if not isinstance(raw, dict):
            raise TypeError(f"{name} settings must be an object")
        if name == "tasks" and "stages" not in raw and (
            "layout_probabilities" in raw
            or "redesign_conditioning_probabilities" in raw
        ):
            raw = {**raw, "stages": ()}
        # A partial section overrides the training recipe, not the generic
        # model dataclass defaults (which can describe a different architecture).
        raw = defaults[name] | raw
        for key in retired.get(name, ()):
            raw.pop(key, None)
        if name == "tasks":
            stages_raw = raw.get("stages", ())
            if not isinstance(stages_raw, (list, tuple)):
                raise TypeError("tasks.stages settings must be a list")
            stages = []
            for item in stages_raw:
                if not isinstance(item, dict):
                    raise TypeError("TaskStageConfig settings must be an object")
                stage_item = dict(item)
                for key in (
                    "layout_probabilities",
                    "redesign_conditioning_probabilities",
                ):
                    if key in stage_item:
                        stage_item[key] = tuple(stage_item[key])
                stages.append(_construct(TaskStageConfig, stage_item))
            raw["stages"] = tuple(stages)
        if name == "structure":
            for key, inner in (
                ("context", ThreeDiContextConfig),
                ("cross_attention", ThreeDiCrossAttentionConfig),
            ):
                if raw.get(key) is not None:
                    inner_raw = raw[key]
                    if not isinstance(inner_raw, dict):
                        raise TypeError(f"structure.{key} settings must be an object")
                    inner_raw = defaults[name][key] | inner_raw
                    if key == "context":
                        inner_raw.pop("gated_skip", None)
                        inner_raw.pop("gate_init", None)
                    raw[key] = _construct(inner, inner_raw)
        for key in (
            "channel_mults",
            "layout_probabilities",
            "redesign_conditioning_probabilities",
        ):
            if key in raw:
                raw[key] = tuple(raw[key])
        values[name] = _construct(cls, raw)
    return _construct(TrainingConfig, values)


def load_config(path: str | Path) -> TrainingConfig:
    with Path(path).open() as handle:
        values = json.load(handle)
    if not isinstance(values, dict):
        raise TypeError("Training config must be a JSON object")
    return config_from_dict(values)
