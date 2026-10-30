"""Explicit schedules for denoising, generation, and structural evaluation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any

PANEL_INDICES = (63, 66, 51, 57, 26, 0, 2, 22, 35, 43)
REGIMES = {
    "full": "300k evaluation · sequences and structures",
    "sequence": "300k sequence panels · folding disabled",
    "loss": "Denoising loss only",
    "off": "Evaluation disabled",
    "custom": "Custom evaluation suites",
}
STRUCTURAL_CHECKS = frozenset({"binder_structure", "realized_threedi"})
CHECKS = STRUCTURAL_CHECKS | {"sequence_validity", "property_mae", "fixed_residues"}


@dataclass
class SamplingConfig:
    sampler: str = "ddim"
    steps: int | None = 50
    ddim_eta: float = 0.0


@dataclass
class DenoisingConfig:
    enabled: bool = True
    every: int = 1000
    batches: int = 3
    batch_size: int = 1024
    noise_samples: int = 4
    noise_bins: int = 10
    seed: int = 20260910
    threedi_utilization: bool = True
    utilization_count: int = 100
    utilization_batch_size: int = 128
    utilization_cohort_seed: int = 20260839
    utilization_noise_seed: int = 20260841
    high_noise_start: float = 0.75


@dataclass
class FoldingConfig:
    endpoint: str = "auto"
    foldseek: str = "foldseek"
    reference_manifest: str | None = None
    reference_root: str | None = None
    cache_dir: str | None = None
    timeout: float = 1800.0
    seed: int = 0
    protocol: dict[str, Any] = field(
        default_factory=lambda: {
            "model": "biohub/ESMFold2-Fast",
            "esmc_model": "biohub/ESMC-6B",
            "revision": "c6c7958d63f5f2f1f0fed0bb9462316f8ccceea6",
            "num_loops": 3,
            "num_sampling_steps": 20,
            "num_diffusion_samples": 1,
        }
    )
    models_dir: str | None = None
    device: str = "auto"


@dataclass
class SuiteConfig:
    enabled: bool = True
    task_mode: str = "conditioned"
    interval_steps: int = 1000
    run_at_end: bool = True
    sample_count: int = 16
    batch_size: int = 16
    cohort_seed: int = 20260821
    task_seed: int = 20260822
    sampling_seed: int = 20260823
    property_guidance_scale: float = 1.0
    validators: tuple[str, ...] = ("sequence_validity", "property_mae")
    dataset_indices: tuple[int, ...] | None = None
    candidates_per_task: int = 1
    partial_mask_seed: int | None = None
    visible_min_fraction: float = 0.35
    visible_max_fraction: float = 0.80
    property_condition_mask: tuple[int, int, int] | None = None
    gap_length: int = 10
    select_checkpoint: bool = False

    @property
    def candidate_count(self):
        return self.sample_count * self.candidates_per_task


def reference_suites() -> dict[str, SuiteConfig]:
    """The 22 enabled suites in the archived 250k-to-300k run."""
    structural = SuiteConfig(
        task_mode="threedi",
        interval_steps=5000,
        sample_count=10,
        batch_size=20,
        dataset_indices=PANEL_INDICES,
        candidates_per_task=10,
        validators=("sequence_validity", "realized_threedi", "binder_structure"),
        select_checkpoint=True,
    )
    suites = {
        "routine": SuiteConfig(),
        "property_steering": SuiteConfig(
            task_mode="property_steering",
            interval_steps=5000,
            sample_count=128,
            cohort_seed=20260824,
            sampling_seed=20260825,
        ),
        "threedi_structure": structural,
        "partial_threedi": replace(
            structural,
            partial_mask_seed=20260925,
            validators=("sequence_validity", "realized_threedi"),
            select_checkpoint=False,
        ),
    }
    combinations = {
        "none": (0, 0, 0),
        "length": (1, 0, 0),
        "charge": (0, 1, 0),
        "hydrophobicity": (0, 0, 1),
        "length_charge": (1, 1, 0),
        "length_hydrophobicity": (1, 0, 1),
        "charge_hydrophobicity": (0, 1, 1),
        "all": (1, 1, 1),
    }
    for prefix in ("full", "partial"):
        for name, mask in combinations.items():
            suites[f"{prefix}_properties_{name}"] = SuiteConfig(
                task_mode="threedi",
                interval_steps=5000,
                cohort_seed=20260824,
                sampling_seed=20260825,
                property_condition_mask=mask,
                partial_mask_seed=20260925 if prefix == "partial" else None,
            )
    inpainting = replace(
        structural,
        task_mode="inpainting",
        interval_steps=1000,
        validators=("sequence_validity", "fixed_residues"),
        select_checkpoint=False,
    )
    suites["inpainting"] = inpainting
    suites["inpainting_structure"] = replace(
        inpainting,
        interval_steps=5000,
        validators=("sequence_validity", "fixed_residues", "binder_structure"),
    )
    return suites


def construct(cls, raw):
    if not isinstance(raw, dict):
        raise TypeError(f"{cls.__name__} settings must be an object")
    unknown = raw.keys() - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(
            f"Unknown {cls.__name__} options: {', '.join(sorted(unknown))}"
        )
    return cls(**raw)


@dataclass
class EvaluationConfig:
    regime: str = "full"
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    denoising: DenoisingConfig = field(default_factory=DenoisingConfig)
    folding: FoldingConfig = field(default_factory=FoldingConfig)
    run_on_start: bool = False
    run_on_resume: bool = True
    # Overrides retain unspecified values from the named suite. Custom starts empty.
    suites: dict[str, dict[str, Any]] = field(default_factory=dict)

    def resolved_suites(self) -> dict[str, SuiteConfig]:
        if self.regime in {"loss", "off"}:
            if self.suites:
                raise ValueError(
                    "Suite overrides require full, sequence, or custom evaluation"
                )
            return {}
        suites = reference_suites() if self.regime != "custom" else {}
        if self.regime == "sequence":
            suites = {
                name: value
                for name, value in suites.items()
                if not STRUCTURAL_CHECKS.intersection(value.validators)
            }
        for name, raw in self.suites.items():
            if not name or any(
                c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in name
            ):
                raise ValueError(f"Invalid evaluation suite name: {name!r}")
            if not isinstance(raw, dict):
                raise TypeError(f"Evaluation suite {name} must be an object")
            base = asdict(suites.get(name, SuiteConfig()))
            suites[name] = construct(SuiteConfig, base | raw)
        return {name: suite for name, suite in suites.items() if suite.enabled}

    def validate(self, timesteps: int) -> None:
        if self.regime not in REGIMES:
            raise ValueError(f"evaluation.regime must be one of {', '.join(REGIMES)}")
        suites = self.resolved_suites()
        s, d = self.sampling, self.denoising
        if s.sampler not in {"ddpm", "ddim"} or not 0 <= s.ddim_eta <= 1:
            raise ValueError("Invalid evaluation sampling settings")
        if (
            suites
            and s.steps is not None
            and (type(s.steps) is not int or not 1 <= s.steps <= timesteps)
        ):
            raise ValueError(
                "Evaluation sampling steps must fit the diffusion schedule"
            )
        if s.sampler == "ddpm" and (
            s.steps not in (None, timesteps) or s.ddim_eta != 0
        ):
            raise ValueError("DDPM evaluation requires all diffusion steps and eta=0")
        for key in (
            "every",
            "batches",
            "batch_size",
            "noise_samples",
            "noise_bins",
            "utilization_count",
            "utilization_batch_size",
        ):
            value = getattr(d, key)
            if type(value) is not int or value < 1:
                raise ValueError(
                    f"evaluation.denoising.{key} must be a positive integer"
                )
        if not 0 <= d.high_noise_start < 1:
            raise ValueError("evaluation.denoising.high_noise_start must be in [0, 1)")
        for name, suite in suites.items():
            for key in (
                "interval_steps",
                "sample_count",
                "batch_size",
                "candidates_per_task",
                "gap_length",
            ):
                value = getattr(suite, key)
                if type(value) is not int or value < 1:
                    raise ValueError(
                        f"Evaluation {name}.{key} must be a positive integer"
                    )
            if suite.task_mode not in {
                "conditioned",
                "property_steering",
                "threedi",
                "inpainting",
            }:
                raise ValueError(f"Unknown evaluation task mode: {suite.task_mode}")
            if not suite.validators or set(suite.validators) - CHECKS:
                raise ValueError(f"Unknown or empty validators for {name}")
            if len(set(suite.validators)) != len(suite.validators):
                raise ValueError(f"Duplicate validators for {name}")
            if (
                not math.isfinite(suite.property_guidance_scale)
                or suite.property_guidance_scale < 0
            ):
                raise ValueError(f"Invalid property guidance for {name}")
            if suite.candidates_per_task != 1 and suite.task_mode not in {
                "threedi",
                "inpainting",
            }:
                raise ValueError(
                    f"Replicate candidates are only supported for structural and inpainting panels: {name}"
                )
            if suite.select_checkpoint and not STRUCTURAL_CHECKS.issubset(
                suite.validators
            ):
                raise ValueError(
                    f"{name}: checkpoint selection requires both structural checks"
                )
        if not math.isfinite(self.folding.timeout) or self.folding.timeout <= 0:
            raise ValueError("Folding timeout must be finite and positive")
        if not isinstance(self.folding.endpoint, str) or not (
            self.folding.endpoint == "auto"
            or self.folding.endpoint.startswith(("unix://", "http://", "https://"))
        ):
            raise ValueError(
                "Folding endpoint must be auto, an HTTP URL, or unix:///socket/path"
            )
        device = self.folding.device
        if not isinstance(device, str) or not (
            device in {"auto", "cpu", "cuda"}
            or (device.startswith("cuda:") and device[5:].isdecimal())
        ):
            raise ValueError("Folding device must be auto, cpu, cuda, or cuda:N")


def evaluation_from_dict(raw: dict) -> EvaluationConfig:
    raw = dict(raw)
    for name, cls in (
        ("sampling", SamplingConfig),
        ("denoising", DenoisingConfig),
        ("folding", FoldingConfig),
    ):
        if name in raw:
            raw[name] = construct(cls, raw[name])
    return construct(EvaluationConfig, raw)
