"""Core diffusion sampling and canvas decoding (`predict`).

Given an already-loaded `ProteinDiffusionUNet` (or a `StructuralGuidance` wrapper)
and a `DiffusionSchedule`, `predict` runs reverse diffusion on a batch of conditioning
tasks and decodes each generated canvas into an amino acid sequence

1. Pack the task dictionary (`seed_3di`, `c`, `condition_mask`, `seed_x`, `edit_mask`)
   into `StructureInput`, `PropertyInput`, and `SeedInput` on the target device (`model_inputs`).
2. Enter `inference_context` (putting the model in `.eval()` mode, resetting any cached
   sampling state, and optionally forking RNG streams with `seed`) and `autocast_context`.
3. Run reverse diffusion via `diffusion.sample` (`ddim` by default for fast sampling,
   or `ddpm` for full 1,000-step binder design and benchmarks).
4. Decode each continuous `[8, L]` canvas back into a sequence and validity report with
   `decode_canvas_sample`, returning `Prediction(vectors, decoded)`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from ..diffusion import DiffusionSchedule


@dataclass
class Prediction:
    """Raw `[B, C, L]` denoised vectors on CPU and per-sample decoded sequence records."""

    vectors: torch.Tensor
    decoded: list[dict[str, Any]]


def predict(
    model: torch.nn.Module,
    diffusion: DiffusionSchedule,
    tasks: Mapping[str, Any],
    *,
    normalized_embeddings: bool,
    device: str | torch.device | None = None,
    seed: int | None = 0,
    self_condition: bool = True,
    precision: str = "fp32",
    sampler: str = "ddim",
    sampling_steps: int | None = 50,
    ddim_eta: float = 0.0,
    property_guidance_scale: float = 1.0,
    on_step: Callable[[int, int], None] | None = None,
) -> Prediction:
    """Sample continuous canvases and decode them into sequences (retaining invalid outputs).

    The caller owns `model` and `diffusion` and places them on the requested device.
    Passing `seed=None` consumes the caller's active RNG stream so multi-batch loops
    can seed once around the entire loop with `inference_context`.
    """
    import torch

    from ..dataset.decode import decode_canvas_sample
    from ..diffusion import sample

    with torch.inference_mode():
        if device is None:
            device = next(model.parameters(), tasks["seed_x"]).device
        device = torch.device(device)

        # 1. Validate and move conditioning tensors to the target device.
        inputs = model_inputs(tasks, device)
        if not getattr(
            getattr(model, "property_config", None), "mask_enabled", False
        ) and not bool(inputs["properties"].mask.eq(1).all()):
            raise ValueError(
                "Masked properties require a model trained with property masks"
            )

        # 2. Run reverse diffusion under the requested RNG seed and precision.
        with (
            inference_context(model, device, seed=seed),
            autocast_context(device, precision),
        ):
            vectors = sample(
                diffusion,
                model,
                **inputs,
                shape=tuple(inputs["seed"].x.shape),
                self_condition=self_condition,
                property_guidance_scale=property_guidance_scale,
                noise_group_ids=tasks.get("noise_group_ids"),
                sampler=sampler,
                sampling_steps=sampling_steps,
                ddim_eta=ddim_eta,
                on_step=on_step,
            )

        # 3. Verify finite outputs and decode each canvas on CPU.
        if not bool(torch.isfinite(vectors).all()):
            raise FloatingPointError("Diffusion produced non-finite predictions")
        vectors = vectors.detach().float().cpu()
        decoded = [
            decode_canvas_sample(row, normalized_embeddings=normalized_embeddings)
            for row in vectors
        ]
        return Prediction(vectors, decoded)


@contextmanager
def inference_context(model, device, *, seed=None):
    """Switch `model` to `.eval()`, reset sampling caches, and optionally fork RNG state."""
    import torch

    device = torch.device(device)
    devices = (
        list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    )
    was_training = model.training
    reset = getattr(model, "reset_sampling_state", None)
    try:
        with (
            torch.random.fork_rng(devices=devices)
            if seed is not None
            else nullcontext()
        ):
            if seed is not None:
                torch.manual_seed(seed)
            model.eval()
            if reset is not None:
                reset()
            yield
    finally:
        if reset is not None:
            reset()
        model.train(was_training)


def model_inputs(tasks: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    """Pack task tensors into `StructureInput`, `PropertyInput`, and `SeedInput` on `device`."""
    import torch

    from ..model import PropertyInput, SeedInput, StructureInput

    def required(name):
        value = tasks.get(name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Prediction requires tensor {name!r}")
        return value.to(device)

    def optional(name):
        value = tasks.get(name)
        return None if value is None else value.to(device)

    values = required("c")
    mask = optional("condition_mask")
    if mask is None:
        mask = torch.ones_like(values)
    if mask.shape != values.shape or not torch.isfinite(values).all():
        raise ValueError(
            "Property values and masks must be finite, matching [batch, 3] tensors"
        )
    if not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError("Property masks must contain only zero or one")
    return {
        "structure": StructureInput(
            required("seed_3di"),
            hidden_mask=optional("threedi_conditioning_hidden_mask"),
        ),
        "properties": PropertyInput(values, mask),
        "seed": SeedInput(required("seed_x"), required("edit_mask")),
    }


def autocast_context(device, precision: str):
    """Return a `torch.autocast` context for `bf16` or `fp16`, or `nullcontext` for `fp32`."""
    import torch

    if precision not in {"fp32", "bf16", "fp16"}:
        raise ValueError("Inference precision must be fp32, bf16, or fp16")
    if precision == "fp32":
        return nullcontext()
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("FP16 inference requires CUDA")
    return torch.autocast(
        device.type, dtype=torch.bfloat16 if precision == "bf16" else torch.float16
    )


__all__ = [
    "Prediction",
    "autocast_context",
    "inference_context",
    "model_inputs",
    "predict",
]
