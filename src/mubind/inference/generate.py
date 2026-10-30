"""Direct sequence sampling from a saved MUbind checkpoint (`generate`, backing `mubind sample`).

Samples sequence candidates directly with DDIM without requiring a target PDB,
Complexa scaffolds, or ESMFold2 folding

1. Validate `GenerationConfig`, resolve device and precision, and load the saved
   training checkpoint (`prepare_generation`).
2. Instantiate `ProteinDiffusionUNet` (preferring EMA weights) and `DiffusionSchedule`.
3. Loop over batches of `generation_tasks`, calling `predict()` under a single
   `inference_context` RNG stream, and write decoded candidates to `candidates.jsonl`,
   valid sequences to `sequences.fasta`, and run metadata to `generation.json`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ..model import ProteinDiffusionUNet
from ..training.config import TrainingConfig, config_from_dict
from ..training.helpers.checkpoint import load_checkpoint
from .config import GenerationConfig, next_output
from .predict import inference_context, predict


@dataclass(frozen=True)
class GenerationResult:
    """Summary returned when `generate` completes (`directory`, `count`, `valid_count`)."""

    directory: Path
    count: int
    valid_count: int


@dataclass
class PreparedGeneration:
    """Validated settings, training config, checkpoint payload, device, precision, and output folder."""

    settings: GenerationConfig
    training: TrainingConfig
    checkpoint: dict
    device: torch.device
    precision: str
    weights: str
    out: Path


def generate(
    settings: GenerationConfig,
    *,
    on_status: Callable[[str], None] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> GenerationResult:
    """Sample sequences from a saved checkpoint and write `sequences.fasta` and `candidates.jsonl`."""
    status = on_status or (lambda message: None)

    # 1. Validate settings and load the checkpoint payload.
    status("Loading checkpoint…")
    prepared = prepare_generation(settings)
    training, device = prepared.training, prepared.device

    # 2. Build the U-Net and diffusion schedule on the target device.
    status(f"Loading {prepared.weights.upper()} weights on {device}…")
    with torch.random.fork_rng(devices=[]):
        model = ProteinDiffusionUNet(
            training.backbone, training.structure, training.properties
        )
    model.load_state_dict(prepared.checkpoint[prepared.weights])
    checkpoint_step = int(prepared.checkpoint["step"])
    prepared.checkpoint.clear()
    model = model.to(device).eval().requires_grad_(False)
    diffusion = training.diffusion.build().to(device)

    prepared.out.mkdir(parents=True, exist_ok=False)
    sampler = settings.sampler.lower()
    steps_per_batch = (
        training.diffusion.timesteps
        if sampler == "ddpm" or settings.sampling_steps is None
        else settings.sampling_steps
    )
    metadata = {
        "settings": {
            **asdict(settings),
            "checkpoint": str(settings.checkpoint.expanduser().resolve()),
            "out": str(prepared.out),
        },
        "checkpoint_step": checkpoint_step,
        "weights": prepared.weights,
        "device": str(device),
        "precision": prepared.precision,
        "sampler": sampler,
        "ddim_eta": settings.ddim_eta if sampler == "ddim" else 0.0,
        "checkpoint_config": training.to_dict(),
        "completed_count": 0,
        "valid_count": 0,
        "status": "running",
    }
    report_path = prepared.out / "generation.json"
    report_path.write_text(json.dumps(metadata, indent=2) + "\n")

    # 3. Sample batches under a single seeded RNG stream and stream results to disk.
    completed = valid = 0
    batch_count = math.ceil(settings.count / settings.batch_size)
    total_steps = batch_count * steps_per_batch
    try:
        with (
            (prepared.out / "sequences.fasta").open("x") as fasta,
            (prepared.out / "candidates.jsonl").open("x") as records,
            inference_context(model, device, seed=settings.seed),
        ):
            for batch_index, start in enumerate(
                range(0, settings.count, settings.batch_size)
            ):
                count = min(settings.batch_size, settings.count - start)
                status(f"Sampling batch {batch_index + 1}/{batch_count}…")

                def progress(step, total, batch_index=batch_index):
                    if on_progress is not None:
                        on_progress(
                            batch_index * steps_per_batch + step, total_steps
                        )

                prediction = predict(
                    model,
                    diffusion,
                    generation_tasks(settings, training, count),
                    normalized_embeddings=training.data.normalize_embeddings,
                    device=device,
                    seed=None,
                    precision=prepared.precision,
                    self_condition=training.self_condition.enabled,
                    sampler=sampler,
                    sampling_steps=None if sampler == "ddpm" else steps_per_batch,
                    ddim_eta=settings.ddim_eta if sampler == "ddim" else 0.0,
                    on_step=progress,
                )
                for index, decoded in enumerate(prediction.decoded, start=start + 1):
                    name = f"candidate_{index:06d}"
                    record = {
                        "id": name,
                        **decoded,
                        "requested_length": settings.length,
                        "length_matches_request": (
                            decoded["generated_length"] == settings.length
                        ),
                    }
                    records.write(json.dumps(record) + "\n")
                    if decoded["sequence_valid"]:
                        fasta.write(
                            f">{name} length={decoded['generated_length']}\n{decoded['generated']}\n"
                        )
                        valid += 1
                    completed += 1
                fasta.flush()
                records.flush()
                metadata.update(completed_count=completed, valid_count=valid)
                report_path.write_text(json.dumps(metadata, indent=2) + "\n")
        metadata["status"] = "complete"
    except KeyboardInterrupt:
        metadata["status"] = "interrupted"
        raise
    except Exception:
        metadata["status"] = "failed"
        raise
    finally:
        metadata.update(completed_count=completed, valid_count=valid)
        report_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return GenerationResult(prepared.out, completed, valid)


def prepare_generation(settings: GenerationConfig) -> PreparedGeneration:
    """Validate `settings` and load the saved checkpoint recipe before allocating model weights."""
    if settings.checkpoint is None:
        raise ValueError("Choose a mubind training checkpoint first.")
    for name in ("count", "length", "batch_size"):
        if type(getattr(settings, name)) is not int or getattr(settings, name) < 1:
            raise ValueError(f"{name} must be a positive integer")
    sampler = str(settings.sampler).lower()
    if sampler not in {"ddim", "ddpm"}:
        raise ValueError("sampler must be ddpm or ddim")
    if settings.sampling_steps is not None and (
        type(settings.sampling_steps) is not int or settings.sampling_steps < 1
    ):
        raise ValueError("sampling_steps must be a positive integer")
    if not math.isfinite(settings.ddim_eta) or not 0 <= settings.ddim_eta <= 1:
        raise ValueError("DDIM eta must be between 0 and 1")
    if sampler == "ddpm" and settings.ddim_eta != 0.0:
        raise ValueError("ddim_eta only applies when sampler='ddim'")
    if type(settings.seed) is not int or not 0 <= settings.seed < 2**63:
        raise ValueError("seed must be an integer between 0 and 2**63 - 1")
    for name in ("charge", "hydrophobicity"):
        value = getattr(settings, name)
        if value is not None and not math.isfinite(value):
            raise ValueError(f"{name} must be finite or omitted")

    out = (settings.out or next_output()).expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"Output already exists: {out}. Choose a new folder.")

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if settings.device == "auto"
        else torch.device(settings.device)
    )
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Generation supports CPU and CUDA devices")
    if device.type == "cuda" and (
        not torch.cuda.is_available()
        or (device.index is not None and device.index >= torch.cuda.device_count())
    ):
        raise ValueError(f"Requested CUDA device is unavailable: {device}")

    checkpoint = load_checkpoint(settings.checkpoint.expanduser())
    training = config_from_dict(checkpoint["config"])
    if settings.length > training.backbone.max_length:
        raise ValueError(
            f"Requested length exceeds this checkpoint's {training.backbone.max_length}-residue canvas."
        )
    if (
        sampler == "ddpm"
        and settings.sampling_steps not in {None, training.diffusion.timesteps}
    ):
        raise ValueError(
            "DDPM uses every diffusion timestep; choose sampler='ddim' to reduce sampling_steps"
        )
    if (
        settings.sampling_steps is not None
        and settings.sampling_steps > training.diffusion.timesteps
    ):
        raise ValueError(
            f"Sampling steps cannot exceed {training.diffusion.timesteps} for this checkpoint."
        )
    if not training.properties.mask_enabled and (
        settings.charge is None or settings.hydrophobicity is None
    ):
        raise ValueError("This checkpoint requires both charge and hydrophobicity.")

    precision = settings.precision
    if precision == "auto":
        precision = training.precision if device.type == "cuda" else "fp32"
        if precision == "bf16":
            with torch.cuda.device(device):
                if not torch.cuda.is_bf16_supported():
                    precision = "fp32"
    if precision not in {"fp32", "bf16", "fp16"}:
        raise ValueError("Precision must be auto, fp32, bf16, or fp16")
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("FP16 generation requires CUDA")
    if precision == "bf16" and device.type == "cuda":
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise ValueError("This CUDA device does not support BF16")

    return PreparedGeneration(
        settings,
        training,
        checkpoint,
        device,
        precision,
        "ema" if checkpoint.get("ema") is not None else "model",
        out,
    )


def generation_tasks(
    settings: GenerationConfig, training: TrainingConfig, count: int
) -> dict[str, torch.Tensor]:
    """Build full-canvas unconditional-3Di tasks with length and explicitly supplied properties."""
    width = training.backbone.max_length
    return {
        "c": torch.tensor(
            [[settings.length, settings.charge or 0.0, settings.hydrophobicity or 0.0]],
            dtype=torch.float32,
        )
        .expand(count, -1)
        .clone(),
        "condition_mask": torch.tensor(
            [[1, settings.charge is not None, settings.hydrophobicity is not None]],
            dtype=torch.float32,
        )
        .expand(count, -1)
        .clone(),
        "seed_x": torch.zeros(count, training.backbone.x_channels, width),
        "edit_mask": torch.ones(count, width, dtype=torch.bool),
        "seed_3di": torch.zeros(count, width, dtype=torch.long),
    }


__all__ = [
    "GenerationConfig",
    "GenerationResult",
    "PreparedGeneration",
    "generate",
    "generation_tasks",
    "next_output",
    "prepare_generation",
]
