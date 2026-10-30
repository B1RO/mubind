"""Pre-flight validation, dataset loading, model/optimizer initialization, and validation helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from ...dataset import ProstT5Dataset
from ...diffusion import DiffusionSchedule
from ...evaluation import evaluate_loss
from ...evaluation.evaluator import (
    Evaluator,
    evaluation_directory,
    structural_checkpoint,
)
from ...model import ProteinDiffusionUNet
from ..config import TrainingConfig, config_from_dict
from ..step import batch_to_device, compute_loss
from ..tasks import TaskCollator
from .checkpoint import (
    checkpoint_paths, load_checkpoint, load_optimizer_state, restore_rng_state,
)
from .compilation import compile_training_model
from .ema import ModelEMA
from .loader import TASK_RNG_SCHEME, training_batches_per_epoch


@dataclass
class PreparedTraining:
    """Validated config, device, datasets, evaluator, and optional checkpoint payload before model allocation."""

    config: TrainingConfig
    device: torch.device
    dataset: ProstT5Dataset
    validation: ProstT5Dataset | None
    checkpoint: dict | None
    data_signature: str
    evaluator: Evaluator | None


@dataclass
class TrainingState:
    """Live model, optimizer, scheduler, GradScaler, EMA, and step/epoch counters during training."""

    model: ProteinDiffusionUNet
    training_model: nn.Module
    diffusion: DiffusionSchedule
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    scaler: torch.amp.GradScaler
    ema: ModelEMA | None
    task_rng: torch.Generator
    step: int
    epoch: int
    batch_in_epoch: int
    best_loss: float | None
    best_3di: tuple | None
    evaluation_signature: str
    batches_per_epoch: int


def prepare_training(
    config: TrainingConfig, *, resume: str | Path | None = None, on_status=None
) -> PreparedTraining:
    """Validate config, device, output paths, and resume compatibility, and load `train`/`valid` datasets."""
    status = on_status or (lambda message: None)
    config = config_from_dict(config.to_dict())
    config.data.path = (
        str(Path(config.data.path).expanduser().resolve()) if config.data.path else ""
    )
    config.validate()
    config.out = str(Path(config.out).expanduser().resolve())
    device = resolve_device(config.device)
    if config.precision == "fp16" and device.type != "cuda":
        raise ValueError("fp16 training requires CUDA; use fp32 or bf16 on CPU")
    if (
        config.precision == "bf16"
        and device.type == "cuda"
        and not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("This CUDA device does not support bf16")

    output, best_output, metrics_path, config_path = checkpoint_paths(config.out)
    continuing_output = (
        resume is not None and output == Path(resume).expanduser().resolve()
    )
    if not continuing_output and any(
        path.exists()
        for path in (
            output,
            best_output,
            metrics_path,
            config_path,
            structural_checkpoint(output),
            evaluation_directory(output),
        )
    ):
        raise FileExistsError(
            "Training output already exists; choose a new --out or resume the run"
        )

    if resume is not None:
        status("Loading checkpoint…")
    checkpoint = load_checkpoint(resume) if resume is not None else None

    data_options = {
        "max_length": config.backbone.max_length,
        "normalize_embeddings": config.data.normalize_embeddings,
        "joint_3di": config.data.joint_3di,
        "property_definition": config.data.property_definition,
    }
    status("Loading data…")
    dataset = ProstT5Dataset(config.data.path, split="train", **data_options)
    if not len(dataset):
        raise ValueError("The training split is empty")
    batches_per_epoch = training_batches_per_epoch(len(dataset), config.data)
    if batches_per_epoch == 0:
        raise ValueError(
            "Training has fewer examples than data.batch_size with data.drop_last=true; "
            "reduce the batch size or set data.drop_last=false"
        )

    validation = (
        ProstT5Dataset(config.data.path, split="valid", **data_options)
        if (config.evaluation.regime == "loss" and config.val_every)
        or config.evaluation.regime not in {"off", "loss"}
        else None
    )
    if validation is not None and not len(validation):
        raise ValueError(
            "The validation split is empty; choose evaluation.regime=off to disable evaluation"
        )
    if config.properties.mean is None:
        mean, std = dataset.property_statistics()
        config.properties.mean, config.properties.std = mean.tolist(), std.tolist()

    manifest = Path(config.data.path) / "prostt5_diffusion.json"
    signature = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if checkpoint is not None:
        _validate_resume_checkpoint(checkpoint, config, signature, batches_per_epoch)

    evaluator = None
    if config.evaluation.regime not in {"off", "loss"}:
        status("Evaluation setup…")
        evaluator = Evaluator(config, dataset, validation)
        evaluator.check_ready()
    return PreparedTraining(
        config, device, dataset, validation, checkpoint, signature, evaluator
    )


def build_training_state(
    prepared: PreparedTraining, *, on_status=None
) -> TrainingState:
    """Instantiate the UNet, diffusion schedule, optimizer, scheduler, GradScaler, and EMA (and restore checkpoint state if resuming)."""
    status = on_status or (lambda message: None)
    config, device = prepared.config, prepared.device
    checkpoint = prepared.checkpoint

    status("Loading the model into memory…")
    torch.manual_seed(config.seed)
    model = ProteinDiffusionUNet(
        config.backbone, config.structure, config.properties
    ).to(device)
    diffusion = config.diffusion.build().to(device)
    optimizer_class = (
        torch.optim.RAdam if config.optimizer.name == "radam" else torch.optim.AdamW
    )
    optimizer = optimizer_class(
        model.parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=config.optimizer.scheduler_gamma
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.precision == "fp16")
    ema = (
        ModelEMA(model, decay=config.ema.decay, warmup=config.ema.warmup)
        if config.ema.enabled
        else None
    )
    task_rng = torch.Generator().manual_seed(config.seed + 1)
    step = epoch = batch_in_epoch = 0
    best_loss = best_3di = None
    evaluation_signature = _evaluation_signature(
        config, prepared.data_signature, prepared.evaluator
    )
    batches_per_epoch = training_batches_per_epoch(len(prepared.dataset), config.data)

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        load_optimizer_state(optimizer, model, checkpoint["optimizer"], checkpoint["model"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        if ema is not None:
            ema.load_checkpoint_state(checkpoint["ema"], checkpoint["ema_metadata"])
        step, epoch, batch_in_epoch = (
            int(checkpoint[key]) for key in ("step", "epoch", "batch_in_epoch")
        )
        if epoch < 0 or not 0 <= batch_in_epoch < batches_per_epoch:
            raise ValueError("Checkpoint contains an invalid data position")
        same_evaluation = (
            checkpoint.get("evaluation_signature", evaluation_signature)
            == evaluation_signature
        )
        # Old checkpoints predate generation evaluation and only store loss.
        if "evaluation_signature" not in checkpoint and prepared.evaluator is not None:
            same_evaluation = False
        if same_evaluation:
            best_loss = checkpoint.get("best_validation_loss")
            saved_selector = checkpoint.get("best_3di_score")
            best_3di = tuple(saved_selector) if saved_selector is not None else None
        task_rng.set_state(checkpoint["task_rng"])
        restore_rng_state(checkpoint["rng"])

    training_model = compile_training_model(model, config, device)
    return TrainingState(
        model=model,
        training_model=training_model,
        diffusion=diffusion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        ema=ema,
        task_rng=task_rng,
        step=step,
        epoch=epoch,
        batch_in_epoch=batch_in_epoch,
        best_loss=best_loss,
        best_3di=best_3di,
        evaluation_signature=evaluation_signature,
        batches_per_epoch=batches_per_epoch,
    )


def prepare_validation_loader(
    validation: ProstT5Dataset | None,
    evaluator: Evaluator | None,
    config: TrainingConfig,
):
    """Build the validation `TaskCollator` and `DataLoader` when `evaluation.regime == 'loss'`."""
    validation_collator = TaskCollator(
        config.tasks, property_dropout=config.property_dropout, training=False
    )
    validation_loader = (
        None
        if validation is None or evaluator is not None
        else DataLoader(
            validation,
            batch_size=config.data.batch_size,
            shuffle=False,
            collate_fn=list,
            num_workers=config.data.num_workers,
            generator=torch.Generator().manual_seed(config.validation_seed),
        )
    )
    return validation_collator, validation_loader


def run_validation_loss(
    state: TrainingState,
    validation_loader: DataLoader,
    validation_collator: TaskCollator,
    config: TrainingConfig,
    device: torch.device,
    best_output: Path,
    save_fn,
) -> float:
    """Evaluate validation MSE loss on `ema.module` (or `model`) and save `best_output` if improved."""
    validation_model = state.ema.module if state.ema is not None else state.model

    def loss_fn(validation_rows, generator, validation_model=validation_model):
        batch = batch_to_device(
            validation_collator.collate(validation_rows, generator=generator),
            device,
        )
        weight = (
            batch["target_length"].sum().item()
            if config.reconstruction.example_weighting == "target_length"
            else len(validation_rows)
        )
        return (
            compute_loss(
                validation_model, state.diffusion, batch, config, training=False
            ),
            weight,
        )

    validation_loader.generator.manual_seed(config.validation_seed)
    result = evaluate_loss(
        validation_model,
        validation_loader,
        loss_fn,
        seed=config.validation_seed,
        max_batches=config.val_batches,
    )
    if state.best_loss is None or result["loss"] < state.best_loss:
        state.best_loss = result["loss"]
        save_fn(best_output)
    return result["loss"]


def run_due_evaluation(
    state: TrainingState,
    evaluator: Evaluator | None,
    config: TrainingConfig,
    device: torch.device,
    output: Path,
    best_output: Path,
    save_fn,
    *,
    force: bool = False,
    on_evaluation=None,
) -> dict:
    """Run due `Evaluator` benchmarks and save `.best.pt` or `.best-3di.pt` when improved."""
    if evaluator is None:
        return {}
    try:
        report = evaluator.run_due(
            step=state.step,
            final_step=config.steps,
            model=state.ema.module if state.ema is not None else state.model,
            diffusion=state.diffusion,
            device=device,
            force=force,
            on_event=on_evaluation,
        )
    except (Exception, KeyboardInterrupt):
        # Evaluation happens between optimizer steps. Preserve that
        # consistent state if a service or evaluation is interrupted.
        save_fn(output)
        raise
    loss_value = report.metrics.get("validation_loss")
    if loss_value is not None and (
        state.best_loss is None or loss_value < state.best_loss
    ):
        state.best_loss = loss_value
        save_fn(best_output)
    if report.best_3di_score is not None and (
        state.best_3di is None or report.best_3di_score > state.best_3di
    ):
        state.best_3di = report.best_3di_score
        save_fn(structural_checkpoint(output))
    return report.metrics


def resolve_device(value: str) -> torch.device:
    """Resolve `'auto'`, `'cpu'`, or `'cuda[:index]'` into a validated `torch.device`."""
    try:
        device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if value == "auto"
            else torch.device(value)
        )
    except RuntimeError as exc:
        raise ValueError(f"Invalid training device: {value}") from exc
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Training currently supports CPU and CUDA devices")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if (
        device.type == "cuda"
        and device.index is not None
        and device.index >= torch.cuda.device_count()
    ):
        raise ValueError(f"CUDA device index {device.index} is unavailable")
    return device


def _validate_resume_checkpoint(
    checkpoint: dict,
    config: TrainingConfig,
    signature: str,
    batches_per_epoch: int,
) -> None:
    """Check that a saved checkpoint matches the current config, dataset manifest hash, and step range."""
    if checkpoint.get("task_rng_scheme", "legacy") not in {
        "legacy",
        TASK_RNG_SCHEME,
    }:
        raise ValueError("Checkpoint uses an unsupported task RNG scheme")
    saved_config = config_from_dict(checkpoint["config"])
    if _resume_settings(config) != _resume_settings(saved_config):
        raise ValueError(
            "Resume settings differ from the saved data/model/objective/optimizer settings"
        )
    if checkpoint["data_signature"] != signature:
        raise ValueError(
            "The prepared dataset manifest changed since the checkpoint"
        )
    if int(checkpoint["step"]) > config.steps:
        raise ValueError(
            "Requested steps is less than the checkpoint's completed step count"
        )
    if (
        int(checkpoint["epoch"]) < 0
        or not 0 <= int(checkpoint["batch_in_epoch"]) < batches_per_epoch
    ):
        raise ValueError("Checkpoint contains an invalid data position")


def _resume_settings(config: TrainingConfig) -> dict:
    """Strip runtime-only fields (`steps`, `out`, `device`, logging intervals, `num_workers`) before comparing configs."""
    settings = config.to_dict()
    for name in (
        "steps",
        "out",
        "device",
        "log_every",
        "save_every",
        "val_every",
        "evaluation",
        "compile",
        "compile_mode",
    ):
        settings.pop(name)
    settings["data"].pop("num_workers")
    return settings


def _evaluation_signature(
    config: TrainingConfig, data_signature: str, evaluator: Evaluator | None
) -> str:
    """Return a SHA-256 hash identifying the validation/evaluation configuration."""
    if evaluator is not None:
        return evaluator.signature
    return hashlib.sha256(
        json.dumps(
            {
                "regime": config.evaluation.regime,
                "batches": config.val_batches,
                "seed": config.validation_seed,
                "data": data_signature,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
