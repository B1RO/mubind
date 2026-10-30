"""Outer diffusion training loop (`train`).

While `step.py` computes the diffusion loss on a single batch and `helpers/` handles
dataset loading, EMA, and checkpoint files, this module runs the outer loop from step `0`
(or a resumed checkpoint step) up to `config.steps`

1. Validate the config, load `ProstT5Dataset` splits, and initialize (or restore) the
   UNet, optimizer, LR scheduler, `GradScaler`, and `ModelEMA` (`prepare_training` and
   `build_training_state` in `helpers/setup.py`).
2. Loop over shuffled epochs (`EpochSampler`) and prepared batches (`make_training_loader`).
3. For each batch, run `_train_step` (`compute_loss` -> `backward()` -> gradient clipping ->
   `optimizer.step()` -> `scheduler.step()` -> `ema.update()`).
4. Periodically evaluate validation loss and structural generation metrics on the smoothed
   `ema.module` weights, append JSONL metrics, and save `.pt` checkpoints.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .config import TrainingConfig
from .helpers.checkpoint import checkpoint_paths, save_training_state
from .helpers.loader import EpochSampler, make_training_loader, next_task_seed
from .helpers.setup import (
    PreparedTraining,
    TrainingState,
    build_training_state,
    prepare_training,
    prepare_validation_loader,
    resolve_device,
    run_due_evaluation,
    run_validation_loss,
)
from .step import batch_to_device, compute_loss


def train(
    config: TrainingConfig,
    *,
    resume: str | Path | None = None,
    log=None,
    on_checkpoint=None,
    on_evaluation=None,
    on_status=None,
    on_step=None,
) -> TrainingResult:
    """Train the diffusion UNet to `config.steps` (or resume a saved checkpoint to that step).

    Progress is logged and appended to `<out>.metrics.jsonl`. Optional callbacks receive
    metric dictionaries (`log`), checkpoint saves (`on_checkpoint`), evaluation progress
    (`on_evaluation`), startup status strings (`on_status`), and per-step updates (`on_step`).
    """
    status = on_status or (lambda message: None)
    log = _default_log if log is None else log

    # 1. Validate config, load datasets, and build (or restore) model, optimizer, and EMA state.
    prepared = prepare_training(config, resume=resume, on_status=on_status)
    config, device = prepared.config, prepared.device
    dataset, validation = prepared.dataset, prepared.validation
    signature, evaluator = prepared.data_signature, prepared.evaluator
    if evaluator is not None:
        evaluator.start_folding(on_status=on_status)
    state = build_training_state(prepared, on_status=on_status)
    # Drop the resume checkpoint dictionary so we do not keep a second copy in RAM.
    del prepared

    output, best_output, metrics_path, config_path = checkpoint_paths(config.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    validation_collator, validation_loader = prepare_validation_loader(
        validation, evaluator, config
    )

    def save(path: Path) -> None:
        save_training_state(path, state, config, signature, on_checkpoint)

    # 2. Run the outer training loop across epochs until reaching config.steps.
    state.model.train()
    with metrics_path.open("a") as metrics:
        if evaluator is not None and (
            (resume is not None and config.evaluation.run_on_resume)
            or (resume is None and config.evaluation.run_on_start)
        ):
            initial = {
                "step": state.step,
                "lr": state.optimizer.param_groups[0]["lr"],
                **run_due_evaluation(
                    state,
                    evaluator,
                    config,
                    device,
                    output,
                    best_output,
                    save,
                    force=True,
                    on_evaluation=on_evaluation,
                ),
            }
            _write_and_log(metrics, log, initial)

        first_batch = True
        while state.step < config.steps:
            sampler = EpochSampler(
                len(dataset),
                config.seed + state.epoch,
                state.batch_in_epoch * config.data.batch_size,
            )
            loader = make_training_loader(
                dataset,
                config,
                sampler,
                state.task_rng,
                device=device,
                epoch=state.epoch,
                start_step=state.step,
            )
            for prepared_batch in loader:
                if first_batch:
                    # torch.compile is lazy and compiles on the first forward + backward pass.
                    status(
                        "Compiling the model…"
                        if state.training_model is not state.model
                        else "Training…"
                    )

                # 3. Run one forward/backward/optimizer/EMA update on the batch.
                batch = batch_to_device(prepared_batch, device)
                loss_value = _train_step(state, batch, config)

                event = {
                    "step": state.step,
                    "loss": loss_value,
                    "lr": state.optimizer.param_groups[0]["lr"],
                }
                if first_batch:
                    first_batch = False
                    if state.training_model is not state.model:
                        status("Training…")
                if on_step is not None:
                    on_step(event.copy())

                # 4. Periodically run validation, log metrics, and save checkpoints.
                if validation_loader is not None and (
                    state.step % config.val_every == 0 or state.step == config.steps
                ):
                    event["validation_loss"] = run_validation_loss(
                        state,
                        validation_loader,
                        validation_collator,
                        config,
                        device,
                        best_output,
                        save,
                    )
                event.update(
                    run_due_evaluation(
                        state,
                        evaluator,
                        config,
                        device,
                        output,
                        best_output,
                        save,
                        on_evaluation=on_evaluation,
                    )
                )
                if _should_log_step(state.step, config, event):
                    _write_and_log(metrics, log, event)
                if config.save_every and state.step % config.save_every == 0:
                    save(output)
                if state.step >= config.steps:
                    break

        save(output)
    return TrainingResult(state.step, output, state.best_loss)


def _train_step(state: TrainingState, batch: dict, config: TrainingConfig) -> float:
    """Run one forward/backward pass, step the optimizer, update EMA, and advance step counters."""
    state.optimizer.zero_grad(set_to_none=True)
    if state.training_model is not state.model:
        torch.compiler.cudagraph_mark_step_begin()

    # 1. Forward pass and diffusion MSE loss (in step.py).
    loss = compute_loss(
        state.training_model, state.diffusion, batch, config, training=True
    )
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError(f"Non-finite training loss at step {state.step + 1}")

    # 2. Backward pass and optional gradient norm clipping.
    state.scaler.scale(loss).backward()
    if config.optimizer.grad_clip is not None:
        state.scaler.unscale_(state.optimizer)
        torch.nn.utils.clip_grad_norm_(
            state.model.parameters(), config.optimizer.grad_clip
        )

    # 3. Optimizer step, LR schedule step, and EMA weight smoothing.
    previous_scale = state.scaler.get_scale()
    state.scaler.step(state.optimizer)
    state.scaler.update()
    if state.scaler.get_scale() >= previous_scale:
        state.scheduler.step()
        if state.ema is not None:
            state.ema.update(state.model)

    # 4. Advance task RNG and step/epoch counters only for consumed batches (never prefetched ones).
    next_task_seed(state.task_rng)
    state.step += 1
    state.batch_in_epoch += 1
    if state.batch_in_epoch == state.batches_per_epoch:
        state.epoch += 1
        state.batch_in_epoch = 0
    return float(loss.detach())


@dataclass(frozen=True)
class TrainingResult:
    """Summary returned when `train` finishes (`step`, final `checkpoint` path, `best_validation_loss`)."""

    step: int
    checkpoint: Path
    best_validation_loss: float | None


def _should_log_step(step: int, config: TrainingConfig, event: dict) -> bool:
    """Return True when `step` hits `log_every`, finishes training, or includes validation/evaluation metrics."""
    return (
        step % config.log_every == 0
        or step == config.steps
        or "validation_loss" in event
        or any(key.startswith("evaluation/") for key in event)
    )


def _write_and_log(metrics_file, log_fn, event: dict) -> None:
    """Append `event` as a JSON line to `.metrics.jsonl` and pass it to `log_fn`."""
    metrics_file.write(json.dumps(event, allow_nan=False) + "\n")
    metrics_file.flush()
    log_fn(event)


def _default_log(event: dict) -> None:
    """Print a single-line step summary (`step`, `loss`, optional `val`, and `lr`) to stdout."""
    parts = [f"step {event['step']}"]
    if "loss" in event:
        parts.append(f"loss {event['loss']:.6f}")
    if "validation_loss" in event:
        parts.append(f"val {event['validation_loss']:.6f}")
    parts.append(f"lr {event['lr']:.3g}")
    print(" | ".join(parts), flush=True)


__all__ = [
    "PreparedTraining",
    "TrainingResult",
    "TrainingState",
    "checkpoint_paths",
    "prepare_training",
    "resolve_device",
    "train",
]
