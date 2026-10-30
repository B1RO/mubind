"""Atomic saving and loading of resumable training checkpoints (`.pt` files).

Every saved checkpoint bundles the UNet weights, smoothed EMA weights, optimizer and
LR scheduler state, FP16 `GradScaler` state, epoch/batch counters, and CPU/CUDA RNG
states into a single dictionary, written atomically via a temporary file so an
interrupted write never corrupts an existing checkpoint.
"""

from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path

import torch

from .loader import TASK_RNG_SCHEME

FORMAT_VERSION = 1


def checkpoint_paths(output: str | Path) -> tuple[Path, Path, Path, Path]:
    """Return the four output paths derived from `--out` (`latest.pt`, `.best.pt`, `.metrics.jsonl`, `.config.json`)."""
    output = Path(output).expanduser().resolve()
    return (
        output,
        output.with_name(f"{output.stem}.best{output.suffix}"),
        output.with_suffix(output.suffix + ".metrics.jsonl"),
        output.with_suffix(output.suffix + ".config.json"),
    )


def save_training_state(
    path: str | Path,
    state,
    config,
    data_signature: str,
    on_checkpoint=None,
) -> None:
    """Pack the current `TrainingState` into a checkpoint dictionary and save it atomically."""
    save_checkpoint(
        path,
        {
            "config": config.to_dict(),
            "model": state.model.state_dict(),
            "optimizer": state.optimizer.state_dict(),
            "scheduler": state.scheduler.state_dict(),
            "scaler": state.scaler.state_dict(),
            "ema": None if state.ema is None else state.ema.module.state_dict(),
            "ema_metadata": None if state.ema is None else state.ema.metadata(),
            "step": state.step,
            "epoch": state.epoch,
            "batch_in_epoch": state.batch_in_epoch,
            "best_validation_loss": state.best_loss,
            "best_3di_score": state.best_3di,
            "evaluation_signature": state.evaluation_signature,
            "rng": capture_rng_state(),
            "task_rng": state.task_rng.get_state(),
            "task_rng_scheme": TASK_RNG_SCHEME,
            "data_signature": data_signature,
        },
    )
    if on_checkpoint is not None:
        on_checkpoint(Path(path), state.step)


def save_checkpoint(path: str | Path, payload: dict):
    """Write `payload` to a temporary sibling file, `fsync` to disk, and atomically replace `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save({**payload, "format_version": FORMAT_VERSION}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_checkpoint(path: str | Path) -> dict:
    """Load and validate a saved training checkpoint dictionary on CPU."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, EOFError) as exc:
        raise ValueError(f"Cannot read training checkpoint at {path} ({exc})") from exc
    if not isinstance(payload, dict) or payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            "Expected a mubind training checkpoint (legacy TUM checkpoints need conversion)"
        )
    required = {
        "config",
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "step",
        "epoch",
        "batch_in_epoch",
        "rng",
        "task_rng",
        "data_signature",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"Incomplete checkpoint; missing {', '.join(sorted(missing))}")
    # Earlier mubind runs kept the short final batch. Preserve their epoch
    # positions when loading them after the new-run default changes.
    if isinstance(payload["config"], dict):
        data = payload["config"].get("data")
        if isinstance(data, dict):
            data.setdefault("drop_last", False)
    return payload


def load_optimizer_state(optimizer, model, optimizer_state, model_state):
    """Restore optimizer moments when context weights move inside the 3Di encoder."""
    if any(name.startswith("threed_context.") for name in model_state):
        current_names = list(dict(model.named_parameters()))
        current_name_set = set(current_names)
        saved_names = []
        for name in model_state:
            if name.startswith("threed_context."):
                name = "threed_conditioning.context." + name[len("threed_context."):]
            if name in current_name_set:
                saved_names.append(name)
        groups = optimizer_state["param_groups"]
        if len(groups) != 1 or len(saved_names) != len(groups[0]["params"]):
            raise ValueError("Cannot match saved optimizer parameters to the model")
        parameter_ids = dict(zip(saved_names, groups[0]["params"]))
        group = {**groups[0], "params": [parameter_ids[name] for name in current_names]}
        if "param_names" in group:
            group["param_names"] = current_names
        optimizer_state = {**optimizer_state, "param_groups": [group]}
    optimizer.load_state_dict(optimizer_state)


def capture_rng_state():
    """Snapshot CPU (and CUDA, if available) random number generator states."""
    state = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    """Restore CPU and CUDA random number generator states from a checkpoint snapshot."""
    torch.set_rng_state(state["cpu"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError(
                "Resume requires the same CUDA device count to restore RNG state"
            )
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])
