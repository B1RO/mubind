"""Load modern MUbind and legacy full-canvas TUM checkpoints for inference."""

from __future__ import annotations

import pickle
import zipfile
from dataclasses import dataclass
from pathlib import Path

import torch

from ...diffusion import DiffusionConfig
from ...model import (
    BackboneConfig,
    PropertyConfig,
    ProteinDiffusionUNet,
    ThreeDiConfig,
    ThreeDiContextConfig,
    ThreeDiCrossAttentionConfig,
)
from ...training.config import TrainingConfig, config_from_dict
from ...training.helpers.checkpoint import FORMAT_VERSION


@dataclass
class InferenceCheckpoint:
    """Loaded checkpoint config, state dict, completed step count, and weight source (`ema` or `model`)."""

    config: TrainingConfig
    state: dict
    step: int
    weights: str

    def build_model(self) -> ProteinDiffusionUNet:
        """Instantiate `ProteinDiffusionUNet` without advancing caller RNG and load `self.state`."""
        with torch.random.fork_rng(devices=[]):
            model = ProteinDiffusionUNet(
                self.config.backbone, self.config.structure, self.config.properties
            )
        model.load_state_dict(self.state, strict=True)
        return model.eval().requires_grad_(False)


def load_inference_checkpoint(
    path: Path, *, metadata_only: bool = False
) -> InferenceCheckpoint:
    """Read an inference checkpoint, optionally without materializing tensor storage.

    Menu inspection (`metadata_only=True`) uses `mmap` and `"meta"` tensors for
    `torch.save` ZIP archives, falling back to `FakeTensorMode` for older non-ZIP files.
    Full inference loads weights onto CPU for strict model loading.
    """
    path = path.expanduser()
    try:
        if metadata_only and zipfile.is_zipfile(path):
            payload = torch.load(path, map_location="meta", weights_only=True, mmap=True)
        elif metadata_only:
            from torch._subclasses.fake_tensor import FakeTensorMode

            with FakeTensorMode():
                payload = torch.load(path, map_location="cpu", weights_only=True)
        else:
            payload = torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, EOFError) as exc:
        raise ValueError(
            f"Cannot read supported inference weights from {path}"
        ) from exc

    if not isinstance(payload, dict):
        raise TypeError("Expected a mubind or full-canvas TUM checkpoint")
    if payload.get("format_version") == FORMAT_VERSION:
        config = config_from_dict(payload["config"])
        key = "ema" if payload.get("ema") is not None else "model"
        state = payload[key]
    elif "args" in payload and "model" in payload:
        key = "ema_model" if payload.get("ema_model") is not None else "model"
        state = payload[key]
        config = _legacy_config(payload, state)
    else:
        raise ValueError("Unsupported checkpoint format")

    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint contains no model weights")
    return InferenceCheckpoint(
        config,
        state,
        int(payload.get("step", 0)),
        "ema" if key.startswith("ema") else "model",
    )


def validate_design_checkpoint(config: TrainingConfig) -> None:
    """Check that `config` meets the requirements of the minibinder design pipeline."""
    from ...common.properties import (
        OMEGAMP_PROPERTY_DEFINITION,
        canonical_property_definition,
    )

    if config.backbone.max_length != 70 or config.backbone.x_channels != 8:
        raise ValueError("Binder design needs an 8-channel checkpoint with a 70-position canvas")
    if config.structure.representation != "foldseek_2d" or config.diffusion.objective != "pred_x0":
        raise ValueError("Binder design needs Foldseek 3Di conditioning and a pred_x0 checkpoint")
    if canonical_property_definition(config.data.property_definition) != OMEGAMP_PROPERTY_DEFINITION:
        raise ValueError("Binder design requires the omegamp_modlamp_v1 property definition")


def _legacy_config(payload: dict, state: dict) -> TrainingConfig:
    """Translate flat legacy checkpoint arguments into a `TrainingConfig`."""
    args = payload["args"]
    required = {
        "max_length",
        "x_channels",
        "hidden_channels",
        "channel_mults",
        "cond_dim",
        "blocks_per_level",
        "threed_dim",
        "threed_representation",
        "attention_mode",
        "normalize_embeddings",
        "property_definition",
        "objective",
        "self_condition",
    }
    if not isinstance(args, dict) or required - args.keys():
        raise ValueError("Legacy checkpoint is missing its full-canvas model settings")
    for key, expected in (
        ("seed_conditioning", "separate_edit_mask"),
        ("seed_preservation", "learned"),
        ("reconstruction_scope", "full_canvas"),
    ):
        if args.get(key) != expected:
            raise ValueError(f"Unsupported legacy checkpoint: {key} must be {expected}")

    config = TrainingConfig()
    mults = args["channel_mults"]
    if isinstance(mults, str):
        mults = [int(value) for value in mults.split(",")]
    config.backbone = BackboneConfig(
        max_length=int(args["max_length"]),
        x_channels=int(args["x_channels"]),
        hidden_channels=int(args["hidden_channels"]),
        channel_mults=tuple(mults),
        cond_dim=int(args["cond_dim"]),
        blocks_per_level=int(args["blocks_per_level"]),
        attention_mode=args["attention_mode"],
        attention_heads=int(args.get("attention_heads", 4)),
    )
    context = None
    if args.get("contextual_threed_conditioning", False):
        context = ThreeDiContextConfig(
            layers=int(args["threed_context_layers"]),
            heads=int(args["threed_context_heads"]),
            feedforward_dim=int(args["threed_context_feedforward_dim"]),
            dropout=float(args.get("threed_context_dropout", 0)),
        )
    cross = None
    if args.get("cross_attention_threed_conditioning", False):
        cross = ThreeDiCrossAttentionConfig(
            heads=int(args["threed_cross_attention_heads"]),
            dim_head=int(args["threed_cross_attention_dim_head"]),
            bottleneck=bool(args.get("threed_cross_attention_bottleneck", False)),
        )
    config.structure = ThreeDiConfig(
        dim=int(args["threed_dim"]),
        representation=args["threed_representation"],
        multiscale=bool(args.get("multiscale_threed_conditioning", False)),
        infer_missing=bool(args.get("threed_infer_missing", False)),
        context=context,
        cross_attention=cross,
    )
    config.properties = PropertyConfig(
        mask_enabled=bool(args.get("condition_mask_enabled", False)),
        mean=args.get("property_mean", state.get("property_mean")),
        std=args.get("property_std", state.get("property_std")),
    )
    config.diffusion = DiffusionConfig(
        timesteps=int(payload["timesteps"]),
        objective=args["objective"],
        loss_weighting=args.get("diffusion_loss_weighting", "uniform"),
    )
    config.data.normalize_embeddings = bool(args["normalize_embeddings"])
    config.data.joint_3di = config.backbone.x_channels == 8
    config.data.property_definition = args["property_definition"]
    config.self_condition.enabled = bool(args["self_condition"])
    config.precision = args.get("precision", "fp32")
    return config
