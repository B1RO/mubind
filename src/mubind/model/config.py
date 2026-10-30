"""Construction options for ProteinDiffusionUNet, grouped like configs/train.yaml."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .conditioning.threedi_2d_representation import validate_threedi_representation


@dataclass
class BackboneConfig:
    max_length: int
    x_channels: int = 6
    hidden_channels: int = 64
    channel_mults: tuple[int, ...] = (1, 2, 4)
    cond_dim: int = 128
    blocks_per_level: int = 2
    # none, bottleneck (bottleneck only), unet_linear, or unet_full.
    attention_mode: str = "none"
    attention_heads: int = 4

    def __post_init__(self) -> None:
        if self.blocks_per_level < 2:
            raise ValueError("blocks_per_level must be at least 2")
        if self.attention_mode not in {"none", "bottleneck", "unet_linear", "unet_full"}:
            raise ValueError(
                "attention_mode must be one of: none, bottleneck, "
                "unet_linear, unet_full"
            )


@dataclass
class ThreeDiContextConfig:
    """Transformer encoder that contextualizes 3Di embeddings before conditioning."""

    layers: int = 4
    heads: int = 4
    # None means four times the 3Di embedding width.
    feedforward_dim: int | None = None
    dropout: float = 0.0


@dataclass
class ThreeDiCrossAttentionConfig:
    """Cross-attention from sequence features to the 3Di embedding after the final block."""

    heads: int = 4
    dim_head: int = 32
    # Also attend from the coarse bottleneck features.
    bottleneck: bool = False


@dataclass
class ThreeDiConfig:
    vocab_size: int = 21
    dim: int = 32
    # learned or foldseek_2d.
    representation: str = "learned"
    multiscale: bool = False
    infer_missing: bool = True
    context: ThreeDiContextConfig | None = None
    cross_attention: ThreeDiCrossAttentionConfig | None = None

    def __post_init__(self) -> None:
        validate_threedi_representation(self.representation)
        if self.representation == "foldseek_2d" and self.vocab_size != 21:
            raise ValueError("foldseek_2d requires 20 3Di tokens plus PAD")


@dataclass
class PropertyConfig:
    # Pair each property with a presence flag so properties can be dropped individually.
    mask_enabled: bool = False
    mean: torch.Tensor | list[float] | tuple[float, ...] | None = None
    std: torch.Tensor | list[float] | tuple[float, ...] | None = None
