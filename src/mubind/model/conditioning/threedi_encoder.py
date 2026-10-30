"""Multi-layer 3Di conditioning encoder for the diffusion U-Net.

ThreeDiConditioningEncoder is a stack of downsampling layers that shrinks the
3Di sequence (70 -> 35 -> 17) and grows the channels (32 -> 64 -> 128) to
produce one feature map for each U-Net resolution. Its very first step is
ContextualThreeDiEncoder, a Transformer pass over the full sequence that lets
each token see the surrounding structure and fills in any masked '?' positions
before downsampling begins.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from ..layers import _resize_availability_mask
from ._pyramid import _conditioning_downs

if TYPE_CHECKING:
    from ..config import ThreeDiContextConfig


class ThreeDiConditioningEncoder(nn.Module):
    """Downsample 3Di tokens into one feature map per U-Net level, starting with sequence attention.

      width = sequence length (70 -> 35 -> 17), height = channels (32 -> 64 -> 128)

      raw 3Di  [B, 128, 70]  A  C  ?  ?  D  E ...
                                    │
                                    ▼  self.context (ContextualThreeDiEncoder)
      threedi  [B, 128, 70]  ████████████████   (a' c' x' y' d' e' ...)
                             ████████████████
                             ████████████████
                             ████████████████
                                    │
                                    ▼  1x1 conv (128 -> 32 channels)
      maps[0]  [B,  32, 70]  ████████████████   -> level 0 (DownBlock 0, UpBlock 2)
                                    │
                                    ▼  stride-2 down (length 70 -> 35)
      maps[1]  [B,  32, 35]  ████████           -> level 1 (DownBlock 1, UpBlock 1)
                                    │
                                    ▼  stride-2 down (length 35 -> 17, 32 -> 64 channels)
      maps[2]  [B,  64, 17]  ████               -> level 2 (DownBlock 2, UpBlock 0)
                             ████
                                    │
                                    ▼  stride-1 conv (length 17 -> 17, 64 -> 128 channels)
      maps[3]  [B, 128, 17]  ████               -> MiddleBlock (bottleneck)
                             ████
                             ████
                             ████
    """

    def __init__(
        self,
        *,
        input_channels: int,
        level_channels: list[int],
        bottleneck_channels: int,
        attention_heads: int,
        context: ContextualThreeDiEncoder | None = None,
        multiscale: bool = True,
    ) -> None:
        super().__init__()
        self.context = context
        if not multiscale:
            self.initial = None
            self.downs = nn.ModuleList()
            return
        if not level_channels:
            raise ValueError("3Di conditioning requires at least one U-Net level")

        self.initial = (
            nn.Identity()
            if input_channels == level_channels[0]
            else nn.Conv1d(input_channels, level_channels[0], kernel_size=1)
        )
        self.downs = _conditioning_downs(
            level_channels, bottleneck_channels, attention_heads,
        )

    @staticmethod
    def make_context(
        config: ThreeDiContextConfig | None, *, channels: int, max_length: int,
    ) -> ContextualThreeDiEncoder | None:
        """Build context separately so the U-Net retains its initialization order."""
        if config is None:
            return None
        return ContextualThreeDiEncoder(
            channels=channels,
            max_length=max_length,
            layers=config.layers,
            heads=config.heads,
            feedforward_dim=(
                channels * 4 if config.feedforward_dim is None else config.feedforward_dim
            ),
            dropout=config.dropout,
        )

    def forward(
        self,
        threedi: torch.Tensor,
        availability_mask: torch.Tensor,
        *,
        context_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        if context_mask is None:
            context_mask = availability_mask
        if self.context is not None:
            if threedi.shape[-1] > self.context.position_embedding.shape[1]:
                raise ValueError("3Di sequence exceeds contextual encoder length")
            threedi = self.context(threedi, availability_mask, context_mask=context_mask)
        if self.initial is None:
            return threedi, None
        condition = self.initial(threedi)
        current_availability = _resize_availability_mask(
            context_mask,
            condition.shape[-1],
        )
        condition = condition * current_availability[:, None, :]
        maps = [condition]

        for block1, block2, attention, resize in self.downs:
            condition = block1(condition)
            condition = block2(condition)
            condition = attention(condition, current_availability)
            condition = resize(condition)
            current_availability = _resize_availability_mask(
                context_mask,
                condition.shape[-1],
            )
            condition = condition * current_availability[:, None, :]
            maps.append(condition)
        return threedi, maps


class ContextualThreeDiEncoder(nn.Module):
    """Initial Transformer pass over the 3Di sequence inside ThreeDiConditioningEncoder.

    Without this pass each position only knows its own 3Di letter (or nothing at
    all when a span is masked out). Running self-attention across the sequence
    lets every visible token absorb surrounding structural context and fills in
    plausible structural features for hidden '?' positions from the visible ones.

      Input 3Di     A   C   ?   ?   D   E   pad pad
      available     1   1   0   0   1   1    0   0   (visible keys in attention)
      context_mask  1   1   1   1   1   1    0   0   (all real peptide positions)
                    │   │   │   │   │   │
                    ▼   ▼   ▼   ▼   ▼   ▼
                 ┌──────────────────────────┐
                 │  4x Transformer Encoder  │
                 └──────────────────────────┘
                    │   │   │   │   │   │
                    ▼   ▼   ▼   ▼   ▼   ▼
      Output 3Di    a'  c'  x'  y'  d'  e'   0   0
      
      Of course attention  is not strictly in spirit of OmegAMP, but we found it improves performance.
      And who am I to argue with empirical feedback from reality.
    """

    def __init__(
        self,
        *,
        channels: int,
        max_length: int,
        layers: int,
        heads: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("contextual 3Di encoder layers must be positive")
        if heads < 1 or channels % heads:
            raise ValueError(
                "contextual 3Di encoder heads must divide the embedding dimension"
            )
        if feedforward_dim < channels:
            raise ValueError(
                "contextual 3Di feedforward dimension must be at least the "
                "embedding dimension"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError("contextual 3Di dropout must be in [0, 1)")

        self.position_embedding = nn.Parameter(
            torch.zeros(1, max_length, channels)
        )
        nn.init.normal_(self.position_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=layers,
            norm=nn.LayerNorm(channels),
            enable_nested_tensor=False,
        )

    def forward(
        self,
        threedi: torch.Tensor,
        availability_mask: torch.Tensor,
        *,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode the [B,C,L] embedding and [B,L] masks prepared by the U-Net."""
        available = availability_mask.bool()
        if context_mask is None:
            context_mask = available
        # Queries may represent deliberately hidden tokens, but their input
        # token values never enter the encoder. With no observed structure,
        # the whole structural branch must still contribute zero context.
        context_mask = context_mask.bool() & available.any(dim=1, keepdim=True)
        # PyTorch attention cannot consume a row whose every key is masked.
        # Unconditional rows receive one temporary zero-valued key and are
        # zeroed again after encoding.
        safe_available = available.clone()
        safe_available[:, 0] |= ~available.any(dim=1)
        observed = threedi.transpose(1, 2).masked_fill(~available[..., None], 0)
        tokens = observed + self.position_embedding[:, : observed.shape[1]]
        tokens = tokens.masked_fill(~context_mask[..., None], 0)
        contextual = self.encoder(
            tokens,
            src_key_padding_mask=~safe_available,
        )
        contextual = contextual.masked_fill(~context_mask[..., None], 0)
        return contextual.transpose(1, 2)
