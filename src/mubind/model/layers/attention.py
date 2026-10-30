"""Choose the attention layer used inside a U-Net block."""

from torch import nn

from .linear_attention import LinearAttention1d
from .self_attention import SelfAttention1d


def make_attention(
    mode: str, channels: int, heads: int, *, bottleneck: bool = False,
) -> nn.Module:
    if bottleneck:
        return nn.Identity() if mode == "none" else SelfAttention1d(channels, heads)
    if mode == "unet_linear":
        return LinearAttention1d(channels, heads)
    if mode == "unet_full":
        return SelfAttention1d(channels, heads)
    return nn.Identity()
