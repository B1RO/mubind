"""Shared multiscale feature pyramid for the conditioning encoders."""

from torch import nn

from ..layers import LinearAttention1d, ResidualBlock1d


def _conditioning_downs(level_channels, bottleneck_channels, attention_heads):
    """Build the shared pyramid while preserving module names and initialization order."""
    downs = nn.ModuleList()
    for index, channels in enumerate(level_channels):
        last = index == len(level_channels) - 1
        next_channels = bottleneck_channels if last else level_channels[index + 1]
        # Resize was initialized before the residual/attention layers in both encoders.
        resize = nn.Conv1d(
            channels, next_channels, kernel_size=3 if last else 4,
            stride=1 if last else 2, padding=1,
        )
        downs.append(nn.ModuleList([
            ResidualBlock1d(channels, channels, None),
            ResidualBlock1d(channels, channels, None),
            LinearAttention1d(channels, attention_heads),
            resize,
        ]))
    return downs
