"""Grouped conditioning inputs for ProteinDiffusionUNet.forward."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class StructureInput:
    """3Di seed tokens [B,L] (0 is PAD) and an optional hidden-token mask."""

    tokens: torch.Tensor
    # True where a token is deliberately hidden, which is distinct from padding.
    hidden_mask: torch.Tensor | None = None


@dataclass
class PropertyInput:
    """Raw global properties [B,3], with optional masking."""

    values: torch.Tensor
    mask: torch.Tensor | None = None


@dataclass
class SeedInput:
    """Seed residues [B,C,L]; edit_mask [B,L] is True where the model redesigns."""

    x: torch.Tensor
    edit_mask: torch.Tensor
