"""Embed 3Di structure tokens via Foldseek's 2D latent-space centroids.

When Foldseek discretizes local protein backbone geometry into the 20 letters
of the 3Di alphabet (A..Y), each letter corresponds to the nearest centroid in
a 2D VQ-VAE latent space, so structurally similar local motifs sit close
together on that 2D plane.

Instead of learning 20 unrelated embedding vectors from scratch,
FoldseekThreeDiEmbedding looks up each token's frozen 2D centroid in a [21, 2]
codebook (with (0.0, 0.0) at index 0 for PAD or masked positions) and projects
it to the 3Di channel width through a single bias-free linear layer.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


# Token IDs 1..20 follow this alphabet order, with 0 reserved for PAD or hidden tokens.
THREEDI_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


# Foldseek Alphabet3Di centroids keyed by the 20 exported 3Di letters
# (from steineggerlab/foldseek lib/3di/structureto3di.h, commit 4aba8c34).
# These are 2D VQ-VAE latent-space centroids rather than physical 3D coordinates.
FOLDSEEK_3DI_CENTROIDS = {
    "A": (-1.0729, -0.3600),
    "C": (-0.1356, -1.8914),
    "D": (0.4948, -0.4205),
    "E": (-0.9874, 0.8128),
    "F": (-1.6621, -0.4259),
    "G": (2.1394, 0.0486),
    "H": (1.5558, -0.1503),
    "I": (2.9179, 1.1437),
    "K": (-2.8814, 0.9956),
    "L": (-1.1400, -2.0068),
    "M": (3.2025, 1.7356),
    "N": (1.7769, -1.3037),
    "P": (0.6901, -1.2554),
    "Q": (-1.1061, -1.3397),
    "R": (2.1495, -0.8030),
    "S": (2.3060, -1.4988),
    "T": (2.5522, 0.6046),
    "V": (0.7786, -2.1660),
    "W": (-2.3030, 0.3813),
    "Y": (1.0290, 0.8772),
}


def foldseek_3di_codebook(
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the [21, 2] centroid lookup table with (0.0, 0.0) at row 0 for PAD."""
    return torch.tensor(
        [(0.0, 0.0), *[FOLDSEEK_3DI_CENTROIDS[t] for t in THREEDI_ALPHABET]],
        device=device,
        dtype=dtype,
    )


def validate_threedi_representation(value: str) -> str:
    """Check that the configured 3Di embedding mode is 'learned' or 'foldseek_2d'."""
    if value not in {"learned", "foldseek_2d"}:
        raise ValueError(
            "3Di representation must be learned or foldseek_2d; "
            f"got {value!r}"
        )
    return value


class FoldseekThreeDiEmbedding(nn.Module):
    """Frozen [21, 2] centroid lookup followed by a learned 2-to-width linear projection.

    Used by ProteinDiffusionUNet when ThreeDiConfig.representation == 'foldseek_2d'.
    The codebook is registered as a non-trainable buffer, and the projection has no
    bias so row 0 (PAD or hidden tokens) maps to an exact zero vector.
    """

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.register_buffer("coordinates", foldseek_3di_codebook())
        self.projection = nn.Linear(2, output_dim, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        coordinates = F.embedding(token_ids, self.coordinates, padding_idx=0)
        return self.projection(coordinates)
