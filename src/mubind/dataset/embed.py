"""Continuous 6-channel amino-acid embedding for peptide diffusion.

Instead of diffusing discrete token IDs, MUbind represents each amino acid as a
fixed 6-number physicochemical vector (Table 8 of the OmegAMP paper) so continuous
Gaussian diffusion can add and remove noise directly in feature space.

  Channel 0  WW*              Wimley-White whole-residue hydrophobicity
  Channel 1  pI               Isoelectric point
  Channel 2  Levitt           Alpha-helix propensity
  Channel 3  TM               Transmembrane tendency
  Channel 4  AASI             Amino-acid selectivity index
  Channel 5  residue_present  1.0 for a real amino acid, 0.0 for padding

  Residue 'A'  ->  [-0.03, 6.01, 1.29, 11.20, 1.89, 1.0]  (or min-max scaled to [-1, 1] on ch 0..4)
  Padding      ->  [ 0.00, 0.00, 0.00,  0.00, 0.00, 0.0]

At the end of sampling, dataset/decode.py converts the denoised [6, L] tensor
back into an amino-acid string by picking the closest row in this 21-entry table
(20 amino acids + PAD) by Euclidean distance.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..common.sequences import AA_ORDER, normalize_sequence
from ..model.conditioning.threedi_2d_representation import THREEDI_ALPHABET, foldseek_3di_codebook

# 1-based lookup for the 20 Foldseek 3Di structural letters (0 is reserved for PAD).
THREEDI_TO_ID = {token: i + 1 for i, token in enumerate(THREEDI_ALPHABET)}

# 5 physicochemical descriptors per amino acid [WW*, pI, Levitt, TM, AASI] from OmegAMP Table 8.
OMEGAMP_AA_SCALE = {
    "A": [-0.03, 6.01, 1.290, 11.200, 1.89],
    "R": [-0.74, 10.76, 0.960, 0.500, 1.91],
    "N": [-0.28, 5.41, 0.900, 2.900, 2.33],
    "D": [-1.23, 2.85, 1.040, 2.900, 3.13],
    "C": [0.71, 5.05, 1.110, 4.100, 1.73],
    "Q": [-0.51, 5.65, 1.270, 1.600, 3.05],
    "E": [-2.02, 3.15, 1.440, 1.800, 3.14],
    "G": [0.37, 6.06, 0.560, 11.800, 2.67],
    "H": [-0.89, 6.00, 1.220, 2.000, 3.00],
    "I": [0.81, 6.05, 0.970, 8.600, 1.97],
    "L": [1.06, 6.01, 1.300, 11.700, 1.74],
    "K": [-0.99, 9.60, 1.230, 0.500, 2.28],
    "M": [0.61, 5.74, 1.470, 1.900, 2.50],
    "F": [1.63, 5.49, 1.070, 5.100, 1.53],
    "P": [-0.38, 6.30, 0.520, 2.700, 0.22],
    "S": [0.17, 5.68, 0.820, 8.000, 2.14],
    "T": [0.07, 5.60, 0.820, 4.900, 2.18],
    "W": [2.35, 5.89, 0.990, 2.200, 2.00],
    "Y": [1.44, 5.64, 0.720, 2.600, 2.01],
    "V": [0.27, 6.00, 0.910, 12.900, 2.37],
}

RESIDUE_FEATURE_DIM = 5
EMBEDDING_DIM = RESIDUE_FEATURE_DIM + 1
JOINT_EMBEDDING_DIM = EMBEDDING_DIM + 2
PAD_VECTOR = [0.0] * EMBEDDING_DIM


def _scale_bounds(
    *, dtype: torch.dtype, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    vectors = torch.tensor(list(OMEGAMP_AA_SCALE.values()), dtype=dtype, device=device)
    return vectors.amin(dim=0), vectors.amax(dim=0)


def normalize_residue_vectors(vectors: torch.Tensor) -> torch.Tensor:
    """Min-max normalize residue features channel-wise to [-1, 1]."""
    if vectors.shape[-1] != RESIDUE_FEATURE_DIM:
        raise ValueError(
            f"expected {RESIDUE_FEATURE_DIM} physicochemical features, got {vectors.shape[-1]}"
        )
    minimum, maximum = _scale_bounds(dtype=vectors.dtype, device=vectors.device)
    return 2 * (vectors - minimum) / (maximum - minimum) - 1


def embed_sequence(
    sequence: str,
    *,
    pad_to: int | None = None,
    dtype: torch.dtype = torch.float32,
    normalized: bool = False,
) -> torch.Tensor:
    """Return an [L, 6] or [pad_to, 6] residue embedding tensor."""
    seq = normalize_sequence(sequence)
    if pad_to is not None and len(seq) > pad_to:
        raise ValueError(f"sequence length {len(seq)} exceeds pad_to={pad_to}")

    rows: list[list[float]] = []
    for aa in seq:
        rows.append(OMEGAMP_AA_SCALE[aa])

    residue_features = torch.tensor(rows, dtype=dtype).reshape(-1, RESIDUE_FEATURE_DIM)
    if normalized and residue_features.numel():
        residue_features = normalize_residue_vectors(residue_features)
    residue_present = torch.ones((len(rows), 1), dtype=dtype)
    embedding = torch.cat([residue_features, residue_present], dim=1)

    # Keep PAD at zero in both representations. This lets variable-length
    # training use the existing zero-vector PAD target.
    if pad_to is not None:
        padding = torch.zeros((pad_to - len(rows), len(PAD_VECTOR)), dtype=dtype)
        embedding = torch.cat([embedding, padding], dim=0)

    return embedding


def encode_3di(tokens: str, *, pad_to: int) -> torch.Tensor:
    """Return padded 3Di token IDs with PAD=0."""
    seq = tokens.strip().upper()
    if len(seq) > pad_to:
        raise ValueError(f"3Di length {len(seq)} exceeds pad_to={pad_to}")

    ids = torch.zeros(pad_to, dtype=torch.long)
    for pos, token in enumerate(seq):
        try:
            ids[pos] = THREEDI_TO_ID[token]
        except KeyError as exc:
            raise ValueError(
                f"unsupported 3Di token {token!r} at position {pos + 1}"
            ) from exc
    return ids


def embed_diffusion_sequence(
    sequence: str,
    threedi: str,
    *,
    pad_to: int,
    normalized: bool = False,
    joint_3di: bool = False,
) -> torch.Tensor:
    """Build [L,6] AA/PAD features, optionally appending two 3Di coordinates.

    Joint targets use the complete aligned 3Di sequence, before any conditioning
    masking. AA normalization leaves the published structural coordinates alone.
    """
    biochemical = embed_sequence(sequence, pad_to=pad_to, normalized=normalized)
    if not joint_3di:
        return biochemical
    if len(sequence.strip()) != len(threedi.strip()):
        raise ValueError("joint diffusion requires one 3Di token per amino acid")
    structural = F.embedding(
        encode_3di(threedi, pad_to=pad_to), foldseek_3di_codebook(), padding_idx=0
    )
    return torch.cat([biochemical, structural], dim=-1)
