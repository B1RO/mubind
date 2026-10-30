"""Nearest-neighbor decoding from continuous diffusion tensors back to amino acids.

Given a denoised [6, L] (or [8, L] joint) canvas from the U-Net, decode.py
compares each position against the 20 canonical amino-acid vectors from
OMEGAMP_AA_SCALE plus the all-zero PAD vector [0, 0, 0, 0, 0, 0] and picks the
nearest entry by Euclidean distance (torch.cdist).
"""

from __future__ import annotations

import torch

from ..model.conditioning.threedi_2d_representation import THREEDI_ALPHABET, foldseek_3di_codebook
from .embed import (
    AA_ORDER,
    EMBEDDING_DIM,
    JOINT_EMBEDDING_DIM,
    OMEGAMP_AA_SCALE,
    RESIDUE_FEATURE_DIM,
    normalize_residue_vectors,
)


def nearest_foldseek_3di(coordinates: torch.Tensor) -> torch.Tensor:
    """Decode [...,2] coordinates to residue IDs 1..20; AA decoding owns PAD."""
    if coordinates.shape[-1] != 2:
        raise ValueError("Foldseek coordinates must have final dimension 2")
    points = coordinates.float()
    codebook = foldseek_3di_codebook(device=points.device)[1:]
    return (points.unsqueeze(-2) - codebook).square().sum(dim=-1).argmin(dim=-1) + 1


def biochemical_embedding_dim(channels: int) -> int:
    """Width of the AA/PAD part of a legacy, biochemical, or joint state."""
    if channels == JOINT_EMBEDDING_DIM:
        return EMBEDDING_DIM
    if channels in {RESIDUE_FEATURE_DIM, EMBEDDING_DIM}:
        return channels
    raise ValueError(
        f"expected {RESIDUE_FEATURE_DIM} legacy, {EMBEDDING_DIM} biochemical, "
        f"or {JOINT_EMBEDDING_DIM} joint channels, got {channels}"
    )


def biochemical_features(state: torch.Tensor) -> torch.Tensor:
    """Select AA/PAD channels from a [C,L] or [B,C,L] diffusion state."""
    return state[..., : biochemical_embedding_dim(state.shape[-2]), :]


def decode_embedding_tokens(
    embedding: torch.Tensor,
    *,
    channels_first: bool = True,
    allow_pad: bool = True,
    normalized: bool = False,
) -> list[str | None]:
    """Decode every canvas position, representing PAD as ``None``."""
    x = embedding.detach().float().cpu()
    if channels_first:
        x = x.transpose(0, 1)

    embedding_dim = biochemical_embedding_dim(x.shape[1])
    x = x[:, :embedding_dim]

    labels = list(AA_ORDER)
    vectors_t = torch.tensor(
        [OMEGAMP_AA_SCALE[aa] for aa in AA_ORDER],
        dtype=torch.float32,
    )
    if normalized:
        vectors_t = normalize_residue_vectors(vectors_t)
    if embedding_dim == EMBEDDING_DIM:
        vectors_t = torch.cat([vectors_t, torch.ones((len(labels), 1))], dim=1)
    if allow_pad:
        labels.append("PAD")
        vectors_t = torch.cat([vectors_t, torch.zeros((1, embedding_dim))], dim=0)
    distances = torch.cdist(x, vectors_t)
    ids = distances.argmin(dim=1).tolist()
    return [None if labels[idx] == "PAD" else labels[idx] for idx in ids]


def decode_canvas_sample(
    sample: torch.Tensor,
    *,
    normalized_embeddings: bool,
) -> dict[str, str | int | bool | list[str | None]]:
    """Decode a canvas and diagnose whether residues form one nonempty block."""
    if sample.ndim != 2:
        raise ValueError("sample must have shape [channels, canvas_length]")
    canvas_tokens = decode_embedding_tokens(
        sample,
        channels_first=True,
        allow_pad=True,
        normalized=normalized_embeddings,
    )
    non_pad = [idx for idx, token in enumerate(canvas_tokens) if token is not None]
    trim_start = non_pad[0] if non_pad else 0
    trim_end = non_pad[-1] + 1 if non_pad else 0
    internal_pad_count = sum(
        token is None for token in canvas_tokens[trim_start:trim_end]
    )
    sequence_valid = bool(non_pad) and internal_pad_count == 0
    validity_reason = (
        "valid" if sequence_valid else "empty" if not non_pad else "internal_pad"
    )
    generated = "".join(
        token for token in canvas_tokens[trim_start:trim_end] if token is not None
    )
    result = {
        "generated": generated,
        "generated_length": len(generated),
        "canvas_length": int(sample.shape[-1]),
        "trim_start": trim_start,
        "trim_end": trim_end,
        "sequence_valid": sequence_valid,
        "sequence_validity_reason": validity_reason,
        "internal_pad_count": internal_pad_count,
        "decoded_canvas": "".join(
            token if token is not None else "-" for token in canvas_tokens
        ),
        "canvas_tokens": canvas_tokens,
    }
    if sample.shape[0] == JOINT_EMBEDDING_DIM:
        structural_ids = (
            nearest_foldseek_3di(
                sample[EMBEDDING_DIM:].detach().float().transpose(0, 1)
            )
            .cpu()
            .tolist()
        )
        structural_tokens = [
            THREEDI_ALPHABET[token_id - 1] if aa is not None else "-"
            for token_id, aa in zip(structural_ids, canvas_tokens, strict=True)
        ]
        # These decode the structural output channels; they are not a Foldseek
        # measurement of the protein obtained by folding the generated sequence.
        result["predicted_3di"] = "".join(t for t in structural_tokens if t != "-")
        result["predicted_3di_canvas"] = "".join(structural_tokens)
    return result
