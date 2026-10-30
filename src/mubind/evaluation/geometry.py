"""Fixed-correspondence backbone metrics used by the 300k evaluation."""

from __future__ import annotations

import numpy as np


def ca_geometry(predicted: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Return Kabsch CA RMSD (Angstrom) and pair-averaged CA-lDDT (0..1).

    Every residue participates in RMSD. lDDT uses distinct reference CA pairs
    closer than 15 A, averaging preserved fractions at 0.5, 1, 2 and 4 A.
    Sequence positions must already correspond; missing residues are rejected.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if (
        predicted.shape != reference.shape
        or reference.ndim != 2
        or reference.shape[1] != 3
        or len(reference) < 2
    ):
        raise ValueError(
            "Fixed-position CA metrics require equal, complete (N, 3) traces with N >= 2"
        )
    if not (np.isfinite(predicted).all() and np.isfinite(reference).all()):
        raise ValueError("Non-finite CA coordinates")
    x = predicted - predicted.mean(axis=0)
    y = reference - reference.mean(axis=0)
    u, _, vt = np.linalg.svd(x.T @ y)
    # Enforce a proper rotation: a mirrored protein must not align perfectly.
    correction = np.diag([1.0, 1.0, np.linalg.det(u @ vt)])
    rmsd = np.sqrt(np.mean(np.sum((x @ (u @ correction @ vt) - y) ** 2, axis=1)))
    reference_distances = np.linalg.norm(
        reference[:, None] - reference[None, :], axis=-1
    )
    pairs = np.triu(reference_distances < 15.0, k=1)
    if not pairs.any():
        raise ValueError("CA-lDDT has no reference pairs within 15 Angstrom")
    predicted_distances = np.linalg.norm(
        predicted[:, None] - predicted[None, :], axis=-1
    )
    errors = np.abs(predicted_distances[pairs] - reference_distances[pairs])
    lddt = np.mean(errors[:, None] < np.array([0.5, 1.0, 2.0, 4.0]))
    return {"ca_rmsd": float(rmsd), "ca_lddt": float(lddt)}


def gap_geometry(predicted, reference, gap):
    predicted = np.asarray(predicted, dtype=float)
    reference = np.asarray(reference, dtype=float)
    gap = np.asarray(gap, dtype=bool)
    if predicted.shape != reference.shape or reference.shape != (len(gap), 3):
        raise ValueError("Need complete traces with fixed sequence correspondence")
    if not np.isfinite(predicted).all() or not np.isfinite(reference).all():
        raise ValueError("Non-finite coordinates")
    if not gap.any() or (~gap).sum() < 3:
        raise ValueError("Need a gap and at least three retained positions")
    p_center = predicted[~gap].mean(0)
    r_center = reference[~gap].mean(0)
    x, y = predicted - p_center, reference - r_center
    u, _, vt = np.linalg.svd(x[~gap].T @ y[~gap])
    rotation = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
    squared = np.square(x @ rotation - y).sum(1)
    return {
        "gap_ca_rmsd": float(np.sqrt(squared[gap].mean())),
        "retained_ca_rmsd": float(np.sqrt(squared[~gap].mean())),
    }
