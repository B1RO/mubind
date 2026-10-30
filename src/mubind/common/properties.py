"""Global biochemical peptide properties (length, net charge, hydrophobicity).

Computes the 3-number property vector [length, charge, hydrophobicity] that
PropertyConditioningEncoder turns into per-resolution U-Net feature maps.
All constants and formulas match modlAMP 4.3.x (used by OmegAMP) so training
and sampling do not need scipy or modlAMP installed at runtime.

  sequence  "ACDEF..."  (length L)
       │
       ├─► length          L
       ├─► charge          net Henderson-Hasselbalch charge at pH 7.0
       │                   (free N-terminus, amidated C-terminus, K/R/H vs D/E/C/Y)
       └─► hydrophobicity  mean residue value on the Eisenberg scale
"""

from __future__ import annotations

import torch

from .sequences import AA_ORDER, normalize_sequence

OMEGAMP_PROPERTY_DEFINITION = "omegamp_modlamp_v1"
DEFAULT_PROPERTY_DEFINITION = OMEGAMP_PROPERTY_DEFINITION

# Per-residue Eisenberg consensus hydrophobicity scale (matching modlAMP calc_H("eisenberg")).
EISENBERG_HYDROPHOBICITY = {
    "A": 0.62,
    "R": -2.53,
    "N": -0.78,
    "D": -0.90,
    "C": 0.29,
    "Q": -0.85,
    "E": -0.74,
    "G": 0.48,
    "H": -0.40,
    "I": 1.38,
    "L": 1.06,
    "K": -1.50,
    "M": 0.64,
    "F": 1.19,
    "P": 0.12,
    "S": -0.18,
    "T": -0.05,
    "W": 0.81,
    "Y": 0.26,
    "V": 1.08,
}

# Side-chain and terminal pKa values matching modlAMP GlobalAnalysis.calc_charge(ph=7.0, amide=True).
OMEGAMP_POSITIVE_PKA = {"K": 10.67, "R": 12.10, "H": 6.04}
OMEGAMP_NEGATIVE_PKA = {"D": 3.71, "E": 4.15, "C": 8.14, "Y": 10.10}
OMEGAMP_N_TERMINUS_PKA = 9.38
OMEGAMP_AMIDATED_C_TERMINUS_PKA = 15.0
OMEGAMP_CHARGE_PH = 7.0


def canonical_property_definition(definition: str | None) -> str:
    """Resolve a property definition alias to the canonical 'omegamp_modlamp_v1' identifier."""
    if definition is None:
        return DEFAULT_PROPERTY_DEFINITION
    aliases = {
        "omegamp": OMEGAMP_PROPERTY_DEFINITION,
        OMEGAMP_PROPERTY_DEFINITION: OMEGAMP_PROPERTY_DEFINITION,
    }
    try:
        return aliases[str(definition).lower()]
    except KeyError as exc:
        raise ValueError(
            f"property definition must be omegamp_modlamp_v1; got {definition!r}"
        ) from exc


def _positive_fraction(pka: float, *, ph: float = OMEGAMP_CHARGE_PH) -> float:
    """Protonated (+1) fraction of a basic group (K, R, H, or N-terminus) at the given pH."""
    ratio = 10.0 ** (pka - ph)
    return ratio / (ratio + 1.0)


def _negative_fraction(pka: float, *, ph: float = OMEGAMP_CHARGE_PH) -> float:
    """Deprotonated (-1) fraction of an acidic group (D, E, C, Y, or C-terminus) at the given pH."""
    ratio = 10.0 ** (ph - pka)
    return ratio / (ratio + 1.0)


def residue_property_scales(
    definition: str | None = None,
) -> tuple[dict[str, float], dict[str, float], float]:
    """Return per-amino-acid charge contributions, Eisenberg hydrophobicities, and terminal net charge."""
    canonical_property_definition(definition)
    charges = {aa: 0.0 for aa in AA_ORDER}
    for aa, pka in OMEGAMP_POSITIVE_PKA.items():
        charges[aa] += _positive_fraction(pka)
    for aa, pka in OMEGAMP_NEGATIVE_PKA.items():
        charges[aa] -= _negative_fraction(pka)
    terminal_offset = _positive_fraction(OMEGAMP_N_TERMINUS_PKA)
    terminal_offset -= _negative_fraction(OMEGAMP_AMIDATED_C_TERMINUS_PKA)
    return charges, dict(EISENBERG_HYDROPHOBICITY), terminal_offset


def sequence_properties(
    sequence: str,
    *,
    property_definition: str | None = None,
) -> dict[str, float]:
    """Compute length, net charge at pH 7.0, and mean Eisenberg hydrophobicity for a sequence."""
    seq = normalize_sequence(sequence)
    length = len(seq)
    definition = canonical_property_definition(property_definition)
    charges, hydrophobicities, terminal_charge = residue_property_scales(definition)
    charge = terminal_charge + sum(charges[aa] for aa in seq) if length else 0.0
    # modlAMP rounds GlobalDescriptor charge to three decimal places.
    charge = round(charge, 3)
    hydrophobicity = sum(hydrophobicities[aa] for aa in seq) / length if length else 0.0
    return {
        "length": float(length),
        "charge": float(charge),
        "hydrophobicity": float(hydrophobicity),
    }


def condition_tensor(
    sequence: str,
    *,
    dtype: torch.dtype = torch.float32,
    property_definition: str | None = None,
) -> torch.Tensor:
    """Compute [length, charge, hydrophobicity] for a sequence as a 1D float tensor of shape [3]."""
    props = sequence_properties(
        sequence,
        property_definition=property_definition,
    )
    return torch.tensor(
        [props["length"], props["charge"], props["hydrophobicity"]],
        dtype=dtype,
    )
