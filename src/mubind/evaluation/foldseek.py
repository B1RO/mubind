"""Read Foldseek output and write the minimal ESMFold mmCIF representation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from Bio.PDB.MMCIF2Dict import MMCIF2Dict


def _column(mmcif: dict[str, Any], *names: str) -> list[str]:
    for name in names:
        if name not in mmcif:
            continue
        value = mmcif[name]
        return [value] if isinstance(value, str) else list(value)
    raise ValueError(f"mmCIF is missing required column {names[0]!r}")


def _atom_name_field(name: str, element: str) -> str:
    stripped = name.strip()
    if len(stripped) < 4 and not stripped[:1].isdigit() and len(element) == 1:
        return f" {stripped:<3}"
    return f"{stripped:>4}"


def write_mmcif_chain_pdb(source: Path, output: Path, *, chain_id: str) -> None:
    """Write one chain from the ESMFold2-Fast minimal mmCIF as standard PDB."""
    mmcif = MMCIF2Dict(str(source))
    atom_names = _column(
        mmcif,
        "_atom_site.auth_atom_id",
        "_atom_site.label_atom_id",
    )
    elements = _column(mmcif, "_atom_site.type_symbol")
    residue_names = _column(
        mmcif,
        "_atom_site.auth_comp_id",
        "_atom_site.label_comp_id",
    )
    chain_ids = _column(
        mmcif,
        "_atom_site.auth_asym_id",
        "_atom_site.label_asym_id",
    )
    residue_ids = _column(
        mmcif,
        "_atom_site.auth_seq_id",
        "_atom_site.label_seq_id",
    )
    x_values = _column(mmcif, "_atom_site.Cartn_x")
    y_values = _column(mmcif, "_atom_site.Cartn_y")
    z_values = _column(mmcif, "_atom_site.Cartn_z")
    b_factors = _column(mmcif, "_atom_site.B_iso_or_equiv")
    model_ids = _column(mmcif, "_atom_site.pdbx_PDB_model_num")
    first_model = model_ids[0]

    lines: list[str] = []
    serial = 1
    for atom, element, residue, chain, residue_id, x, y, z, b, model in zip(
        atom_names,
        elements,
        residue_names,
        chain_ids,
        residue_ids,
        x_values,
        y_values,
        z_values,
        b_factors,
        model_ids,
        strict=True,
    ):
        if chain != chain_id or model != first_model:
            continue
        element = element.strip().upper()
        lines.append(
            f"ATOM  {serial:5d} {_atom_name_field(atom, element)} "
            f"{residue.strip():>3} A{int(float(residue_id)):4d}    "
            f"{float(x):8.3f}{float(y):8.3f}{float(z):8.3f}"
            f"{1.0:6.2f}{float(b):6.2f}          {element:>2}"
        )
        serial += 1
    if not lines:
        raise ValueError(f"chain {chain_id!r} is absent from {source}")
    output.write_text("\n".join([*lines, "TER", "END"]) + "\n")


def read_foldseek_records(database: Path) -> dict[str, tuple[str, str]]:
    lookup: dict[int, str] = {}
    with Path(f"{database}.lookup").open() as handle:
        for line in handle:
            record_id, name, _ = line.rstrip("\n").split("\t")
            lookup[int(record_id)] = name

    indexes: dict[int, tuple[int, int]] = {}
    with Path(f"{database}.index").open() as handle:
        for line in handle:
            record_id, offset, length = line.rstrip("\n").split("\t")
            indexes[int(record_id)] = (int(offset), int(length))

    ss_indexes: dict[int, tuple[int, int]] = {}
    with Path(f"{database}_ss.index").open() as handle:
        for line in handle:
            record_id, offset, length = line.rstrip("\n").split("\t")
            ss_indexes[int(record_id)] = (int(offset), int(length))

    aa_data = database.read_bytes()
    ss_data = Path(f"{database}_ss").read_bytes()
    result: dict[str, tuple[str, str]] = {}
    for record_id, name in lookup.items():
        aa_offset, aa_length = indexes[record_id]
        ss_offset, ss_length = ss_indexes[record_id]
        amino_acids = (
            aa_data[aa_offset : aa_offset + aa_length].rstrip(b"\0\n").decode()
        )
        threedi = ss_data[ss_offset : ss_offset + ss_length].rstrip(b"\0\n").decode()
        result[name] = (amino_acids, threedi)
    return result


def token_identity(left: str, right: str) -> tuple[int, int, float]:
    """Return positionwise identity, counting absent terminal tokens as errors."""
    token_count = max(len(left), len(right))
    if token_count == 0:
        raise ValueError("cannot score two empty 3Di strings")
    matches = sum(a == b for a, b in zip(left, right))
    return matches, token_count, matches / token_count
