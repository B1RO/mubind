"""Target PDB discovery, chain/residue range parsing, hotspot checks, and chain extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ...common.sequences import normalize_sequence


@dataclass(frozen=True)
class Target:
    """Validated target chain, residue range, sequence, and hotspot residues."""

    path: Path
    spec: str
    chain: str
    residue_ids: tuple[int, ...]
    sequence: str
    hotspots: tuple[str, ...]


def discover_targets(directory: Path) -> list[Path]:
    """Return sorted `.pdb` files in `directory`."""
    return sorted(
        (
            p
            for p in directory.expanduser().glob("*")
            if p.is_file() and p.suffix.lower() == ".pdb"
        ),
        key=lambda p: p.name.lower(),
    )


def first_model(path: Path):
    """Parse `path` and return its first BioPython PDB model."""
    from Bio.PDB import PDBParser
    from Bio.PDB.PDBExceptions import PDBException

    try:
        structure = PDBParser(QUIET=True).get_structure("protein", str(path))
    except PDBException as exc:
        raise ValueError(f"Cannot parse target PDB {path}: {exc}") from exc
    try:
        return next(structure.get_models())
    except StopIteration as exc:
        raise ValueError(f"PDB contains no model: {path}") from exc


def chain_sequence(chain) -> str:
    """Extract the normalized 1-letter amino acid sequence of standard residues in `chain`."""
    from Bio.SeqUtils import seq1

    return normalize_sequence(
        "".join(seq1(res.resname) for res in chain if res.id[0] == " ")
    )


def inspect_target(path: Path, spec: str | None = None, hotspots=()) -> Target:
    """Validate a target PDB, optional chain/residue range (`'A10-120'`), and hotspot list."""
    from Bio.SeqUtils import seq1

    path = path.expanduser().resolve()
    model = first_model(path)
    if spec:
        match = re.fullmatch(r"([A-Za-z0-9])(-?\d+)-(-?\d+)", spec.strip())
        if match is None:
            raise ValueError(
                "Target range must be a chain and residue range, such as A10-120"
            )
        chain_id, start, end = match[1], int(match[2]), int(match[3])
        if start > end or chain_id not in model:
            raise ValueError("Target range is reversed or the chain does not exist")
        residues = [
            r for r in model[chain_id] if r.id[0] == " " and start <= r.id[1] <= end
        ]
        if not residues or residues[0].id[1] != start or residues[-1].id[1] != end:
            raise ValueError("Both target range endpoints must be present in the PDB")
    else:
        chains = [(chain.id, [r for r in chain if r.id[0] == " "]) for chain in model]
        if not chains or not max(len(rows) for _, rows in chains):
            raise ValueError("Target PDB has no protein residues")
        chain_id, residues = max(chains, key=lambda item: len(item[1]))

    if not re.fullmatch(r"[A-Za-z0-9]", chain_id):
        raise ValueError("Target chain must have a single letter or digit identifier")
    if any(r.id[2] != " " for r in residues):
        raise ValueError(
            "Selected residues contain insertion codes; renumber this target PDB first"
        )
    sequence = normalize_sequence("".join(seq1(r.resname) for r in residues))
    residue_ids = tuple(r.id[1] for r in residues)
    permitted = {f"{chain_id}{index}" for index in residue_ids}
    hotspots = tuple(
        dict.fromkeys(str(value).strip() for value in hotspots if str(value).strip())
    )
    if set(hotspots) - permitted:
        raise ValueError(
            "Every hotspot must name a residue in the selected target range"
        )
    return Target(
        path,
        f"{chain_id}{residue_ids[0]}-{residue_ids[-1]}",
        chain_id,
        residue_ids,
        sequence,
        hotspots,
    )


def write_chain(path: Path, out: Path, chain_id: str, residue_ids=None) -> None:
    """Write standard residues of `chain_id` (optionally restricted to `residue_ids`) to `out`."""
    from Bio.PDB import PDBIO, Select

    model = first_model(path)
    if chain_id not in model:
        raise ValueError(f"Chain {chain_id} is missing from {path}")

    class Chosen(Select):
        def accept_chain(self, chain):
            return chain.id == chain_id

        def accept_residue(self, residue):
            return residue.id[0] == " " and (
                residue_ids is None or residue.id[1] in residue_ids
            )

    writer = PDBIO()
    writer.set_structure(model)
    writer.save(str(out), Chosen())
