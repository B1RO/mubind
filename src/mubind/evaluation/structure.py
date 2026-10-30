"""Monomer structure evaluation, ported from the 300k run."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import zip_longest
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..model.conditioning.threedi_2d_representation import FOLDSEEK_3DI_CENTROIDS, THREEDI_ALPHABET
from ..setup.tools import find_foldseek
from .folding import FoldingClient
from .geometry import ca_geometry, gap_geometry

FOLDSEEK_3DI_MAX_DISTANCE = max(
    math.dist(a, b)
    for a in FOLDSEEK_3DI_CENTROIDS.values()
    for b in FOLDSEEK_3DI_CENTROIDS.values()
)


def find_foldseek_executable(value):
    return find_foldseek(value)


def score_gap_geometry(records):
    """Fit retained positions, then measure the gap; keep failed cases visible."""
    scored, errors = [], []
    for record in records:
        try:
            sequence = record["native_binder"]
            offset = record.get("target_offset", 0)
            canvas = record["decoded_canvas"]
            occupied = np.array([aa != "-" for aa in canvas])
            expected = np.zeros(len(canvas), dtype=bool)
            expected[offset : offset + len(sequence)] = True
            if not np.array_equal(occupied, expected):
                raise ValueError(
                    "Generated residue positions differ from the requested canvas"
                )
            gap = np.array([flag == "1" for flag in record["edit_mask"]])[expected]
            reference, ref_sequence = _ca_trace(
                Path(record["reference_structure_path"]), chain_id="A"
            )
            predicted, pred_sequence = _ca_trace(
                Path(record["generated_monomer_structure_path"]), chain_id=None
            )
            native, native_sequence = _ca_trace(
                Path(record["native_monomer_structure_path"]), chain_id=None
            )
            if (
                ref_sequence != sequence
                or pred_sequence != record["generated"]
                or native_sequence != sequence
            ):
                raise ValueError("Structure sequence mismatch")
            values = gap_geometry(predicted, reference, gap)
            baseline = gap_geometry(native, reference, gap)
            scored.append(
                {
                    "case_id": record["case_id"],
                    **values,
                    "native_gap_ca_rmsd": baseline["gap_ca_rmsd"],
                    "fixed_residues_preserved": record["fixed_residues_preserved"],
                }
            )
        except (KeyError, ValueError, OSError) as exc:
            errors.append({"case_id": record.get("case_id"), "error": str(exc)})
    summary = {
        "candidate_count": len(records),
        "scored_count": len(scored),
        "error_count": len(errors),
    }
    if scored:
        for key in ("gap_ca_rmsd", "retained_ca_rmsd", "native_gap_ca_rmsd"):
            summary["mean_" + key] = float(np.mean([row[key] for row in scored]))
        summary["median_gap_ca_rmsd"] = float(
            np.median([row["gap_ca_rmsd"] for row in scored])
        )
        summary["gap_within_2a_fraction_all_candidates"] = sum(
            row["gap_ca_rmsd"] <= 2 for row in scored
        ) / len(records)
    return {"summary": summary, "records": scored, "errors": errors}


def _visible_threedi_strings(
    realized: str,
    canvas: str,
    requested_tokens: Sequence[int],
) -> tuple[str, str]:
    """Align supplied positions to folded residues, keeping missing ones as gaps."""

    if len(canvas) != len(requested_tokens):
        raise ValueError("Partial 3Di canvas lengths do not match")
    if sum(aa != "-" for aa in canvas) != len(realized):
        raise ValueError("Foldseek residue count differs from generated sequence")
    requested, predicted = [], []
    residue_index = 0
    for aa, token in zip(canvas, requested_tokens, strict=True):
        if token:
            requested.append(THREEDI_ALPHABET[token - 1])
            predicted.append(realized[residue_index] if aa != "-" else "-")
        if aa != "-":
            residue_index += 1
    if not requested:
        raise ValueError("Partial 3Di task has no supplied constraints")
    return "".join(requested), "".join(predicted)


def _batch_item(value: Any, index: int) -> Any:
    if isinstance(value, torch.Tensor):
        item = value[index].detach().cpu()
        return item.item() if item.ndim == 0 else item
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value[index]
    raise ValueError("structural validation metadata must be batched")


def _fingerprint(*values: str) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def _metric(metrics: Mapping[str, Any], name: str) -> float | None:
    value = metrics.get(name)
    return None if value is None else float(value)


def _difference(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else left - right


def _case_metadata(
    tasks: Mapping[str, Any],
    index: int,
) -> tuple[int, str]:
    dataset_index = (
        int(_batch_item(tasks["dataset_index"], index))
        if "dataset_index" in tasks
        else index
    )
    case_id = (
        str(_batch_item(tasks["validation_case_id"], index))
        if "validation_case_id" in tasks
        else f"row_{dataset_index}"
    )
    return dataset_index, case_id


def _mmcif_column(
    mmcif: Mapping[str, Any],
    *names: str,
) -> list[str]:
    for name in names:
        if name not in mmcif:
            continue
        values = mmcif[name]
        return [values] if isinstance(values, str) else list(values)
    raise ValueError(f"mmCIF is missing required column {names[0]!r}")


def _mmcif_ca_trace(path: Path, *, chain_id: str | None) -> tuple[np.ndarray, str]:
    """Read the CA trace without requiring optional atom-site columns."""
    from Bio.Data.PDBData import protein_letters_3to1
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict

    mmcif = MMCIF2Dict(str(path))
    atom_names = _mmcif_column(
        mmcif,
        "_atom_site.label_atom_id",
        "_atom_site.auth_atom_id",
    )
    residue_names = _mmcif_column(
        mmcif,
        "_atom_site.auth_comp_id",
        "_atom_site.label_comp_id",
    )
    chain_ids = _mmcif_column(
        mmcif,
        "_atom_site.auth_asym_id",
        "_atom_site.label_asym_id",
    )
    residue_ids = _mmcif_column(
        mmcif,
        "_atom_site.label_seq_id",
        "_atom_site.auth_seq_id",
    )
    x_values = _mmcif_column(mmcif, "_atom_site.Cartn_x")
    y_values = _mmcif_column(mmcif, "_atom_site.Cartn_y")
    z_values = _mmcif_column(mmcif, "_atom_site.Cartn_z")

    atom_count = len(atom_names)
    required_columns = {
        "residue name": residue_names,
        "chain ID": chain_ids,
        "residue ID": residue_ids,
        "x coordinate": x_values,
        "y coordinate": y_values,
        "z coordinate": z_values,
    }
    malformed = {
        name: len(values)
        for name, values in required_columns.items()
        if len(values) != atom_count
    }
    if malformed:
        raise ValueError(
            f"inconsistent mmCIF atom-site column lengths in {path}: "
            f"atom names={atom_count}, {malformed}"
        )

    model_ids = mmcif.get("_atom_site.pdbx_PDB_model_num")
    if model_ids is None:
        model_ids = ["1"] * atom_count
    elif isinstance(model_ids, str):
        model_ids = [model_ids]
    else:
        model_ids = list(model_ids)
    if len(model_ids) != atom_count:
        raise ValueError(f"inconsistent mmCIF model column length in {path}")
    first_model = model_ids[0] if model_ids else "1"

    ca_rows: list[tuple[str, str, str, str, str, str]] = []
    for index, atom_name in enumerate(atom_names):
        if model_ids[index] != first_model or atom_name.strip().upper() != "CA":
            continue
        residue_name = residue_names[index].strip().upper()
        amino_acid = protein_letters_3to1.get(residue_name)
        if amino_acid is None:
            continue
        ca_rows.append(
            (
                chain_ids[index],
                residue_ids[index],
                amino_acid,
                x_values[index],
                y_values[index],
                z_values[index],
            )
        )

    available_chains = list(dict.fromkeys(row[0] for row in ca_rows))
    if chain_id is None:
        if len(available_chains) != 1:
            raise ValueError(
                f"expected one protein chain in {path}, found {len(available_chains)}"
            )
        selected_chain = available_chains[0]
    else:
        if chain_id not in available_chains:
            raise ValueError(f"chain {chain_id!r} not found in {path}")
        selected_chain = chain_id

    coordinates: list[np.ndarray] = []
    residues: list[str] = []
    seen_residues: set[str] = set()
    for row_chain, residue_id, amino_acid, x, y, z in ca_rows:
        if row_chain != selected_chain or residue_id in seen_residues:
            continue
        seen_residues.add(residue_id)
        residues.append(amino_acid)
        try:
            coordinates.append(
                np.asarray([float(x), float(y), float(z)], dtype=np.float64)
            )
        except ValueError as error:
            raise ValueError(
                f"invalid CA coordinates for residue {residue_id!r} in {path}"
            ) from error
    if not coordinates:
        raise ValueError(f"no protein CA trace found in {path}")
    return np.stack(coordinates), "".join(residues)


def _ca_trace(path: Path, *, chain_id: str | None) -> tuple[np.ndarray, str]:
    from Bio.Data.PDBData import protein_letters_3to1
    from Bio.PDB import PDBParser

    if path.suffix.lower() in {".cif", ".mmcif"}:
        return _mmcif_ca_trace(path, chain_id=chain_id)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(path.stem, str(path))
    model = next(structure.get_models())
    chains = list(model.get_chains())
    if chain_id is None:
        if len(chains) != 1:
            raise ValueError(f"expected one chain in {path}, found {len(chains)}")
        chain = chains[0]
    else:
        chain = next((value for value in chains if value.id == chain_id), None)
        if chain is None:
            raise ValueError(f"chain {chain_id!r} not found in {path}")

    coordinates: list[np.ndarray] = []
    residues: list[str] = []
    for residue in chain:
        amino_acid = protein_letters_3to1.get(residue.resname.upper())
        if amino_acid is None or "CA" not in residue:
            continue
        residues.append(amino_acid)
        coordinates.append(residue["CA"].coord.astype(np.float64))
    if not coordinates:
        raise ValueError(f"no protein CA trace found in {path}")
    return np.stack(coordinates), "".join(residues)


def _aligned_residue_count(left: str, right: str) -> int:
    return sum(a != "-" and b != "-" for a, b in zip(left, right, strict=True))


def score_threedi_centroid_distance(requested: str, realized: str) -> dict[str, float]:
    """Compare aligned token centroids in the published Foldseek latent space.

    Mean Euclidean distance is in [0, max_distance]; soft agreement is
    1 - mean_distance / max_distance. Missing residues (including terminal
    length differences) and unknown tokens receive maximum distance and zero
    credit. This is separate from the substitution-matrix soft identity.
    """
    token_count = max(len(requested), len(realized))
    if not token_count:
        raise ValueError("cannot score two empty 3Di strings")
    distances = (
        math.dist(FOLDSEEK_3DI_CENTROIDS[left], FOLDSEEK_3DI_CENTROIDS[right])
        if left in FOLDSEEK_3DI_CENTROIDS and right in FOLDSEEK_3DI_CENTROIDS
        else FOLDSEEK_3DI_MAX_DISTANCE
        for left, right in zip_longest(requested, realized, fillvalue="-")
    )
    mean_distance = math.fsum(distances) / token_count
    return {
        "mean_distance": mean_distance,
        "soft_agreement": max(
            0.0, min(1.0, 1.0 - mean_distance / FOLDSEEK_3DI_MAX_DISTANCE)
        ),
    }


@dataclass
class StructuralEvaluator:
    out_dir: Path
    baseline_cache_dir: Path
    client: FoldingClient
    foldseek_executable: str
    seed: int = 0
    _native_monomer_scores: dict[str, dict[str, Any]] = field(
        default_factory=dict, init=False, repr=False
    )

    def _fold_monomer(
        self,
        *,
        directory: Path,
        sequence: str,
    ) -> tuple[Path, dict[str, Any]]:
        protocol = self.client.protocol()
        directory = directory / _fingerprint(
            json.dumps(protocol, sort_keys=True), sequence
        )
        structure_path = directory / "monomer.cif"
        metrics_path = directory / "metrics.json"
        request_path = directory / "request.json"
        expected_request = {"sequence": sequence, "protocol": protocol}
        cache_matches = (
            request_path.exists()
            and json.loads(request_path.read_text()) == expected_request
        )
        if cache_matches and structure_path.exists() and metrics_path.exists():
            return structure_path, json.loads(metrics_path.read_text())
        directory.mkdir(parents=True, exist_ok=True)
        structure_path = structure_path.resolve()
        metrics_path = metrics_path.resolve()
        job = {
            "sequence": sequence,
            "out": str(structure_path),
            "metrics_json": str(metrics_path),
            "seed": self.seed,
        }
        metrics = self.client.fold_monomer(job)
        request_path.write_text(json.dumps(expected_request, indent=2) + "\n")
        return structure_path, metrics

    def _native_monomer_baseline(
        self,
        sequence: str,
        reference_path: Path,
        reference_coords: np.ndarray,
    ) -> dict[str, Any]:
        from tmtools import tm_align

        reference_sha = hashlib.sha256(reference_path.read_bytes()).hexdigest()
        protocol = self.client.protocol()
        key = _fingerprint(
            sequence, reference_sha, json.dumps(protocol, sort_keys=True)
        )
        if key not in self._native_monomer_scores:
            path, _ = self._fold_monomer(
                directory=self.baseline_cache_dir
                / "native_monomers"
                / _fingerprint(sequence),
                sequence=sequence,
            )
            coords, folded_sequence = _ca_trace(path, chain_id=None)
            if folded_sequence != sequence or len(coords) != len(reference_coords):
                raise ValueError(
                    "Original-sequence fold has missing or mismatched residues"
                )
            geometry = ca_geometry(coords, reference_coords)
            alignment = tm_align(coords, reference_coords, sequence, sequence)
            self._native_monomer_scores[key] = {
                "native_monomer_scored": True,
                "native_monomer_tm_score": float(alignment.tm_norm_chain2),
                "native_monomer_ca_rmsd": geometry["ca_rmsd"],
                "native_monomer_ca_lddt": geometry["ca_lddt"],
                "native_monomer_structure_path": str(path),
                "reference_structure_sha256": reference_sha,
                "monomer_fold_protocol": protocol,
            }
        return dict(self._native_monomer_scores[key])

    def binder_structure(
        self,
        tasks: Mapping[str, Any],
        outputs: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        if "dataset_index" not in tasks:
            raise ValueError(
                "binder_structure requires raw task metadata 'dataset_index'"
            )
        # A populated manifest path that disappeared is a dataset/configuration
        # failure, not a failed generated fold. Do not turn it into a zero TM
        # score in failure-adjusted summaries. Check the whole batch before
        # scoring anything; empty paths retain their intentional-unavailable
        # behavior for datasets without reference structures.
        if "reference_pdb" in tasks:
            for index in range(len(outputs)):
                reference_path = str(_batch_item(tasks["reference_pdb"], index))
                if reference_path and not Path(reference_path).is_file():
                    raise FileNotFoundError(
                        "Configured validation reference is unavailable: "
                        f"{reference_path}. Restore the reference before scoring."
                    )
        from tmtools import tm_align

        results: list[dict[str, Any]] = []
        for index, output in enumerate(outputs):
            record = {
                "binder_structure_scored": False,
                "binder_structure_error": None,
                "binder_tm_score": None,
                "binder_tm_score_generated_normalized": None,
                "binder_tm_rmsd": None,
                "binder_tm_aligned_residue_count": None,
                "generated_monomer_plddt": None,
                "binder_backbone_scored": False,
                "binder_backbone_error": None,
                "binder_ca_rmsd": None,
                "binder_ca_lddt": None,
                "binder_tm_gap": None,
                "native_monomer_scored": False,
                "native_monomer_error": None,
                "native_monomer_tm_score": None,
                "native_monomer_ca_rmsd": None,
                "native_monomer_ca_lddt": None,
            }
            results.append(record)
            _, case_id = _case_metadata(tasks, index)
            try:
                if "reference_pdb" in tasks:
                    reference_path = str(_batch_item(tasks["reference_pdb"], index))
                    if not reference_path or not Path(reference_path).is_file():
                        raise RuntimeError(
                            "Verified source structure is unavailable for this validation row"
                        )
                    native_path = Path(reference_path)
                else:
                    raise ValueError(
                        "ProstT5 structural evaluation requires a reference manifest"
                    )
                native_coords, native_sequence = _ca_trace(native_path, chain_id="A")
                if "reference_pdb" in tasks and native_sequence != str(
                    _batch_item(tasks["binder_aa"], index)
                ):
                    raise RuntimeError(
                        "Source structure sequence does not match the validation sequence"
                    )
                if not np.isfinite(native_coords).all():
                    raise ValueError("Non-finite reference coordinates")
            except (OSError, RuntimeError, ValueError) as error:
                record["binder_structure_error"] = f"{type(error).__name__}: {error}"
                continue

            # Score originals even when a generated candidate is invalid, so
            # baseline coverage does not depend on which designs succeeded.
            record["reference_structure_path"] = str(native_path)
            try:
                record.update(
                    self._native_monomer_baseline(
                        native_sequence, native_path, native_coords
                    )
                )
            except (KeyError, OSError, RuntimeError, ValueError) as error:
                record["native_monomer_error"] = f"{type(error).__name__}: {error}"
            if not bool(output["sequence_valid"]):
                continue
            generated = str(output["generated"])
            try:
                generated_path, generated_metrics = self._fold_monomer(
                    directory=self.out_dir
                    / "binder_monomers"
                    / f"{case_id}_{_fingerprint(generated)}",
                    sequence=generated,
                )
                generated_coords, generated_sequence = _ca_trace(
                    generated_path, chain_id=None
                )
                if (
                    generated_sequence != generated
                    or not np.isfinite(generated_coords).all()
                ):
                    raise ValueError(
                        "Generated fold has missing/mismatched residues or non-finite coordinates"
                    )
                alignment = tm_align(
                    generated_coords, native_coords, generated_sequence, native_sequence
                )
                record.update(
                    {
                        "binder_structure_scored": True,
                        "binder_tm_score": float(alignment.tm_norm_chain2),
                        "binder_tm_score_generated_normalized": float(
                            alignment.tm_norm_chain1
                        ),
                        "binder_tm_rmsd": float(alignment.rmsd),
                        "binder_tm_aligned_residue_count": _aligned_residue_count(
                            alignment.seqxA, alignment.seqyA
                        ),
                        "generated_monomer_plddt": _metric(generated_metrics, "plddt"),
                    }
                )
            except (KeyError, OSError, RuntimeError, ValueError) as error:
                record["binder_structure_error"] = f"{type(error).__name__}: {error}"
                continue
            record["generated_monomer_structure_path"] = str(generated_path)
            record["binder_tm_gap"] = _difference(
                record["binder_tm_score"], record["native_monomer_tm_score"]
            )
            try:
                geometry = ca_geometry(generated_coords, native_coords)
                record.update(
                    {
                        "binder_backbone_scored": True,
                        "binder_ca_rmsd": geometry["ca_rmsd"],
                        "binder_ca_lddt": geometry["ca_lddt"],
                    }
                )
            except ValueError as error:
                record["binder_backbone_error"] = str(error)
        return results

    def realized_threedi(
        self,
        tasks: Mapping[str, Any],
        outputs: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Score refolded AA against full 3Di or only supplied partial positions."""
        from .foldseek import (
            read_foldseek_records,
            token_identity,
            write_mmcif_chain_pdb,
        )
        from .structure import find_foldseek_executable

        results = [
            {
                "realized_threedi_scored": False,
                "realized_threedi_error": None,
                "realized_threedi_matches": None,
                "realized_threedi_token_count": None,
                "realized_threedi_identity": None,
                "realized_threedi_mean_distance": None,
                "realized_threedi_soft_agreement": None,
                "realized_threedi_plddt": None,
                "realized_threedi_ptm": None,
            }
            for _ in outputs
        ]
        selected = [
            index
            for index, output in enumerate(outputs)
            if bool(output["sequence_valid"])
            and str(_batch_item(tasks["threedi_conditioning"], index))
            in {"full_3di", "partial_3di"}
        ]
        if not selected:
            return results
        foldseek = find_foldseek_executable(self.foldseek_executable)
        if foldseek is None:
            error = "Foldseek executable was not found"
            for index in selected:
                results[index]["realized_threedi_error"] = error
            return results

        try:
            with tempfile.TemporaryDirectory(prefix="realized_3di_") as temporary:
                temporary_path = Path(temporary)
                pdb_dir = temporary_path / "pdb"
                pdb_dir.mkdir()
                for index in selected:
                    _, case_id = _case_metadata(tasks, index)
                    generated = str(outputs[index]["generated"])
                    generated_path, generated_metrics = self._fold_monomer(
                        directory=(
                            self.out_dir
                            / "binder_monomers"
                            / f"{case_id}_{_fingerprint(generated)}"
                        ),
                        sequence=generated,
                    )
                    write_mmcif_chain_pdb(
                        generated_path,
                        pdb_dir / f"generated__{index:04d}.pdb",
                        chain_id="B",
                    )
                    results[index]["realized_threedi_plddt"] = _metric(
                        generated_metrics, "plddt"
                    )
                    results[index]["realized_threedi_ptm"] = _metric(
                        generated_metrics, "ptm"
                    )
                database = temporary_path / "structures"
                subprocess.run(
                    [
                        foldseek,
                        "createdb",
                        str(pdb_dir),
                        str(database),
                        "--chain-name-mode",
                        "1",
                        "--threads",
                        "8",
                        "-v",
                        "1",
                    ],
                    check=True,
                )
                encoded = read_foldseek_records(database)
                for index in selected:
                    _, realized = encoded[f"generated__{index:04d}_A"]
                    requested = str(_batch_item(tasks["binder_3di"], index))
                    if (
                        str(_batch_item(tasks["threedi_conditioning"], index))
                        == "partial_3di"
                    ):
                        requested, realized = _visible_threedi_strings(
                            realized,
                            str(outputs[index]["decoded_canvas"]),
                            tasks["seed_3di"][index].detach().cpu().tolist(),
                        )
                    matches, token_count, identity = token_identity(realized, requested)
                    distances = score_threedi_centroid_distance(requested, realized)
                    results[index] = {
                        **results[index],
                        "realized_threedi_scored": True,
                        "realized_threedi_error": None,
                        "realized_threedi_matches": matches,
                        "realized_threedi_token_count": token_count,
                        "realized_threedi_identity": identity,
                        "realized_threedi_mean_distance": distances["mean_distance"],
                        "realized_threedi_soft_agreement": distances["soft_agreement"],
                    }
        except (
            KeyError,
            OSError,
            RuntimeError,
            subprocess.SubprocessError,
            ValueError,
        ) as error:
            message = f"{type(error).__name__}: {error}"
            for index in selected:
                if not results[index]["realized_threedi_scored"]:
                    results[index]["realized_threedi_error"] = message
        return results
