"""Complexa scaffold extraction, ESMFold2 scaffold ranking, selection, and Foldseek 3Di encoding."""

from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from ...common.properties import OMEGAMP_PROPERTY_DEFINITION, sequence_properties
from ...dataset.embed import encode_3di
from ...runtime import get_runtime
from ...setup.config import load_config as load_setup
from ..config import DesignConfig, MoreScaffolds, RedesignChoice
from ..services.complexa import run_complexa
from .inpainting import parse_edit_residues
from .targets import Target, chain_sequence, first_model, write_chain


def write_json(path: Path, data: Any) -> None:
    """Atomically write `data` as formatted JSON to `path` via a `.tmp` sibling."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, allow_nan=False, default=str) + "\n"
    )
    temporary.replace(path)


def requested_properties(
    scaffolds: list[dict], overrides: dict | None = None, *, edit_residues: str | None = None
) -> dict[str, dict[str, float]]:
    """Validate per-scaffold target `[length, charge, hydrophobicity]` values (and inpainting length)."""
    overrides = overrides or {}
    if set(overrides) - {s["id"] for s in scaffolds}:
        raise ValueError("Property overrides must refer to selected scaffolds")
    result = {}
    for scaffold in scaffolds:
        values = overrides.get(scaffold["id"], scaffold["properties"])
        if set(values) != {"length", "charge", "hydrophobicity"}:
            raise ValueError("Specify length, charge and hydrophobicity")
        if any(
            isinstance(value, bool) or not math.isfinite(float(value))
            for value in values.values()
        ):
            raise ValueError("Requested properties must be finite numbers")
        length = float(values["length"])
        if not length.is_integer() or not 1 <= length <= 70:
            raise ValueError("Requested length must be an integer between 1 and 70")
        if edit_residues is not None:
            try:
                parse_edit_residues(edit_residues, length=len(scaffold["sequence"]))
            except ValueError as exc:
                raise ValueError(f"{scaffold['id']}: {exc}") from exc
            if length != len(scaffold["sequence"]):
                raise ValueError(
                    f"Inpainting keeps {scaffold['id']} at its original length of "
                    f"{len(scaffold['sequence'])} residues"
                )
        result[scaffold["id"]] = {key: float(value) for key, value in values.items()}
    return result


def describe_scaffolds(
    pdbs: list[Path], target: Target, out: Path, on_status, *, start_index: int = 0
) -> list[dict]:
    """Read target (`A`) and binder (`B`) chains from Complexa PDBs and compute binder properties."""
    on_status("1/5 · Reading scaffold sequences and properties…")
    scaffolds = []
    for index, pdb in enumerate(pdbs, start_index):
        model = first_model(pdb)
        if "A" not in model or "B" not in model:
            raise ValueError(
                f"Complexa scaffold must contain target A and binder B: {pdb}"
            )
        target_seq, sequence = chain_sequence(model["A"]), chain_sequence(model["B"])
        if target_seq != target.sequence:
            raise ValueError(
                f"Scaffold target does not match the selected PDB range: {pdb}"
            )
        if not 1 <= len(sequence) <= 70:
            raise ValueError(
                f"Scaffold binder does not fit the 70-position canvas: {pdb}"
            )
        name = f"scaffold_{index:02d}"
        scaffolds.append(
            {
                "idx": index,
                "id": name,
                "pdb": str(pdb),
                "target_seq": target_seq,
                "sequence": sequence,
                "properties": sequence_properties(
                    sequence, property_definition=OMEGAMP_PROPERTY_DEFINITION
                ),
                "property_definition": OMEGAMP_PROPERTY_DEFINITION,
            }
        )
    write_json(out / "scaffolds.json", scaffolds)
    return scaffolds


def confidence(metrics: dict) -> list[float]:
    """Validate and return `[binder_plddt, iptm]` in `0..1` units."""
    values = []
    for name in ("binder_plddt", "iptm"):
        value = metrics.get(name)
        if value is None or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Folding must return {name} in 0–1 units")
        values.append(float(value))
    return values


def ranking_score(metrics: dict) -> float:
    """Compute the equal-weight complex ranking score `0.5 * binder_plddt + 0.5 * iptm`."""
    plddt, iptm = confidence(metrics)
    return 0.5 * plddt + 0.5 * iptm


def fold_pair(client, target: str, binder: str, directory: Path) -> tuple[dict, str]:
    """Fold a `(target, binder)` complex with ESMFold2 and write `complex.cif` and `metrics.json`."""
    directory.mkdir(parents=True)
    structure = directory / "complex.cif"
    metrics = client.fold_complex(
        {
            "target": target,
            "binder": binder,
            "out": str(structure),
            "metrics_json": str(directory / "metrics.json"),
            "seed": 0,
        }
    )
    confidence(metrics)
    if not structure.is_file() or not structure.stat().st_size:
        raise RuntimeError("Folding service did not write the complex structure")
    write_json(directory / "metrics.json", metrics)
    return metrics, str(structure)


def stage_progress(callback, stage: str, completed: int, total: int, **details) -> None:
    """Emit a structured stage progress event when `callback` is provided."""
    if callback:
        callback(
            dict(
                stage=stage,
                completed=completed,
                total=total,
                at=time.monotonic(),
                **details,
            )
        )


def rank_scaffolds(
    scaffolds: list[dict], client, out: Path, on_status, on_stage_progress=None
) -> list[dict]:
    """Fold any unfolded scaffolds with ESMFold2 and rank succeeded scaffolds by `ranking_score`."""
    pending = [
        s for s in scaffolds if "scaffold_score" not in s and "fold_error" not in s
    ]
    for index, scaffold in enumerate(pending):
        on_status(f"2/5 · Folding scaffold {index + 1}/{len(pending)}…")
        stage_progress(on_stage_progress, "scaffold_folding", index, len(pending))
        failed = False
        try:
            metrics, path = fold_pair(
                client,
                scaffold["target_seq"],
                scaffold["sequence"],
                out / "scaffold_folds" / scaffold["id"],
            )
            scaffold.update(
                scaffold_metrics=metrics,
                scaffold_score=ranking_score(metrics),
                scaffold_structure=path,
            )
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            scaffold["fold_error"] = str(exc)
            failed = True
        write_json(out / "scaffold_results.json", scaffolds)
        stage_progress(
            on_stage_progress,
            "scaffold_folding",
            index + 1,
            len(pending),
            failed=failed,
        )
    successes = sorted(
        (s for s in scaffolds if "scaffold_score" in s),
        key=lambda s: s["scaffold_score"],
        reverse=True,
    )
    if not successes:
        raise RuntimeError(
            f"No scaffold folded successfully. See {out / 'scaffold_results.json'}"
        )
    write_json(
        out / "scaffold_ranking.json",
        [dict(scaffold, rank=rank) for rank, scaffold in enumerate(successes, 1)],
    )
    return successes


def select_scaffolds(
    scaffolds: list[dict],
    settings: DesignConfig,
    client,
    out: Path,
    on_status,
    on_stage_progress=None,
) -> list[dict]:
    """Select scaffolds non-interactively by explicit index, `--all-scaffolds`, or top-N ranking."""
    if settings.scaffold_index is not None:
        if settings.scaffold_index >= len(scaffolds):
            raise ValueError("Scaffold index is outside the available final scaffolds")
        chosen = [scaffolds[settings.scaffold_index]]
    elif settings.all_scaffolds:
        chosen = scaffolds
    else:
        ranked = rank_scaffolds(scaffolds, client, out, on_status, on_stage_progress)
        count = settings.top_scaffolds or 1
        RedesignChoice(count, settings.num_diffusion).validate(len(ranked))
        chosen = ranked[:count]
    write_json(out / "selected_scaffolds.json", chosen)
    return chosen


def interactive_select_scaffolds(
    scaffolds: list[dict],
    settings: DesignConfig,
    target: Target,
    target_path: Path,
    out: Path,
    runtime,
    seed: int,
    device,
    folding,
    manifest: dict,
    on_status,
    on_scaffolds,
    on_complexa_progress=None,
    on_stage_progress=None,
) -> tuple[list[dict], dict[str, dict[str, float]], DesignConfig]:
    """Fold and rank scaffolds, prompting `on_scaffolds` (and running `MoreScaffolds` batches if requested)."""
    batch_number = 1
    while True:
        ranked = rank_scaffolds(scaffolds, folding, out, on_status, on_stage_progress)
        manifest.update(status="awaiting_selection", scaffold_count=len(scaffolds))
        write_json(out / "run.json", manifest)
        choice = on_scaffolds(ranked)
        if choice is None:
            raise KeyboardInterrupt
        if not isinstance(choice, MoreScaffolds):
            break
        if type(choice.count) is not int or choice.count < 1:
            raise ValueError("Additional scaffold count must be a positive integer")
        if device.type != "cuda":
            raise ValueError(
                "Generating additional Complexa scaffolds requires CUDA"
            )
        if runtime is None:
            runtime = get_runtime(
                "complexa",
                models_dir=settings.models_dir or load_setup().models_dir,
            )
        batch_number += 1
        batch_out = out / "scaffold_batches" / f"batch-{batch_number:03d}"
        batch_out.mkdir(parents=True, exist_ok=False)
        batch_seed = (seed + batch_number - 1) % 2**32
        manifest.setdefault("additional_scaffold_batches", []).append(
            {
                "directory": str(batch_out),
                "count": choice.count,
                "seed": batch_seed,
            }
        )
        manifest["status"] = "running"
        write_json(out / "run.json", manifest)
        on_status(
            f"1/5 · Starting Complexa for {choice.count} additional scaffolds…"
        )
        new_pdbs = run_complexa(
            replace(settings, num_complexa=choice.count),
            target,
            target_path,
            batch_out,
            runtime,
            batch_seed,
            device,
            on_status,
            on_progress=on_complexa_progress,
        )
        scaffolds.extend(
            describe_scaffolds(
                new_pdbs,
                target,
                batch_out,
                on_status,
                start_index=len(scaffolds),
            )
        )
        write_json(out / "scaffolds.json", scaffolds)

    selected = choice.select(ranked)
    property_targets = requested_properties(
        selected, choice.properties, edit_residues=settings.edit_residues
    )
    settings = replace(
        settings,
        top_scaffolds=choice.top_scaffolds,
        num_diffusion=choice.sequences_per_scaffold,
        scaffold_index=None,
        all_scaffolds=False,
    )
    manifest["selected_scaffold_ranks"] = list(
        choice.ranks or range(1, choice.top_scaffolds + 1)
    )
    write_json(out / "selected_scaffolds.json", selected)
    return selected, property_targets, settings


def encode_scaffolds(
    scaffolds: list[dict], out: Path, foldseek: str, on_status
) -> None:
    """Extract binder chain `B` for each selected scaffold and encode its 3Di string with Foldseek."""
    on_status(
        f"3/5 · Encoding {len(scaffolds)} selected scaffold backbones with Foldseek…"
    )
    directory = out / "scaffold_binders"
    directory.mkdir()
    for scaffold in scaffolds:
        binder_path = directory / f"{scaffold['id']}.pdb"
        write_chain(Path(scaffold["pdb"]), binder_path, "B")
        scaffold["binder_pdb"] = str(binder_path)
    database = out / "scaffold_foldseek"
    with (out / "foldseek.log").open("w") as log:
        result = subprocess.run(
            [
                foldseek,
                "createdb",
                str(directory),
                str(database),
                "--chain-name-mode",
                "1",
                "--threads",
                "4",
                "-v",
                "1",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"Foldseek failed. Details: {out / 'foldseek.log'}")
    from ...evaluation.foldseek import read_foldseek_records

    records = read_foldseek_records(database)
    for scaffold in scaffolds:
        matching = []
        for key, value in records.items():
            name = key
            for suffix in (".pdb_B", ".pdb", "_B"):
                name = name.removesuffix(suffix)
            if name == scaffold["id"]:
                matching.append(value)
        if len(matching) != 1 or matching[0][0] != scaffold["sequence"]:
            raise ValueError(
                f"Foldseek sequence does not match chain B of {scaffold['id']}"
            )
        tokens = matching[0][1].upper()
        if len(tokens) != len(scaffold["sequence"]):
            raise ValueError(f"Foldseek returned incomplete 3Di for {scaffold['id']}")
        encode_3di(tokens, pad_to=70)  # Validate the alphabet before sampling.
        scaffold["input_3di"] = tokens
    write_json(out / "selected_scaffolds.json", scaffolds)
