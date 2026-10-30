"""Fresh benchmark structure prediction and scoring with the persistent folding service."""

from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path

from .common import (
    prostt5_targets,
    new_run,
    read,
    require,
    runtime_versions,
    save,
    sha256,
)
from .metrics import (
    confidence,
    threedi_agreement,
)


def resolve_foldseek(value, models_dir=None):
    from ..setup.tools import find_foldseek

    executable = find_foldseek(value, models_dir=models_dir)
    require(
        executable is not None,
        "Foldseek is unavailable; run mubind setup --only foldseek --yes",
    )
    version = subprocess.run(
        [str(executable), "version"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    ).stdout.strip()
    require(
        version.startswith("941cd33") or version == "10-941cd33",
        f"Paper scoring requires Foldseek 10-941cd33; found {version!r}. Select that release with --foldseek.",
    )
    return executable


def encode_structures(rows, folder, foldseek):
    from ..evaluation.foldseek import read_foldseek_records, write_mmcif_chain_pdb

    folder.mkdir(parents=True, exist_ok=True)
    pdbs = folder / "pdb"
    pdbs.mkdir(exist_ok=True)
    for row in rows:
        write_mmcif_chain_pdb(
            Path(row["structure"]), pdbs / (row["case_id"] + ".pdb"), chain_id="B"
        )
    database = folder / "db"
    subprocess.run(
        [
            str(foldseek),
            "createdb",
            str(pdbs),
            str(database),
            "--chain-name-mode",
            "1",
            "--threads",
            "4",
            "-v",
            "1",
        ],
        check=True,
    )
    encoded = read_foldseek_records(database)
    require(
        len(encoded) == len(rows),
        "Foldseek record count differs from the folded candidate pool",
    )
    return encoded


def score_structure(rows, native, folder, foldseek):
    from tmtools import tm_align

    from ..evaluation.structure import _ca_trace

    targets = {r["target_id"]: r for r in prostt5_targets()}
    require(
        len(native) == 92 and all(r["fold_success"] for r in native),
        "All 92 native controls must fold successfully before reference scoring",
    )
    native_encoded = encode_structures(native, folder / "native", foldseek)
    native_coords, native_scores = {}, []
    for row in native:
        target = targets[row["target_id"]]
        aa, three = native_encoded[row["case_id"] + "_A"]
        xyz, seq = _ca_trace(Path(row["structure"]), chain_id="B")
        require(
            aa == seq == target["native_sequence"], "Native refold sequence mismatch"
        )
        native_coords[row["target_id"]] = xyz
        native_scores.append(
            {
                **row,
                "native_sequence": seq,
                "native_3di": three,
                "target_3di": target["target_3di"],
                **threedi_agreement(target["target_3di"], three),
            }
        )
    scores = []
    for arm in ("baseline", "guidance5"):
        pool = [r for r in rows if r["arm"] == arm and r["fold_success"]]
        encoded = encode_structures(pool, folder / arm, foldseek) if pool else {}
        for row in (r for r in rows if r["arm"] == arm):
            if not row["fold_success"]:
                scores.append(row)
                continue
            target = targets[row["target_id"]]
            aa, three = encoded[row["case_id"] + "_A"]
            xyz, seq = _ca_trace(Path(row["structure"]), chain_id="B")
            require(aa == seq == row["generated"], "Generated refold sequence mismatch")
            aligned = tm_align(
                xyz, native_coords[row["target_id"]], seq, target["native_sequence"]
            )
            scores.append(
                {
                    **row,
                    "generated_3di": three,
                    "target_3di": target["target_3di"],
                    "tm_to_native_refold": float(aligned.tm_norm_chain2),
                    **threedi_agreement(target["target_3di"], three),
                }
            )
    return {"mode": "structure", "records": scores, "native": native_scores}


def score_complex(rows, baselines):
    from tmtools import tm_align

    from ..evaluation.structure import _ca_trace

    references = {}
    for row in rows:
        if not row.get("fold_success", True) or not row["sequence_valid"]:
            continue
        ident = row["ppi_id"]
        if ident not in references:
            baseline = baselines[ident]
            reference_path = Path(baseline['reference_binder_pdb'])
            require(sha256(reference_path) == baseline['reference_binder_sha256'], 'Input binder reference changed')
            references[ident] = _ca_trace(
                reference_path, chain_id=baseline['reference_binder_chain']
            )
        reference, ref_sequence = references[ident]
        require(
            ref_sequence == baselines[ident]["binder_observed_sequence"],
            "Reference sequence mismatch",
        )
        prediction, sequence = _ca_trace(Path(row["structure"]), chain_id="B")
        require(sequence == row["generated"], "Predicted binder sequence mismatch")
        # Normalize TM-score by the supplied reference binder's length.
        aligned = tm_align(reference, prediction, ref_sequence, sequence)
        row["folds"]["complex"].update(
            tm_score=float(aligned.tm_norm_chain1), rmsd=float(aligned.rmsd)
        )
    return rows


def validate_service(client, endpoint):
    """Check the pinned paper protocol, also before fresh generation."""
    protocol = client.protocol()
    status = client.status()
    reported = status.get("protocol", status)
    revision = reported.get("revision") or reported.get("model_revisions", {}).get(
        "biohub/ESMFold2-Fast"
    )
    require(
        revision == client.config.protocol["revision"],
        "Paper folding needs a service reporting the pinned ESMFold2 revision; use the managed runtime or expose its revision in /health",
    )
    expected_esmc = "45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a"
    esmc_revision = reported.get("esmc_revision") or reported.get(
        "model_revisions", {}
    ).get("biohub/ESMC-6B")
    if endpoint == "auto":
        # The managed runtime verifies both pinned snapshots during setup.
        esmc_revision = expected_esmc
    require(
        esmc_revision == expected_esmc,
        "External paper folding must report the pinned ESMC-6B revision in /health",
    )
    require(
        reported.get("num_independent_trunks", 1) == 1,
        "Paper folding uses one independent trunk",
    )
    protocol.update(esmc_revision=esmc_revision, num_independent_trunks=1)
    return protocol


def fold_run(run, endpoint="auto", models_dir=None, foldseek="foldseek", device="auto", *, out=None, on_progress=None, on_status=None):
    from ..evaluation.config import FoldingConfig
    from ..evaluation.folding import FoldingClient

    status = on_status or (lambda message: print(message, flush=True))
    run = run.resolve()
    samples = read(run / "samples.json")
    mode = samples["mode"]
    require(
        mode in ("structure", "full-design", "inpainting"),
        "This mode does not require structure prediction",
    )
    if mode != "structure":
        require(samples.get("complexes"), "This run has no complex inputs; start a new evaluation with a PDB folder")
    baselines = {r["ppi_id"]: r for r in samples.get("complexes", [])}
    executable = resolve_foldseek(foldseek, models_dir) if mode == "structure" else None
    config = FoldingConfig(
        endpoint=endpoint,
        models_dir=None if models_dir is None else str(models_dir),
        device=device,
    )
    client = FoldingClient(config)
    if mode != "structure":
        client.check_complex_ready()
    client.start(on_status=status)
    protocol = validate_service(client, endpoint)
    work = new_run(
        out if out is not None else run / "folding",
        {
            "operation": "fold",
            "mode": mode,
            "samples": str(run / "samples.json"),
            "protocol": protocol,
            "scoring_versions": runtime_versions("numpy", "biopython", "tmtools"),
            "foldseek": str(executable) if executable else None,
        },
    )
    targets = prostt5_targets() if mode == "structure" else []
    total_folds = len(samples["records"]) + len(targets)
    completed_folds = 0

    def report_fold(row):
        nonlocal completed_folds
        completed_folds += 1
        if on_progress is not None:
            on_progress({
                "stage": "folding", "completed": completed_folds, "total": total_folds,
                "detail": f"{completed_folds:,}/{total_folds:,} fold attempts", "minimum": 1,
                "failed": not row["fold_success"],
            })
        return row

    def fold(row, name, target=None):
        if on_progress is not None:
            on_progress({
                "stage": "folding", "completed": completed_folds, "total": total_folds,
                "description": "Folding reference proteins" if name.startswith("native/") else "Folding candidates",
                "detail": f"{completed_folds:,}/{total_folds:,} fold attempts", "minimum": 1,
            })
        destination = work / name
        destination.mkdir(parents=True, exist_ok=True)
        result = {"fold_success": False}
        if row["sequence_valid"]:
            structure = destination / "prediction.cif"
            job = {
                "out": str(structure),
                "metrics_json": str(destination / "metrics.json"),
                "seed": 0,
            }
            try:
                if target is None:
                    metrics = client.fold_monomer({**job, "sequence": row["generated"]})
                else:
                    metrics = client.fold_complex(
                        {**job, "target": target, "binder": row["generated"]}
                    )
                for key in ("plddt",) if target is None else ("binder_plddt", "iptm"):
                    confidence(metrics.get(key), key)
                if target is not None:
                    pae = metrics.get("interface_pae")
                    require(
                        isinstance(pae, (int, float))
                        and math.isfinite(pae)
                        and pae >= 0,
                        "Folding service returned invalid interface PAE",
                    )
                from ..evaluation.structure import _ca_trace

                _, sequence = _ca_trace(structure, chain_id="B")
                require(
                    sequence == row["generated"],
                    "Folding service returned a different binder sequence",
                )
                result.update(
                    fold_success=True,
                    structure=str(structure),
                    metrics=metrics,
                )
            except Exception as exc:  # noqa: BLE001 — keep failed attempts in the reported candidate pool
                result["error"] = f"{type(exc).__name__}: {exc}"
        else:
            result["error"] = "Invalid sequence canvas"
        save(destination / "result.json", result)
        return report_fold({**row, **result})

    rows, native = [], []
    if mode == "structure":
        for target in targets:
            row = {
                "case_id": target["target_id"] + "_native",
                "target_id": target["target_id"],
                "generated": target["native_sequence"],
                "sequence_valid": True,
            }
            result = fold(row, "native/" + row["case_id"])
            if result["fold_success"]:
                result["plddt"] = result["metrics"]["plddt"]
            native.append(result)
    for index, original in enumerate(samples["records"]):
        row = dict(original)
        if mode == "structure":
            row["target_id"] = row["threedi_target_id"]
            row = fold(row, row["arm"] + "/" + row["case_id"])
            if row["fold_success"]:
                row["plddt"] = row["metrics"]["plddt"]
        else:
            row = fold(
                row,
                row["ppi_id"] + "/" + row["candidate_id"],
                baselines[row["ppi_id"]]["target_sequence"],
            )
            row["folds"] = (
                {"complex": {"metrics": row["metrics"]}} if row["fold_success"] else {}
            )
        rows.append(row)
        if on_progress is None and (index + 1) % 10 == 0:
            print(
                f"Folded {index + 1}/{len(samples['records'])} candidate attempts",
                flush=True,
            )
    save(work / "folded.json", {"mode": mode, "records": rows, "native": native})
    status("Measuring structural recovery…")
    if mode == "structure":
        with tempfile.TemporaryDirectory(prefix="paper-score-") as temporary:
            result = score_structure(rows, native, Path(temporary), executable)
    else:
        result = {"mode": mode, "records": score_complex(rows, baselines)}
    save(work / "scores.json", result)
    # Persistent services deliberately remain loaded.
    return {"directory": str(work), "attempts": len(rows), "folded": sum(r["fold_success"] for r in rows)}
