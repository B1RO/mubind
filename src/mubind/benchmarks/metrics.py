"""Paper metrics, with explicit candidate selection and denominators."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from .common import DATA, PROPERTIES, STD, prostt5_targets, training_corpus, require, sha256


def properties(sequence):
    from ..common.properties import sequence_properties

    return sequence_properties(sequence, property_definition="omegamp_modlamp_v1")


def confidence(value, name):
    require(
        isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1,
        f"{name} must be a finite confidence value on the 0–1 scale",
    )
    return value


def valid_canvas(record):
    canvas = record["decoded_canvas"]
    sequence = canvas.strip("-")
    valid = bool(sequence) and set(sequence) <= set("ACDEFGHIKLMNPQRSTVWY")
    require(len(canvas) == 70, "The paper uses a 70-position canvas")
    require(
        record["sequence_valid"] == valid,
        "Stored validity disagrees with the decoded canvas",
    )
    require(
        record["generated"] == sequence.replace("-", ""),
        "Stored sequence disagrees with the canvas",
    )
    return valid


def ks(first, second):
    first, second = np.sort(first), np.sort(second)
    require(len(first) > 0 and len(second) > 0, "KS requires two nonempty samples")
    x = np.unique(np.concatenate((first, second)))
    return float(
        np.max(
            np.abs(
                np.searchsorted(first, x, side="right") / len(first)
                - np.searchsorted(second, x, side="right") / len(second)
            )
        )
    )


def composition(sequences):
    counts = Counter("".join(sequences))
    total = sum(counts.values())
    require(total > 0, "Cannot compute composition of empty sequences")
    return np.array([counts[a] / total for a in "ACDEFGHIKLMNPQRSTVWY"])


def js_bits(first, second):
    midpoint = (first + second) / 2
    return sum(
        float(0.5 * np.sum(v[v > 0] * np.log2(v[v > 0] / midpoint[v > 0])))
        for v in (first, second)
    )


def pairwise(sequences):
    from rapidfuzz import process
    from rapidfuzz.distance import Levenshtein

    require(len(sequences) > 1, "Pairwise diversity requires at least two sequences")
    distances = process.cdist(
        sequences,
        sequences,
        scorer=Levenshtein.normalized_distance,
        dtype=np.float64,
        workers=4,
    )
    return float(distances[np.triu_indices(len(sequences), 1)].mean())


def training_sequences(dataset):
    """Decode the complete, verified paper corpus once for either novelty metric."""
    expected = training_corpus()
    require(
        sha256(dataset / "train/aa.npy") == expected["sha256"],
        "The training sequence array differs from the paper corpus",
    )
    matrix = np.load(dataset / "train/aa.npy", mmap_mode="r")
    require(len(matrix) == expected["rows"], "Expected all 1,505,527 training rows")
    alphabet = "ALGVSREDTIPKFQNYMHWC"
    translation = bytes([ord(alphabet[t - 3]) if 3 <= t <= 22 else 0 for t in range(256)])
    training = sorted({
        row.tobytes().translate(translation).rstrip(b"\0").decode()
        for row in matrix
    })
    require(len(training) == expected["unique_sequences"], "Training sequences differ from the paper corpus")
    require(all(s and "\0" not in s for s in training), "Invalid training tokens")
    return training


def unconditional_summary(records, test_sequences, dataset=None, metrics=None, *, on_progress=None, on_status=None):
    from .unconditional import CORPUS_METRICS, selected_metrics

    selected = set(selected_metrics(metrics))
    status = on_status or (lambda message: None)
    require(len(records) == 1024, "Unconditional protocol requires all 1,024 attempts")
    require(
        len({r["case_id"] for r in records}) == len(records),
        "Repeated unconditional candidate ID",
    )
    valid = [r for r in records if valid_canvas(r)]
    sequences = [r["generated"] for r in valid]
    # These counts describe the shared filtering and denominators, even when the
    # validity metric itself was not requested.
    result = {"attempts": len(records), "valid": len(valid)}
    if "validity" in selected:
        result["valid_fraction"] = len(valid) / len(records)
    if selected - {"validity"}:
        require(sequences, "Selected sequence metrics need at least one valid output")
    if "distinctness" in selected:
        result.update(unique=len(set(sequences)), unique_fraction=len(set(sequences)) / len(valid))
    training = None
    if dataset is not None and selected & CORPUS_METRICS:
        status("Reading paper training sequences…")
        training = training_sequences(dataset)
    if "exact-novelty" in selected:
        status("Checking exact novelty…")
        if training is not None:
            training_set = set(training)
            for row in valid:
                row["exact_train_novel"] = row["generated"] not in training_set
        require(
            all("exact_train_novel" in r for r in valid),
            "Fresh exact novelty needs --dataset with the complete prepared training split",
        )
        for row in valid:
            if training is None and "nearest_train_sequence" in row:
                require(
                    row["exact_train_novel"] == (row["generated"] != row["nearest_train_sequence"]),
                    "Exact novelty disagrees with the nearest training sequence",
                )
        result["exact_novel"] = sum(r["exact_train_novel"] for r in valid)
        result["exact_novel_fraction"] = result["exact_novel"] / len(valid)
    if "nearest-training" in selected:
        from rapidfuzz.distance import Levenshtein

    if "nearest-training" in selected and training is not None:
        from rapidfuzz import process

        def report_nearest(completed):
            if on_progress is not None:
                on_progress({
                    "stage": "metrics", "work": "nearest-training",
                    "description": "Finding nearest training sequences",
                    "completed": completed, "total": len(valid),
                    "detail": f"{completed:,}/{len(valid):,} sequences", "minimum": 1,
                })
            elif completed:
                print(f"Nearest training sequence: {completed}/{len(valid)}", flush=True)

        report_nearest(0)
        for start in range(0, len(valid), 16):
            rows = valid[start : start + 16]
            distances = process.cdist(
                [r["generated"] for r in rows],
                training,
                scorer=Levenshtein.normalized_distance,
                dtype=np.float64,
                workers=4,
            )
            for row, i in zip(rows, distances.argmin(axis=1), strict=True):
                row["nearest_train_sequence"] = training[int(i)]
                row["nearest_train_normalized_levenshtein"] = (
                    Levenshtein.normalized_distance(row["generated"], training[int(i)])
                )
            report_nearest(min(start + 16, len(valid)))
    if "nearest-training" in selected:
        require(
            all("nearest_train_sequence" in r for r in valid),
            "Fresh nearest-training scoring needs --dataset with the complete prepared training split",
        )
        for r in valid:
            measured = Levenshtein.normalized_distance(r["generated"], r["nearest_train_sequence"])
            require(
                math.isclose(measured, r["nearest_train_normalized_levenshtein"], abs_tol=1e-12),
                "Stored nearest-training distance disagrees with its sequence pair",
            )
        result["nearest_training_distance"] = float(
            np.mean([r["nearest_train_normalized_levenshtein"] for r in valid])
        )
        result["nearest_training_search"] = (
            "recomputed" if training is not None
            else "archived complete-corpus search; sequence-pair distances rechecked"
        )
    if "diversity" in selected:
        status("Computing pairwise sequence diversity…")
        result["pairwise_distance"] = pairwise(sequences)
        result["test_pairwise_distance"] = pairwise(test_sequences)
    requested_properties = [p for p in PROPERTIES if p in selected]
    if requested_properties:
        status("Comparing selected property distributions…")
        # Charge and hydrophobicity share the paper's property extraction;
        # only the requested distribution comparisons are computed and saved.
        values = properties if selected & {"charge", "hydrophobicity"} else lambda s: {"length": len(s)}
        generated_values = [values(s) for s in sequences]
        test_values = [values(s) for s in test_sequences]
        result["properties"] = {
            p: {
                "generated_mean": float(np.mean([r[p] for r in generated_values])),
                "test_mean": float(np.mean([r[p] for r in test_values])),
                "ks": ks([r[p] for r in generated_values], [r[p] for r in test_values]),
            }
            for p in requested_properties
        }
    if "composition" in selected:
        status("Comparing amino-acid composition…")
        result["composition_js_bits"] = js_bits(composition(sequences), composition(test_sequences))
    return result


def property_summary(records, training_std=STD):
    require(len(training_std) == 3 and all(math.isfinite(value) and value > 0 for value in training_std),
            "Training property standard deviations must be three positive finite values")
    require(len(records) == 4000, "Property protocol requires all 4,000 attempts")
    require(
        len({r["case_id"] for r in records}) == len(records),
        "Repeated property candidate ID",
    )
    grouped = defaultdict(list)
    for row in records:
        valid_canvas(row)
        grouped[row["property_conditioning"]].append(row)
    masks = {
        "length_only": "100",
        "charge_only": "010",
        "hydrophobicity_only": "001",
        "all_properties": "111",
    }
    require(set(grouped) == set(masks), "Expected the four paper property settings")
    result, exact, total = {}, 0, 0
    for name, mask in masks.items():
        rows = grouped[name]
        require(len(rows) == 1000, f"{name} must have 1,000 attempts")
        require(
            all(r["condition_mask"] == mask for r in rows), "Property mask mismatch"
        )
        valid = [r for r in rows if r["sequence_valid"]]
        require(valid, f"No valid outputs in {name}; property error is undefined")
        measured = [properties(r["generated"]) for r in valid]
        metrics = {}
        for i, prop in enumerate(PROPERTIES):
            if mask[i] == "0":
                continue
            errors = [
                abs(v[prop] - r[f"requested_{prop}"]) for v, r in zip(measured, valid)
            ]
            metrics[prop] = {
                "mae": float(np.mean(errors)),
                "standardized_mae": float(np.mean(errors)) / training_std[i],
            }
            if prop == "length":
                exact += sum(e == 0 for e in errors)
                total += len(errors)
        result[name] = {"attempts": len(rows), "valid": len(valid), "metrics": metrics}
    return {
        "settings": result,
        "length_exact": exact,
        "length_requested_valid": total,
        "length_exact_fraction": exact / total,
        "training_std": dict(zip(PROPERTIES, training_std)),
    }


def substitution_matrix():
    lines = [
        line.split()
        for line in (DATA / "3di_matrix.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    alphabet = lines[0]
    return {
        (row[0], column): float(value)
        for row in lines[1:]
        for column, value in zip(alphabet, row[1:], strict=True)
    }


def threedi_agreement(requested, realized):
    matrix = substitution_matrix()
    require(
        set(requested + realized) <= set("ACDEFGHIKLMNPQRSTVWY"),
        "Unknown 3Di state; cannot silently omit positions",
    )
    denominator = max(len(requested), len(realized))
    require(denominator > 0, "Empty 3Di comparison")
    matches = sum(a == b for a, b in zip(requested, realized))
    soft = (
        sum(
            max(0, min(1, 2 * matrix[a, b] / (matrix[a, a] + matrix[b, b])))
            for a, b in zip(requested, realized)
        )
        / denominator
    )
    return {"identity": matches / denominator, "matrix_soft_identity": soft}


def structure_summary(rows, native):
    targets = {r["target_id"] for r in prostt5_targets()}
    groups = defaultdict(list)
    for row in rows:
        groups[row["target_id"]].append(row)
    require(set(groups) == targets, "Structure evaluation must include all 92 targets")
    selected, missing = [], []
    for target, candidates in groups.items():
        require(
            len(candidates) == 10
            and {r["candidate_replicate"] for r in candidates} == set(range(1, 11)),
            f"Missing or repeated replicate for {target}",
        )
        eligible = [r for r in candidates if r["sequence_valid"] and r["fold_success"]]
        if not eligible:
            missing.append(target)
            continue
        for row in eligible:
            confidence(row["plddt"], "plddt")
        winner = min(eligible, key=lambda r: (-r["plddt"], r["candidate_replicate"]))
        realized = winner["generated_3di"]
        selected.append({**winner, **threedi_agreement(winner["target_3di"], realized)})
    require(selected, "No successfully folded structural candidates")
    require(
        not missing,
        f"No eligible structural candidate for {missing}; do not average a smaller panel",
    )
    require(
        {r["target_id"] for r in native} == targets and len(native) == 92,
        "Expected one native refold per test target",
    )
    native_scores = [
        threedi_agreement(r["target_3di"], r["native_3di"]) for r in native
    ]
    return {
        "targets": len(targets),
        "attempts": len(rows),
        "selected": len(selected),
        "missing_targets": missing,
        "folded": sum(r["fold_success"] for r in rows),
        "mean": {
            key: float(np.mean([r[key] for r in selected]))
            for key in ("identity", "matrix_soft_identity", "tm_to_native_refold")
        },
        "native_mean": {
            key: float(np.mean([r[key] for r in native_scores]))
            for key in ("identity", "matrix_soft_identity")
        },
        "selected_ids": [r["case_id"] for r in selected],
    }


def annotate_pipeline(row, baseline):
    valid_canvas(row)
    row["generated_properties"] = properties(row["generated"])
    row["property_absolute_errors"] = {
        p: abs(row["generated_properties"][p] - baseline["requested_properties"][p])
        for p in PROPERTIES
    }
    if "edit_mask" in row:
        sequence, mask = baseline["binder_observed_sequence"], row["edit_mask"]
        canvas = row["decoded_canvas"]
        require(len(mask) == len(sequence), "Inpainting mask length mismatch")
        retained = sum(a != canvas[i] for i, a in enumerate(sequence) if mask[i] == "0")
        changed = sum(a != canvas[i] for i, a in enumerate(sequence) if mask[i] == "1")
        layout = "-" not in canvas[: len(sequence)] and set(
            canvas[len(sequence) :]
        ) <= {"-"}
        row.update(
            retained_changes_count=retained,
            edited_changes_count=changed,
            edited_residue_count=mask.count("1"),
            fixed_residues_preserved=retained == 0,
            fixed_layout_preserved=layout,
            completion_specification_preserved=retained == 0 and layout,
        )


def selection_score(row):
    m = row["folds"]["complex"]["metrics"]
    e = row["property_absolute_errors"]
    for key in ("binder_plddt", "iptm"):
        confidence(m.get(key), key)
    # The adopted score uses these rounded constants, as recorded in its protocol.
    return (
        0.8 * m["binder_plddt"]
        + 0.2 * m["iptm"]
        - 0.02 * (e["charge"] / 4.508427 + e["hydrophobicity"] / 0.204651)
    )


def pipeline_summary(rows, inpainting=False, baselines=None):
    require(baselines, "Redesign and inpainting scoring require the run’s complex inputs")
    grouped = defaultdict(list)
    for row in rows:
        require(row["ppi_id"] in baselines, "Unexpected pipeline target")
        annotate_pipeline(row, baselines[row["ppi_id"]])
        grouped[row["ppi_id"]].append(row)
    require(
        set(grouped) == set(baselines),
        "Pipeline comparison requires every input complex",
    )
    count = 20 if inpainting else 50
    selected, missing, preserving = [], [], 0
    for ident in baselines:
        candidates = grouped[ident]
        require(
            len(candidates) == count
            and {r["candidate_id"] for r in candidates}
            == {f"candidate_{i:02d}" for i in range(1, count + 1)},
            f"Missing or duplicate candidates for {ident}",
        )
        preserving += sum(
            bool(r.get("completion_specification_preserved")) for r in candidates
        )
        eligible = [
            r
            for r in candidates
            if r["sequence_valid"]
            and "complex" in r.get("folds", {})
            and (not inpainting or r["completion_specification_preserved"])
        ]
        if not eligible:
            missing.append(ident)
            continue
        winner = min(
            eligible,
            key=lambda r: (-selection_score(r), int(r["candidate_id"].split("_")[-1])),
        )
        selected.append(winner)
    require(selected, "No eligible pipeline candidates")
    require(
        not missing,
        f"No eligible pipeline candidate for {missing}; do not average a smaller panel",
    )
    result = {
        "cases": len(baselines),
        "candidates": len(rows),
        "selected": len(selected),
        "missing_cases": missing,
        "mean_binder_tm": float(
            np.mean([r["folds"]["complex"]["tm_score"] for r in selected])
        ),
        "mean_iptm": float(
            np.mean([r["folds"]["complex"]["metrics"]["iptm"] for r in selected])
        ),
        "property_mae": {
            p: float(np.mean([r["property_absolute_errors"][p] for r in selected]))
            for p in PROPERTIES
        },
        "selected_ids": [r["ppi_id"] + "/" + r["candidate_id"] for r in selected],
    }
    if inpainting:
        changed = sum(r["edited_changes_count"] for r in selected)
        masked = sum(r["edited_residue_count"] for r in selected)
        result.update(
            context_preserving_candidates=preserving,
            changed_masked_residues=changed,
            masked_residues=masked,
            changed_masked_fraction=changed / masked,
        )
    return result
