"""Summarize candidates from an evaluation run."""

from __future__ import annotations

from .common import (
    MODES,
    prostt5_targets,
    STD,
    new_run,
    read,
    require,
    save,
)
from .metrics import (
    pipeline_summary,
    property_summary,
    structure_summary,
    unconditional_summary,
)


def score_run(run, out, dataset=None, metrics=None, *, folds=None, on_progress=None, on_status=None):
    status = on_status or (lambda message: print(message, flush=True))
    samples = read(run / "samples.json")
    mode, rows = samples["mode"], samples["records"]
    require(mode in MODES, f"Unsupported evaluation mode: {mode}")
    if mode in {"full-design", "inpainting"}:
        require(samples.get("complexes"), "This run has no complex inputs; start a new evaluation with a PDB folder")
    protocol = {
        "operation": "score",
        "mode": mode,
        "samples": str((run / "samples.json").resolve()),
        "folds": str(folds.resolve()) if folds is not None else None,
    }
    if mode == "unconditional":
        from .unconditional import ESMC_METRICS, selected_metrics

        metrics = tuple(key for key in selected_metrics(metrics) if key not in ESMC_METRICS)
        require(metrics, "Choose at least one sequence metric to summarize")
        protocol["metrics"] = list(metrics)
    else:
        require(metrics is None, "Metric selection applies to unconditional evaluation")
    out = new_run(out, protocol)
    if mode == "unconditional":
        test = [r["native_sequence"] for r in prostt5_targets()]
        result = unconditional_summary(rows, test, dataset, metrics, on_progress=on_progress, on_status=on_status)
    elif mode == "properties":
        result = property_summary(rows, samples.get("property_inputs", {}).get("training_std", STD))
    else:
        scored = read((folds if folds is not None else run) / "scores.json")
        require(scored["mode"] == mode, "Generation and folding modes differ")
        if mode == "structure":
            result = {
                arm: structure_summary(
                    [r for r in scored["records"] if r["arm"] == arm], scored["native"]
                )
                for arm in ("baseline", "guidance5")
            }
        else:
            baselines = {case['ppi_id']: case for case in samples['complexes']}
            result = pipeline_summary(scored["records"], mode == "inpainting", baselines)
    save(out / "summary.json", result)
    if mode == "unconditional":
        omitted = set()
        if "exact-novelty" not in metrics:
            omitted.add("exact_train_novel")
        if "nearest-training" not in metrics:
            omitted.update(("nearest_train_sequence", "nearest_train_normalized_levenshtein"))
        rows = [{key: value for key, value in row.items() if key not in omitted} for row in rows]
        save(out / "sequence_metrics.json", {"records": rows, "summary": result})
    status(f"Metrics saved to {out}")
    return result
