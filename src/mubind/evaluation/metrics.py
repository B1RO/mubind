"""Generation metrics preserving the 300k run's names and denominators.

Source: prostt5_learned_inpainting_250k_to300k_20261007/source/validation.py.
No training, sampling, or folding side effects: structural checks use supplied services.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from ..common.properties import (
    DEFAULT_PROPERTY_DEFINITION,
    canonical_property_definition,
    sequence_properties,
)
from ..dataset.decode import decode_embedding_tokens
from ..training.tasks.config import REDESIGN_CONDITIONING_NAMES
from .panels import PROPERTY_STEERING_CONDITIONS

PROPERTY_NAMES = ("length", "charge", "hydrophobicity")
ValidationRecord = dict[str, Any]
CheckFunction = Callable[..., list[dict[str, Any]]]
SummaryFunction = Callable[[Sequence[ValidationRecord]], dict[str, Any]]
_CHECKS: dict[str, ValidationCheck] = {}


@dataclass(frozen=True)
class ValidationContext:
    normalized_embeddings: bool
    property_definition: str = DEFAULT_PROPERTY_DEFINITION
    services: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationCheck:
    score: CheckFunction
    summarize: SummaryFunction


def register_check(
    name: str,
    *,
    summarize: SummaryFunction,
) -> Callable[[CheckFunction], CheckFunction]:
    """Register one stable validation name."""

    def decorator(function: CheckFunction) -> CheckFunction:
        if name in _CHECKS:
            raise ValueError(f"validation check already registered: {name}")
        _CHECKS[name] = ValidationCheck(score=function, summarize=summarize)
        return function

    return decorator


def available_checks() -> tuple[str, ...]:
    return tuple(sorted(_CHECKS))


def require_checks(names: Iterable[str]) -> tuple[str, ...]:
    selected = tuple(str(name) for name in names)
    unknown = sorted(set(selected) - set(_CHECKS))
    if unknown:
        raise ValueError(
            f"unknown validation checks: {unknown}; available={list(available_checks())}"
        )
    if len(set(selected)) != len(selected):
        raise ValueError("validation checks must not contain duplicates")
    return selected


def _mean(values: Iterable[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else None


def _batch_size(tasks: Mapping[str, Any]) -> int:
    edit_mask = tasks.get("edit_mask")
    if not isinstance(edit_mask, torch.Tensor) or edit_mask.ndim != 2:
        raise ValueError("task batch must contain edit_mask [B,L]")
    return int(edit_mask.shape[0])


def _batch_item(value: Any, index: int) -> Any:
    if isinstance(value, torch.Tensor):
        item = value[index].detach().cpu()
        return item.item() if item.ndim == 0 else item
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value[index]
    raise ValueError("task metadata must be a tensor or per-item sequence")


def _base_records(
    tasks: Mapping[str, Any],
    outputs: Sequence[Mapping[str, Any]],
) -> list[ValidationRecord]:
    size = _batch_size(tasks)
    if len(outputs) != size:
        raise ValueError("generated outputs must align one-to-one with task batch")
    records: list[ValidationRecord] = []
    for index, output in enumerate(outputs):
        record: ValidationRecord = {
            "batch_index": index,
            "generated": str(output["generated"]),
        }
        if "dataset_index" in tasks:
            record["dataset_index"] = int(_batch_item(tasks["dataset_index"], index))
        if "validation_case_id" in tasks:
            record["case_id"] = str(_batch_item(tasks["validation_case_id"], index))
        if "layout_id" in tasks:
            record["layout_id"] = int(_batch_item(tasks["layout_id"], index))
        if "redesign_conditioning_id" in tasks:
            mode_id = int(_batch_item(tasks["redesign_conditioning_id"], index))
            record["redesign_conditioning_id"] = mode_id
            record["redesign_conditioning"] = (
                REDESIGN_CONDITIONING_NAMES[mode_id] if mode_id >= 0 else None
            )
        for metadata_name in (
            "property_conditioning",
            "threedi_conditioning",
            "threedi_pair_id",
            "threedi_target_id",
        ):
            if metadata_name in tasks:
                record[metadata_name] = str(_batch_item(tasks[metadata_name], index))
        if "condition_mask" in tasks:
            condition_mask = _batch_item(tasks["condition_mask"], index)
            if not isinstance(condition_mask, torch.Tensor) or condition_mask.shape != (
                3,
            ):
                raise ValueError("task condition_mask rows must contain three values")
            record["condition_mask"] = "".join(
                "1" if float(value) > 0.5 else "0" for value in condition_mask
            )
        for name in ("target_offset", "target_length", "visible_start", "visible_end"):
            if name in tasks:
                record[name] = int(_batch_item(tasks[name], index))
        if "candidate_replicate" in tasks:
            record["candidate_replicate"] = int(
                _batch_item(tasks["candidate_replicate"], index)
            )
        if "binder_aa" in tasks:
            record["native_binder"] = str(_batch_item(tasks["binder_aa"], index))
        if "target_aa" in tasks:
            record["target_sequence"] = str(_batch_item(tasks["target_aa"], index))
        edit_mask = tasks["edit_mask"][index].detach().cpu().bool().tolist()
        record["edit_mask"] = "".join("1" if value else "0" for value in edit_mask)
        seed_3di = tasks.get("seed_3di")
        if isinstance(seed_3di, torch.Tensor):
            visible_3di = seed_3di[index].detach().cpu().ne(0).tolist()
            record["visible_3di_mask"] = "".join(
                "1" if value else "0" for value in visible_3di
            )
            record["visible_3di_token_count"] = sum(visible_3di)
        records.append(record)
    return records


def _run_checks_with_timing(
    tasks: Mapping[str, Any],
    outputs: Sequence[Mapping[str, Any]],
    names: Iterable[str],
    *,
    normalized_embeddings: bool,
    property_definition: str = DEFAULT_PROPERTY_DEFINITION,
    services: Mapping[str, Any] | None = None,
) -> tuple[list[ValidationRecord], dict[str, Any]]:
    selected = require_checks(names)
    records_started = time.perf_counter()
    records = _base_records(tasks, outputs)
    record_building_seconds = time.perf_counter() - records_started
    context = ValidationContext(
        normalized_embeddings=normalized_embeddings,
        property_definition=canonical_property_definition(property_definition),
        services={} if services is None else services,
    )
    check_seconds: dict[str, float] = {}
    for name in selected:
        check_started = time.perf_counter()
        fields = _CHECKS[name].score(tasks, outputs, context)
        if len(fields) != len(records):
            raise RuntimeError(f"validation check {name!r} returned wrong record count")
        for record, additions in zip(records, fields, strict=True):
            overlap = set(record) & set(additions)
            if overlap:
                raise RuntimeError(
                    f"validation check {name!r} overwrote fields: {sorted(overlap)}"
                )
            record.update(additions)
        check_seconds[name] = time.perf_counter() - check_started
    return records, {
        "record_building_seconds": record_building_seconds,
        "checks": check_seconds,
    }


def run_checks(
    tasks: Mapping[str, Any],
    outputs: Sequence[Mapping[str, Any]],
    names: Iterable[str],
    *,
    normalized_embeddings: bool,
    services: Mapping[str, Any] | None = None,
) -> list[ValidationRecord]:
    """Run named checks and merge their aligned per-candidate fields."""
    records, _ = _run_checks_with_timing(
        tasks,
        outputs,
        names,
        normalized_embeddings=normalized_embeddings,
        services=services,
    )
    return records


def summarize_checks(
    records: Sequence[ValidationRecord],
    names: Iterable[str],
) -> dict[str, Any]:
    selected = require_checks(names)
    return {
        "candidate_count": len(records),
        "checks": {name: _CHECKS[name].summarize(records) for name in selected},
    }


def _summarize_sequence_validity(records: Sequence[ValidationRecord]) -> dict[str, Any]:
    return {
        "valid_count": sum(bool(record["sequence_valid"]) for record in records),
        "valid_fraction": _mean(
            float(bool(record["sequence_valid"])) for record in records
        ),
        "mean_internal_pad_count": _mean(
            float(record["internal_pad_count"]) for record in records
        ),
    }


@register_check("sequence_validity", summarize=_summarize_sequence_validity)
def _score_sequence_validity(
    tasks: Mapping[str, Any],
    outputs: Sequence[Mapping[str, Any]],
    context: ValidationContext,
) -> list[dict[str, Any]]:
    del tasks, context
    return [
        {
            "sequence_valid": bool(output["sequence_valid"]),
            "sequence_validity_reason": str(output["sequence_validity_reason"]),
            "generated_length": int(output["generated_length"]),
            "internal_pad_count": int(output["internal_pad_count"]),
            "decoded_canvas": str(output["decoded_canvas"]),
        }
        for output in outputs
    ]


def _property_mae_summary(records: Sequence[ValidationRecord]) -> dict[str, Any]:
    return {
        "candidate_count": len(records),
        "scored_candidate_count": sum(
            any(
                record.get(f"{name}_absolute_error") is not None
                for name in PROPERTY_NAMES
            )
            for record in records
        ),
        "mae": {
            name: _mean(
                float(record[f"{name}_absolute_error"])
                for record in records
                if record.get(f"{name}_absolute_error") is not None
            )
            for name in PROPERTY_NAMES
        },
        "standardized_mae": {
            name: _mean(
                float(record[f"{name}_standardized_absolute_error"])
                for record in records
                if record.get(f"{name}_standardized_absolute_error") is not None
            )
            for name in PROPERTY_NAMES
        },
    }


def _summarize_property_mae(records: Sequence[ValidationRecord]) -> dict[str, Any]:
    summary = _property_mae_summary(records)
    condition_names = [
        name
        for name, _ in PROPERTY_STEERING_CONDITIONS
        if any(record.get("property_conditioning") == name for record in records)
    ]
    if condition_names:
        summary["by_conditioning"] = {
            name: _property_mae_summary(
                [
                    record
                    for record in records
                    if record.get("property_conditioning") == name
                ]
            )
            for name in condition_names
        }
    return summary


@register_check("property_mae", summarize=_summarize_property_mae)
def _score_property_mae(
    tasks: Mapping[str, Any],
    outputs: Sequence[Mapping[str, Any]],
    context: ValidationContext,
) -> list[dict[str, Any]]:
    requested = tasks.get("c")
    if not isinstance(requested, torch.Tensor) or requested.shape != (len(outputs), 3):
        raise ValueError("property_mae requires task c [B,3]")
    property_std = tasks.get("property_std")
    if not isinstance(property_std, torch.Tensor) or property_std.shape != (3,):
        raise ValueError("property_mae requires training property_std [3]")
    standard_deviations = property_std.detach().float().cpu().tolist()
    condition_mask = tasks.get("condition_mask")
    if condition_mask is None:
        condition_mask = torch.ones_like(requested)
    if (
        not isinstance(condition_mask, torch.Tensor)
        or condition_mask.shape != requested.shape
    ):
        raise ValueError("property_mae condition_mask must match task c [B,3]")
    results: list[dict[str, Any]] = []
    for index, output in enumerate(outputs):
        requested_values = requested[index].detach().float().cpu().tolist()
        valid = bool(output["sequence_valid"])
        observed = (
            sequence_properties(
                str(output["generated"]),
                property_definition=context.property_definition,
            )
            if valid
            else None
        )
        fields: dict[str, Any] = {}
        for property_index, name in enumerate(PROPERTY_NAMES):
            conditioned = bool(condition_mask[index, property_index].item() > 0.5)
            target = float(requested_values[property_index])
            value = None if observed is None else float(observed[name])
            error = None if value is None or not conditioned else abs(value - target)
            fields[f"conditioned_{name}"] = conditioned
            fields[f"requested_{name}"] = target
            fields[f"observed_{name}"] = value
            fields[f"{name}_absolute_error"] = error
            fields[f"{name}_standardized_absolute_error"] = (
                None
                if error is None
                else error / max(float(standard_deviations[property_index]), 1e-8)
            )
        results.append(fields)
    return results


def _summarize_fixed_residues(records: Sequence[ValidationRecord]) -> dict[str, Any]:
    total = sum(int(record["fixed_residue_count"]) for record in records)
    matches = sum(int(record["fixed_residue_matches"]) for record in records)
    applicable = [
        record for record in records if int(record["fixed_residue_count"]) > 0
    ]
    return {
        "residue_accuracy": matches / total if total else None,
        "exact_preservation_fraction": _mean(
            float(bool(record["fixed_residues_preserved"])) for record in applicable
        ),
        "applicable_candidate_count": len(applicable),
    }


@register_check("fixed_residues", summarize=_summarize_fixed_residues)
def _score_fixed_residues(
    tasks: Mapping[str, Any],
    outputs: Sequence[Mapping[str, Any]],
    context: ValidationContext,
) -> list[dict[str, Any]]:
    reference = tasks.get("x")
    edit_mask = tasks.get("edit_mask")
    residue_mask = tasks.get("residue_mask")
    if not all(
        isinstance(value, torch.Tensor)
        for value in (reference, edit_mask, residue_mask)
    ):
        raise ValueError(
            "fixed_residues requires x, edit_mask, and residue_mask tensors"
        )
    if reference.shape[0] != len(outputs):
        raise ValueError("fixed_residues task/output batch mismatch")

    results: list[dict[str, Any]] = []
    for index, output in enumerate(outputs):
        reference_tokens = decode_embedding_tokens(
            reference[index],
            channels_first=True,
            allow_pad=True,
            normalized=context.normalized_embeddings,
        )
        generated_tokens = list(output["canvas_tokens"])
        fixed_mask = (
            residue_mask[index].detach().cpu().bool()
            & ~edit_mask[index].detach().cpu().bool()
        )
        positions = fixed_mask.nonzero(as_tuple=False).flatten().tolist()
        matches = sum(
            generated_tokens[position] == reference_tokens[position]
            for position in positions
        )
        results.append(
            {
                "fixed_residue_count": len(positions),
                "fixed_residue_matches": matches,
                "fixed_residues_preserved": matches == len(positions),
            }
        )
    return results


def _binder_structure_metrics(
    records: Sequence[ValidationRecord],
) -> dict[str, Any]:
    scored = [record for record in records if bool(record["binder_structure_scored"])]
    errors = [record for record in records if record.get("binder_structure_error")]
    # Pool all candidates, including failed evaluations as zero, just as in
    # failure_adjusted_mean_tm_score and the best-of-N dashboard quartiles.
    scores = [
        float(record["binder_tm_score"]) if record["binder_structure_scored"] else 0.0
        for record in records
    ]
    quartiles = (
        torch.quantile(
            torch.tensor(scores, dtype=torch.float64),
            torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64),
        ).tolist()
        if scores
        else [None, None, None]
    )
    return {
        "tm_score_q25": quartiles[0],
        "median_tm_score": quartiles[1],
        "tm_score_q75": quartiles[2],
        "scored_candidate_count": len(scored),
        "unscored_error_count": len(errors),
        "mean_tm_score": _mean(float(record["binder_tm_score"]) for record in scored),
        "tm_score_above_0_5_count": sum(
            float(record["binder_tm_score"]) > 0.5 for record in scored
        ),
        "tm_score_above_0_5_fraction": (
            sum(float(record["binder_tm_score"]) > 0.5 for record in scored)
            / len(records)
            if records
            else None
        ),
        "failure_adjusted_mean_tm_score": (
            sum(float(record["binder_tm_score"]) for record in scored) / len(records)
            if records
            else None
        ),
        "mean_rmsd": _mean(float(record["binder_tm_rmsd"]) for record in scored),
        "mean_aligned_residue_count": _mean(
            float(record["binder_tm_aligned_residue_count"]) for record in scored
        ),
        "mean_generated_monomer_plddt": _mean(
            float(record["generated_monomer_plddt"])
            for record in scored
            if record.get("generated_monomer_plddt") is not None
        ),
    }


def _backbone_structure_metrics(records: Sequence[ValidationRecord]) -> dict[str, Any]:
    """Calibrated metrics, with originals counted once per scaffold."""
    groups: dict[str, list[ValidationRecord]] = {}
    for index, record in enumerate(records):
        target = record.get(
            "threedi_target_id",
            record.get("dataset_index", index),
        )
        groups.setdefault(str(target), []).append(record)
    native = [
        next(record for record in group if record.get("native_monomer_scored"))
        for group in groups.values()
        if any(record.get("native_monomer_scored") for record in group)
    ]

    def generated_summary(group):
        scored = [r for r in group if r.get("binder_backbone_scored")]
        calibrated = [r for r in group if r.get("native_monomer_scored")]
        paired = [r for r in calibrated if r.get("binder_tm_gap") is not None]
        return {
            "candidate_count": len(group),
            "scored_count": len(scored),
            "scored_fraction": len(scored) / len(group) if group else None,
            "paired_tm_count": len(paired),
            "paired_tm_fraction": len(paired) / len(group) if group else None,
            "mean_tm_gap": _mean(float(r["binder_tm_gap"]) for r in paired),
            "failure_adjusted_mean_tm_gap": _mean(
                (
                    float(r["binder_tm_score"])
                    if r.get("binder_structure_scored")
                    else 0.0
                )
                - float(r["native_monomer_tm_score"])
                for r in calibrated
            ),
            # RMSD failures are missing, never a misleading perfect zero.
            "mean_ca_rmsd": _mean(float(r["binder_ca_rmsd"]) for r in scored),
            "mean_ca_lddt": _mean(float(r["binder_ca_lddt"]) for r in scored),
            "failure_adjusted_mean_ca_lddt": (
                sum(float(r["binder_ca_lddt"]) for r in scored) / len(group)
                if group
                else None
            ),
            "fraction_ca_rmsd_within_2a": (
                sum(float(r["binder_ca_rmsd"]) <= 2.0 for r in scored) / len(group)
                if group
                else None
            ),
        }

    per_target = []
    for target, group in groups.items():
        original = next((r for r in group if r.get("native_monomer_scored")), {})
        per_target.append(
            {
                "target_id": target,
                **generated_summary(group),
                "native_tm_score": original.get("native_monomer_tm_score"),
                "native_ca_rmsd": original.get("native_monomer_ca_rmsd"),
                "native_ca_lddt": original.get("native_monomer_ca_lddt"),
            }
        )
    return {
        **generated_summary(records),
        "native_scored_target_count": len(native),
        "native_scored_fraction": len(native) / len(groups) if groups else None,
        "mean_native_tm_score": _mean(
            float(r["native_monomer_tm_score"]) for r in native
        ),
        "mean_native_ca_rmsd": _mean(
            float(r["native_monomer_ca_rmsd"]) for r in native
        ),
        "mean_native_ca_lddt": _mean(
            float(r["native_monomer_ca_lddt"]) for r in native
        ),
        "native_fraction_ca_rmsd_within_2a": (
            sum(float(r["native_monomer_ca_rmsd"]) <= 2.0 for r in native) / len(groups)
            if groups
            else None
        ),
        "per_target": per_target,
    }


def _summarize_binder_structure(
    records: Sequence[ValidationRecord],
) -> dict[str, Any]:
    summary = _binder_structure_metrics(records)
    if any("binder_backbone_scored" in record for record in records):
        summary["backbone"] = _backbone_structure_metrics(records)
    labeled = [
        record
        for record in records
        if record.get("threedi_conditioning") in {"full_3di", "no_3di"}
    ]
    if not labeled:
        return summary

    grouped = {
        label: [
            record for record in labeled if record.get("threedi_conditioning") == label
        ]
        for label in ("full_3di", "no_3di")
    }
    by_pair: dict[str, dict[str, ValidationRecord]] = {}
    for record in labeled:
        pair_id = record.get("threedi_pair_id")
        if pair_id is None:
            continue
        by_pair.setdefault(str(pair_id), {})[str(record["threedi_conditioning"])] = (
            record
        )

    tm_gains: list[float] = []
    plddt_gains: list[float] = []
    for pair in by_pair.values():
        if set(pair) != {"full_3di", "no_3di"}:
            continue
        full = pair["full_3di"]
        control = pair["no_3di"]
        if not (
            bool(full["binder_structure_scored"])
            and bool(control["binder_structure_scored"])
        ):
            continue
        tm_gains.append(
            float(full["binder_tm_score"]) - float(control["binder_tm_score"])
        )
        if (
            full.get("generated_monomer_plddt") is not None
            and control.get("generated_monomer_plddt") is not None
        ):
            plddt_gains.append(
                float(full["generated_monomer_plddt"])
                - float(control["generated_monomer_plddt"])
            )

    full_metrics = _binder_structure_metrics(grouped["full_3di"])
    summary.update(full_metrics)
    summary["by_threedi_conditioning"] = {
        label: _binder_structure_metrics(group) for label, group in grouped.items()
    }
    summary["paired"] = {
        "scored_pair_count": len(tm_gains),
        "mean_tm_score_gain": _mean(tm_gains),
        "positive_tm_score_gain_fraction": _mean(float(gain > 0) for gain in tm_gains),
        "mean_generated_monomer_plddt_gain": _mean(plddt_gains),
    }
    return summary


@register_check("binder_structure", summarize=_summarize_binder_structure)
def _score_binder_structure(
    tasks: Mapping[str, Any],
    outputs: Sequence[Mapping[str, Any]],
    context: ValidationContext,
) -> list[dict[str, Any]]:
    service = context.services.get("binder_structure")
    if service is None:
        raise RuntimeError(
            "binder_structure requires a configured ESMFold2/TM-align service"
        )
    results = service(tasks, outputs)
    if not isinstance(results, list) or not all(
        isinstance(result, dict) for result in results
    ):
        raise TypeError("binder_structure service must return list[dict]")
    return results


def _summarize_realized_threedi(
    records: Sequence[ValidationRecord],
) -> dict[str, Any]:
    full_threedi = [
        record
        for record in records
        if record.get("threedi_conditioning") in {"full_3di", "partial_3di"}
    ]

    def summarize_subset(subset: Sequence[ValidationRecord]) -> dict[str, Any]:
        scored = [
            record for record in subset if bool(record.get("realized_threedi_scored"))
        ]
        matches = sum(int(record["realized_threedi_matches"]) for record in scored)
        token_count = sum(
            int(record["realized_threedi_token_count"]) for record in scored
        )
        expected_token_count = sum(
            int(record.get("visible_3di_token_count") or 0)
            if record.get("threedi_conditioning") == "partial_3di"
            else int(record.get("target_length") or 0)
            for record in subset
        )
        return {
            "candidate_count": len(subset),
            "scored_candidate_count": len(scored),
            "scored_fraction": len(scored) / len(subset) if subset else None,
            "unscored_error_count": sum(
                record.get("realized_threedi_error") is not None for record in subset
            ),
            "mean_identity": _mean(
                float(record["realized_threedi_identity"]) for record in scored
            ),
            # Match mean_identity's equal weighting of scored candidates.
            # Historical results without distance metrics remain unscored here.
            "mean_distance": _mean(
                float(record["realized_threedi_mean_distance"])
                for record in scored
                if record.get("realized_threedi_mean_distance") is not None
            ),
            "soft_agreement": _mean(
                float(record["realized_threedi_soft_agreement"])
                for record in scored
                if record.get("realized_threedi_soft_agreement") is not None
            ),
            "mean_generated_plddt": _mean(
                float(record["realized_threedi_plddt"])
                for record in subset
                if record.get("realized_threedi_plddt") is not None
            ),
            "mean_generated_ptm": _mean(
                float(record["realized_threedi_ptm"])
                for record in subset
                if record.get("realized_threedi_ptm") is not None
            ),
            "micro_identity": matches / token_count if token_count else None,
            # Missing or invalid generations contribute zero matches here,
            # preventing checkpoint scores from improving merely by dropping
            # difficult proteins from the denominator.
            "failure_adjusted_micro_identity": (
                matches / expected_token_count if expected_token_count else None
            ),
            "matched_token_count": matches,
            "token_count": token_count,
            "expected_token_count": expected_token_count,
        }

    summary = summarize_subset(full_threedi)
    summary["by_target_length"] = {
        "short_lt_96": summarize_subset(
            [
                record
                for record in full_threedi
                if int(record.get("target_length") or 0) < 96
            ]
        ),
        "long_ge_96": summarize_subset(
            [
                record
                for record in full_threedi
                if int(record.get("target_length") or 0) >= 96
            ]
        ),
    }
    return summary


@register_check("realized_threedi", summarize=_summarize_realized_threedi)
def _score_realized_threedi(
    tasks: Mapping[str, Any],
    outputs: Sequence[Mapping[str, Any]],
    context: ValidationContext,
) -> list[dict[str, Any]]:
    service = context.services.get("realized_threedi")
    if service is None:
        raise RuntimeError(
            "realized_threedi requires a configured ESMFold2/Foldseek service"
        )
    results = service(tasks, outputs)
    if not isinstance(results, list) or not all(
        isinstance(result, dict) for result in results
    ):
        raise TypeError("realized_threedi service must return list[dict]")
    return results


def summarize_threedi_best_of(
    records: Sequence[ValidationRecord],
    *,
    candidates_per_task: int,
) -> dict[str, Any]:
    """Summarize best folded-structure recovery per requested 3Di target."""
    if candidates_per_task < 1:
        raise ValueError("candidates_per_task must be positive")
    grouped: dict[str, list[ValidationRecord]] = {}
    for record in records:
        target_id = record.get("threedi_target_id")
        if target_id is None:
            continue
        grouped.setdefault(str(target_id), []).append(record)

    targets: list[dict[str, Any]] = []
    for target_id, candidates in grouped.items():
        scored = [
            record
            for record in candidates
            if record.get("binder_structure_scored") and record.get("binder_tm_score") is not None
        ]
        winner = (
            max(scored, key=lambda record: float(record["binder_tm_score"]))
            if scored
            else None
        )
        # Geometry belongs to the TM-selected sequence; never select a second
        # winner by RMSD/lDDT or substitute a runner-up with available geometry.
        winner_backbone_scored = (
            winner is not None
            and bool(winner.get("binder_backbone_scored"))
            and all(
                isinstance(winner.get(field), (int, float))
                and math.isfinite(float(winner[field]))
                for field in ("binder_ca_rmsd", "binder_ca_lddt")
            )
        )
        targets.append(
            {
                "target_id": target_id,
                "dataset_index": (
                    int(candidates[0]["dataset_index"])
                    if candidates and candidates[0].get("dataset_index") is not None
                    else None
                ),
                "candidate_count": len(candidates),
                "scored_candidate_count": len(scored),
                "best_tm_score": (
                    float(winner["binder_tm_score"]) if winner is not None else None
                ),
                "winner_backbone_scored": winner_backbone_scored,
                "winner_ca_rmsd": (
                    float(winner["binder_ca_rmsd"]) if winner_backbone_scored else None
                ),
                "winner_ca_lddt": (
                    float(winner["binder_ca_lddt"]) if winner_backbone_scored else None
                ),
                "best_candidate_replicate": (
                    int(winner["candidate_replicate"])
                    if winner is not None
                    and winner.get("candidate_replicate") is not None
                    else None
                ),
                "best_case_id": (
                    str(winner["case_id"])
                    if winner is not None and winner.get("case_id") is not None
                    else None
                ),
                "best_generated_monomer_plddt": (
                    float(winner["generated_monomer_plddt"])
                    if winner is not None
                    and winner.get("generated_monomer_plddt") is not None
                    else None
                ),
            }
        )

    best_scores = sorted(
        float(target["best_tm_score"])
        for target in targets
        if target["best_tm_score"] is not None
    )
    midpoint = len(best_scores) // 2
    median_best = (
        None
        if not best_scores
        else best_scores[midpoint]
        if len(best_scores) % 2
        else (best_scores[midpoint - 1] + best_scores[midpoint]) / 2
    )
    backbone_winners = [
        target for target in targets if target["winner_backbone_scored"]
    ]
    return {
        "requested_target_count": len(grouped),
        "scored_target_count": len(best_scores),
        "candidates_per_target": candidates_per_task,
        "tm_score_field": "binder_tm_score",
        "mean_best_tm_score": _mean(best_scores),
        "mean_winner_ca_rmsd": _mean(
            target["winner_ca_rmsd"] for target in backbone_winners
        ),
        "mean_winner_ca_lddt": _mean(
            target["winner_ca_lddt"] for target in backbone_winners
        ),
        "winner_tm_scored_fraction": len(best_scores) / len(targets)
        if targets
        else None,
        "winner_backbone_scored_target_count": len(backbone_winners),
        "winner_backbone_scored_fraction": len(backbone_winners) / len(targets)
        if targets
        else None,
        "failure_adjusted_mean_best_tm_score": sum(best_scores) / len(grouped)
        if grouped
        else None,
        "best_tm_above_0p5_fraction_all_targets": (
            sum(value > 0.5 for value in best_scores) / len(grouped)
            if grouped
            else None
        ),
        "median_best_tm_score": median_best,
        "min_best_tm_score": min(best_scores) if best_scores else None,
        "max_best_tm_score": max(best_scores) if best_scores else None,
        "best_tm_at_least_0p5_fraction": _mean(
            float(value >= 0.5) for value in best_scores
        ),
        "best_tm_at_least_0p7_fraction": _mean(
            float(value >= 0.7) for value in best_scores
        ),
        "best_tm_at_least_0p8_fraction": _mean(
            float(value >= 0.8) for value in best_scores
        ),
        "targets": targets,
    }


def _summarize_threedi_selection(
    records: Sequence[ValidationRecord],
    *,
    candidates_per_task: int,
    selection_metric: str,
) -> dict[str, Any]:
    """Select using an available inference signal, then score those structures.

    Reference scores and their availability never affect candidate selection.
    Ties keep the first candidate in the supplied order. Missing winner geometry
    remains missing; a runner-up is never substituted.
    """
    if candidates_per_task < 1:
        raise ValueError("candidates_per_task must be positive")
    if selection_metric not in {"mean_monomer_plddt", "realized_threedi_identity"}:
        raise ValueError(f"Unknown selection metric: {selection_metric}")

    def finite(value: Any) -> bool:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )

    def confidence(record: ValidationRecord) -> float | None:
        for key in ("generated_monomer_plddt", "realized_threedi_plddt"):
            if finite(record.get(key)):
                return float(record[key])
        return None

    def agreement(record: ValidationRecord) -> float | None:
        value = record.get("realized_threedi_identity")
        if record.get("realized_threedi_scored", True) and finite(value):
            return float(value)
        return None

    ranking_score = (
        confidence if selection_metric == "mean_monomer_plddt" else agreement
    )
    grouped: dict[str, list[ValidationRecord]] = {}
    for record in records:
        if record.get("threedi_target_id") is not None:
            grouped.setdefault(str(record["threedi_target_id"]), []).append(record)
    targets = []
    for target_id, candidates in grouped.items():
        eligible = [
            (record, ranking_score(record))
            for record in candidates
            if record.get("sequence_valid", True)
        ]
        eligible = [(record, score) for record, score in eligible if score is not None]
        winner, score = (
            max(eligible, key=lambda item: item[1]) if eligible else (None, None)
        )
        selected = winner or {}
        tm_scored = bool(selected.get("binder_structure_scored")) and finite(
            selected.get("binder_tm_score")
        )
        backbone_scored = bool(selected.get("binder_backbone_scored")) and all(
            finite(selected.get(field))
            for field in ("binder_ca_rmsd", "binder_ca_lddt")
        )
        targets.append(
            {
                "target_id": target_id,
                "candidate_count": len(candidates),
                "rankable_candidate_count": len(eligible),
                "selected_case_id": selected.get("case_id"),
                "selected_candidate_replicate": selected.get("candidate_replicate"),
                "selected_score": score,
                "selected_plddt": confidence(selected),
                "selected_threedi_identity": agreement(selected),
                "selected_threedi_mean_distance": (
                    float(selected["realized_threedi_mean_distance"])
                    if selected.get("realized_threedi_scored")
                    and finite(selected.get("realized_threedi_mean_distance"))
                    else None
                ),
                "selected_threedi_soft_agreement": (
                    float(selected["realized_threedi_soft_agreement"])
                    if selected.get("realized_threedi_scored")
                    and finite(selected.get("realized_threedi_soft_agreement"))
                    else None
                ),
                "selected_tm_score": float(selected["binder_tm_score"])
                if tm_scored
                else None,
                "selected_ca_rmsd": float(selected["binder_ca_rmsd"])
                if backbone_scored
                else None,
                "selected_ca_lddt": float(selected["binder_ca_lddt"])
                if backbone_scored
                else None,
                "tm_scored": tm_scored,
                "backbone_scored": backbone_scored,
            }
        )
    selected_targets = [
        target for target in targets if target["selected_score"] is not None
    ]
    tm_scores = [
        target["selected_tm_score"] for target in targets if target["tm_scored"]
    ]
    backbone_targets = [target for target in targets if target["backbone_scored"]]
    return {
        "selection_metric": selection_metric,
        "requested_target_count": len(targets),
        "candidates_per_target": candidates_per_task,
        "selected_target_count": len(selected_targets),
        "tm_scored_target_count": len(tm_scores),
        "backbone_scored_target_count": len(backbone_targets),
        "selection_fraction": len(selected_targets) / len(targets) if targets else None,
        "tm_scored_fraction": len(tm_scores) / len(targets) if targets else None,
        "backbone_scored_fraction": len(backbone_targets) / len(targets)
        if targets
        else None,
        "mean_selected_plddt": _mean(
            target["selected_plddt"]
            for target in selected_targets
            if target["selected_plddt"] is not None
        ),
        "mean_selected_threedi_identity": _mean(
            target["selected_threedi_identity"]
            for target in selected_targets
            if target["selected_threedi_identity"] is not None
        ),
        "mean_selected_threedi_distance": _mean(
            target["selected_threedi_mean_distance"]
            for target in selected_targets
            if target["selected_threedi_mean_distance"] is not None
        ),
        "mean_selected_threedi_soft_agreement": _mean(
            target["selected_threedi_soft_agreement"]
            for target in selected_targets
            if target["selected_threedi_soft_agreement"] is not None
        ),
        "mean_tm_score": _mean(tm_scores),
        "mean_ca_rmsd": _mean(
            target["selected_ca_rmsd"] for target in backbone_targets
        ),
        "mean_ca_lddt": _mean(
            target["selected_ca_lddt"] for target in backbone_targets
        ),
        "failure_adjusted_mean_tm_score": sum(tm_scores) / len(targets)
        if targets
        else None,
        "failure_adjusted_mean_ca_lddt": (
            sum(target["selected_ca_lddt"] for target in backbone_targets)
            / len(targets)
            if targets
            else None
        ),
        "tm_above_0p5_fraction_all_targets": (
            sum(score > 0.5 for score in tm_scores) / len(targets) if targets else None
        ),
        "targets": targets,
    }


def summarize_threedi_agreement_selection(
    records: Sequence[ValidationRecord],
    *,
    candidates_per_task: int,
) -> dict[str, Any]:
    """Select by refolded 3Di agreement with the supplied string, without coordinates."""
    return _summarize_threedi_selection(
        records,
        candidates_per_task=candidates_per_task,
        selection_metric="realized_threedi_identity",
    )


def summarize_threedi_plddt_selection(
    records: Sequence[ValidationRecord],
    *,
    candidates_per_task: int,
) -> dict[str, Any]:
    """Select by mean monomer pLDDT alone, then score those same candidates."""
    return _summarize_threedi_selection(
        records,
        candidates_per_task=candidates_per_task,
        selection_metric="mean_monomer_plddt",
    )


def score_generation_validation_outputs(
    task_batches: Sequence[Mapping[str, Any]],
    output_batches: Sequence[Sequence[Mapping[str, Any]]],
    *,
    validators: Iterable[str],
    normalized_embeddings: bool,
    sampling_seed: int,
    property_definition: str = DEFAULT_PROPERTY_DEFINITION,
    suite_name: str = "routine",
    property_std: torch.Tensor | None = None,
    services: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score already-generated outputs without running diffusion again."""
    if len(task_batches) != len(output_batches):
        raise ValueError("task and generated-output batch counts must match")

    total_started = time.perf_counter()
    selected = require_checks(validators)
    records: list[ValidationRecord] = []
    timing_seconds: dict[str, Any] = {
        "batch_setup_seconds": 0.0,
        "sampling_seconds": 0.0,
        "decoding_seconds": 0.0,
        "record_building_seconds": 0.0,
        "checks": {name: 0.0 for name in selected},
    }
    for tasks, decoded in zip(task_batches, output_batches, strict=True):
        scoring_tasks = dict(tasks)
        if isinstance(property_std, torch.Tensor):
            scoring_tasks["property_std"] = property_std.detach().float().cpu()
        batch_records, batch_timing = _run_checks_with_timing(
            scoring_tasks,
            decoded,
            selected,
            normalized_embeddings=normalized_embeddings,
            property_definition=property_definition,
            services=services,
        )
        records.extend(batch_records)
        timing_seconds["record_building_seconds"] += batch_timing[
            "record_building_seconds"
        ]
        for name, seconds in batch_timing["checks"].items():
            timing_seconds["checks"][name] += seconds

    summary_started = time.perf_counter()
    summary = summarize_checks(records, selected)
    timing_seconds["summarization_seconds"] = time.perf_counter() - summary_started
    timing_seconds["total_seconds"] = time.perf_counter() - total_started
    accounted_seconds = sum(
        float(timing_seconds[name])
        for name in (
            "batch_setup_seconds",
            "sampling_seconds",
            "decoding_seconds",
            "record_building_seconds",
            "summarization_seconds",
        )
    ) + sum(float(seconds) for seconds in timing_seconds["checks"].values())
    timing_seconds["other_seconds"] = max(
        0.0,
        float(timing_seconds["total_seconds"]) - accounted_seconds,
    )
    return {
        "suite": suite_name,
        "sampling_seed": sampling_seed,
        "property_definition": canonical_property_definition(property_definition),
        "validators": list(selected),
        "summary": summary,
        "timing_seconds": timing_seconds,
        "records": records,
    }
