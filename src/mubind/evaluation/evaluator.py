"""Schedule fixed evaluation suites through Python functions."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from ..inference.predict import inference_context, predict
from ..paths import existing_artifact
from ..training.tasks import TaskCollator
from .config import STRUCTURAL_CHECKS, EvaluationConfig
from .denoising import evaluate_denoising, evaluation_tasks, inpainting_denoising
from .folding import FoldingClient
from .metrics import (
    score_generation_validation_outputs,
    summarize_threedi_agreement_selection,
    summarize_threedi_best_of,
    summarize_threedi_plddt_selection,
)
from .panels import (
    materialize_inpainting_task_batches,
    materialize_property_steering_task_batches,
    materialize_task_batches,
    materialize_threedi_task_batches,
)


def evaluation_directory(output):
    path = Path(output)
    return path.with_suffix(path.suffix + ".evaluation")


def structural_checkpoint(output):
    path = Path(output)
    return path.with_name(f"{path.stem}.best-3di{path.suffix}")


def clean_json(value):
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_result(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".evaluation-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(clean_json(result), handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def numeric_metrics(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(item, dict):
            result.update(numeric_metrics(item, name))
        elif (
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(item)
        ):
            result[name] = item
    return result


def structural_score(summary):
    """Original lexicographic selector: failure-adjusted 3Di, TM, pLDDT."""
    checks = summary.get("checks", {})
    structure, realized = (
        checks.get("binder_structure", {}),
        checks.get("realized_threedi", {}),
    )
    if not all(
        item.get(key, 0) > 0
        for item, key in (
            (realized, "scored_candidate_count"),
            (realized, "expected_token_count"),
            (structure, "scored_candidate_count"),
        )
    ):
        return None
    values = (
        realized.get("failure_adjusted_micro_identity"),
        structure.get("mean_tm_score"),
        structure.get("mean_generated_monomer_plddt"),
    )
    if all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        for value in values
    ):
        return tuple(float(value) for value in values)
    return None


@dataclass
class EvaluationReport:
    step: int
    metrics: dict[str, float] = field(default_factory=dict)
    files: list[Path] = field(default_factory=list)
    best_3di_score: tuple[float, float, float] | None = None


class Evaluator:
    def __init__(self, config, train_data, validation_data, *, fold_client=None):
        self.training_config = config
        self.config: EvaluationConfig = config.evaluation
        self.suites = self.config.resolved_suites()
        self.train_data = self._evaluation_view(train_data, include_raw=False)
        self.validation_data = self._evaluation_view(validation_data)
        self.fold_client = fold_client or FoldingClient(
            self.config.folding, training_device=config.device
        )
        self.panels = {}
        for name, suite in self.suites.items():
            common = {
                "sample_count": suite.sample_count,
                "batch_size": suite.batch_size,
                "cohort_seed": suite.cohort_seed,
            }
            if suite.task_mode == "conditioned":
                batches = materialize_task_batches(
                    self.validation_data,
                    TaskCollator(evaluation_tasks(), training=False),
                    **common,
                    task_seed=suite.task_seed,
                )
            elif suite.task_mode == "property_steering":
                batches = materialize_property_steering_task_batches(
                    self.train_data, **common
                )
            elif suite.task_mode == "inpainting":
                batches = materialize_inpainting_task_batches(
                    self.validation_data, suite
                )
            else:
                batches = materialize_threedi_task_batches(
                    self.validation_data,
                    **common,
                    candidates_per_task=suite.candidates_per_task,
                    dataset_indices=suite.dataset_indices,
                    partial_mask_seed=suite.partial_mask_seed,
                    visible_min_fraction=suite.visible_min_fraction,
                    visible_max_fraction=suite.visible_max_fraction,
                    property_condition_mask=suite.property_condition_mask,
                )
            self.panels[name] = batches
        d = self.config.denoising
        self.utilization = (
            materialize_threedi_task_batches(
                self.validation_data,
                sample_count=d.utilization_count,
                batch_size=d.utilization_batch_size,
                cohort_seed=d.utilization_cohort_seed,
            )
            if d.enabled and d.threedi_utilization
            else None
        )
        # Include actual cases and reference contents as well as settings.
        digest = hashlib.sha256(
            json.dumps(asdict(self.config), sort_keys=True).encode()
        )
        reference_hashes = {}
        for name, batches in self.panels.items():
            digest.update(name.encode())
            for batch in batches:
                for key, value in sorted(batch.items()):
                    if key == "reference_pdb":
                        for path in value:
                            if path not in reference_hashes:
                                reference_hashes[path] = (
                                    hashlib.sha256(Path(path).read_bytes()).digest()
                                    if path and Path(path).is_file()
                                    else b"missing-reference"
                                )
                            digest.update(reference_hashes[path])
                        continue
                    digest.update(key.encode())
                    digest.update(
                        value.numpy().tobytes()
                        if isinstance(value, torch.Tensor)
                        else json.dumps(value, sort_keys=True).encode()
                    )
        digest.update((self.train_data.path / "prostt5_diffusion.json").read_bytes())
        digest.update(json.dumps(config.to_dict()["data"], sort_keys=True).encode())
        self.signature = digest.hexdigest()
        self.output = evaluation_directory(config.out) / self.signature[:12]
        self._checked = False

    def _evaluation_view(self, dataset, *, include_raw=True):
        view = copy.copy(dataset)
        view.include_raw = include_raw
        reference = self.config.folding.reference_manifest
        path = (
            existing_artifact(Path(reference))
            if reference
            else dataset.path / "validation_references.json"
        )
        # Full evaluation has a fixed ten-protein panel. Its bundled coordinates
        # must not be shadowed by an old dataset's machine-specific manifest.
        bundled = not reference and self.config.regime == "full"
        if bundled:
            from ..resources import resource_directory
            path = resource_directory("validation") / "validation_references.json"
        view.references = None
        if dataset.split == "valid" and path.exists():
            view.references = json.loads(path.read_text())
            if not isinstance(view.references, dict):
                raise ValueError(
                    "Validation references must be a JSON object keyed by row index"
                )
            for item in view.references.values():
                if not isinstance(item, dict):
                    raise TypeError("Invalid validation reference record")
                if item.get("path"):
                    source = Path(item["path"]).expanduser()
                    if self.config.folding.reference_root:
                        source = (
                            existing_artifact(Path(self.config.folding.reference_root))
                            / source.name
                        )
                    elif not source.is_absolute():
                        source = path.parent / source
                    if (
                        bundled
                        and not self.config.folding.reference_root
                        and hashlib.sha256(source.read_bytes()).hexdigest()
                        != item["sha256"]
                    ):
                        raise ValueError(
                            f"Bundled validation reference checksum mismatch: {source}"
                        )
                    item["path"] = str(source.resolve())
        elif (reference or bundled) and dataset.split == "valid":
            raise FileNotFoundError(f"Validation reference manifest is missing: {path}")
        return view

    def check_ready(self, *, check_service=True):
        """Read-only validation before any optimizer work or output writes."""
        if self.config.regime == "full":
            from ..resources import resource_directory
            reference = json.loads(
                (resource_directory("validation") / "reference_panel.json").read_text()
            )
            if self.training_config.backbone.max_length != reference["max_length"]:
                raise ValueError("The 300k evaluation requires a 70-position canvas")
            for dataset in (self.train_data, self.validation_data):
                if len(dataset) != reference["split_rows"][dataset.split]:
                    raise ValueError(
                        "The 300k evaluation requires its original ProstT5 50–70 subset; use custom panels for another subset"
                    )
            for item in reference["rows"]:
                _, aa, di = self.validation_data.rows[item["dataset_index"]]
                if (
                    hashlib.sha256(aa.encode()).hexdigest() != item["sequence_sha256"]
                    or hashlib.sha256(di.encode()).hexdigest() != item["threedi_sha256"]
                ):
                    raise ValueError(
                        "The fixed 300k reference panel does not match this prepared dataset"
                    )
        if not self.training_config.properties.mask_enabled and (
            self.config.denoising.enabled
            or any(
                not bool(batch["condition_mask"].eq(1).all())
                for batches in self.panels.values()
                for batch in batches
            )
        ):
            raise ValueError(
                "These evaluation panels require property masks in the model"
            )
        structural = [
            (name, suite)
            for name, suite in self.suites.items()
            if STRUCTURAL_CHECKS.intersection(suite.validators)
        ]
        if structural:
            from .structure import _ca_trace, find_foldseek_executable

            for package in ("Bio", "tmtools"):
                if importlib.util.find_spec(package) is None:
                    raise ValueError(
                        f"Full evaluation requires {package}; install the project's evaluation dependencies"
                    )
            if any(
                "realized_threedi" in s.validators for _, s in structural
            ) and not find_foldseek_executable(self.config.folding.foldseek):
                raise ValueError(
                    "Full evaluation requires Foldseek. Run mubind setup --only foldseek, "
                    "or set evaluation.folding.foldseek to an existing executable."
                )
            checked = set()
            for name, suite in structural:
                if "binder_structure" not in suite.validators:
                    continue
                for batch in self.panels[name]:
                    if "reference_pdb" not in batch:
                        raise ValueError(
                            "Full evaluation needs validation_references.json or evaluation.folding.reference_manifest"
                        )
                    for source, sequence in zip(
                        batch["reference_pdb"], batch["binder_aa"], strict=True
                    ):
                        if not source or not Path(source).is_file():
                            raise ValueError(
                                f"Validation reference is missing: {source or '(unset)'}. Set evaluation.folding.reference_root to relocate the structures"
                            )
                        if source not in checked:
                            _, actual = _ca_trace(Path(source), chain_id="A")
                            if actual != sequence:
                                raise ValueError(
                                    f"Validation reference sequence mismatch: {source}"
                                )
                            checked.add(source)
            if check_service:
                try:
                    check = getattr(
                        self.fold_client, "check_ready", self.fold_client.protocol
                    )
                    check()
                except (OSError, RuntimeError, ValueError) as exc:
                    raise ValueError(
                        f"Full evaluation cannot use folding service {self.config.folding.endpoint}: {exc}"
                    ) from exc
        self._checked = True

    def start_folding(self, on_status=None):
        """Load managed folding weights only when an actual run is starting."""
        if any(
            STRUCTURAL_CHECKS.intersection(s.validators) for s in self.suites.values()
        ):
            start = getattr(self.fold_client, "start", None)
            if start is not None:
                start(on_status=on_status)
            else:
                self.fold_client.protocol()

    def run_due(
        self, *, step, final_step, model, diffusion, device, force=False, on_event=None
    ):
        if not self._checked:
            self.check_ready()
        report = EvaluationReport(step)
        notify = on_event or (lambda event: None)
        d = self.config.denoising
        if d.enabled and (force or step % d.every == 0 or step == final_step):
            notify({"phase": "start", "suite": "denoising", "step": step})
            result = evaluate_denoising(
                model,
                diffusion,
                self.validation_data,
                self.training_config,
                d,
                device=device,
                utilization_batches=self.utilization,
            )
            report.metrics.update(numeric_metrics(result, "validation"))
            report.metrics["validation_loss"] = result["loss"]
            path = self.output / "denoising" / f"step_{step}.json"
            write_result(
                path,
                {
                    "step": step,
                    "summary": result,
                    "evaluation_signature": self.signature,
                },
            )
            report.files.append(path)
            notify(
                {"phase": "complete", "suite": "denoising", "step": step, "path": path}
            )
        for name, suite in self.suites.items():
            if not (
                force
                or step % suite.interval_steps == 0
                or (step == final_step and suite.run_at_end)
            ):
                continue
            notify(
                {
                    "phase": "start",
                    "suite": name,
                    "step": step,
                    "candidates": suite.candidate_count,
                }
            )
            start = time.monotonic()
            batches = self.panels[name]
            outputs = []
            sampling = self.config.sampling
            with inference_context(model, device, seed=suite.sampling_seed):
                for index, batch in enumerate(batches):
                    prediction = predict(
                        model,
                        diffusion,
                        batch,
                        normalized_embeddings=self.training_config.data.normalize_embeddings,
                        device=device,
                        seed=None,
                        self_condition=self.training_config.self_condition.enabled,
                        precision=self.training_config.precision,
                        sampler=sampling.sampler,
                        sampling_steps=sampling.steps,
                        ddim_eta=sampling.ddim_eta,
                        property_guidance_scale=suite.property_guidance_scale,
                    )
                    outputs.append(prediction.decoded)
                    notify(
                        {
                            "phase": "sampling",
                            "suite": name,
                            "step": step,
                            "completed_batches": index + 1,
                            "batches": len(batches),
                        }
                    )
            services = {}
            if STRUCTURAL_CHECKS.intersection(suite.validators):
                from .structure import StructuralEvaluator

                backend = StructuralEvaluator(
                    out_dir=self.output / "folds",
                    baseline_cache_dir=Path(self.config.folding.cache_dir).expanduser()
                    if self.config.folding.cache_dir
                    else self.output / "native_baselines",
                    client=self.fold_client,
                    foldseek_executable=self.config.folding.foldseek,
                    seed=self.config.folding.seed,
                )
                services = {
                    key: getattr(backend, key)
                    for key in STRUCTURAL_CHECKS.intersection(suite.validators)
                }
                notify({"phase": "scoring", "suite": name, "step": step})
            std = (
                torch.tensor(self.training_config.properties.std)
                if self.training_config.properties.std is not None
                else None
            )
            result = score_generation_validation_outputs(
                batches,
                outputs,
                validators=suite.validators,
                normalized_embeddings=self.training_config.data.normalize_embeddings,
                sampling_seed=suite.sampling_seed,
                property_definition=self.training_config.data.property_definition,
                suite_name=name,
                property_std=std,
                services=services,
            )
            result.update(
                step=step,
                task_mode=suite.task_mode,
                cohort_seed=suite.cohort_seed,
                evaluation_signature=self.signature,
                sampling={
                    "sampler": sampling.sampler,
                    "sampling_steps": sampling.steps,
                    "ddim_eta": sampling.ddim_eta,
                    "property_guidance_scale": suite.property_guidance_scale,
                    "self_condition": self.training_config.self_condition.enabled,
                    "precision": self.training_config.precision,
                },
            )
            if suite.task_mode == "conditioned":
                result["task_seed"] = suite.task_seed
            if suite.task_mode == "property_steering":
                result["property_target_split"] = "train"
            if suite.task_mode == "threedi" and STRUCTURAL_CHECKS.intersection(
                suite.validators
            ):
                for key, fn in (
                    ("best_of", summarize_threedi_best_of),
                    ("threedi_selection", summarize_threedi_agreement_selection),
                    ("plddt_selection", summarize_threedi_plddt_selection),
                ):
                    if (
                        key == "threedi_selection"
                        and "realized_threedi" not in suite.validators
                    ):
                        continue
                    summary = fn(
                        result["records"], candidates_per_task=suite.candidates_per_task
                    )
                    result[key] = summary
                    result["summary"][key] = {
                        k: v for k, v in summary.items() if k != "targets"
                    }
            if (
                suite.task_mode == "inpainting"
                and "binder_structure" not in suite.validators
            ):
                result["summary"]["denoising"] = inpainting_denoising(
                    model,
                    diffusion,
                    batches,
                    device=device,
                    seed=d.seed,
                    repeats=d.noise_samples,
                    self_condition=self.training_config.self_condition.enabled,
                    precision=self.training_config.precision,
                )
            if (
                suite.task_mode == "inpainting"
                and "binder_structure" in suite.validators
            ):
                from .structure import score_gap_geometry

                result["gap_geometry"] = score_gap_geometry(result["records"])
                result["summary"]["gap_geometry"] = result["gap_geometry"]["summary"]
            result["timing_seconds"]["suite_total_seconds"] = time.monotonic() - start
            path = self.output / name / f"step_{step}.json"
            write_result(path, result)
            report.files.append(path)
            report.metrics.update(
                numeric_metrics(result["summary"], f"evaluation/{name}")
            )
            if suite.select_checkpoint:
                score = structural_score(result["summary"])
                if score is not None and (
                    report.best_3di_score is None or score > report.best_3di_score
                ):
                    report.best_3di_score = score
            notify({"phase": "complete", "suite": name, "step": step, "path": path})
        return report
