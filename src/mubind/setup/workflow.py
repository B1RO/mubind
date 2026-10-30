"""Setup operations shared by interactive and unattended entry points."""

from __future__ import annotations

import json
from pathlib import Path

from filelock import FileLock, Timeout

from ..preprocessing.prostt5.steps._io import write_manifest
from ..preprocessing.prostt5.steps.validate import discover_datasets, is_ready_for_training
from ..preprocessing.prostt5.preprocess import prepare_prostt5
from ..common.progress import ProgressCallback, ProgressEvent
from .config import SetupConfig
from .checkpoint import checkpoint_path, checkpoint_ready, setup_checkpoint
from .models import (
    CCD_FILE,
    CCD_REPOSITORY,
    CCD_REVISION,
    COMPLEXA_FILES,
    HF_MODELS,
    asset_files,
    download_models,
    fingerprints,
    weights_available,
)
from .runtimes import (
    ensure_runtime,
    runtime_directory,
    runtime_identity,
    runtime_identity_matches,
    runtime_ready,
    verify_esmfold2_inputs,
)
from .tools import ensure_foldseek, foldseek_ready


def model_manifest(component: str, root: Path) -> Path:
    return root / ".mubind" / f"{component}.json"


def model_identity(component: str) -> dict:
    assets = (
        {f.name: f.sha256 for f in COMPLEXA_FILES}
        if component == "complexa"
        else {
            **HF_MODELS,
            "ccd": {
                "repository": CCD_REPOSITORY,
                "revision": CCD_REVISION,
                "filename": CCD_FILE.name,
                "sha256": CCD_FILE.sha256,
            },
        }
    )
    return json.loads(
        json.dumps({"assets": assets, "runtime": runtime_identity(component)})
    )


def ready_model(component: str, root: Path) -> dict | None:
    try:
        report = json.loads(model_manifest(component, root).read_text())
        recorded = report["identity"]
        if (
            report["state"] != "complete"
            or not isinstance(recorded, dict)
            or recorded.get("assets") != model_identity(component)["assets"]
            or not runtime_identity_matches(component, recorded.get("runtime"))
        ):
            return None
        python = Path(report["python"])
        if (
            python != runtime_directory(component, root) / "venv" / "bin" / "python"
            or not python.is_file()
        ):
            return None
        if not runtime_ready(component, root):
            return None
        if fingerprints(asset_files(component, report["paths"])) != report["files"]:
            return None
        return report
    except (OSError, ValueError, KeyError, TypeError):
        return None


def component_status(component: str, config: SetupConfig) -> str:
    if component == "mubind":
        if checkpoint_ready(config.models_dir):
            return "Ready"
        if checkpoint_path(config.models_dir).exists():
            return "Weights found · verify checkpoint"
        return "Download checkpoint"
    if component == "foldseek":
        return "Ready" if foldseek_ready(config.models_dir) else "Install Foldseek"
    if component == "prostt5":
        datasets = discover_datasets(config.data_dir.expanduser().resolve())
        source = config.source.expanduser().resolve() if config.source else None
        if any(
            is_ready_for_training(info, config.min_length, config.max_length, source)
            for info in datasets
        ):
            return "Ready"
        if (source is not None and source.is_dir()) or any(
            info.state == "ready" for info in datasets
        ):
            return "Local data available"
        return "Download + prepare"
    root = config.models_dir.expanduser().resolve()
    if ready_model(component, root):
        return "Ready"
    weights = weights_available(component, root)
    if runtime_ready(component, root):
        return "Environment ready · " + (
            "verify weights" if weights else "download weights"
        )
    if weights:
        return "Weights found · set up environment"
    return "Set up environment + weights"


def setup_component(
    component: str, config: SetupConfig, emit: ProgressCallback
) -> dict[str, str]:
    config.validated()
    if component == "mubind":
        return setup_checkpoint(config.models_dir, emit)
    if component == "foldseek":
        executable = ensure_foldseek(config.models_dir.expanduser().resolve(), emit)
        return {"executable": str(executable)}
    if component == "prostt5":
        path = prepare_prostt5(
            config.data_dir,
            source=config.source,
            min_length=config.min_length,
            max_length=config.max_length,
            on_progress=emit,
        )
        return {"dataset": str(path)}
    if component not in ("complexa", "esmfold2"):
        raise ValueError(f"Unknown component: {component}")
    root = config.models_dir.expanduser().resolve()
    directory = root / ".mubind"
    if directory.is_symlink():
        raise ValueError(f"Setup directory must not be a symlink: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(directory / f"{component}.lock", timeout=0):
            ready = ready_model(component, root)
            python = ensure_runtime(component, root, emit)
            if ready:
                emit(ProgressEvent("reuse", message="Environment and weights ready"))
                return {**ready["paths"], "python": str(python)}
            manifest = model_manifest(component, root)
            report = {"state": "preparing", "identity": model_identity(component)}
            write_manifest(manifest, report)
            try:
                paths = download_models(component, root, emit)
                if component == "esmfold2":
                    verify_esmfold2_inputs(python, root, paths, emit)
                report.update(
                    state="complete",
                    paths=paths,
                    python=str(python),
                    files=fingerprints(asset_files(component, paths)),
                )
                write_manifest(manifest, report)
            except BaseException as exc:
                report.update(state="failed", error=str(exc))
                write_manifest(manifest, report)
                raise
            return {**paths, "python": str(python)}
    except Timeout as exc:
        raise RuntimeError(f"Another setup is already preparing {component}.") from exc
