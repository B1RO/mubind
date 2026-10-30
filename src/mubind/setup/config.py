"""Remember setup choices without tying them to a source checkout."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from filelock import FileLock

from ..preprocessing.prostt5.steps._io import write_manifest
from ..paths import DATASETS, MODELS, existing_artifact

COMPONENTS = ("mubind", "prostt5", "complexa", "esmfold2", "foldseek")
DEFAULT_COMPONENTS = tuple(c for c in COMPONENTS if c != "prostt5")
LABELS = {
    "mubind": "mubind checkpoint",
    "prostt5": "ProstT5 training data",
    "complexa": "Proteina-Complexa",
    "esmfold2": "ESMFold2 + ESMC",
    "foldseek": "Foldseek",
}


def default_config_path() -> Path:
    return (
        Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
        / "mubind"
        / "setup.json"
    )


@dataclass(frozen=True)
class SetupConfig:
    data_dir: Path = field(default_factory=lambda: DATASETS.absolute())
    models_dir: Path = field(default_factory=lambda: MODELS.absolute())
    min_length: int = 50
    max_length: int = 70
    components: tuple[str, ...] = DEFAULT_COMPONENTS
    source: Path | None = None

    def validated(self, *, check_paths: bool = True) -> SetupConfig:
        if not 1 <= self.min_length <= self.max_length:
            raise ValueError("Sequence lengths must satisfy 1 <= minimum <= maximum.")
        if not self.components or any(c not in COMPONENTS for c in self.components):
            raise ValueError("Select at least one of " + ", ".join(COMPONENTS) + ".")
        if not check_paths:
            return self
        for path in (self.data_dir, self.models_dir):
            if path.exists() and not path.is_dir():
                raise ValueError(f"Expected a directory: {path}")
        if "prostt5" in self.components and self.source is not None and not self.source.is_dir():
            raise ValueError(f"Local dataset does not exist: {self.source}")
        return self


def load_config(path: Path | None = None) -> SetupConfig:
    path = path or default_config_path()
    if not path.exists():
        return SetupConfig()
    try:
        report = json.loads(path.read_text())
        values = report["settings"]
        for name in ("data_dir", "models_dir", "source"):
            if values.get(name) is not None:
                values[name] = existing_artifact(Path(values[name]))
        values["components"] = tuple(values["components"])
        if report.get("version", 1) == 1 and set(values["components"]) == {
            "prostt5",
            "complexa",
            "esmfold2",
        }:
            values["components"] += ("foldseek",)
        if report.get("version", 1) < 3 and set(values["components"]) == {
            "prostt5", "complexa", "esmfold2", "foldseek",
        }:
            values["components"] = ("mubind", *values["components"])
        if report.get("version", 1) < 4 and set(values["components"]) == set(COMPONENTS):
            # Older defaults included training data without an explicit choice.
            # Keep custom selections and all saved paths, filters and source data.
            values["components"] = DEFAULT_COMPONENTS
        # The CLI applies overrides before validation, so a moved source can
        # still be replaced with --source or cleared with --automatic-source.
        return SetupConfig(**values)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid setup configuration at {path}: {exc}") from exc


def save_config(config: SetupConfig, path: Path | None = None) -> None:
    config.validated()
    path = path or default_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    values = asdict(config)
    for name in ("data_dir", "models_dir", "source"):
        values[name] = (
            str(values[name].expanduser().absolute())
            if values[name] is not None
            else None
        )
    with FileLock(str(path) + ".lock", timeout=5):
        write_manifest(path, {"version": 4, "settings": values})
