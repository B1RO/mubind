"""Resolve installed runtimes for persistent workers without starting them."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .setup.config import load_config
from .setup.runtimes import runtime_environment
from .setup.workflow import ready_model
from .paths import existing_artifact


@dataclass(frozen=True)
class ModelRuntime:
    python: Path
    paths: dict[str, str]
    models_dir: Path

    def environment(self) -> dict[str, str]:
        """An environment for a new worker; never mutates the current process."""
        env = runtime_environment(self.models_dir)
        env.update(
            HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
        )
        if "model" in self.paths:
            env.update(
                ESMFOLD2_MODEL=self.paths["model"],
                ESMC_MODEL=self.paths["esmc_model"],
                ESMFOLD2_PYTHON=str(self.python),
                ESMCFOLD_CCD_PATH=self.paths["ccd"],
            )
        return env

    def complexa_config(self) -> dict[str, str]:
        checkpoint = Path(self.paths["checkpoint"])
        return {
            "ckpt_path": str(checkpoint.parent),
            "ckpt_name": checkpoint.name,
            "autoencoder_ckpt_path": self.paths["autoencoder"],
        }


def get_runtime(
    component: str, *, models_dir: Path | str | None = None
) -> ModelRuntime:
    """Return verified paths for a worker's interpreter and local model files."""
    if component not in ("complexa", "esmfold2"):
        raise ValueError(f"Unknown model runtime: {component}")
    root = (
        existing_artifact(Path(models_dir if models_dir is not None else load_config().models_dir))
        .expanduser()
        .resolve()
    )
    report = ready_model(component, root)
    if report is None:
        raise RuntimeError(
            f"{component} is not ready. Run mubind setup --only {component}."
        )
    return ModelRuntime(Path(report["python"]), dict(report["paths"]), root)
