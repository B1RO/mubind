"""Training orchestration; diffusion math lives in :mod:`mubind.diffusion`."""

from .config import DataConfig, TrainingConfig, config_from_dict, load_config


def __getattr__(name):
    # Evaluation reuses task definitions without importing the training loop.
    if name in {"TrainingResult", "train"}:
        from .trainer import TrainingResult, train

        return {"TrainingResult": TrainingResult, "train": train}[name]
    raise AttributeError(name)


__all__ = [
    "DataConfig",
    "TrainingConfig",
    "TrainingResult",
    "config_from_dict",
    "load_config",
    "train",
]
