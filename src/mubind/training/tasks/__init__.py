"""Construct training tasks independently of data loading and the network."""

from .collate import TaskCollator, collate_task_rows
from .masking import mask_training_threedi_conditioning, property_mask
from .config import (
    DEFAULT_TASK_STAGES,
    PropertyDropoutConfig,
    TaskConfig,
    TaskStageConfig,
    ThreeDiMaskingConfig,
)
from .sampling import sample_task_batch
from .spans import (
    sample_edit_masks_for_budgets,
    sample_edit_spans_for_budget,
    sample_internal_edit_spans,
)

__all__ = [
    "DEFAULT_TASK_STAGES",
    "PropertyDropoutConfig",
    "TaskCollator",
    "TaskConfig",
    "TaskStageConfig",
    "ThreeDiMaskingConfig",
    "collate_task_rows",
    "mask_training_threedi_conditioning",
    "property_mask",
    "sample_edit_masks_for_budgets",
    "sample_edit_spans_for_budget",
    "sample_internal_edit_spans",
    "sample_task_batch",
]
