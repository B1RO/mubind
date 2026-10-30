"""UI-independent progress events for long-running preparation and setup work."""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ProgressEvent:
    stage: str
    completed: int = 0
    total: int | None = None
    retained: int | None = None
    unit: str = "sequences"
    message: str = ""
    # Latest observed subprocess milestone. Stage "output" carries only this and
    # marks that the running step produced output, without starting a new step.
    activity: str = ""


ProgressCallback = Callable[[ProgressEvent], None]
