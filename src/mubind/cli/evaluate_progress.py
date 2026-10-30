"""Evaluate uses the same live stage display and timing as Generate."""

from rich.text import Text

from .generation_progress import PipelineProgress
from ..benchmarks.workflow import evaluation_stages


class EvaluationProgress(PipelineProgress):
    def __init__(self, console, settings):
        stages = evaluation_stages(settings)
        self.stage_numbers = {key: number for number, (key, _) in enumerate(stages, 1)}
        super().__init__(console, dict(enumerate((label for _, label in stages), 1)), name="Evaluation")

    def event(self, task, event):
        self.activate(self.stage_numbers[event["stage"]])
        if event.get("completed") is None:
            self.status(task, event["description"])
            return
        if event.get("failed"):
            self.failures[self.current] += 1
        self.advance(
            task, event.get("work", event["stage"]), event["completed"], event["total"],
            description=event.get("description"), detail=event.get("detail"),
            step_detail=event.get("step_detail", ""),
            minimum=event.get("minimum", 3), failed=event.get("failed", False),
        )


def run_with_progress(settings, console):
    from ..benchmarks.workflow import run_evaluation

    results = []
    progress = EvaluationProgress(console, settings)
    task = progress.add_task("Checking checkpoint and evaluation requirements…", total=None, detail="")
    try:
        with progress:
            result = run_evaluation(
                settings,
                on_status=lambda message: progress.status(task, message),
                on_progress=lambda event: progress.event(task, event),
                on_result=results.append,
            )
            progress.complete(task, "Evaluation complete")
        return result
    finally:
        # Results remain visible after the live display, also if a later stage fails.
        for message in results:
            console.print(Text(message))
