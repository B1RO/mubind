"""Stage-local timing and a bounded terminal display for generation."""

from __future__ import annotations

import math
import time
from collections import deque

from rich.console import Group
from rich.constrain import Constrain
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.spinner import Spinner
from rich.table import Column, Table
from rich.text import Text

from .provider_display import short_duration
from .terminal_live import track_reflow


def approximate_duration(seconds):
    # Deliberately avoid a precise countdown from an approximate throughput.
    if seconds < 60:
        return f"{max(5, math.ceil(seconds / 5) * 5)}s"
    if seconds < 3600:
        return f"{math.ceil(seconds / 60)}m"
    return f"{math.ceil(seconds / 360) / 10:g}h"


class StageEstimate:
    """Estimate only like-for-like work observed in this run, never five equal stages."""

    def __init__(self):
        self.started = time.monotonic()
        self.finished = None
        self.stage = None
        self.completed = self.total = 0
        self.samples = deque(maxlen=64)
        self.active = False
        self.minimum = 3

    def observe(self, stage, completed, total, *, at=None, failed=False, minimum=3):
        at = time.monotonic() if at is None else at
        if (stage != self.stage or total != self.total
                or completed < self.completed):
            self.samples.clear()
        self.stage, self.completed, self.total = stage, completed, total
        self.active, self.minimum = True, minimum
        # Fast failures are not representative of successful folding throughput.
        if failed:
            self.samples.clear()
        if not self.samples or completed > self.samples[-1][0] or failed:
            point = (completed, at)
            # Fast diffusion callbacks must still leave a useful time window.
            if len(self.samples) > 1 and at - self.samples[-2][1] < 0.5 and not failed:
                self.samples[-1] = point
            else:
                self.samples.append(point)

    def eta(self, now=None):
        now = time.monotonic() if now is None else now
        if not self.active or not self.total:
            return "stage ETA pending"
        if self.completed >= self.total:
            return "stage finishing"
        if len(self.samples) <= self.minimum:
            return "stage ETA warming up"
        first, start = self.samples[0]
        last, end = self.samples[-1]
        elapsed = end - start
        if elapsed < 2 or last <= first:
            return "stage ETA warming up"
        per_unit = elapsed / (last - first)
        prediction = per_unit * (self.total - last)
        remaining = prediction - max(0, now - end)
        # A stall, queued request or much slower fold invalidates the countdown.
        if remaining <= 0 or now - end > max(15, per_unit * 3):
            return "stage ETA updating"
        return f"stage ETA ~{approximate_duration(remaining)}"

    def text(self, *, show_eta=True):
        now = self.finished if self.finished is not None else time.monotonic()
        elapsed = f"run elapsed {short_duration(now - self.started)}"
        return elapsed if self.finished is not None or not show_eta else f"{elapsed} · {self.eta(now)}"


class GenerationProgress(Progress):
    """One persistent task, two progress lines and an optional tip below them."""

    def __init__(self, console):
        self.display_console = console
        self.estimate = StageEstimate()
        self.step_detail = ""
        self.waiting_message = ""
        self.tip = ""
        self.last_log = 0.0
        super().__init__(
            SpinnerColumn(finished_text="✓"),
            TextColumn("{task.description}", table_column=Column(
                ratio=1, no_wrap=True, overflow="ellipsis")),
            BarColumn(bar_width=20 if console.width >= 80 else 10,
                      table_column=Column(min_width=8)),
            TextColumn("{task.fields[detail]}", table_column=Column(no_wrap=True)),
            console=console, disable=not console.is_terminal,
        )
        track_reflow(self.live)

    def start(self):
        if not self.live.is_started:
            # Prompts may have moved the cursor since stop(). The previous
            # frame is now scrollback, so restarting must not erase its height.
            track_reflow(self.live)
            self.live.vertical_overflow = "ellipsis"
        super().start()

    def get_renderable(self):
        timing = self.estimate.text(show_eta=not self.waiting_message)
        rows = [
            Text(self.waiting_message) if self.waiting_message else super().get_renderable(),
            Text(timing, style="#808080", no_wrap=True, overflow="ellipsis"),
        ]
        if self.tip:
            rows.append(Text(f"Tip: {self.tip}", style="#808080"))
        return Constrain(Group(*rows), max(1, self.display_console.width - 2))

    def show_tip(self, message):
        if self.tip == message:
            return
        self.tip = message
        if self.console.is_terminal:
            self.refresh()
        else:
            self.console.print(Text(f"Tip: {message}", style="#808080"))

    def status(self, task, message, *, waiting=False):
        self.estimate.active = False
        self.step_detail = ""
        self.waiting_message = message if waiting else ""
        self.reset(task, description=message, completed=0, total=None, detail="")
        # Do not leave the last sampling frame visible until the refresh thread runs.
        self.refresh()
        if not self.console.is_terminal:
            self.console.print(Text(message))

    def advance(self, task, stage, completed, total, *, description=None,
                detail=None, at=None, failed=False, minimum=3, step_detail=""):
        self.waiting_message = ""
        self.estimate.observe(stage, completed, total, at=at, failed=failed,
                              minimum=minimum)
        self.step_detail = step_detail
        detail = detail or f"{completed:,}/{total:,}"
        if step_detail:
            detail += f" · {step_detail}"
        self.update(task, completed=completed, total=total,
                    description=description,
                    detail=detail)
        now = time.monotonic()
        if not self.console.is_terminal and (now - self.last_log >= 10 or completed == total):
            current = self.tasks[0]
            self.console.print(Text(
                f"{current.description} · {current.fields['detail']} · {self.estimate.text()}"
            ))
            self.last_log = now

    def complete(self, task, message):
        self.estimate.finished = time.monotonic()
        self.step_detail = ""
        self.waiting_message = ""
        self.update(task, description=message, total=1, completed=1, detail="", refresh=True)
        self.stop_task(task)


class PipelineProgress(GenerationProgress):
    """The shared stage list, active bar and timing used by Generate and Evaluate."""

    def __init__(self, console, labels, *, skipped=(), name="Run"):
        self.labels, self.name = labels, name
        self.skipped = set(skipped)
        self.states = {stage: "skipped" if stage in self.skipped else "pending" for stage in self.labels}
        self.failures = {stage: 0 for stage in self.labels}
        self.current = None
        self.pipeline_finished = False
        self.preparation_state = "running"
        self.spinner = Spinner("dots", style="cyan")
        super().__init__(console)
        # The spinner belongs to the stage row, not to each detail line.
        self.columns = self.columns[1:]

    def start(self):
        if not self.console.is_terminal:
            self.console.print(self.stage_table())
        super().start()

    def stage_table(self):
        table = Table.grid(padding=(0, 1), expand=True)
        table.add_column(width=1)
        table.add_column(ratio=1)
        for stage, label in self.labels.items():
            state = self.states[stage]
            icon, style = {
                "pending": ("○", "dim"), "running": (self.spinner, "cyan"),
                "complete": ("✓", "green"), "skipped": ("–", "dim"),
                "failed": ("✗", "red"), "interrupted": ("!", "yellow"),
                "partial": ("!", "yellow"),
            }[state]
            suffix = f" · {state}" if state not in {"running", "complete"} else ""
            if self.failures[stage]:
                count = self.failures[stage]
                suffix += f" · {count} failed fold{'s' if count != 1 else ''}"
            table.add_row(icon if isinstance(icon, Spinner) else Text(icon, style=style),
                          Text(label + suffix, style=style, no_wrap=True, overflow="ellipsis"))
        return table

    def get_renderable(self):
        detail = super().get_renderable()
        if self.current is None:
            preparing = Table.grid(padding=(0, 1), expand=True)
            preparing.add_column(width=1)
            preparing.add_column(ratio=1)
            icon = Text("?", style="cyan") if self.waiting_message else self.spinner
            if self.pipeline_finished:
                icon = Text("✗" if self.preparation_state == "failed" else "!" if self.preparation_state == "interrupted" else "✓")
            preparing.add_row(icon, detail)
            detail = preparing
        # Stage rows keep their positions as the active stage changes.
        return Group(self.stage_table(), Text(), detail)

    def finish_stage(self):
        if self.current is not None and self.states[self.current] == "running":
            self.states[self.current] = "partial" if self.failures[self.current] else "complete"

    def wait_for_input(self, task, message):
        # The callback runs after ranking and its output writes have succeeded,
        # not merely when the final fold counter reaches its total.
        self.finish_stage()
        super().status(task, message, waiting=True)

    def activate(self, stage):
        if stage == self.current:
            return
        self.finish_stage()
        if stage is not None:
            # Interactive requests can revisit scaffolding or redesign.
            for later in self.labels:
                if later >= stage and later not in self.skipped:
                    self.states[later] = "pending"
                    self.failures[later] = 0
            self.states[stage] = "running"
        self.current = stage

    def complete(self, task, message):
        self.finish_stage()
        # Never report an unvisited stage as completed (e.g. no valid candidates).
        for stage, state in self.states.items():
            if state == "pending":
                self.states[stage] = "skipped"
        self.pipeline_finished = True
        super().complete(task, message)

    def failed(self, task, message, *, interrupted=False):
        if self.current is not None:
            # Cancelling selection must not turn finished folding into a failure.
            if self.states[self.current] == "running":
                self.states[self.current] = "interrupted" if interrupted else "failed"
        else:
            self.preparation_state = "interrupted" if interrupted else "failed"
        self.pipeline_finished = True
        self.estimate.finished = time.monotonic()
        self.status(task, message)

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None and not self.pipeline_finished and self.tasks:
            self.failed(self.tasks[0].id, str(exc_value) or f"{self.name} interrupted",
                        interrupted=issubclass(exc_type, KeyboardInterrupt))
        return super().__exit__(exc_type, exc_value, traceback)


class DesignProgress(PipelineProgress):
    """Keep Generate's actual five-stage pipeline visible, including configured skips."""

    def __init__(self, console, settings):
        labels = {
            1: "Reuse Complexa scaffolds" if settings.reuse_scaffolds else "Generate scaffolds · Proteina Complexa",
            2: "Fold and select scaffolds · ESMFold2",
            3: "Encode selected scaffolds · Foldseek",
            4: "Redesign sequences · mubind",
            5: "Fold and rank candidates · ESMFold2",
        }
        skipped = {2} if settings.all_scaffolds or settings.scaffold_index is not None else set()
        super().__init__(console, labels, skipped=skipped, name="Design")

    @property
    def design_finished(self):
        return self.pipeline_finished

    def status(self, task, message):
        if len(message) >= 3 and message[0] in "12345" and message[1:3] == "/5":
            stage = int(message[0])
            self.activate(None if stage in self.skipped else stage)
            message = message.partition(" · ")[2] or message
        super().status(task, message)

    def advance(self, task, stage, completed, total, **options):
        number = {"scaffolds": 1, "scaffold_folding": 2, "redesign": 4,
                  "candidate_folding": 5}.get(stage)
        if number is not None:
            self.activate(number)
            if options.get("failed"):
                self.failures[number] += 1
        super().advance(task, stage, completed, total, **options)
