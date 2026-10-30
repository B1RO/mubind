"""The mubind terminal interface: Typer commands, Questionary menus, Rich output."""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from enum import Enum
from pathlib import Path
from typing import Annotated

# A local copy of the small, pure-Python UI avoids hundreds of network-volume
# imports. Models and all native dependencies remain in the selected environment.
from .startup import finish_provider_startup, start_cli_startup
from .ui_cache import use_local_ui

start_cli_startup(sys.argv[1:])
use_local_ui()

import questionary
import typer
from rich.console import Console, Group, RenderableType
from rich.filesize import decimal
from rich.panel import Panel
from rich.progress import (
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.progress_bar import ProgressBar
from rich.table import Column, Table
from rich.text import Text

from .terminal_ui import aligned_choices, select_prompt, short_path
from ..examples import ensure_example_target
from ..paths import TARGETS, existing_artifact
from ..common.progress import ProgressEvent
from ..setup.config import COMPONENTS, LABELS, SetupConfig, load_config, save_config
from ..setup.runtimes import runtime_directory
from ..setup.tools import find_foldseek, foldseek_executable
from ..setup.workflow import component_status, setup_component

app = typer.Typer(
    name="mubind",
    help="Your mubind workspace.",
    no_args_is_help=False,
    rich_markup_mode="rich",
    pretty_exceptions_enable=False,
)
console = Console()


@app.command(
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
        "help_option_names": [],
    }
)
def evaluate(ctx: typer.Context):
    """Evaluate a checkpoint on fixed benchmarks."""
    if not ctx.args and interactive_terminal():
        from .evaluate_ui import EvaluationUI

        EvaluationUI(console, STYLE).screen()
        return
    from .evaluate import main

    main(ctx.args)


STYLE = questionary.Style(
    [
        ("qmark", "fg:#67e8f9 bold"),
        ("question", "fg:#67e8f9 bold"),
        ("answer", "fg:#67e8f9"),
        ("pointer", "fg:#67e8f9 bold"),
        ("highlighted", "fg:#67e8f9 bold"),
        ("selected", "fg:#a5b4fc"),
        ("instruction", "fg:#808080"),
        ("text", ""),
        ("ready", "fg:#808080"),
        ("ok", "fg:#86efac"),
        ("pending", "fg:#fcd34d"),
        ("muted", "fg:#808080"),
        ("error", "fg:#f87171"),
    ]
)
HOME_STYLE = questionary.Style(
    [*STYLE.style_rules, ("question", "fg:#d787ff bold")]
)


class Component(str, Enum):
    mubind = "mubind"
    prostt5 = "prostt5"
    complexa = "complexa"
    esmfold2 = "esmfold2"
    foldseek = "foldseek"


def interactive_terminal() -> bool:
    return sys.stdin.isatty() and console.is_terminal


def select(message, choices, *, style=STYLE, erase=False, readiness=None, **options):
    if readiness is not None:
        options["before_render"] = readiness.render
    prompt = select_prompt(
        message, choices=choices, style=style, qmark="›", pointer="❯", **options
    )
    # An erased menu leaves no answered copy above the screen it opens.
    prompt.application.erase_when_done = erase
    if readiness is not None:
        readiness.attach(prompt.application)
    try:
        return prompt.ask()
    finally:
        if readiness is not None:
            readiness.close()


def setup_statuses(config: SetupConfig) -> dict[str, str]:
    statuses = {}
    for component in COMPONENTS:
        try:
            statuses[component] = component_status(component, config)
        except OSError as exc:
            # A saved directory may be inaccessible on this machine. Keep
            # Settings reachable so it can be corrected before installation.
            statuses[component] = f"Check setup paths: {exc}"
    return statuses


def setup_ready(config: SetupConfig, statuses: dict[str, str]) -> bool:
    return all(statuses.get(component) == "Ready" for component in config.components)


def setup_choice(title: str, value: str, ready: bool) -> questionary.Choice:
    return questionary.Choice(
        [("class:ready", title)] if ready else title, value
    )


def status_text(status: str) -> Text:
    if status == "Ready":
        return Text(status, style="green")
    if status == "Local data available":
        status = "Local data available · prepare dataset"
    if " · " in status:
        completed, pending = status.split(" · ", 1)
        return Text.assemble((completed, "green"), (" · ", "dim"), (pending, "yellow"))
    return Text(status, style="dim" if status == "Not selected" else "yellow")


def overview(config: SetupConfig, statuses: dict[str, str] | None = None) -> None:
    statuses = setup_statuses(config) if statuses is None else statuses
    locations = [
        Text.assemble(("Models will be stored at: ", "dim"), str(config.models_dir)),
    ]
    if "prostt5" in config.components:
        locations.extend([
            Text.assemble(("Training data will be stored at: ", "dim"), str(config.data_dir)),
            Text.assemble(
                ("The dataset will be subset to sequences of: ", "dim"),
                f"{config.min_length}–{config.max_length} residues",
            ),
        ])
    else:
        locations.append(Text(
            "Training data is optional. Prepare it with mubind setup --only prostt5.",
            style="dim",
        ))
    environments = [
        Text.assemble(("The mubind environment is located at: ", "dim"), sys.prefix)
    ]
    for component in config.components:
        if component not in ("complexa", "esmfold2"):
            continue
        directory = (
            runtime_directory(component, config.models_dir.expanduser().resolve())
            / "venv"
        )
        ready = statuses[component] == "Ready" or statuses[component].startswith(
            "Environment ready"
        )
        action = (
            "is ready"
            if ready
            else "will be reused"
            if (directory / "pyvenv.cfg").is_file()
            else "will be created"
        )
        label = "Proteina-Complexa" if component == "complexa" else "ESMFold2"
        environments.extend(
            [
                Text.assemble(
                    ("✓ " if ready else "", "green"),
                    (f"The {label} environment {action} at:", "dim"),
                ),
                Text(str(directory)),
            ]
        )
    if "foldseek" in config.components:
        ready = statuses["foldseek"] == "Ready"
        executable = (
            find_foldseek(models_dir=config.models_dir)
            if ready
            else foldseek_executable(config.models_dir.expanduser().resolve())
        )
        action = "is ready" if ready else "will be installed"
        environments.extend(
            [
                Text.assemble(
                    ("✓ " if ready else "", "green"),
                    (f"Foldseek {action} at:", "dim"),
                ),
                Text(str(executable)),
            ]
        )
    components = Table.grid(padding=(0, 2), expand=True)
    components.add_column(width=2)
    components.add_column()
    components.add_column(justify="right")
    for component in COMPONENTS:
        enabled = component in config.components
        status = statuses[component] if enabled else "Not selected"
        ready = status == "Ready"
        partial = " · " in status or status == "Local data available"
        components.add_row(
            "[green]✓[/]"
            if ready
            else "[yellow]◐[/]"
            if enabled and partial
            else "[yellow]○[/]"
            if enabled
            else "[dim]○[/]",
            Text(
                LABELS[component], style="green" if ready else "" if enabled else "dim"
            ),
            status_text(status),
        )
    completed = sum(statuses[component] == "Ready" for component in config.components)
    console.print()
    console.print(
        Panel(
            Group(
                Text("mubind", style="bold cyan"),
                Text("The following configuration will be used:", style="bold"),
                Text(),
                Group(*locations),
                Text(),
                *environments,
                Text(),
                components,
                Text(),
                Text(
                    f"{completed} of {len(config.components)} selected components ready",
                    style="green" if completed == len(config.components) else "dim",
                ),
            ),
            border_style="cyan",
            padding=(1, 3),
            title="[dim]SETUP[/]",
            title_align="left",
        )
    )


def configure(config: SetupConfig, field=None) -> SetupConfig:
    """Edit one choice at a time; cancelling an edit leaves its value intact."""
    while True:
        choice = field or select(
            "Configure",
            [
                questionary.Choice("Dataset directory", "data_dir"),
                questionary.Choice("Model directory", "models_dir"),
                questionary.Choice("Sequence length filter", "lengths"),
                questionary.Choice("Components", "components"),
                questionary.Choice("Use an existing dataset", "source"),
                questionary.Choice("Back to overview", "back"),
            ],
        )
        if choice in (None, "back"):
            return config
        if choice in ("data_dir", "models_dir", "source"):
            current = getattr(config, choice)
            message = (
                "Local Parquet dataset (leave empty for automatic)"
                if choice == "source"
                else "Directory"
            )
            answer = questionary.path(
                message,
                default=str(current) if current else "",
                only_directories=True,
                style=STYLE,
            ).ask()
            if answer is None:
                if field is not None:
                    return config
                continue
            if not answer.strip() and choice != "source":
                if field is not None:
                    return config
                continue
            value = Path(answer).expanduser().absolute() if answer.strip() else None
            updated = replace(config, **{choice: value})
        elif choice == "components":
            answer = questionary.checkbox(
                "Set up",
                choices=[
                    questionary.Choice(LABELS[c], c, checked=c in config.components)
                    for c in COMPONENTS
                ],
                validate=lambda values: (
                    bool(values) or "Select at least one component."
                ),
                style=STYLE,
            ).ask()
            if answer is None:
                if field is not None:
                    return config
                continue
            updated = replace(config, components=tuple(answer))
        else:
            minimum = questionary.text(
                "Minimum residues",
                default=str(config.min_length),
                validate=lambda s: (
                    s.isdecimal() and int(s) >= 1 or "Enter a positive integer."
                ),
                style=STYLE,
            ).ask()
            if minimum is None:
                if field is not None:
                    return config
                continue
            maximum = questionary.text(
                "Maximum residues",
                default=str(config.max_length),
                validate=lambda s, minimum=minimum: (
                    s.isdecimal()
                    and int(s) >= int(minimum)
                    or "Maximum must be at least the minimum."
                ),
                style=STYLE,
            ).ask()
            if maximum is None:
                if field is not None:
                    return config
                continue
            updated = replace(config, min_length=int(minimum), max_length=int(maximum))
        try:
            # Several saved paths may need repair. Validate them together when
            # saving/installing, after the user has finished editing Settings.
            config = updated.validated(check_paths=False)
        except ValueError as exc:
            console.print(Text(str(exc), style="red"))
        if field is not None:
            return config


QUIET_AFTER = 30
STAGE_LABELS = {
    "inspect": "Inspecting",
    "download": "Downloading",
    "filter": "Filtering",
    "prepare": "Preparing arrays",
    "verify": "Verifying",
    "runtime": "Environment",
    "reuse": "Reusing",
}


def short_duration(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes}m {seconds:02d}s" if minutes else f"{seconds}s"


class StepStatus(SpinnerColumn):
    def render(self, task: Task) -> RenderableType:
        state = task.fields.get("state")
        if state == "failed":
            return Text("✗", style="red")
        if state == "interrupted":
            return Text("■", style="yellow")
        return super().render(task)


class StepDetail(ProgressColumn):
    """Counts or the latest installer activity, plus how long the step has been quiet.

    A bar appears only for measured progress, never as an indeterminate pulse.
    """

    def render(self, task: Task) -> RenderableType:
        text = Text(task.fields.get("detail", ""), style="dim", no_wrap=True)
        if task.finished:
            return text
        if task.total is not None:
            row = Table.grid(padding=(0, 1))
            row.add_row(
                ProgressBar(total=task.total, completed=task.completed, width=20),
                text,
            )
            return row
        quiet = time.monotonic() - task.fields.get("last_output", time.monotonic())
        if quiet < QUIET_AFTER:
            return text
        # Lead with the quiet time so narrow terminals truncate the activity instead.
        note = Text(f"no new output for {short_duration(quiet)}", no_wrap=True)
        if text:
            note.append(" · ", style="dim")
            note.append_text(text)
        return note


class SetupSteps:
    """Show each setup step as its own row; finished rows keep their duration."""

    def __init__(self, progress: Progress) -> None:
        self.progress = progress
        self.task: TaskID | None = None
        self.stage = ""
        self.message = ""

    def emit(self, event: ProgressEvent) -> None:
        now = time.monotonic()
        if event.stage == "output":
            if self.task is not None:
                fields = {"detail": event.activity} if event.activity else {}
                self.progress.update(self.task, last_output=now, **fields)
            return
        # A message-less event announces a stage whose first named update follows.
        if (
            self.task is None
            or event.stage != self.stage
            or (event.message != self.message and self.message)
        ):
            self.finish("done")
            self.task = self.progress.add_task(
                "", total=None, detail="", last_output=now, state="running"
            )
        self.stage, self.message = event.stage, event.message
        word = STAGE_LABELS.get(event.stage, event.stage)
        if event.stage in ("runtime", "verify", "inspect") and event.total is None:
            label = event.message or word
        else:
            label = f"{word} · {event.message}" if event.message else word
        fields = {"description": label, "last_output": now}
        if event.total is not None:
            format_count = decimal if event.unit == "bytes" else lambda n: f"{n:,}"
            detail = f"{format_count(event.completed)} / {format_count(event.total)}"
            if event.retained is not None:
                detail += f" · {event.retained:,} kept"
            fields.update(
                detail=detail, total=event.total, completed=event.completed
            )
        elif event.activity:
            fields["detail"] = event.activity
        self.progress.update(self.task, **fields)

    def finish(self, state: str) -> None:
        if self.task is None:
            return
        task = next(task for task in self.progress.tasks if task.id == self.task)
        total = task.total or 1
        self.progress.update(self.task, total=total, completed=total, state=state)
        self.task = None

    def complete(self) -> None:
        if self.task is None and not self.progress.tasks:
            self.task = self.progress.add_task("Ready", total=None, detail="")
        self.finish("done")


def perform_setup(config: SetupConfig, *, interactive: bool) -> int:
    ensure_example_target(TARGETS)
    results: dict[str, dict[str, str]] = {}
    failed: list[str] = []
    for index, component in enumerate(config.components, 1):
        while True:
            console.print(f"\n[bold cyan]{index:02d}[/]  [bold]{LABELS[component]}[/]")
            if component in ("complexa", "esmfold2"):
                log = (
                    config.models_dir.expanduser().resolve()
                    / ".mubind"
                    / "logs"
                    / f"{component}.log"
                )
                console.print(Text.assemble(("    Full log  ", "dim"), str(log)))
            progress = Progress(
                StepStatus(finished_text="[green]✓[/]"),
                TextColumn("{task.description}"),
                StepDetail(
                    table_column=Column(ratio=1, no_wrap=True, overflow="ellipsis")
                ),
                TimeElapsedColumn(),
                console=console,
                expand=True,
            )
            steps = SetupSteps(progress)

            try:
                with progress:
                    try:
                        results[component] = setup_component(
                            component, config, steps.emit
                        )
                    except KeyboardInterrupt:
                        steps.finish("interrupted")
                        raise
                    except BaseException:
                        steps.finish("failed")
                        raise
                    steps.complete()
                break
            except KeyboardInterrupt:
                console.print(
                    "\n[yellow]Setup interrupted. Run mubind setup to resume.[/]"
                )
                return 130
            except Exception as exc:  # noqa: BLE001 - component failures belong in the retry UI
                console.print(
                    Panel(
                        Text(str(exc)),
                        title=f"{LABELS[component]} needs attention",
                        border_style="red",
                    )
                )
                if not interactive:
                    failed.append(component)
                    break
                choice = select(
                    "Next step",
                    [
                        questionary.Choice("Retry this component", "retry"),
                        questionary.Choice("Continue with the others", "continue"),
                        questionary.Choice("Exit setup", "exit"),
                    ],
                )
                if choice == "retry":
                    continue
                failed.append(component)
                if choice in (None, "exit"):
                    return 1
                break
    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold")
    summary.add_column(overflow="fold")
    for component, paths in results.items():
        location = (
            paths.get("dataset") or paths.get("executable") or str(config.models_dir)
        )
        summary.add_row(f"[green]✓[/] {LABELS[component]}", Text(location))
    for component in failed:
        summary.add_row(f"[yellow]○[/] {LABELS[component]}", "Run setup again to retry")
    console.print(
        Panel(
            summary,
            title="Ready to use" if not failed else "Setup partially complete",
            border_style="green" if not failed else "yellow",
            padding=(1, 2),
        )
    )
    return 1 if failed else 0


def component_choice(component: str, included: bool, ready: bool, width: int) -> questionary.Choice:
    label = f"{LABELS[component]:<{width}}  "
    if not included:
        title = [("class:status class:muted", "– "), ("class:muted", label), ("class:status class:muted", "Skipped")]
    elif ready:
        title = [("class:status class:ok", "✓ "), ("class:text", label), ("class:status class:ok", "Ready")]
    else:
        title = [("class:status class:pending", "○ "), ("class:text", label), ("class:status class:pending", "Not installed")]
    return questionary.Choice(title, component)


def setup_settings_screen(config: SetupConfig) -> SetupConfig:
    while True:
        entries = [
            ("components", "Included components", f"{len(config.components)} of {len(COMPONENTS)}"),
            ("models_dir", "Model folder", config.models_dir),
            ("data_dir", "Data folder", config.data_dir),
            ("lengths", "Training lengths", f"{config.min_length}–{config.max_length} residues"),
            ("source", "Existing dataset", config.source or "Automatic"),
        ]
        choice = select(
            "",
            [
                *aligned_choices(console, entries),
                questionary.Choice("Back", "back"),
            ],
            instruction="mubind · Setup settings",
        )
        if choice in (None, "back"):
            return config
        config = configure(config, field=choice)


def setup_screen(config: SetupConfig, config_path: Path | None = None) -> int:
    width = max(len(LABELS[c]) for c in COMPONENTS)
    focus = "start"
    while True:
        statuses = setup_statuses(config)
        missing = sum(statuses.get(c) != "Ready" for c in config.components)
        install_choices = []
        if missing:
            install_choices = [
                questionary.Choice(f"Install missing ({missing})", "start"),
                questionary.Separator(" "),
            ]
        elif focus == "start":
            focus = COMPONENTS[0]
        choice = select(
            "",
            [
                *install_choices,
                *(
                    component_choice(
                        c,
                        c in config.components,
                        statuses.get(c) == "Ready",
                        width,
                    )
                    for c in COMPONENTS
                ),
                questionary.Separator(" "),
                questionary.Choice("Settings", "settings"),
                questionary.Choice("Back", "back"),
            ],
            instruction="mubind · Setup",
            default=focus,
        )
        if choice in (None, "back"):
            return 0
        if choice in COMPONENTS:
            config = replace(
                config,
                components=tuple(c for c in COMPONENTS if c in config.components or c == choice),
            )
            save_config(config, config_path)
            result = perform_setup(replace(config, components=(choice,)), interactive=True)
            if result:
                return result
            focus = choice
        elif choice == "settings":
            config = setup_settings_screen(config)
            focus = "settings"
        elif choice == "start":
            save_config(config, config_path)
            return perform_setup(config, interactive=True)


def checkpoint_line() -> str:
    """Name the shared checkpoint without loading it, so the menu opens instantly."""
    from ..setup.checkpoint import default_checkpoint

    try:
        path = default_checkpoint()
    except (ValueError, TypeError, OSError):
        path = None
    if path is None:
        return "checkpoint  none chosen"
    line = f"checkpoint  {path.name} · {short_path(path.parent)}"
    room = max(console.width - 4, 20)
    return line if len(line) <= room else line[: room - 1] + "…"


def change_checkpoint() -> None:
    """One choice shared by Generate and Evaluate, saved across restarts."""
    from ..inference.config import DesignConfig, default_design_checkpoint
    from .design import DesignUI

    DesignUI(console, STYLE).choose_checkpoint(
        DesignConfig(checkpoint=default_design_checkpoint())
    )


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand:
        return
    if not interactive_terminal():
        console.print("Open mubind in a terminal to design minibinders, train or evaluate. Use mubind --help for all commands.")
        return
    from .home import HomeReadiness

    while True:
        readiness = HomeReadiness(load_config, component_status, checkpoint_line, LABELS)
        # Actions first; the checkpoint in use is shown above them, and its
        # change option sits with Setup because both are settings.
        choices = [
            questionary.Choice("Generate", "generate"),
            questionary.Choice("Evaluate", "evaluate"),
            questionary.Choice("Train", "train"),
            questionary.Separator(" "),
            questionary.Choice("Change checkpoint", "checkpoint"),
            questionary.Choice(readiness.title, "setup"),
            questionary.Choice("Exit", "exit"),
        ]
        finish_provider_startup()
        choice = select(
            "mubind",
            choices,
            instruction=readiness.instruction,
            style=HOME_STYLE,
            default="generate",
            erase=True,
            readiness=readiness,
        )
        if choice in (None, "exit"):
            return
        try:
            if choice == "train":
                from .train import TrainingUI, build_session

                TrainingUI(console, STYLE).screen(build_session(interactive=True))
            elif choice == "generate":
                from .design import DesignUI

                DesignUI(console, STYLE).screen()
            elif choice == "evaluate":
                from .evaluate_ui import EvaluationUI

                EvaluationUI(console, STYLE).screen()
            elif choice == "checkpoint":
                change_checkpoint()
            elif choice == "setup":
                setup_screen(load_config())
        except (ValueError, TypeError, OSError, RuntimeError) as exc:
            console.print(Text(str(exc), style="red"))


@app.command("setup")
def setup_command(
    data_dir: Annotated[
        Path | None, typer.Option(help="Training data directory. Default: artifacts/datasets.")
    ] = None,
    models_dir: Annotated[
        Path | None, typer.Option(help="Model directory. Default: artifacts/models.")
    ] = None,
    min_length: Annotated[
        int | None, typer.Option(min=1, help="Minimum residues to keep. Default: 50.")
    ] = None,
    max_length: Annotated[
        int | None, typer.Option(min=1, help="Maximum residues to keep. Default: 70.")
    ] = None,
    source: Annotated[
        Path | None,
        typer.Option(help="Existing raw or filtered ProstT5 Parquet directory."),
    ] = None,
    automatic_source: Annotated[
        bool,
        typer.Option(
            "--automatic-source", help="Clear a previously saved local source."
        ),
    ] = False,
    only: Annotated[
        list[Component] | None,
        typer.Option(help="Set up only this component; repeat to select several. Use prostt5 to prepare training data."),
    ] = None,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes", "-y", help="Start with these settings without prompting."
        ),
    ] = False,
    config_path: Annotated[
        Path | None, typer.Option("--config", help="Use a separate settings file.")
    ] = None,
) -> None:
    """Set up models and tools for binder design. Training data is optional (--only prostt5)."""
    try:
        config = load_config(config_path)
        updates = {
            name: value
            for name, value in {
                "data_dir": data_dir,
                "models_dir": models_dir,
                "min_length": min_length,
                "max_length": max_length,
                "source": source,
            }.items()
            if value is not None
        }
        for name in ("data_dir", "models_dir", "source"):
            if name in updates:
                updates[name] = updates[name].expanduser().absolute()
        if automatic_source:
            if source is not None:
                raise ValueError("Choose --source or --automatic-source, not both.")
            updates["source"] = None
        if only:
            updates["components"] = tuple(dict.fromkeys(c.value for c in only))
        config = replace(config, **updates).validated(
            check_paths=yes or not interactive_terminal()
        )
        if yes or not interactive_terminal():
            overview(config)
        if yes:
            save_config(config, config_path)
            raise typer.Exit(perform_setup(config, interactive=False))
        if not interactive_terminal():
            console.print(
                "[yellow]Interactive setup needs a terminal. Use --yes to start.[/]"
            )
            raise typer.Exit(2)
        raise typer.Exit(setup_screen(config, config_path))
    except (ValueError, OSError) as exc:
        console.print(Text(str(exc), style="red"))
        raise typer.Exit(1) from exc


@app.command("sample")
def sample_command(
    checkpoint: Annotated[
        Path | None, typer.Argument(help="mubind training checkpoint to sample.")
    ] = None,
    count: Annotated[
        int, typer.Option(min=1, help="Number of candidates to sample.")
    ] = 16,
    length: Annotated[
        int, typer.Option(min=1, help="Requested sequence length (a soft condition).")
    ] = 60,
    out: Annotated[
        Path | None,
        typer.Option(
            help="New output folder; defaults to a timestamped folder in artifacts/generations/."
        ),
    ] = None,
    seed: Annotated[
        int, typer.Option(min=0, help="Random seed for reproducible sampling.")
    ] = 0,
    sampler: Annotated[
        str, typer.Option(help="Diffusion sampler: ddim or ddpm.")
    ] = "ddim",
    steps: Annotated[int, typer.Option(min=1, help="DDIM sampling steps.")] = 50,
    ddim_eta: Annotated[
        float, typer.Option(min=0, help="DDIM stochasticity (0 is deterministic).")
    ] = 0.0,
    batch_size: Annotated[
        int, typer.Option(min=1, help="Candidates per sampling batch.")
    ] = 16,
    charge: Annotated[
        float | None, typer.Option(help="Optional charge condition in training units.")
    ] = None,
    hydrophobicity: Annotated[
        float | None,
        typer.Option(help="Optional hydrophobicity condition in training units."),
    ] = None,
    device: Annotated[
        str, typer.Option(help="auto, cpu, or a CUDA device such as cuda:0.")
    ] = "auto",
    precision: Annotated[
        str,
        typer.Option(help="auto (checkpoint precision on CUDA), fp32, bf16, or fp16."),
    ] = "auto",
) -> None:
    """Generate sequence candidates. With no checkpoint, open the generation screen."""
    from ..inference.config import GenerationConfig
    from .sample import GenerationUI

    settings = GenerationConfig(
        checkpoint=existing_artifact(checkpoint) if checkpoint is not None else None,
        out=out,
        count=count,
        length=length,
        charge=charge,
        hydrophobicity=hydrophobicity,
        batch_size=batch_size,
        sampler=sampler,
        sampling_steps=steps,
        ddim_eta=ddim_eta,
        seed=seed,
        device=device,
        precision=precision,
    )
    ui = GenerationUI(console, STYLE)
    try:
        if checkpoint is None:
            if interactive_terminal():
                raise typer.Exit(ui.screen(settings))
            console.print(
                "Choose a checkpoint: mubind sample CHECKPOINT.pt, or open mubind sample in a terminal."
            )
            raise typer.Exit(2)
        ui.overview(settings)
        raise typer.Exit(ui.run(settings))
    except typer.Exit:
        raise
    except (ValueError, TypeError, OSError, RuntimeError, KeyError) as exc:
        console.print(Text(str(exc), style="red"))
        raise typer.Exit(1) from exc


@app.command("provider")
def provider_command(
    models_dir: Annotated[
        Path | None, typer.Option(help="Override the model directory saved by setup.")
    ] = None,
    device: Annotated[
        str, typer.Option(help="CUDA device shared with generation, such as cuda:0.")
    ] = "auto",
    timeout: Annotated[
        float, typer.Option(min=1, help="Seconds to wait for each model to load.")
    ] = 1800,
) -> None:
    """Keep ESMFold2 and Complexa loaded for repeated experiments until Ctrl+C."""
    from .startup import finish_provider_startup, start_provider_startup

    start_provider_startup()
    try:
        from ..inference.services import WorkerFailed, run_provider
        from .provider_display import ProviderDisplay

        try:
            run_provider(
                models_dir=models_dir,
                device=device,
                timeout=timeout,
                display=ProviderDisplay(console),
            )
        except WorkerFailed as exc:
            raise typer.Exit(1) from exc
        except (ValueError, TypeError, OSError, RuntimeError, KeyError) as exc:
            finish_provider_startup()
            console.print(Text(str(exc), style="red"))
            raise typer.Exit(1) from exc
    finally:
        finish_provider_startup()


@app.command("generate")
def generate_command(
    target: Annotated[
        Path | None, typer.Argument(help="Target PDB; omit to open the target picker.")
    ] = None,
    checkpoint: Annotated[
        Path | None,
        typer.Option(help="300k learned-inpainting or compatible mubind checkpoint."),
    ] = None,
    targets_dir: Annotated[
        Path, typer.Option(help="Folder shown by the target picker.")
    ] = TARGETS,
    out: Annotated[Path | None, typer.Option(help="New output folder.")] = None,
    target_spec: Annotated[
        str | None,
        typer.Option(
            help="Chain/range, such as A10-120; default is the longest chain."
        ),
    ] = None,
    hotspots: Annotated[
        str, typer.Option(help="Comma-separated target residue IDs.")
    ] = "",
    num_complexa: Annotated[
        int,
        typer.Option(min=1, help="Final Complexa scaffold count (including replicas)."),
    ] = 100,
    binder_length: Annotated[
        int, typer.Option(min=1, max=70, help="Complexa binder length.")
    ] = 64,
    num_diffusion: Annotated[
        int, typer.Option(min=1, help="Redesign candidates per selected scaffold.")
    ] = 5,
    redesign_batch_size: Annotated[
        int, typer.Option(min=1, help="Maximum redesign sequences sampled together, across scaffolds.")
    ] = 2048,
    complexa_batch_size: Annotated[
        int, typer.Option(min=1, help="Maximum Complexa candidates sampled together; lower to save VRAM.")
    ] = 16,
    complexa_steps: Annotated[
        int, typer.Option(min=1, help="Complexa diffusion steps.")
    ] = 400,
    replicas: Annotated[
        int, typer.Option(min=1, help="Complexa best-of replicas.")
    ] = 4,
    structural_guidance: Annotated[
        float, typer.Option(min=0, help="AA/PAD structural guidance scale.")
    ] = 5.0,
    sampler: Annotated[
        str, typer.Option(help="Redesign diffusion sampler: ddpm or ddim.")
    ] = "ddpm",
    sampling_steps: Annotated[
        int | None,
        typer.Option(min=1, help="DDIM sampling steps when --sampler ddim is used."),
    ] = None,
    ddim_eta: Annotated[
        float, typer.Option(min=0, help="DDIM stochasticity (0 is deterministic).")
    ] = 0.0,
    scaffold_index: Annotated[
        int | None,
        typer.Option(min=0, help="Force a final scaffold index, starting at 0."),
    ] = None,
    all_scaffolds: Annotated[
        bool, typer.Option(help="Redesign every final scaffold.")
    ] = False,
    top_scaffolds: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Redesign the top N folded scaffolds; skips interactive selection.",
        ),
    ] = None,
    reuse_scaffolds: Annotated[
        Path | None,
        typer.Option(
            "--reuse-scaffolds",
            "--skip-complexa",
            help="Reuse an existing Complexa folder with rewards CSVs.",
        ),
    ] = None,
    edit_residues: Annotated[
        str | None,
        typer.Option(
            help="Inpaint binder positions, e.g. 12-20,35-40 (1-based, inclusive). "
            "Retain all other residues and the original length; applies to each selected scaffold."
        ),
    ] = None,
    seed: Annotated[
        int | None,
        typer.Option(min=0, help="Random seed; otherwise generated and saved."),
    ] = None,
    device: Annotated[
        str, typer.Option(help="auto, cpu (reuse only), or cuda:0.")
    ] = "auto",
    folding_endpoint: Annotated[
        str, typer.Option(help="auto, a Unix socket URL, or HTTP URL.")
    ] = "auto",
    models_dir: Annotated[
        Path | None, typer.Option(help="Override the model directory saved by setup.")
    ] = None,
    foldseek: Annotated[
        str, typer.Option(help="Override the Foldseek executable.")
    ] = "foldseek",
) -> None:
    """Design minibinders for a target: Complexa → redesign → complex ranking."""
    from ..inference.config import DesignConfig, default_design_checkpoint
    from ..inference.helpers.inpainting import parse_edit_residues
    from .design import DesignUI

    try:
        if edit_residues is not None:
            parse_edit_residues(edit_residues)
        settings = DesignConfig(
            target=existing_artifact(target) if target is not None else None,
            checkpoint=existing_artifact(checkpoint) if checkpoint is not None else default_design_checkpoint(models_dir),
            targets_dir=targets_dir,
            out=out,
            target_spec=target_spec,
            hotspots=tuple(x.strip() for x in hotspots.split(",") if x.strip()),
            num_complexa=num_complexa,
            binder_length=binder_length,
            num_diffusion=num_diffusion,
            redesign_batch_size=redesign_batch_size,
            complexa_batch_size=complexa_batch_size,
            complexa_steps=complexa_steps,
            replicas=replicas,
            structural_guidance=structural_guidance,
            sampler=sampler,
            sampling_steps=sampling_steps,
            ddim_eta=ddim_eta,
            scaffold_index=scaffold_index,
            all_scaffolds=all_scaffolds,
            top_scaffolds=top_scaffolds,
            reuse_scaffolds=existing_artifact(reuse_scaffolds) if reuse_scaffolds is not None else None,
            edit_residues=edit_residues,
            seed=seed,
            device=device,
            folding_endpoint=folding_endpoint,
            models_dir=models_dir,
            foldseek=foldseek,
        )
        ui = DesignUI(console, STYLE)
        if target is None:
            if interactive_terminal():
                raise typer.Exit(ui.screen(settings))
            console.print(
                "Choose a target: mubind generate TARGET.pdb --checkpoint MODEL.pt, or open mubind generate in a terminal."
            )
            raise typer.Exit(2)
        ui.overview(settings)
        raise typer.Exit(ui.run(settings, interactive=interactive_terminal()))
    except typer.Exit:
        raise
    except (ValueError, TypeError, OSError, RuntimeError, KeyError) as exc:
        console.print(Text(str(exc), style="red"))
        raise typer.Exit(1) from exc


@app.command("view")
def view_command(
    run: Annotated[
        Path | None,
        typer.Argument(help="Generation run folder. Defaults to the latest viewable run."),
    ] = None,
    port: Annotated[
        int, typer.Option(min=1, max=65535, help="Local viewer port; forward this port over SSH.")
    ] = 8000,
    export: Annotated[
        Path | None,
        typer.Option(help="Save a standalone HTML file and exit without starting a server."),
    ] = None,
    browser: Annotated[
        bool, typer.Option("--browser/--no-browser", help="Open a local browser when available.")
    ] = True,
) -> None:
    """Explore predicted complexes in a browser, locally or through SSH forwarding."""
    from .viewer import load_preview, open_browser, render_html, save_html, viewer_server

    try:
        preview = load_preview(run)
        document = render_html(preview)
        console.print(Text(f"{len(preview.candidates)} designs · {preview.directory}"))
        if preview.skipped:
            console.print(
                f"[dim]Skipped {preview.skipped} candidates with unfinished or missing structures.[/]"
            )
        if export is not None:
            path = save_html(export, document)
            console.print(Text(f"Saved {path} · open in any browser, including offline."))
            return
        with viewer_server(document, port) as server:
            path = save_html(preview.directory / "view.html", document)
            url = f"http://localhost:{port}"
            console.print(Text(f"Open {url} in your browser.", style="green"))
            console.print("For SSH, run this in a separate terminal on your laptop:")
            console.print(Text(f"  ssh -N -L {port}:127.0.0.1:{port} user@server"))
            console.print("Replace user@server with your usual SSH destination.")
            console.print(Text(f"Offline copy: {path}"))
            console.print("Keep this command running while viewing. Ctrl+C stops the viewer.")
            if browser:
                open_browser(url)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                console.print("\nViewer stopped.")
    except (ValueError, OSError) as exc:
        console.print(Text(str(exc), style="red"))
        raise typer.Exit(1) from exc


@app.command("train")
def train_command(
    dataset: Annotated[
        Path | None,
        typer.Argument(
            help="Override the prepared ProstT5 directory from saved setup settings."
        ),
    ] = None,
    config_path: Annotated[
        Path | None, typer.Option("--config", help="Training JSON settings.")
    ] = None,
    out: Annotated[Path | None, typer.Option(help="Output checkpoint path.")] = None,
    steps: Annotated[
        int | None,
        typer.Option(min=1, help="Total steps to reach, including resumed steps."),
    ] = None,
    batch_size: Annotated[
        int | None, typer.Option(min=1, help="Examples per training batch.")
    ] = None,
    device: Annotated[
        str | None, typer.Option(help="auto, cpu, or a CUDA device such as cuda:0.")
    ] = None,
    precision: Annotated[
        str | None, typer.Option(help="fp32, bf16, or fp16 (CUDA only).")
    ] = None,
    evaluation: Annotated[
        str | None,
        typer.Option(help="Evaluation regime: full, sequence, loss, or off."),
    ] = None,
    resume: Annotated[
        Path | None, typer.Option(help="Resume a mubind training checkpoint.")
    ] = None,
    setup_config: Annotated[
        Path | None,
        typer.Option(help="Resolve ProstT5 using an alternative setup settings file."),
    ] = None,
) -> None:
    """Train on ProstT5. With no options, open the interactive training screen."""
    # Keep setup/help usable without importing the training stack.
    from .train import TrainingUI, build_session

    try:
        direct = any(
            value is not None
            for value in (
                dataset,
                config_path,
                out,
                steps,
                batch_size,
                device,
                precision,
                evaluation,
                resume,
            )
        )
        session = build_session(
            dataset=dataset,
            config_path=config_path,
            resume=resume,
            setup_path=setup_config,
            interactive=not direct,
        )
        config = session.config
        if out is not None:
            config.out = str(out)
        if steps is not None:
            config.steps = steps
        if batch_size is not None:
            config.data.batch_size = batch_size
        if device is not None:
            config.device = device
        if precision is not None:
            config.precision = precision
        if evaluation is not None:
            from dataclasses import replace

            config.evaluation = replace(config.evaluation, regime=evaluation, suites={})
        ui = TrainingUI(console, STYLE)
        if not direct:
            if interactive_terminal():
                raise typer.Exit(ui.screen(session, setup_config))
            ui.overview(session)
            console.print(
                "Interactive training needs a terminal. Use --config or --steps "
                "to start directly."
            )
            raise typer.Exit(2)
        if session.resume is not None and config.steps <= session.completed_steps:
            raise ValueError(
                f"Checkpoint is already at step {session.completed_steps:,}. "
                "Use --steps with a larger total to continue training."
            )
        ui.overview(session)
        raise typer.Exit(ui.run(session))
    except typer.Exit:
        raise
    except (
        ValueError,
        TypeError,
        OSError,
        FloatingPointError,
        RuntimeError,
        KeyError,
    ) as exc:
        console.print(Text(str(exc), style="red"))
        raise typer.Exit(1) from exc
