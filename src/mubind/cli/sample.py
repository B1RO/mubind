"""Interactive checkpoint selection and sequence generation."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import questionary
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..paths import GENERATIONS
from ..inference.config import GenerationConfig, next_output
from .generation_progress import GenerationProgress
from .terminal_ui import aligned_choices, screen_intro, select_prompt, short_path


def generate(*args, **kwargs):
    from ..inference.generate import generate as run

    return run(*args, **kwargs)


class GenerationUI:
    def __init__(self, console: Console, style=None):
        self.console, self.style = console, style

    def select(self, message, choices, **options):
        return select_prompt(
            message, choices=choices, style=self.style, qmark="›", pointer="❯", **options
        ).ask()

    def path(self, message, current=""):
        answer = questionary.path(message, default=str(current), style=self.style).ask()
        return (
            Path(answer).expanduser() if answer is not None and answer.strip() else None
        )

    def overview(self, settings: GenerationConfig):
        rows = Table.grid(padding=(0, 2))
        rows.add_column(style="dim")
        rows.add_column(overflow="fold")
        rows.add_row(
            "Checkpoint", Text(str(settings.checkpoint or "Choose a trained model"))
        )
        rows.add_row(
            "Candidates",
            f"{settings.count:,} · requested length {settings.length} residues",
        )
        rows.add_row(
            "Sampling",
            f"{settings.sampler.upper()} · {settings.sampling_steps} steps · EMA when available"
            if settings.sampler == "ddim"
            else "DDPM · full schedule · EMA when available",
        )
        conditions = []
        if settings.charge is not None:
            conditions.append(f"charge {settings.charge:g}")
        if settings.hydrophobicity is not None:
            conditions.append(f"hydrophobicity {settings.hydrophobicity:g}")
        if conditions:
            rows.add_row("Properties", ", ".join(conditions))
        rows.add_row(
            "Save to",
            Text(
                str(settings.out)
                if settings.out
                else f"{GENERATIONS}/ · current date and time"
            ),
        )
        self.console.print(
            Panel(
                Group(
                    rows,
                    Text(),
                    Text("Sequence candidates; binding is not evaluated.", style="dim"),
                ),
                title="[bold cyan]mubind · Sample sequences[/]",
                border_style="cyan",
                padding=(1, 2),
            )
        )

    def choose_checkpoint(self, settings: GenerationConfig) -> GenerationConfig:
        from .checkpoint_catalog import discover_design_checkpoints

        paths = [option.path for option in discover_design_checkpoints(settings.checkpoint, check=False)]
        selected = self.select(
            "Choose checkpoint",
            [
                *(questionary.Choice(str(path), path) for path in paths),
                questionary.Choice("Choose checkpoint path…", "path"),
                questionary.Choice("Back", "back"),
            ],
        )
        if selected in (None, "back"):
            return settings
        path = (
            self.path("mubind checkpoint file", settings.checkpoint or "")
            if selected == "path"
            else selected
        )
        if path is None:
            return settings
        if not path.is_file():
            raise ValueError(f"Checkpoint file not found: {path}")
        return replace(settings, checkpoint=path)

    def configure(self, settings: GenerationConfig, field=None) -> GenerationConfig:
        integers = {
            "count": "Number of candidates",
            "length": "Requested length (residues)",
            "sampling_steps": "Sampling steps",
            "batch_size": "Batch size",
            "seed": "Random seed",
        }
        while True:
            choice = field or self.select(
                "Sampling settings",
                [
                    *(
                        questionary.Choice(f"{label} · {getattr(settings, key)}", key)
                        for key, label in integers.items()
                    ),
                    questionary.Choice(f"Charge · {settings.charge if settings.charge is not None else 'No condition'}", "charge"),
                    questionary.Choice(f"Hydrophobicity · {settings.hydrophobicity if settings.hydrophobicity is not None else 'No condition'}", "hydrophobicity"),
                    questionary.Choice(f"Sampler · {settings.sampler.upper()}", "sampler"),
                    questionary.Choice(f"Device · {settings.device}", "device"),
                    questionary.Choice(f"Precision · {settings.precision}", "precision"),
                    questionary.Choice(f"Output folder · {settings.out or 'Automatic'}", "out"),
                    questionary.Choice("Back", "back"),
                ],
            )
            if choice in (None, "back"):
                return settings
            if choice in integers:
                minimum = 0 if choice == "seed" else 1
                answer = questionary.text(
                    integers[choice],
                    default=str(getattr(settings, choice)),
                    style=self.style,
                    validate=lambda value, minimum=minimum: (
                        value.isdecimal()
                        and int(value) >= minimum
                        or f"Enter an integer of at least {minimum}."
                    ),
                ).ask()
                if answer is not None:
                    settings = replace(settings, **{choice: int(answer)})
            elif choice in {"charge", "hydrophobicity"}:

                def validate(value):
                    try:
                        return (
                            not value.strip()
                            or math.isfinite(float(value))
                            or "Enter a finite number."
                        )
                    except ValueError:
                        return "Enter a number, or leave empty for no condition."

                current = getattr(settings, choice)
                answer = questionary.text(
                    f"{choice.capitalize()} (empty = no condition)",
                    default="" if current is None else str(current),
                    style=self.style,
                    validate=validate,
                ).ask()
                if answer is not None:
                    settings = replace(
                        settings, **{choice: float(answer) if answer.strip() else None}
                    )
            elif choice == "out":
                value = self.path("New output folder", settings.out or next_output())
                if value is not None:
                    settings = replace(settings, out=value)
            else:
                options = (
                    ["ddim", "ddpm"]
                    if choice == "sampler"
                    else ["auto", "cpu", "cuda"]
                    if choice == "device"
                    else ["auto", "fp32", "bf16", "fp16"]
                )
                value = self.select(choice.capitalize(), options)
                if value is not None:
                    settings = replace(settings, **{choice: value})
            if field is not None:
                return settings

    def menu(self, settings):
        sampler_summary = (
            f"DDIM {settings.sampling_steps} steps"
            if settings.sampler == "ddim"
            else "DDPM"
        )
        return [
            questionary.Choice("Start sampling", "start", disabled=None if settings.checkpoint and settings.checkpoint.is_file() else "Choose a checkpoint"),
            questionary.Separator(" "),
            *aligned_choices(self.console, [
                ("checkpoint", "Checkpoint", short_path(settings.checkpoint) if settings.checkpoint else "Choose a trained model"),
                ("count", "Candidates", f"{settings.count:,} sequences"),
                ("length", "Requested length", f"{settings.length} residues · soft condition"),
                ("charge", "Charge", settings.charge if settings.charge is not None else "No condition"),
                ("hydrophobicity", "Hydrophobicity", settings.hydrophobicity if settings.hydrophobicity is not None else "No condition"),
                ("out", "Save to", short_path(settings.out) if settings.out else f"{GENERATIONS}/ · new folder"),
                ("configure", "Settings", f"{sampler_summary} · batch {settings.batch_size} · {settings.device} · {settings.precision}"),
            ]),
            questionary.Choice("Back", "back"),
        ]

    def screen(self, settings: GenerationConfig | None = None) -> int:
        from ..setup.checkpoint import default_checkpoint

        settings = settings or GenerationConfig()
        settings = replace(settings, checkpoint=settings.checkpoint or default_checkpoint())
        while True:
            choice = self.select("", self.menu(settings), instruction=screen_intro(
                self.console, "mubind · Sample sequences",
                "Generate sequences from a checkpoint, with optional property conditions. Sampling uses no target or structure folding.",
            ))
            if choice in (None, "back"):
                return 0
            try:
                if choice == "checkpoint":
                    settings = self.choose_checkpoint(settings)
                elif choice == "configure":
                    settings = self.configure(settings)
                elif choice in {"count", "length", "charge", "hydrophobicity", "out"}:
                    settings = self.configure(settings, field=choice)
                elif choice == "start":
                    return self.run(settings)
            except (ValueError, TypeError, OSError, RuntimeError, KeyError) as exc:
                self.console.print(Text(str(exc), style="red"))

    def run(self, settings: GenerationConfig) -> int:
        with GenerationProgress(self.console) as progress:
            task = progress.add_task("Loading checkpoint…", total=None, detail="")

            def status(message):
                progress.status(task, message)

            def update(completed, total):
                progress.advance(task, "sampling", completed, total,
                                 detail=f"{completed:,}/{total:,} sampling steps")

            try:
                result = generate(settings, on_status=status, on_progress=update)
                progress.complete(task, "Generation complete")
            except KeyboardInterrupt:
                self.console.print(
                    "[yellow]Generation interrupted. Completed batches remain in the output folder.[/]"
                )
                return 130
        self.console.print(
            Text(
                f"Generated {result.count} candidates · {result.valid_count} valid sequences saved to {result.directory / 'sequences.fasta'}",
                style="green" if result.valid_count else "yellow",
            )
        )
        self.console.print(
            Text(
                f"All samples and validity details: {result.directory / 'candidates.jsonl'}",
                style="dim",
            )
        )
        return 0
