"""The ProstT5 training screen and terminal progress display."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from textwrap import fill

import questionary
import torch
from prompt_toolkit.styles import merge_styles
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from ..paths import CHECKPOINTS, existing_artifact
from .terminal_ui import aligned_choices, select_prompt, short_path
from ..evaluation.config import REGIMES, STRUCTURAL_CHECKS
from ..evaluation.evaluator import evaluation_directory, structural_checkpoint
from ..preprocessing.prostt5.steps.validate import (
    DatasetInfo,
    discover_datasets,
    inspect_dataset,
    is_ready_for_training,
)
from ..setup.config import load_config as load_setup_config
from ..setup.tools import find_foldseek
from ..training.helpers.checkpoint import load_checkpoint
from ..training.config import TrainingConfig, config_from_dict
from ..training.trainer import checkpoint_paths, resolve_device, train

EVALUATION_LABELS = {
    "full": "Full · sequences and structures",
    "sequence": "Sequences only",
    "loss": "Denoising loss only",
    "off": "Off",
    "custom": "Custom evaluation suites",
}


TITLE = "mubind · Train"
# The screen title matches the purple home title; rows keep the shared style.
TITLE_STYLE = questionary.Style([("question", "fg:#d787ff bold")])


@dataclass
class TrainingSession:
    config: TrainingConfig
    settings: str = "300k recipe · new run"
    resume: Path | None = None
    completed_steps: int = 0
    problem: str = ""


def prepared_prostt5(setup_path: Path | None = None) -> DatasetInfo:
    """Resolve the same prepared ProstT5 subset that setup would reuse."""
    setup = load_setup_config(setup_path)
    source = setup.source.expanduser().resolve() if setup.source is not None else None
    for info in discover_datasets(setup.data_dir.expanduser().resolve()):
        if is_ready_for_training(info, setup.min_length, setup.max_length, source):
            return info
    raise ValueError(
        f"Prepared ProstT5 data for {setup.min_length}–{setup.max_length} residues "
        f"was not found in {setup.data_dir}. Run mubind setup --only prostt5"
        + (f" --config {setup_path}" if setup_path is not None else "")
        + "."
    )


def next_checkpoint(directory: Path = CHECKPOINTS) -> Path:
    index = 1
    while True:
        path = directory / f"experiment-{index:03d}.pt"
        if not any(
            candidate.exists()
            for candidate in (
                *checkpoint_paths(path),
                evaluation_directory(path),
                structural_checkpoint(path),
            )
        ):
            return path
        index += 1


def build_session(
    *,
    dataset: Path | None = None,
    config_path: Path | None = None,
    resume: Path | None = None,
    setup_path: Path | None = None,
    interactive: bool = False,
) -> TrainingSession:
    session = TrainingSession(TrainingConfig())
    supplied = {}
    if resume is not None:
        resume = existing_artifact(resume).resolve()
        payload = load_checkpoint(resume)
        session = TrainingSession(
            config_from_dict(payload["config"]),
            "From checkpoint",
            resume,
            int(payload["step"]),
        )
        session.config.out = str(resume)
    if config_path is not None:
        supplied = json.loads(config_path.expanduser().read_text())
        if not isinstance(supplied, dict):
            raise ValueError("Training settings must be a JSON object")
        session.config = config_from_dict(supplied)
        session.settings = str(config_path)
        if resume is not None and "out" not in supplied:
            session.config.out = str(resume)
    elif interactive and resume is None:
        session.config.out = str(next_checkpoint())
    if dataset is not None:
        session.config.data.path = str(existing_artifact(dataset).resolve())
    elif session.config.data.path:
        session.config.data.path = str(existing_artifact(Path(session.config.data.path)))
    if setup_path is not None and session.config.evaluation.folding.models_dir is None:
        session.config.evaluation.folding.models_dir = str(
            load_setup_config(setup_path).models_dir.expanduser().resolve()
        )
    if not session.config.data.path:
        try:
            info = prepared_prostt5(setup_path)
            session.config.data.path = str(info.path)
            # Storage padding is not the model's canvas. The default 300k
            # recipe stays at 70 even when prepared arrays have 128 columns.
        except (ValueError, OSError) as exc:
            session.problem = str(exc)
    if (
        setup_path is not None
        and session.config.evaluation.folding.foldseek == "foldseek"
    ):
        executable = find_foldseek(models_dir=load_setup_config(setup_path).models_dir)
        if executable is not None:
            session.config.evaluation.folding.foldseek = executable
    return session


def dataset_info(session: TrainingSession) -> DatasetInfo:
    if session.problem:
        raise ValueError(session.problem)
    info = inspect_dataset(existing_artifact(Path(session.config.data.path)).resolve())
    if info is None or info.kind != "Prepared" or info.state != "ready":
        detail = f" ({info.problem})" if info is not None and info.problem else ""
        raise ValueError(
            f"Prepared ProstT5 data is not ready at {session.config.data.path}{detail}. "
            "Run mubind setup --only prostt5."
        )
    return info


def device_description(requested: str) -> str:
    try:
        device = resolve_device(requested)
        description = "CPU" if device.type == "cpu" else str(device)
        if device.type == "cuda":
            description += f" · {torch.cuda.get_device_name(device)}"
        return f"Auto → {description}" if requested == "auto" else description
    except ValueError as exc:
        return str(exc)


def training_summary(session: TrainingSession, info: DatasetInfo | None) -> str:
    """How a run with these settings will proceed, in plain sentences."""
    config = session.config
    if session.resume is not None:
        start = (
            f"mubind will continue {session.resume.name} from step "
            f"{session.completed_steps:,} to {config.steps:,}"
        )
    else:
        start = f"mubind will train a new model for {config.steps:,} steps"
    if info is None:
        data = "on prepared ProstT5 data"
    else:
        count = (info.counts or {}).get("train")
        data = (
            f"on {f'{count:,} ' if count else ''}ProstT5 sequences of "
            f"{info.minimum}–{info.maximum} residues"
        )
    try:
        device = resolve_device(config.device)
        where = (
            "on the CPU"
            if device.type == "cpu"
            else "on the GPU" if device.index is None else f"on GPU {device.index}"
        )
    except ValueError:
        where = f"on {config.device}"
    return " ".join([
        f"Given the current config, {start} {data}, in batches of "
        f"{config.data.batch_size:,} {where} with {config.precision.upper()} precision.",
        f"It will save the checkpoint to {short_path(config.out)}.",
        evaluation_summary(session),
        *([] if info is not None else ["Choose prepared ProstT5 data below to start."]),
    ])


def evaluation_summary(session: TrainingSession) -> str:
    """What the selected regime will check during training, and how often."""
    config = session.config
    evaluation = config.evaluation
    if evaluation.regime == "off" or (
        evaluation.regime == "loss" and not config.val_every
    ):
        return "It will not evaluate the model during training."
    if evaluation.regime == "loss":
        return (
            f"Every {config.val_every:,} steps it will measure denoising loss on "
            f"{config.val_batches:,} validation batches and keep the lowest-loss checkpoint."
        )
    try:
        suites = evaluation.resolved_suites()
    except (TypeError, ValueError):
        return "It will run its custom evaluation suites during training."
    structural = [
        suite for suite in suites.values()
        if STRUCTURAL_CHECKS.intersection(suite.validators)
    ]
    sequence = [
        suite for suite in suites.values()
        if not STRUCTURAL_CHECKS.intersection(suite.validators)
    ]
    checks = []
    if evaluation.denoising.enabled:
        checks.append((evaluation.denoising.every, "measure denoising loss"))
    if sequence:
        checks.append((
            min(suite.interval_steps for suite in sequence), "check generated sequences"
        ))
    if structural:
        checks.append((
            min(suite.interval_steps for suite in structural),
            "fold samples with ESMFold2 to check their structures",
        ))
    if not checks:
        return "It will not evaluate the model during training."
    groups = {}
    for every, check in sorted(checks, key=lambda item: item[0]):
        groups.setdefault(every, []).append(check)
    sentence = ", and ".join(
        f"every {every:,} steps it will {' and '.join(group)}"
        for every, group in groups.items()
    )
    sentence = sentence[0].upper() + sentence[1:] + "."
    if session.resume is not None and evaluation.run_on_resume:
        sentence += " It will also check the model once before resuming."
    elif session.resume is None and evaluation.run_on_start:
        sentence += " It will also check the model once before the first step."
    if any(suite.select_checkpoint for suite in structural):
        sentence += " The checkpoint with the best structures will also be kept."
    return sentence


class TrainingUI:
    def __init__(self, console: Console, style=None):
        self.console, self.style = console, style

    def select(self, message, choices, *, erase=False, **options):
        prompt = select_prompt(
            message,
            choices=choices,
            style=options.pop("style", self.style),
            qmark="›",
            pointer="❯",
            **options,
        )
        # An erased menu is redrawn fresh instead of piling up answered copies.
        prompt.application.erase_when_done = erase
        return prompt.ask()

    def number(self, message, current: int, minimum: int = 1):
        answer = questionary.text(
            message,
            default=str(current),
            style=self.style,
            validate=lambda value: (
                value.isdecimal()
                and int(value) >= minimum
                or f"Enter an integer of at least {minimum}."
            ),
        ).ask()
        return None if answer is None else int(answer)

    def path(self, message, current=""):
        answer = questionary.path(message, default=str(current), style=self.style).ask()
        return (
            Path(answer).expanduser() if answer is not None and answer.strip() else None
        )

    def overview(self, session: TrainingSession) -> bool:
        """Keep the launch screen focused on the choices needed to start a run."""
        config = session.config
        rows = Table.grid(padding=(0, 2))
        rows.add_column(style="dim")
        rows.add_column(overflow="fold")
        problem = ""
        try:
            info = dataset_info(session)
            rows.add_row("Data", f"ProstT5 · {info.minimum}–{info.maximum} residues")
        except ValueError as exc:
            problem = str(exc)
            rows.add_row("Data", "[yellow]ProstT5 · not ready[/]")
        rows.add_row(
            "Training",
            f"{config.steps:,} steps · batch size {config.data.batch_size:,}",
        )
        if session.resume is not None:
            rows.add_row("Resume", f"From step {session.completed_steps:,}")
        rows.add_row("Device", Text(device_description(config.device)))
        evaluation = config.evaluation
        label = EVALUATION_LABELS.get(evaluation.regime, evaluation.regime)
        if evaluation.regime == "loss" and not config.val_every:
            label = "Off · loss checks disabled"
        elif evaluation.suites and evaluation.regime in {"full", "sequence"}:
            label += " · customized"
        rows.add_row("Evaluation", Text(label))
        path = Path(config.out).expanduser().absolute()
        if path.is_relative_to(Path.cwd()):
            path = path.relative_to(Path.cwd())
        rows.add_row("Save to", Text(str(path)))
        self.console.print(
            Panel(
                Group(
                    rows,
                    *([Text(), Text(problem, style="yellow")] if problem else []),
                ),
                title="[bold cyan]mubind · Train[/]",
                title_align="left",
                border_style="cyan",
                padding=(1, 2),
            )
        )
        return not problem

    def details(self, session: TrainingSession) -> None:
        """Show technical settings and the full evaluation plan on request."""
        config = session.config
        rows = Table.grid(padding=(0, 2))
        rows.add_column(style="dim")
        rows.add_column(overflow="fold")
        problem = ""
        try:
            info = dataset_info(session)
            rows.add_row("Dataset", f"ProstT5 · {info.minimum}–{info.maximum} residues")
            rows.add_row("Prepared data", Text(str(info.path)))
            rows.add_row("Training examples", f"{info.counts['train']:,}")
        except ValueError as exc:
            problem = str(exc)
            rows.add_row("Dataset", "[yellow]ProstT5 · not ready[/]")
        rows.add_row("Settings", Text(session.settings))
        rows.add_row(
            "Model",
            f"U-Net · {config.backbone.x_channels} output channels · "
            f"{config.backbone.blocks_per_level} blocks per level · canvas {config.backbone.max_length}",
        )
        structure = config.structure
        structure_description = f"{structure.representation} · width {structure.dim}"
        if structure.context is not None:
            structure_description += f" · {structure.context.layers} contextual layers"
        if structure.cross_attention is not None:
            structure_description += " · cross-attention"
            if structure.cross_attention.bottleneck:
                structure_description += " at output and bottleneck"
        rows.add_row("3Di conditioning", structure_description)
        rows.add_row(
            "Denoising target",
            f"{config.diffusion.objective} · {config.diffusion.loss_weighting} loss weighting",
        )
        rows.add_row("Total steps", f"{config.steps:,}")
        rows.add_row("Batch size", f"{config.data.batch_size:,}")
        rows.add_row("Device", Text(device_description(config.device)))
        rows.add_row("Precision", config.precision.upper())
        rows.add_row(
            "Optimizer",
            f"{config.optimizer.name.upper()} · weight decay {config.optimizer.weight_decay:g}",
        )
        rows.add_row("Learning rate", f"{config.optimizer.lr:g}")
        rows.add_row("LR decay per step", f"× {config.optimizer.scheduler_gamma:.15g}")
        rows.add_row("Checkpoint", Text(str(Path(config.out).expanduser().absolute())))
        if session.resume is not None:
            rows.add_row("Resume from", Text(str(session.resume)))
            rows.add_row("Completed steps", f"{session.completed_steps:,}")
        self.console.print(
            Panel(
                Group(
                    rows,
                    *([Text(), Text(problem, style="yellow")] if problem else []),
                ),
                title="[bold cyan]Training details[/]",
                title_align="left",
                border_style="cyan",
                padding=(1, 3),
            )
        )
        self.evaluation_overview(session)

    def evaluation_overview(self, session: TrainingSession):
        config = session.config
        evaluation = config.evaluation
        lines = [Text(REGIMES.get(evaluation.regime, evaluation.regime), style="bold")]
        if evaluation.suites:
            lines.append(Text("Using suite settings from the training configuration."))
        if evaluation.regime == "off":
            lines.append(Text("No evaluation will run."))
        elif evaluation.regime == "loss":
            lines.append(
                Text(
                    f"Every {config.val_every:,} steps and at completion · up to {config.val_batches} batches · diffusion loss"
                    if config.val_every
                    else "Loss evaluation is disabled (val_every=0)."
                )
            )
        else:
            table = Table(
                "Evaluation", "Every", "Work per check", box=None, padding=(0, 1)
            )
            d = evaluation.denoising
            if d.enabled:
                table.add_row(
                    "Denoising",
                    f"{d.every:,} steps + end",
                    f"up to {d.batches} × {d.batch_size} examples × {d.noise_samples} corruptions",
                )
                if d.threedi_utilization:
                    table.add_row(
                        "3Di utilization",
                        f"{d.every:,} steps + end",
                        f"{d.utilization_count} fixed examples",
                    )
            suites = evaluation.resolved_suites()
            grouped = {}
            for name, suite in suites.items():
                family = (
                    "Full 3Di + property subsets"
                    if name.startswith("full_properties_")
                    else "Partial 3Di + property subsets"
                    if name.startswith("partial_properties_")
                    else {
                        "threedi_structure": "Full 3Di structures",
                        "partial_threedi": "Partial 3Di structures",
                    }.get(name, name.replace("_", " ").capitalize())
                )
                key = (family, suite.interval_steps, suite.run_at_end)
                grouped[key] = grouped.get(key, 0) + suite.candidate_count
            for (label, every, at_end), count in grouped.items():
                table.add_row(
                    label,
                    f"{every:,} steps" + (" + end" if at_end else ""),
                    f"{count} candidates",
                )
            s = evaluation.sampling
            lines.extend(
                [
                    table,
                    Text(
                        f"{'EMA' if config.ema.enabled else 'Model'} weights · {s.sampler.upper()} · "
                        f"{s.steps or config.diffusion.timesteps} sampling steps · fixed panels and seeds"
                    ),
                    Text(
                        f"{config.precision.upper()} · self-conditioning {'on' if config.self_condition.enabled else 'off'} · "
                        f"eta {s.ddim_eta:g}"
                    ),
                ]
            )
            if any(
                STRUCTURAL_CHECKS.intersection(s.validators) for s in suites.values()
            ):
                endpoint = evaluation.folding.endpoint
                lines.append(
                    Text(
                        "Folding service: managed ESMFold2 · starts automatically"
                        if endpoint == "auto"
                        else f"Folding service: {endpoint}"
                    )
                )
                lines.append(
                    Text(
                        "Requires matching reference structures and Foldseek; checked before training."
                    )
                )
            if session.resume and evaluation.run_on_resume:
                lines.append(Text("All enabled checks also run at the resumed step."))
            elif not session.resume and evaluation.run_on_start:
                lines.append(
                    Text("All enabled checks also run before the first update.")
                )
            lines.append(
                Text(f"Reports: {evaluation_directory(config.out)}", style="dim")
            )
        self.console.print(
            Panel(
                Group(*lines),
                title="[dim]EVALUATION[/]",
                title_align="left",
                border_style="cyan",
            )
        )

    def configure(self, session: TrainingSession, setup_path=None, field=None) -> TrainingSession:
        while True:
            locked = "Saved in checkpoint" if session.resume is not None else None
            config = session.config
            choice = field
            if choice is None:
                rows = aligned_choices(self.console, [
                    ("batch", "Batch size", f"{config.data.batch_size:,}"),
                    ("device", "Device", device_description(config.device)),
                    ("precision", "Precision", config.precision.upper()),
                    ("evaluation", "Validation", EVALUATION_LABELS.get(config.evaluation.regime, config.evaluation.regime)),
                    ("settings", "Recipe", session.settings),
                ])
                for row in rows:
                    if row.value in {"batch", "precision", "settings"}:
                        row.disabled = locked
                choice = self.select(
                    "Configure training",
                    [
                        *rows,
                        questionary.Separator(" "),
                        questionary.Choice("Architecture and validation details", "details"),
                        questionary.Choice("Back", "back"),
                    ],
                )
            if choice in (None, "back"):
                return session
            try:
                if choice == "details":
                    self.details(session)
                    self.select("", [questionary.Choice("Back", "back")])
                elif choice in ("steps", "batch"):
                    value = self.number(
                        "Total steps" if choice == "steps" else "Batch size",
                        config.steps if choice == "steps" else config.data.batch_size,
                        session.completed_steps + 1 if choice == "steps" else 1,
                    )
                    if value is not None:
                        if choice == "steps":
                            config.steps = value
                        else:
                            config.data.batch_size = value
                elif choice == "settings":
                    path = self.path("Training settings JSON file")
                    if path is not None:
                        updated = build_session(config_path=path, setup_path=setup_path)
                        # Preserve the chosen run location when the file leaves it unspecified.
                        if "out" not in json.loads(path.read_text()):
                            updated.config.out = config.out
                        session = updated
                elif choice == "out":
                    path = self.path("Checkpoint file", config.out)
                    if path is not None:
                        if path.is_dir():
                            raise ValueError(
                                "Choose a checkpoint file, not a directory"
                            )
                        config.out = str(path)
                elif choice == "device":
                    choices = [
                        questionary.Choice("Auto", "auto"),
                        questionary.Choice("CPU", "cpu"),
                    ]
                    if torch.cuda.is_available():
                        choices.extend(
                            questionary.Choice(
                                f"CUDA {i} · {torch.cuda.get_device_name(i)}",
                                f"cuda:{i}",
                            )
                            for i in range(torch.cuda.device_count())
                        )
                    value = self.select("Training device", choices)
                    if value is not None:
                        config.device = value
                elif choice == "precision":
                    cuda = resolve_device(config.device).type == "cuda"
                    value = self.select(
                        "Training precision",
                        [
                            questionary.Choice("FP32", "fp32"),
                            questionary.Choice("BF16", "bf16"),
                            questionary.Choice(
                                "FP16",
                                "fp16",
                                disabled=None if cuda else "Requires a CUDA device",
                            ),
                        ],
                    )
                    if value is not None:
                        config.precision = value
                elif choice == "evaluation":
                    value = self.select(
                        "Evaluation regime",
                        [
                            questionary.Choice(label, value)
                            for value, label in EVALUATION_LABELS.items()
                            if value != "custom"
                        ],
                    )
                    if value is not None:
                        config.evaluation = replace(
                            config.evaluation, regime=value, suites={}
                        )
            except (ValueError, TypeError, OSError, KeyError) as exc:
                self.console.print(Text(str(exc), style="red"))
            if field is not None:
                return session

    def choose_resume(self, session: TrainingSession, setup_path=None) -> TrainingSession:
        directory = Path(session.config.out).expanduser().parent
        files = sorted(
            directory.glob("*.pt"), key=lambda path: path.stat().st_mtime, reverse=True
        )[:20]
        selected = self.select(
            "Run",
            [
                questionary.Choice("Train a new model", "new"),
                *(questionary.Choice(f"Resume {short_path(path)}", path) for path in files),
                questionary.Choice("Resume another checkpoint…", "path"),
                questionary.Choice("Back", "back"),
            ],
        )
        if selected in (None, "back"):
            return session
        if selected == "new":
            return session if session.resume is None else build_session(setup_path=setup_path, interactive=True)
        path = self.path("Checkpoint file") if selected == "path" else selected
        if path is None:
            return session
        updated = build_session(resume=path)
        steps = self.number(
            "New total step count",
            max(updated.config.steps, updated.completed_steps + 1),
            updated.completed_steps + 1,
        )
        if steps is not None:
            updated.config.steps = steps
            return updated
        return session

    def menu(self, session, info, problem):
        """The run, its data, length and destination, then Start; the rest waits in Configure."""
        config = session.config
        rows = aligned_choices(self.console, [
            (
                "resume",
                "Run",
                f"Resume {session.resume.name} from step {session.completed_steps:,}"
                if session.resume
                else "New model",
            ),
            (
                "dataset",
                "Data",
                f"ProstT5 · {info.minimum}–{info.maximum} residues"
                if info is not None
                else "Not ready · run mubind setup --only prostt5",
            ),
            ("steps", "Steps", f"{config.steps:,}"),
            ("out", "Save to", short_path(config.out)),
        ])
        label = "Resume training" if session.resume else "Start training"
        # Until data is ready, Start is struck through, names what is missing in
        # red, and leads there when chosen. The status class keeps it red under
        # the cursor instead of turning cyan like an available row.
        start = (
            [
                ("class:status class:error strike", label),
                ("class:status class:error", " [Needs prepared ProstT5 data]"),
            ]
            if problem
            else label
        )
        return [
            *rows,
            questionary.Separator(" "),
            questionary.Choice("Configure training", "configure"),
            questionary.Separator(" "),
            questionary.Choice(start, "start"),
            questionary.Separator(" "),
            questionary.Choice("Back", "back"),
        ]

    def screen(self, session: TrainingSession, setup_path=None) -> int:
        focus = None
        while True:
            try:
                info, problem = dataset_info(session), None
            except (ValueError, OSError) as exc:
                info, problem = None, str(exc)
            # Point at the data when it is missing, otherwise at Start.
            if focus in (None, "dataset", "start"):
                focus = "dataset" if problem else "start"
            summary = fill(
                training_summary(session, info),
                width=max(30, self.console.width - 4),
                initial_indent="  ",
                subsequent_indent="  ",
            )
            choice = self.select(
                TITLE,
                self.menu(session, info, problem),
                instruction="\n" + summary + "\n",
                style=merge_styles([self.style, TITLE_STYLE]) if self.style else TITLE_STYLE,
                default=focus,
                erase=True,
            )
            if choice in (None, "back"):
                return 0
            focus = choice
            if choice == "start" and problem:
                choice = "dataset"
            try:
                if choice == "configure":
                    session = self.configure(session, setup_path)
                elif choice in {"steps", "out"}:
                    session = self.configure(session, setup_path, field=choice)
                elif choice == "dataset":
                    path = self.path("Prepared ProstT5 folder", session.config.data.path)
                    if path is not None:
                        updated = replace(session, config=replace(session.config, data=replace(session.config.data, path=str(path))), problem="")
                        dataset_info(updated)
                        session = updated
                elif choice == "resume":
                    session = self.choose_resume(session, setup_path)
                elif choice == "start":
                    dataset_info(session)
                    return self.run(session)
            except (ValueError, TypeError, OSError, KeyError, RuntimeError) as exc:
                self.console.print(Text(str(exc), style="red"))

    def run(self, session: TrainingSession) -> int:
        dataset_info(session)
        config = session.config
        self.console.print(
            Text(
                "W&B logging is not connected in this trainer yet. Metrics will be saved locally.",
                style="yellow",
            )
        )
        started = time.monotonic()
        last_validation = "—"
        last_loss = "—"
        last_saved = (
            (session.resume, session.completed_steps) if session.resume else None
        )
        terminal = self.console.is_terminal
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[cyan]{task.completed:,.0f}/{task.total:,.0f}[/]"),
            BarColumn(bar_width=None),
            TextColumn("{task.fields[metrics]}"),
            TimeElapsedColumn(),
            console=self.console,
            disable=not terminal,
            transient=False,
        )
        task = progress.add_task(
            "Training",
            total=config.steps,
            completed=session.completed_steps,
            metrics="Loading data…",
        )

        def status(message):
            if terminal:
                progress.update(task, metrics=message, refresh=True)
            else:
                self.console.print(Text(message, style="dim"))

        def report(event):
            nonlocal last_validation, last_loss
            if "validation_loss" in event:
                last_validation = f"{event['validation_loss']:.5f}"
            if "loss" in event:
                last_loss = f"{event['loss']:.5f}"
            metrics = f"loss {last_loss} · val {last_validation} · lr {event['lr']:.3g}"
            if last_loss != "—":
                metrics = f"Training · {metrics}"
            if terminal:
                progress.update(task, completed=event["step"], metrics=metrics)
            else:
                elapsed = time.monotonic() - started
                self.console.print(
                    Text(
                        f"step {event['step']:,}/{config.steps:,} | {metrics} | {elapsed:.0f}s"
                    )
                )

        def step_completed(event):
            if terminal:
                report(event)

        def saved(path, step):
            nonlocal last_saved
            if last_saved == (path, step):
                return
            last_saved = (path, step)
            self.console.print(Text(f"Saved step {step:,} → {path}", style="dim"))

        def evaluating(event):
            name = event["suite"].replace("_", " ")
            if event["phase"] == "start":
                self.console.print(
                    Text(f"Step {event['step']:,} · evaluating {name}", style="cyan")
                )
            elif event["phase"] == "complete":
                self.console.print(
                    Text(f"Evaluation saved → {event['path']}", style="dim")
                )
            if terminal:
                progress.update(
                    task,
                    completed=event["step"],
                    metrics=f"evaluating {name} · {event['phase']}",
                )

        try:
            with progress:
                result = train(
                    config,
                    resume=session.resume,
                    log=report,
                    on_checkpoint=saved,
                    on_evaluation=evaluating,
                    on_status=status,
                    on_step=step_completed,
                )
                progress.update(task, completed=result.step)
        except KeyboardInterrupt:
            self.console.print("[yellow]Training interrupted.[/]")
            if last_saved is not None:
                path, step = last_saved
                self.console.print(
                    Text(
                        f"Latest checkpoint: {path} (step {step:,}). Resume with mubind train --resume PATH --steps TOTAL."
                    )
                )
            else:
                self.console.print("No checkpoint was saved yet.")
            return 130
        self.console.print(
            Text(f"Training complete at step {result.step:,}.", style="green")
        )
        return 0
