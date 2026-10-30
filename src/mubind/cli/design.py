"""A target-first menu for the complete minibinder workflow."""

from __future__ import annotations

import math
from textwrap import fill
from dataclasses import replace
from pathlib import Path

import questionary
from prompt_toolkit.styles import merge_styles
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..paths import GENERATIONS
from ..examples import ensure_example_target, example_target_options
from .terminal_ui import select_prompt, value_choice
from .sample import GenerationUI
from .checkpoint_catalog import (
    checkpoint_folders,
    discover_design_checkpoints,
    inspect_design_checkpoint,
)
from ..inference.config import (
    DesignConfig,
    MoreCandidates,
    MoreScaffolds,
    RedesignChoice,
    default_design_checkpoint,
    next_output,
    parse_scaffold_ranks,
)
from .generation_progress import DesignProgress
from .preferences import load_target_notes, save_checkpoint_choice, save_target_note, target_note
from .ranking_ui import CandidateBrowser, ERROR_NOTE, RANKING_NOTE, candidate_table


def design(*args, **kwargs):
    from ..inference.design import design as run

    return run(*args, **kwargs)


def inspect_target(*args, **kwargs):
    from ..inference.helpers.targets import inspect_target as inspect

    return inspect(*args, **kwargs)


TITLE = "mubind · Generate"
# The screen title matches the purple home title; rows keep the shared style.
TITLE_STYLE = questionary.Style([("question", "fg:#d787ff bold")])

PROVIDER_HINT = (
    "You should probably run mubind provider in a separate tmux session and keep "
    "it running, if you intend to do multiple generations"
)
DEFAULT_OUTPUT = f"{GENERATIONS}/ · current date and time"


def display_path(path):
    """Show a path relative to the working folder or home when it lies inside one."""
    path = Path(path).expanduser()
    for base, prefix in ((Path.cwd, ""), (Path.home, "~/")):
        try:
            return prefix + path.relative_to(base()).as_posix()
        except (ValueError, OSError, RuntimeError):
            pass  # Outside this folder, or the folder cannot be determined.
    return str(path)


def fit(text, width):
    return text if len(text) <= width else text[: max(width, 1) - 1] + "…"


def fit_path(text, width):
    """Drop leading folders first so the file name and its folder stay readable."""
    if len(text) <= width:
        return text
    parts = text.split("/")
    for start in range(1, len(parts)):
        shorter = "…/" + "/".join(parts[start:])
        if len(shorter) <= width:
            return shorter
    name, keep = parts[-1], max(width - 1, 0)
    return "…" + (name[len(name) - keep :] if keep else "")


def path_with_facts(path, facts, room):
    """A path followed by short facts; without a width limit, the full path."""
    suffix = "".join(f" · {fact}" for fact in facts)
    if room is None:
        return str(path) + suffix
    return fit_path(display_path(path), max(room - len(suffix), 12)) + suffix


def target_value(settings, room):
    if settings.target is None:
        return "Choose a PDB file"
    hotspots = len(settings.hotspots)
    return path_with_facts(
        settings.target,
        [
            *filter(None, [target_note(settings.target)]),
            settings.target_spec or "longest chain",
            f"{hotspots} hotspot{'s' if hotspots != 1 else ''}" if hotspots else "no hotspots",
        ],
        room,
    )


def checkpoint_facts(option):
    facts = []
    if option.step is not None:
        facts.append(f"step {option.step:,}")
    if option.weights:
        facts.append(f"{option.weights.upper()} weights")
    return facts


def checkpoint_value(settings, selected, room):
    if settings.checkpoint is None or selected is None:
        return "Choose a checkpoint"
    facts = ["unavailable"] if selected.problem else ["checked on start"] if not selected.checked else []
    return path_with_facts(selected.path, facts, room)


def scaffold_value(settings, room):
    if settings.reuse_scaffolds is not None:
        folder = fit_path(display_path(settings.reuse_scaffolds), room - 6) if room else str(settings.reuse_scaffolds)
        return f"Reuse {folder}"
    return f"Generate {settings.num_complexa:,} with Proteina Complexa"


def redesign_plan(settings):
    sequences = f"{settings.num_diffusion} sequences each"
    if settings.all_scaffolds:
        return f"All scaffolds · {sequences}"
    if settings.scaffold_index is not None:
        return f"Scaffold index {settings.scaffold_index} · {settings.num_diffusion} sequences"
    if settings.top_scaffolds is not None:
        pick = "Best scaffold" if settings.top_scaffolds == 1 else f"Top {settings.top_scaffolds} scaffolds"
        return f"{pick} · {sequences}"
    return "Choose scaffolds after ranking"


def selection_value(settings):
    if settings.all_scaffolds:
        return "redesign all"
    if settings.scaffold_index is not None:
        return f"redesign index {settings.scaffold_index}"
    if settings.top_scaffolds == 1:
        return "redesign the best"
    if settings.top_scaffolds is not None:
        return f"redesign top {settings.top_scaffolds}"
    return "pick after ranking"


def plan_summary(settings, ready):
    """How a run with these settings will proceed, in plain sentences."""
    against = f"against {settings.target.name}" if settings.target else "against your target"
    if settings.reuse_scaffolds is None:
        first = (
            f"Given the current config, Proteina Complexa will build {settings.num_complexa:,} binder scaffolds "
            f"of {settings.binder_length} residues {against}, and ESMFold2 will fold and rank them."
        )
    else:
        first = (
            f"Given the current config, mubind will reuse the scaffolds in "
            f"{fit_path(display_path(settings.reuse_scaffolds), 40)} {against} and rank them."
        )
    pick = {
        "pick after ranking": "You will then pick which scaffolds to redesign",
        "redesign all": "Every scaffold will then be redesigned",
        "redesign the best": "The best scaffold will then be redesigned",
    }.get(selection_value(settings))
    if pick is None:
        pick = (
            f"Scaffold {settings.scaffold_index} will then be redesigned"
            if settings.scaffold_index is not None
            else f"The top {settings.top_scaffolds} scaffolds will then be redesigned"
        )
    sequences = settings.num_diffusion
    second = (
        f"{pick} into {sequences} new sequence{'s' if sequences != 1 else ''} each, "
        "and every candidate will be folded and ranked again."
    )
    return " ".join([first, second, *([] if ready else ["Choose what is missing below to start."])])


def output_value(settings):
    return display_path(settings.out) if settings.out else DEFAULT_OUTPUT


class DesignUI(GenerationUI):
    def heading(self):
        if not getattr(self, "_heading_shown", False):
            self.console.print()
            self.console.print(Text(TITLE, style="bold #d787ff"))
            self._heading_shown = True

    def select(self, message, choices, instruction=None, *, erase=False, **options):
        prompt = select_prompt(
            message,
            choices=choices,
            style=options.pop("style", self.style),
            qmark="›",
            pointer="❯",
            instruction=instruction,
            **options,
        )
        # An erased menu is redrawn fresh instead of piling up answered copies.
        prompt.application.erase_when_done = erase
        return prompt.ask()

    def active_checkpoint(self, settings, *, refresh=False, check=True):
        if settings.checkpoint is None:
            return None
        if not check:
            return inspect_design_checkpoint(settings.checkpoint, refresh=refresh, check=False)
        with self.console.status("Checking active checkpoint…"):
            return inspect_design_checkpoint(settings.checkpoint, refresh=refresh)

    def aligned(self, entries):
        """Rows of `label  value` whose values line up and fit the terminal.

        A value may be a function of the columns left for it, so paths can shrink.
        """
        label = max(len(name) for _, name, _ in entries)
        room = max(self.console.width - label - 7, 20)
        return [
            value_choice(f"{name:<{label}}  ", fit(value(room) if callable(value) else value, room), key)
            for key, name, value in entries
        ]

    def menu(self, settings, selected):
        """The three choices every run needs, then Start; the rest waits in Options."""
        missing = [
            *([] if settings.target else ["a target"]),
            *([] if selected and not selected.problem else ["a compatible checkpoint"]),
        ]
        # Until the run is complete, Start is struck through, names what is missing
        # in red, and leads there when chosen. The status class keeps it red under
        # the cursor instead of turning cyan like an available row.
        start = (
            [
                ("class:status class:error strike", "Start design"),
                ("class:status class:error", f" [Needs {' and '.join(missing)}]"),
            ]
            if missing
            else "Start design"
        )
        return [
            *self.aligned([
                ("target", "Target", lambda room: target_value(settings, room)),
                ("checkpoint", "Checkpoint", lambda room: checkpoint_value(settings, selected, room)),
            ]),
            questionary.Separator(" "),
            questionary.Choice("Configure generation", "configure"),
            questionary.Separator(" "),
            questionary.Choice(start, "start"),
            questionary.Separator(" "),
            questionary.Choice("Back", "back"),
        ]

    def checkpoint_row(self, option, active):
        name = option.path.name
        detail = ("  in use · " if active else "  ")
        # Leave room for the "› Checkpoint " that precedes the row once chosen.
        detail += fit_path(display_path(option.path.parent), max(self.console.width - len(name) - len(detail) - 20, 12))
        return [("class:text", name), ("class:muted", detail)]

    def checkpoint_locations(self, settings):
        places = checkpoint_folders(settings.checkpoint, settings.models_dir)
        self.console.print(Text("Searching " + " · ".join(map(str, places)), style="dim", overflow="fold"))

    def choose_checkpoint(self, settings):
        self.heading()
        refresh = False
        while True:
            with self.console.status("Finding checkpoint files…"):
                options = discover_design_checkpoints(
                    settings.checkpoint, settings.models_dir, check=False
                )
            active = self.active_checkpoint(settings, refresh=refresh, check=False)
            current = active.path if active else None
            if not options:
                self.checkpoint_locations(settings)
                self.console.print(Text("No checkpoint files found in these locations.", style="dim"))
            # Keep the active file first, with its checked metadata. Listing the
            # other files must not deserialize a whole training run's checkpoints.
            options = sorted(options, key=lambda option: option.path != current)
            options = [active if option.path == current else option for option in options]
            choice = self.select(
                "Checkpoint",
                [
                    *(questionary.Choice(
                        self.checkpoint_row(option, option.path == current),
                        option.path,
                        disabled=option.problem[:120] if option.problem else None,
                    ) for option in options),
                    questionary.Choice("Choose checkpoint path…", "path"),
                    questionary.Choice("Refresh folders", "refresh"),
                    questionary.Choice("Back", "back"),
                ],
                instruction="Used by Generate and Evaluate until you change it",
            )
            if choice in (None, "back"):
                return settings
            if choice == "refresh":
                refresh = True
                continue
            refresh = False
            path = self.path("mubind redesign checkpoint file", current or "") if choice == "path" else choice
            if path is None:
                continue
            with self.console.status(f"Checking {path.name}…"):
                option = inspect_design_checkpoint(path, refresh=True)
            if option.problem:
                self.console.print(Text(f"Cannot select {option.path.name}\n{option.problem}", style="yellow"))
                continue
            selected = replace(settings, checkpoint=option.path)
            break
        try:
            save_checkpoint_choice(selected.checkpoint)
        except OSError as exc:
            self.console.print(
                Text(
                    f"Checkpoint selected for this session, but could not save the choice: {exc}",
                    style="yellow",
                )
            )
        self.console.print(Text(
            f"Checkpoint in use · {option.path.name}",
            style="green",
        ))
        return selected

    def overview(self, settings):
        """Summarize a run that starts without a menu, such as `generate TARGET.pdb`."""
        self.heading()
        selected = self.active_checkpoint(settings, check=False)
        rows = Table.grid(padding=(0, 2))
        rows.add_column(style="dim")
        rows.add_column(overflow="fold")
        rows.add_row("Target", Text(target_value(settings, None)))
        rows.add_row("Checkpoint", Text(checkpoint_value(settings, selected, None)))
        rows.add_row("Scaffolds", Text(scaffold_value(settings, None)))
        rows.add_row("Redesign", Text(redesign_plan(settings)))
        if settings.edit_residues is not None:
            rows.add_row("Inpainting", Text(f"Binder positions {settings.edit_residues} · original length retained"))
        rows.add_row("Save to", Text(output_value(settings)))
        self.console.print(rows)
        if selected is not None and selected.problem:
            self.console.print(Text(f"Checkpoint unavailable · {selected.problem}", style="yellow"))
        return selected

    def settings_details(self, settings, selected=None):
        selected = selected or self.active_checkpoint(settings, check=False)
        rows = Table.grid(padding=(0, 2))
        rows.add_column(style="dim")
        rows.add_column(overflow="fold")
        rows.add_row("Target", Text(str(settings.target or "Choose a PDB")))
        rows.add_row("Region", Text(settings.target_spec or "Longest protein chain"))
        rows.add_row("Hotspots", Text(", ".join(settings.hotspots) or "None"))
        rows.add_row("Binder length", f"{settings.binder_length} residues")
        rows.add_row(
            "Proteina Complexa",
            f"{settings.num_complexa} candidates · {settings.complexa_steps} steps · guidance 1",
        )
        rows.add_row("Folding", "ESMFold2 · 3 loops")
        rows.add_row("Redesign", redesign_plan(settings))
        facts = " · ".join(checkpoint_facts(selected)) if selected else ""
        rows.add_row(
            "Checkpoint",
            Text("\n".join(filter(None, [str(settings.checkpoint or "Not selected"), facts]))),
        )
        rows.add_row("Checkpoint folders", Text("\n".join(
            str(path) for path in checkpoint_folders(settings.checkpoint, settings.models_dir)
        )))
        rows.add_row("Properties", "From each scaffold's chain B")
        rows.add_row("Sampling", f"{settings.replicas} best-of replicas · structural guidance {settings.structural_guidance:g}")
        rows.add_row("Device", Text(settings.device))
        rows.add_row("Seed", str(settings.seed) if settings.seed is not None else "Random")
        rows.add_row("Folding endpoint", Text(settings.folding_endpoint))
        rows.add_row("Scaffolds", Text(str(settings.reuse_scaffolds or "Generate fresh scaffolds")))
        rows.add_row("Target PDB folder", Text(str(settings.targets_dir.expanduser().resolve())))
        rows.add_row("Save to", Text(str(settings.out) if settings.out else DEFAULT_OUTPUT))
        self.console.print(
            Panel(
                Group(
                    rows,
                    Text(),
                    Text(
                        "Scaffolds → fold and select → Foldseek encoding → redesign → ranking",
                        style="dim",
                    ),
                ),
                title="[bold cyan]Settings and paths[/]",
                border_style="cyan",
                padding=(1, 2),
            )
        )
        self.select("", [questionary.Choice("Back", "back")])

    def choose_target(self, settings):
        from ..inference.helpers.targets import discover_targets

        directory = settings.targets_dir.expanduser().resolve()
        ensure_example_target(directory)
        current = settings.target.expanduser().resolve() if settings.target else None
        while True:
            self.console.print(
                Text(f"Place target PDB files in: {directory}", style="cyan")
            )
            paths = discover_targets(directory)
            notes = load_target_notes()
            width = max((len(p.name) for p in paths), default=0)

            def row(p):
                # The file name stays white; its saved note and marker are grey.
                detail = " · ".join(filter(None, [
                    notes.get(str(p.resolve()), ""),
                    "current" if p.resolve() == current else "",
                ]))
                return value_choice(f"{p.name:<{width}}  ", detail, p) if detail else questionary.Choice(p.name, p)

            choice = self.select(
                "Target PDB",
                [
                    *(row(p) for p in paths),
                    questionary.Choice("Choose another PDB path…", "path"),
                    questionary.Choice("Refresh folder", "refresh"),
                    questionary.Choice("Back", "back"),
                ],
            )
            if choice in (None, "back"):
                return settings
            if choice == "refresh":
                continue
            path = self.path("Target PDB file") if choice == "path" else choice
            if path is None:
                continue
            options = example_target_options(path)
            spec = options.get("target_spec")
            hotspots = options.get("hotspots", ())
            inspect_target(path, spec, hotspots)
            return replace(
                settings,
                target=path,
                target_spec=spec,
                hotspots=hotspots,
                binder_length=options.get("binder_length", settings.binder_length),
            )

    def target_menu(self, settings):
        """Once a target is chosen, its file, region and hotspots are edited together."""
        if settings.target is None:
            return self.choose_target(settings)
        while True:
            choice = self.select(
                "Target",
                [
                    *self.aligned([
                        ("file", "PDB file", lambda room: fit_path(display_path(settings.target), room)),
                        ("note", "Note", target_note(settings.target) or "None"),
                        ("target_spec", "Region", settings.target_spec or "Longest chain"),
                        ("hotspots", "Hotspots", ", ".join(settings.hotspots) or "None"),
                    ]),
                    questionary.Choice("Back", "back"),
                ],
            )
            if choice in (None, "back"):
                return settings
            if choice == "file":
                settings = self.choose_target(settings)
                continue
            if choice == "note":
                answer = questionary.text(
                    f"Note for {settings.target.name} (empty = none)",
                    default=target_note(settings.target),
                    style=self.style,
                ).ask()
                if answer is not None:
                    try:
                        save_target_note(settings.target, answer)
                    except OSError as exc:
                        self.console.print(Text(f"Could not save the note: {exc}", style="yellow"))
                continue
            answer = questionary.text(
                "Chain and range, e.g. A10-120 (empty = longest chain)"
                if choice == "target_spec"
                else "Hotspot residues, comma separated (empty = none)",
                default=(settings.target_spec or "")
                if choice == "target_spec"
                else ",".join(settings.hotspots),
                style=self.style,
            ).ask()
            if answer is None:
                continue
            value = (
                answer.strip() or None
                if choice == "target_spec"
                else tuple(v.strip() for v in answer.split(",") if v.strip())
            )
            updated = replace(settings, **{choice: value})
            inspect_target(updated.target, updated.target_spec, updated.hotspots)
            settings = updated

    def number(self, message, current, minimum=1, maximum=None):
        answer = questionary.text(
            message,
            default=str(current),
            style=self.style,
            validate=lambda value: (
                value.isdecimal()
                and int(value) >= minimum
                and (maximum is None or int(value) <= maximum)
                or (
                    f"Enter an integer between {minimum} and {maximum}."
                    if maximum is not None
                    else f"Enter an integer of at least {minimum}."
                )
            ),
        ).ask()
        return None if answer is None else int(answer)

    def choose_scaffolds(self, settings):
        action = self.select(
            "Scaffold source",
            [
                questionary.Choice("Generate new with Proteina Complexa", "fresh"),
                questionary.Choice("Reuse a scaffold folder…", "folder"),
                questionary.Choice("Back", "back"),
            ],
        )
        if action == "fresh":
            value = self.number("Scaffold count", settings.num_complexa)
            return settings if value is None else replace(settings, num_complexa=value, reuse_scaffolds=None)
        if action == "folder":
            value = self.path("Scaffold folder", settings.reuse_scaffolds or "")
            return settings if value is None else replace(settings, reuse_scaffolds=value)
        return settings

    def choose_selection(self, settings):
        mode = self.select(
            "Scaffolds to redesign",
            [
                questionary.Choice("Pick after ranking", "ask"),
                questionary.Choice("The best by folding score", "best"),
                questionary.Choice("Top N by folding score", "top"),
                questionary.Choice("All scaffolds", "all"),
                questionary.Choice("One by index", "index"),
            ],
        )
        if mode in {"ask", "best", "all"}:
            settings = replace(
                settings,
                all_scaffolds=mode == "all",
                scaffold_index=None,
                top_scaffolds=1 if mode == "best" else None,
            )
        elif mode == "top":
            value = self.number("Number of top scaffolds", settings.top_scaffolds or 5)
            if value is not None:
                settings = replace(
                    settings,
                    top_scaffolds=value,
                    all_scaffolds=False,
                    scaffold_index=None,
                )
        elif mode == "index":
            value = self.number(
                "Scaffold index (starting at 0)", settings.scaffold_index or 0, 0
            )
            if value is not None:
                settings = replace(
                    settings,
                    scaffold_index=value,
                    all_scaffolds=False,
                    top_scaffolds=None,
                )
        return settings

    def configure(self, settings):
        """Every generation setting in one flat list, the scaffold plan first."""
        from .provider_ui import worker_status

        numbers = {
            "num_diffusion": "Sequences per scaffold",
            "binder_length": "Binder length",
            "seed": "Random seed",
            "complexa_steps": "Proteina Complexa steps",
            "replicas": "Best-of replicas",
            "redesign_batch_size": "Redesign batch size",
            "complexa_batch_size": "Complexa batch size",
        }
        while True:
            choice = self.select(
                "Configure generation",
                [
                    *self.aligned([
                        ("source", "Scaffolds", lambda room: scaffold_value(settings, room)),
                        ("selection", "Redesign", selection_value(settings).capitalize()),
                        ("num_diffusion", "Sequences", f"{settings.num_diffusion} per scaffold"),
                        ("binder_length", numbers["binder_length"], f"{settings.binder_length} residues"),
                        ("out", "Output folder", output_value(settings)),
                        ("seed", numbers["seed"], str(settings.seed) if settings.seed is not None else "Random"),
                        ("device", "Device", settings.device),
                        ("complexa_steps", numbers["complexa_steps"], str(settings.complexa_steps)),
                        ("replicas", numbers["replicas"], str(settings.replicas)),
                        ("redesign_batch_size", numbers["redesign_batch_size"], str(settings.redesign_batch_size)),
                        ("complexa_batch_size", numbers["complexa_batch_size"], str(settings.complexa_batch_size)),
                        ("folding_endpoint", "Folding endpoint", settings.folding_endpoint),
                        ("targets_dir", "Target folder", display_path(settings.targets_dir)),
                        ("workers", "Model workers", worker_status(settings.models_dir, settings.device)),
                    ]),
                    questionary.Separator(" "),
                    questionary.Choice("All settings and paths", "details"),
                    questionary.Choice("Back", "back"),
                ],
            )
            if choice in (None, "back"):
                return settings
            if choice == "source":
                settings = self.choose_scaffolds(settings)
            elif choice == "selection":
                settings = self.choose_selection(settings)
            elif choice in numbers:
                value = self.number(
                    numbers[choice],
                    getattr(settings, choice) or 0,
                    0 if choice == "seed" else 1,
                )
                if value is not None:
                    settings = replace(settings, **{choice: value})
            elif choice == "out":
                value = self.path("New output folder", settings.out or next_output())
                if value is not None:
                    settings = replace(settings, out=value)
            elif choice == "targets_dir":
                value = self.path("Folder", settings.targets_dir)
                if value is not None:
                    settings = replace(settings, targets_dir=value)
            elif choice == "workers":
                from .provider_ui import provider_screen

                provider_screen(self.console, self.style, models_dir=settings.models_dir, device=settings.device)
            elif choice == "details":
                self.settings_details(settings)
            else:
                value = questionary.text(
                    "Device (auto, cpu, cuda:0)"
                    if choice == "device"
                    else "Folding endpoint (auto, unix:///path, or HTTP URL)",
                    default=getattr(settings, choice),
                    style=self.style,
                ).ask()
                if value is not None and value.strip():
                    settings = replace(settings, **{choice: value.strip()})

    def screen(self, settings=None):
        self.heading()
        settings = settings or DesignConfig()
        settings = replace(
            settings,
            checkpoint=settings.checkpoint or default_design_checkpoint(settings.models_dir),
        )
        settings.targets_dir.expanduser().mkdir(parents=True, exist_ok=True)

        focus = None
        while True:
            selected = self.active_checkpoint(settings, check=False)
            if selected is not None and selected.problem:
                self.console.print(Text(f"Checkpoint unavailable · {selected.problem}", style="yellow"))
            ready = bool(settings.target and selected and not selected.problem)
            # Point at the next missing choice, or at Start once nothing is missing.
            if focus in (None, "target", "checkpoint", "start"):
                focus = "start" if ready else "checkpoint" if settings.target else "target"
            # A grey summary of what Start would do sits under the title.
            summary = fill(
                plan_summary(settings, ready),
                width=max(30, self.console.width - 4),
                initial_indent="  ",
                subsequent_indent="  ",
            )
            choice = self.select(
                "Generate settings",
                self.menu(settings, selected),
                instruction="\n" + summary + "\n",
                style=merge_styles([self.style, TITLE_STYLE]),
                default=focus,
                erase=True,
            )
            if choice in (None, "back"):
                return 0
            focus = choice
            if choice == "start" and not ready:
                choice = "checkpoint" if settings.target else "target"
            try:
                if choice == "target":
                    settings = self.target_menu(settings)
                elif choice == "checkpoint":
                    settings = self.choose_checkpoint(settings)
                elif choice == "configure":
                    settings = self.configure(settings)
                elif choice == "start":
                    # The file may have been replaced while the prompt was open.
                    # Show changed metadata and require another start selection.
                    current = self.active_checkpoint(settings, check=False)
                    if current is None or current.fingerprint != selected.fingerprint or current.problem:
                        self.console.print(Text(
                            "Checkpoint changed or is unavailable. Review the active checkpoint before starting.",
                            style="yellow",
                        ))
                        continue
                    return self.run(settings, interactive=True)
            except (ValueError, TypeError, OSError, RuntimeError, KeyError) as exc:
                self.console.print(
                    Panel(Text(str(exc)), title="Design error", border_style="red")
                )

    def choose_redesign(self, ranked, settings):
        import torch

        if self.console.is_terminal:
            self.console.clear()
        title = f"Scaffolds · top 10 of {len(ranked)}" if len(ranked) > 10 else "Scaffolds"
        table = Table(title=title)
        for column in ("Rank", "Scaffold", "pLDDT", "ipTM"):
            table.add_column(
                column, justify="left" if column == "Scaffold" else "right"
            )
        for index, scaffold in enumerate(ranked[:10], 1):
            metrics = scaffold["scaffold_metrics"]
            table.add_row(
                str(index),
                scaffold["id"],
                f"{metrics['binder_plddt']:.3f}",
                f"{metrics['iptm']:.3f}",
            )
        self.console.print(table)
        self.console.print(
            Text("3 = top 3 · 1,2,5 = ranks · more = generate more", style="dim")
        )
        can_generate = settings.device.startswith("cuda") or (
            settings.device == "auto" and torch.cuda.is_available()
        )

        def validate(value):
            if value.strip().lower() == "more":
                return (
                    True if can_generate else "Generating more scaffolds requires CUDA."
                )
            try:
                parse_scaffold_ranks(value, len(ranked))
                return True
            except ValueError as exc:
                return str(exc)

        answer = questionary.text(
            "Scaffolds",
            default=str(min(5, len(ranked))),
            style=self.style,
            validate=validate,
        ).ask()
        if answer is None:
            return None
        if answer.strip().lower() == "more":
            if not can_generate:
                raise ValueError("Generating more scaffolds requires CUDA")
            count = self.number("Additional Proteina Complexa scaffolds", settings.num_complexa)
            return MoreScaffolds(count) if count is not None else None
        ranks = parse_scaffold_ranks(answer, len(ranked))
        sequences = self.number(
            "Sequences to generate per scaffold", settings.num_diffusion
        )
        if sequences is None:
            return None
        properties = self.choose_properties(
            [ranked[rank - 1] for rank in ranks],
            **({"fixed_length": True} if settings.edit_residues is not None else {}),
        )
        if properties is None:
            return None
        label = (
            f"Ranks {','.join(map(str, ranks))}"
            if "," in answer
            else f"Top {len(ranks)}"
        )
        self.console.print(
            Text(
                f"{label} × {sequences} sequences = {len(ranks) * sequences:,} redesigns",
                style="cyan",
            )
        )
        return RedesignChoice(
            len(ranks), sequences, ranks if "," in answer else None, properties
        )

    def choose_properties(self, scaffolds, *, fixed_length=False):
        selected = {}
        for scaffold in scaffolds:
            self.console.print(
                Text(f"Requested properties · {scaffold['id']}", style="cyan")
            )
            values = {}
            for name in ("length", "charge", "hydrophobicity"):
                original = scaffold["properties"][name]
                if name == "length" and fixed_length:
                    values[name] = original
                    self.console.print(Text(f"Length {int(original)} residues · fixed for inpainting", style="dim"))
                    continue
                default = str(int(original)) if name == "length" else f"{original:.6g}"

                def validate(value, name=name):
                    try:
                        number = float(value)
                    except ValueError:
                        return "Enter a number."
                    if not math.isfinite(number):
                        return "Enter a finite number."
                    if name == "length" and (
                        not number.is_integer() or not 1 <= number <= 70
                    ):
                        return "Enter a whole length between 1 and 70."
                    return True

                answer = questionary.text(
                    name.capitalize(),
                    default=default,
                    style=self.style,
                    validate=validate,
                ).ask()
                if answer is None:
                    return None
                values[name] = original if answer == default else float(answer)
            selected[scaffold["id"]] = values
        return selected

    def review_candidates(self, ranked, count):
        if self.console.is_terminal:
            action = CandidateBrowser(ranked, self.console).run()
        else:
            successes = [row for row in ranked if row.get("fold_status") == "complete"]
            table = candidate_table(successes, len(ranked), width=self.console.width)
            if len(successes) > 10:
                table.caption = "Showing the top 10; the full ranking is in scores.csv."
            self.console.print(table)
            self.console.print(Text(RANKING_NOTE, style="#808080"))
            self.console.print(Text(ERROR_NOTE, style="#808080"))
            action = self.select(
                "Next step",
                [
                    questionary.Choice("Finish and save", "finish"),
                    questionary.Choice(
                        "Generate more sequences for the selected scaffolds", "more"
                    ),
                ],
            )
        if action == "more":
            extra = self.number("Additional sequences per selected scaffold", count)
            if extra is None:
                return None
            latest = {}
            for row in ranked:
                if row["scaffold"] not in latest or row.get(
                    "generation_round", 1
                ) > latest[row["scaffold"]].get("generation_round", 1):
                    latest[row["scaffold"]] = row
            properties = self.choose_properties(
                [
                    {"id": name, "properties": latest[name]["target_properties"]}
                    for name in sorted(latest)
                ],
                **({"fixed_length": True} if any("edit_mask" in row for row in ranked) else {}),
            )
            return MoreCandidates(extra, properties) if properties is not None else None
        return None

    def run(self, settings, *, interactive=False):
        self.heading()
        progress = DesignProgress(self.console, settings)
        task = progress.add_task("Loading generation dependencies…", total=None, detail="")
        with progress:
            progress.refresh()
            redesign_count = settings.num_diffusion
            from .provider_ui import worker_status

            # Without a running provider, every run loads Complexa from scratch.
            provider_hint = "Complexa warm" not in worker_status(
                settings.models_dir, settings.device
            )

            def status(message):
                nonlocal provider_hint
                progress.status(task, message)
                if provider_hint and message.startswith("1/5 · Starting Complexa"):
                    provider_hint = False
                    progress.show_tip(PROVIDER_HINT)

            def complexa_update(event):
                description = event["message"]
                step_detail = ""
                if "candidate_first" in event and "step" in event:
                    description = f"Candidates {event['candidate_first']}–{event['candidate_last']}/{event['candidates']}"
                    step_detail = f"step {event['step']}/{event['steps']}"
                progress.advance(
                    task, "scaffolds", event["completed"], event["total"],
                    description=description,
                    detail=f"{event['completed'] / max(event['total'], 1):.0%}",
                    step_detail=step_detail,
                    at=event.get("at"),
                )

            def stage_update(event):
                detail = f"{event['completed']:,}/{event['total']:,} folds"
                step_detail = ""
                if event["stage"] == "redesign":
                    detail = f"{event['outputs']:,}/{event['output_total']:,} redesigns"
                    step_detail = f"batch step {event['step']:,}/{event['steps']:,}"
                progress.advance(
                    task, event["stage"], event["completed"], event["total"],
                    detail=detail, step_detail=step_detail, at=event.get("at"),
                    failed=event.get("failed", False),
                    minimum=3 if event["stage"] == "redesign" else 1,
                )

            def choose(ranked):
                nonlocal redesign_count
                progress.wait_for_input(task, "Waiting for scaffold selection")
                progress.stop()
                try:
                    choice = self.choose_redesign(ranked, settings)
                    if isinstance(choice, RedesignChoice):
                        redesign_count = choice.sequences_per_scaffold
                    return choice
                finally:
                    progress.status(task, "Continuing design…")
                    progress.start()

            def review(ranked):
                nonlocal redesign_count
                progress.wait_for_input(task, "Waiting for candidate review")
                progress.stop()
                try:
                    count = self.review_candidates(ranked, redesign_count)
                    if count is not None:
                        redesign_count = count.count
                    return count
                finally:
                    progress.status(task, "Continuing design…")
                    progress.start()

            def properties_only(scaffolds):
                progress.wait_for_input(task, "Waiting for requested properties")
                progress.stop()
                try:
                    return self.choose_properties(
                        scaffolds,
                        **({"fixed_length": True} if settings.edit_residues is not None else {}),
                    )
                finally:
                    progress.status(task, "Continuing design…")
                    progress.start()

            ask = (
                interactive
                and settings.top_scaffolds is None
                and settings.scaffold_index is None
                and not settings.all_scaffolds
            )

            try:
                result = design(
                    settings,
                    on_status=status,
                    on_complexa_progress=complexa_update,
                    on_stage_progress=stage_update,
                    on_scaffolds=choose if ask else None,
                    on_candidates=review if interactive else None,
                    on_properties=properties_only if interactive and not ask else None,
                )
                progress.complete(task, "Design complete")
            except KeyboardInterrupt:
                progress.failed(task, "Design interrupted", interrupted=True)
                self.console.print(
                    "[yellow]Design interrupted. Completed stages remain in the output folder.[/]"
                )
                return 130
        self.console.print(
            Text(
                f"{result.ranked}/{result.candidates} candidates ranked · {result.directory}",
                style="green" if result.ranked else "yellow",
            )
        )
        if result.best:
            self.console.print(
                Text(
                    f"Top candidate: {result.best['scaffold']}/{result.best['candidate_id']}"
                )
            )
            metrics, errors = result.best["metrics"], result.best["property_errors"]
            self.console.print(
                Text(
                    f"pLDDT {metrics['binder_plddt']:.3f} · ipTM {metrics['iptm']:.3f} · "
                    f"length error {errors['length']:g} · charge error {errors['charge']:.3f} · "
                    f"hydrophobicity error {errors['hydrophobicity']:.3f}"
                )
            )
        self.console.print("Results: results.json · scores.csv · candidates.fasta")
        if result.ranked:
            import shlex

            self.console.print(
                Text(f"View designs: mubind view {shlex.quote(str(result.directory))}")
            )
        return 0 if result.ranked else 1
