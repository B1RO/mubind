"""Guided checkpoint evaluation through the shared workflow."""

from __future__ import annotations

import subprocess
from pathlib import Path
from textwrap import fill

import questionary
from prompt_toolkit.styles import merge_styles
from rich.console import Console
from rich.table import Table
from rich.text import Text

from ..paths import EVALUATIONS
from .terminal_ui import select_prompt
from ..benchmarks.common import MODES, unused_output
from ..benchmarks.property_inputs import default_property_dataset, has_property_dataset
from ..benchmarks.unconditional import COMPUTATIONS, CORPUS_METRICS, ESMC_METRICS

PAPER_SECTIONS = {
    "4.1": (
        "Unconditional evaluation",
        "Choose sequence, novelty, distribution and ESMC computations in a submenu.",
        ("unconditional",),
    ),
    "4.2": (
        "De novo biochemical property steering",
        "Measure property accuracy under individual and joint conditioning.",
        ("properties",),
    ),
    "4.3": (
        "3Di token steering",
        "Measure fold and 3Di recovery with standard and guided sampling.",
        ("structure",),
    ),
    "4.4": (
        "End-to-end pipeline evaluation",
        "Assess binder-fold and interface recovery after sequence redesign.",
        ("full-design", "inpainting"),
    ),
}

# These summaries describe what Start will execute, like Generate's plan summary.
# The fixed scientific protocols remain in the runner.
BENCHMARK_SUMMARIES = {
    "unconditional": (
        "mubind will generate 1,024 unconditioned sequences for the selected computations."
    ),
    "properties": (
        "mubind will randomly draw property requests from the selected training dataset "
        "and generate 1,000 sequences for each of four conditioning settings, "
        "specifying length, charge or hydrophobicity individually, or all three jointly. "
        "The report will compare requested and generated properties using raw and "
        "standardized mean absolute errors. Each run chooses a fresh seed unless you set one."
    ),
    "structure": (
        "mubind will generate 10 candidates with standard sampling and 10 with stronger "
        "3Di guidance for each of 92 reference proteins. ESMFold2 will fold the candidates. "
        "The highest-pLDDT candidate in each setting will be assessed by TM-score and "
        "3Di recovery."
    ),
    "full-design": (
        "mubind will generate 50 candidates per PDB complex in the selected folder, "
        "using the supplied binder's 3Di and properties. ESMFold2 will predict each candidate "
        "with its target, followed by ranking and measurement of property accuracy, "
        "fold recovery and interface confidence."
    ),
    "inpainting": (
        "mubind will generate 20 completions per PDB complex in the selected folder, with half the binder residues "
        "masked in contiguous spans. Candidates will be folded with their targets and "
        "ranked among those preserving all supplied residues. The report will assess "
        "property accuracy, fold recovery and interface confidence."
    ),
}

BENCHMARK_LABELS = {
    "unconditional": "Unconditional evaluation…",
    "properties": "Property steering",
    "structure": "3Di token steering",
    "full-design": "Full binder redesign",
    "inpainting": "Binder inpainting",
}

TITLE_STYLE = questionary.Style([("question", "fg:#d787ff bold")])


def default_paper_corpus():
    """Find the reference split prepared by setup, without loading model code."""
    from ..setup.config import load_config
    from ..preprocessing.prostt5.steps.validate import discover_datasets, is_ready_for_training

    try:
        for info in discover_datasets(load_config().data_dir.expanduser()):
            if is_ready_for_training(info, 50, 70):
                return info.path
    except (ValueError, OSError):
        pass
    return None


class EvaluationUI:
    """Review a checkpoint and paper benchmark together before an explicit start."""

    def __init__(self, console, style=None):
        self.console, self.style = console, style
        self.checkpoint = None
        self.recipe = None
        self.problems = {}

    def heading(self):
        if not getattr(self, "_heading_shown", False):
            self.console.print()
            self.console.print(Text("mubind · Evaluate", style="bold #d787ff"))
            self._heading_shown = True

    def select(self, message, choices, **options):
        descriptions = options.pop("descriptions", ())
        prompt = select_prompt(
            message, choices=choices, style=options.pop("style", self.style),
            qmark="›", pointer="❯", **options
        )
        if descriptions:
            from questionary.prompts.common import InquirerControl

            control = next(
                item for item in prompt.application.layout.find_all_controls()
                if isinstance(item, InquirerControl)
            )
            original_tokens = control._get_choice_tokens
            # Questionary gives headings and descriptions the same separator style.
            # Use Generate's gray instruction style for description lines only.
            control.text = lambda: [
                ("class:instruction", *token[1:])
                if token[0] == "class:separator" and token[1] in descriptions
                else token for token in original_tokens()
            ]
        # Match Generate's transient menus instead of leaving answered menus above
        # every new screen.
        prompt.application.erase_when_done = True
        return prompt.ask()

    def path(self, label, default=""):
        answer = questionary.path(label, default=str(default or ""), style=self.style).ask()
        return Path(answer.strip()).expanduser() if answer and answer.strip() else None

    def inspect(self, path):
        from ..benchmarks.checkpoint_generation import inspect_checkpoint

        with self.console.status("Reading checkpoint settings…"):
            recipe, problems = inspect_checkpoint(path)
        self.checkpoint = Path(path).resolve()
        self.recipe, self.problems = recipe, problems
        return problems

    def choose_checkpoint(self):
        from .checkpoint_catalog import discover_design_checkpoints
        from .terminal_ui import short_path

        while True:
            options = discover_design_checkpoints(self.checkpoint, check=False)
            choice = self.select(
                "Checkpoint",
                [
                    *(questionary.Choice(short_path(option.path), option.path) for option in options),
                    questionary.Choice("Choose checkpoint path…", "path"),
                    questionary.Choice("Back", "back"),
                ],
            )
            if choice in (None, "back"):
                return None
            path = self.path("mubind checkpoint", self.checkpoint) if choice == "path" else choice
            if path is None:
                continue
            try:
                problems = self.inspect(path)
                # Share the choice with Generate and later sessions.
                from .preferences import save_checkpoint_choice

                save_checkpoint_choice(self.checkpoint)
                return problems
            except (ValueError, TypeError, OSError, RuntimeError, KeyError) as exc:
                self.console.print(Text(f"Cannot use this checkpoint\n{exc}", style="red"))

    def instruction(self, description):
        return "\n" + fill(
            description, width=max(30, self.console.width - 4),
            initial_indent="  ", subsequent_indent="  ",
        ) + "\n"

    def choose_benchmark(self, settings=None):
        """Paper protocols; section 4.1 opens its computation submenu."""
        choices = []
        descriptions = set()
        for section, (title, purpose, tests) in PAPER_SECTIONS.items():
            description = fill(
                purpose, width=max(24, self.console.width - 6),
                subsequent_indent="   ",
            )
            descriptions.add(description)
            choices.extend([
                questionary.Separator(f"{section} {title}"),
                questionary.Separator(description),
            ])
            for test in tests:
                choices.append(questionary.Choice(
                    BENCHMARK_LABELS[test], test, disabled=self.problems.get(test) or None,
                ))
            choices.append(questionary.Separator(" "))
        selected = settings.mode if settings else None
        available = [choice.value for choice in choices if not choice.disabled]
        return self.select(
            "Benchmark", [*choices, questionary.Choice("Back", "back")],
            instruction=self.instruction("Select a protocol for the current checkpoint."),
            default=selected if selected in available else None,
            descriptions=descriptions,
        )

    def choose_computations(self, current=()):
        """Toggle an explicit selection, applying it only on Done."""
        selected = set(current)
        focus = next(iter(COMPUTATIONS))
        while True:
            choices = []
            width = max(len(label) for label, _ in COMPUTATIONS.values())
            for key, (label, description) in COMPUTATIONS.items():
                mark = "[x]" if key in selected else "[ ]"
                # Keep explanations gray even while their computation is focused.
                # Wrap explanations on narrow terminals.
                prefix = f"{mark} {label:<{width}}  "
                room = self.console.width - len(prefix) - 4
                if room >= 24:
                    detail = fill(description, width=room, subsequent_indent="")
                    detail = detail.replace("\n", "\n" + " " * (len(prefix) + 2))
                else:
                    prefix = f"{mark} {label}\n     "
                    detail = fill(description, width=max(24, self.console.width - 9), subsequent_indent="     ")
                choices.append(questionary.Choice([
                    ("class:text", prefix),
                    ("class:status class:instruction", detail),
                ], key))
            choices.extend([
                questionary.Separator(" "),
                questionary.Choice("Select all", "all"),
                questionary.Choice("Clear selection", "clear"),
                questionary.Choice(f"Done · {len(selected)} selected", "done", disabled=None if selected else "Choose at least one computation"),
                questionary.Choice("Back", "back"),
            ])
            choice = self.select(
                "4.1 Unconditional evaluation", choices, default=focus,
                instruction=self.instruction(
                    "Choose which measurements to run on 1,024 generated sequences. "
                    "Empty outputs and sequences with internal gaps are excluded from subsequent measurements."
                ) + self.instruction(
                    "Press Enter to toggle an option, then choose Done to apply your selection."
                ),
            )
            if choice in (None, "back"):
                return None
            if choice == "done" and selected:
                return tuple(key for key in COMPUTATIONS if key in selected)
            if choice == "all":
                selected = set(COMPUTATIONS)
            elif choice == "clear":
                selected.clear()
            elif choice in COMPUTATIONS:
                selected.symmetric_difference_update((choice,))
            focus = choice if choice != "done" else next(iter(COMPUTATIONS))

    def benchmark_name(self, settings):
        if not settings.mode:
            return "Choose a protocol"
        section = next(key for key, (_, _, tests) in PAPER_SECTIONS.items() if settings.mode in tests)
        name = BENCHMARK_LABELS[settings.mode].rstrip("…")
        return f"{section} · {name}"

    def stages(self, settings):
        from ..benchmarks.workflow import STRUCTURAL

        stages = "Generate → fold → score" if settings.mode in STRUCTURAL else "Generate"
        if settings.mode == "properties" or set(settings.computations) - ESMC_METRICS:
            stages += " → score"
        return stages + (" → ESMC comparison" if settings.needs_esmc else "")

    def start_problem(self, settings):
        if not self.checkpoint:
            return "Choose a checkpoint"
        if not settings.mode:
            return "Choose a benchmark"
        if self.problems.get(settings.mode):
            return self.problems[settings.mode]
        if settings.mode == "unconditional":
            if not settings.computations:
                return "Choose computations"
        if settings.needs_corpus:
            if settings.dataset is None or not (settings.dataset / "train/aa.npy").is_file():
                return "Choose the paper reference sequences"
        if settings.mode == "properties" and not has_property_dataset(settings.dataset):
            return "Choose a prepared training dataset"
        if settings.mode in {'full-design', 'inpainting'}:
            folder = settings.complexes_dir.expanduser()
            if not folder.is_dir() or not any(p.is_file() and p.suffix.lower() == '.pdb' for p in folder.iterdir()):
                return 'Choose a folder containing PDB complexes'
        return None

    def menu(self, settings):
        from .terminal_ui import aligned_choices, short_path

        checkpoint = short_path(self.checkpoint) if self.checkpoint else "Choose a checkpoint"
        if self.recipe:
            checkpoint += f" · step {self.recipe.step:,} · {self.recipe.weights.upper()}"
        entries = [
            ("benchmark", "Benchmark", self.benchmark_name(settings)),
            ("checkpoint", "Checkpoint", checkpoint),
        ]
        if settings.mode == "unconditional":
            entries.append(("metrics", "Computations", f"{len(settings.computations)} selected · choose metrics…"))
        if settings.needs_corpus:
            entries.append(("dataset", "Reference sequences", short_path(settings.dataset) if settings.dataset else "Missing · choose the original paper corpus"))
        if settings.mode == "properties":
            entries.append(("dataset", "Training dataset", short_path(settings.dataset) if settings.dataset else "Missing · choose prepared training data"))
        if settings.mode in {'full-design', 'inpainting'}:
            entries.append(('complexes', 'Complexes folder', short_path(settings.complexes_dir)))
        problem = self.start_problem(settings)
        start = [
            ("class:status class:error strike", "Start evaluation"),
            ("class:status class:error", f" [{problem}]"),
        ] if problem else "Start evaluation"
        return [
            *aligned_choices(self.console, entries),
            questionary.Separator(" "),
            questionary.Choice("Configure evaluation", "settings"),
            questionary.Separator(" "),
            questionary.Choice(start, "start"),
            questionary.Separator(" "),
            questionary.Choice("Back", "back"),
        ]

    def introduction(self, settings):
        purpose = (
            "Evaluate a checkpoint with sequence, property or structural benchmarks. "
            "Binder redesign and inpainting use a folder of PDB complexes. "
            "Select a benchmark below to review the run."
        )
        if settings.mode:
            purpose = BENCHMARK_SUMMARIES[settings.mode]
            if settings.mode in {'full-design', 'inpainting'}:
                purpose += ' By default, this uses the 45 bundled PPIRef complexes. Each contains an experimentally observed binder and its target.'
            if settings.mode == "unconditional":
                purpose += " Selected computations · " + ", ".join(
                    COMPUTATIONS[key][0].lower() for key in settings.computations
                ) + "."
                if set(settings.computations) - {"validity"}:
                    purpose += " Invalid canvases are excluded from sequence analyses."
                if "esmc-shuffle" in settings.computations:
                    purpose += " Shuffle controls include scores for their unshuffled parents."
        return self.instruction(purpose)

    def review(self, settings):
        from ..benchmarks.workflow import STRUCTURAL

        self.console.print(Text("Evaluation settings", style="bold cyan"))
        self.console.print(Text(self.introduction(settings).strip(), style="#808080"))
        rows = Table.grid(padding=(0, 2))
        rows.add_row("Checkpoint", Text(str(self.checkpoint or "Not selected")))
        rows.add_row("Benchmark", self.benchmark_name(settings))
        rows.add_row("Save to", Text(str(settings.out)))
        rows.add_row("Device", settings.device)
        if settings.mode:
            rows.add_row("Stages", self.stages(settings))
        if settings.needs_corpus:
            rows.add_row("Reference sequences", Text(str(settings.dataset or "Not selected")))
        if settings.mode == "properties":
            rows.add_row("Training dataset", Text(str(settings.dataset or "Not selected")))
            rows.add_row("Sampling seed", str(settings.seed) if settings.seed is not None else "Random for each run")
        if settings.mode in {'full-design', 'inpainting'}:
            rows.add_row('Complexes folder', Text(str(settings.complexes_dir)))
        if settings.mode in STRUCTURAL or settings.needs_esmc:
            rows.add_row("Models", Text(str(settings.models_dir or "Setup's model folder")))
        if settings.mode in STRUCTURAL:
            rows.add_row("Folding", settings.endpoint)
        if settings.mode in STRUCTURAL:
            rows.add_row("Foldseek", settings.foldseek)
        self.console.print(rows)
        if settings.needs_corpus:
            self.console.print(Text(
                "The corpus is separate from model weights. Choose the original folder containing train/aa.npy; its identity is checked before sampling.", style="#808080"
            ))
        if settings.mode in {"full-design", "inpainting"}:
            self.console.print(Text("Predicted scores do not establish experimental binding.", style="dim"))

    def fields(self, settings):
        from ..benchmarks.workflow import STRUCTURAL

        fields = [("Output folder", "out"), ("Device", "device")]
        if settings.mode in {'full-design', 'inpainting'}:
            fields.insert(0, ('Complexes folder', 'complexes'))
        if settings.needs_corpus:
            fields += [("Reference sequences", "dataset")]
        if settings.mode in STRUCTURAL or settings.needs_esmc:
            fields += [("Model folder", "models_dir")]
        if settings.mode in STRUCTURAL:
            fields += [("Folding service", "endpoint")]
        if settings.mode in STRUCTURAL:
            fields += [("Foldseek executable", "foldseek")]
        if settings.mode == "properties":
            fields += [("Training dataset", "dataset"), ("Sampling seed", "seed"), ("Save figures", "figures")]
        return fields

    def edit(self, settings, choice):
        from dataclasses import replace

        current = settings.complexes_dir if choice == 'complexes' else getattr(settings, choice)
        label = dict((key, label) for label, key in self.fields(settings))[choice]
        if choice == "figures":
            from .terminal_ui import screen_intro

            purpose = "Save plots comparing requested and generated properties."
            value = self.select(
                "", [questionary.Choice("On", True), questionary.Choice("Off", False)],
                instruction=screen_intro(self.console, label, purpose), default=current,
            )
        elif choice in {"out", "dataset", "models_dir", "complexes"}:
            if choice == "dataset" and settings.mode == "properties":
                self.console.print(Text(
                    "Choose prepared training data with prostt5_diffusion.json and train/properties.npy. Property requests are sampled from its training rows.", style="#808080"
                ))
            elif choice == "dataset":
                self.console.print(Text(
                    "Choose the original paper training sequences, used to measure novelty. This is separate from your checkpoint.", style="#808080"
                ))
                label = "Reference folder containing train/aa.npy"
            elif choice == 'complexes':
                self.console.print(Text('Each PDB must contain two protein chains in file order, binder first and target second.', style='#808080'))
            default_dataset = default_property_dataset if settings.mode == "properties" else default_paper_corpus
            value = self.path(label, current or (default_dataset() if choice == "dataset" else ""))
            if value is not None and choice == "dataset":
                if settings.mode == "properties":
                    if not has_property_dataset(value):
                        self.console.print(Text("Choose a prepared dataset containing prostt5_diffusion.json and train/properties.npy.", style="yellow"))
                        return settings
                elif not (value / "train/aa.npy").is_file():
                    self.console.print(Text(f"Reference sequences not found at {value / 'train/aa.npy'}. Choose another folder.", style="yellow"))
                    return settings
            if value is not None and choice == 'complexes' and not value.is_dir():
                self.console.print(Text(f'Complexes folder not found at {value}. Choose another folder.', style='yellow'))
                return settings
        elif choice == "device":
            value = self.select("Device", ["auto", "cpu", "cuda"])
        elif choice == "seed":
            answer = questionary.text("Sampling seed (blank for a fresh seed)", default=str(current) if current is not None else "", style=self.style).ask()
            if answer is None:
                return settings
            if not answer.strip():
                return replace(settings, seed=None)
            try:
                value = int(answer)
                if not 0 <= value < 2**63:
                    raise ValueError
            except ValueError:
                self.console.print(Text("Enter an integer from 0 to 2^63 - 1, or leave blank.", style="yellow"))
                return settings
        else:
            value = questionary.text(label, default=str(current), style=self.style).ask()
        return replace(settings, **{choice: value}) if value is not None else settings

    def configure(self, settings):
        from .terminal_ui import aligned_choices

        while True:
            entries = []
            for label, key in self.fields(settings):
                value = settings.complexes_dir if key == 'complexes' else getattr(settings, key)
                if isinstance(value, bool):
                    value = "On" if value else "Off"
                entries.append((key, label, value if value is not None else ("Fresh random seed" if key == "seed" else "Automatic")))
            choice = self.select(
                "Configure evaluation",
                [
                    *aligned_choices(self.console, entries),
                    questionary.Separator(" "),
                    questionary.Choice("All settings and paths", "details"),
                    questionary.Choice("Back", "back"),
                ],
                instruction=self.instruction(
                    "Set runtime and output options for the selected protocol. "
                    "Sample counts and conditioning settings remain fixed."
                ),
            )
            if choice in (None, "back"):
                return settings
            if choice == "details":
                self.review(settings)
                self.select("", [questionary.Choice("Back", "back")])
            else:
                settings = self.edit(settings, choice)

    def default_output(self, mode, esmc=False):
        stem = self.checkpoint.stem if self.checkpoint else "checkpoint"
        return unused_output(EVALUATIONS / f"{stem}-{mode or 'benchmark'}{'-esmc' if esmc else ''}")

    def screen(self):
        from dataclasses import replace
        from ..setup.checkpoint import default_checkpoint
        from ..benchmarks.workflow import EvaluationConfig
        from .evaluate_progress import run_with_progress

        self.heading()
        path = self.checkpoint or default_checkpoint()
        if path:
            try:
                self.inspect(path)
            except (ValueError, TypeError, OSError, RuntimeError, KeyError) as exc:
                self.console.print(Text(f"Choose another checkpoint · {exc}", style="yellow"))
        settings = EvaluationConfig(self.checkpoint, "", self.default_output(""))
        unconditional_metrics = ()
        datasets = {}
        automatic_output = settings.out
        focus = None
        while True:
            problem = self.start_problem(settings)
            required = (
                "checkpoint" if not self.checkpoint else
                "benchmark" if not settings.mode else
                "checkpoint" if self.problems.get(settings.mode) else
                "metrics" if settings.mode == "unconditional" and not settings.computations else
                "complexes" if problem and settings.mode in {'full-design', 'inpainting'} else
                "dataset" if problem else None
            )
            if focus in (None, "checkpoint", "benchmark", "metrics", "dataset", "complexes", "start"):
                focus = required or "start"
            choice = self.select(
                "Evaluate settings", self.menu(settings), default=focus,
                instruction=self.introduction(settings),
                style=merge_styles([self.style, TITLE_STYLE] if self.style else [TITLE_STYLE]),
            )
            if choice in (None, "back"):
                return
            focus = choice
            if choice == "start" and required:
                choice = required
            try:
                if choice == "checkpoint":
                    self.choose_checkpoint()
                    settings = replace(settings, checkpoint=self.checkpoint)
                elif choice == "benchmark":
                    test = self.choose_benchmark(settings)
                    if test in (None, "back"):
                        continue
                    mode = test
                    if settings.mode in {"properties", "unconditional"}:
                        datasets[settings.mode] = settings.dataset
                    if mode == "unconditional":
                        chosen = self.choose_computations(unconditional_metrics)
                        if chosen is None:
                            continue
                        unconditional_metrics = chosen
                    settings = replace(
                        settings, mode=mode, esmc=False,
                        metrics=unconditional_metrics if mode == "unconditional" else None,
                        figures=settings.figures if mode == "properties" else False,
                        seed=settings.seed if mode == "properties" else None,
                        dataset=datasets.get(mode) or (
                            default_property_dataset() if mode == "properties" else
                            default_paper_corpus() if mode == "unconditional" and set(unconditional_metrics) & CORPUS_METRICS else None
                        ),
                    )
                elif choice == "metrics":
                    chosen = self.choose_computations(settings.computations)
                    if chosen is not None:
                        unconditional_metrics = chosen
                        settings = replace(settings, metrics=chosen, esmc=False)
                        if settings.needs_corpus and settings.dataset is None:
                            settings = replace(settings, dataset=default_paper_corpus())
                elif choice in {"out", "dataset", "complexes"}:
                    settings = self.edit(settings, choice)
                elif choice == "settings":
                    settings = self.configure(settings)
                elif choice == "start":
                    problem = self.start_problem(settings)
                    if problem:
                        self.console.print(Text(problem, style="yellow"))
                        continue
                    run_with_progress(settings, self.console)
                    # Stay in Evaluate after completion, with a fresh output for another run.
                    settings = replace(settings, out=unused_output(settings.out))
                    automatic_output = settings.out
                if choice in {"checkpoint", "benchmark", "metrics"} and settings.out == automatic_output:
                    automatic_output = self.default_output(settings.mode, settings.needs_esmc)
                    settings = replace(settings, out=automatic_output)
            except KeyboardInterrupt:
                self.console.print(Text("Evaluation interrupted. Saved outputs remain; Start begins a fresh run in a new folder.", style="yellow"))
            except (ValueError, TypeError, OSError, RuntimeError, KeyError, ImportError, subprocess.CalledProcessError) as exc:
                self.console.print(Text(f"Evaluation could not finish\n{exc}", style="red"))
