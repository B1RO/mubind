"""Small shared pieces for compact, editable terminal menus."""

from copy import copy
from pathlib import Path
from textwrap import fill

import questionary
from prompt_toolkit.styles import merge_styles
from prompt_toolkit.filters import IsDone
from prompt_toolkit.layout import ConditionalContainer, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from questionary.prompts.common import InquirerControl


SELECTION_STYLE = questionary.Style(
    [(name, "fg:#67e8f9 bold") for name in ("pointer", "highlighted", "selected")]
)


def select_prompt(message, choices, *, style=None, **options):
    """Keep the focused label cyan, including labels with custom formatting."""
    before_render = options.pop("before_render", None)
    if isinstance(options.get("default"), questionary.Choice):
        options["default"] = options["default"].value
    instruction = options.get("instruction")
    if callable(instruction):
        options["instruction"] = " "
    prompt = questionary.select(
        message,
        choices=[copy(choice) for choice in choices],
        style=merge_styles([style, SELECTION_STYLE] if style else [SELECTION_STYLE]),
        **options,
    )
    control = next(
        item for item in prompt.application.layout.find_all_controls()
        if isinstance(item, InquirerControl)
    )
    if callable(instruction):
        # Keep live status in its own prompt-toolkit window. Questionary's
        # instruction string is fixed when the question is constructed.
        prompt.application.layout.container.children.insert(1, ConditionalContainer(
            Window(FormattedTextControl(lambda: [("class:instruction", instruction())]),
                   dont_extend_height=True),
            filter=~IsDone(),
        ))
    # In a single-choice menu, the default is only the initial cursor position.
    control.selected_options.clear()
    titles = [(choice, choice.title) for choice in control.choices]

    def highlight_focused_label(_):
        if before_render is not None:
            before_render()
        focused = control.get_pointed_at()
        for choice, title in titles:
            if choice is focused and not choice.disabled and isinstance(title, list):
                choice.title = [
                    (fragment[0] + " class:highlighted", *fragment[1:])
                    if "class:status" not in fragment[0].split() else fragment
                    for fragment in title
                ]
            else:
                choice.title = title

    prompt.application.before_render += highlight_focused_label
    return prompt


def short_path(path):
    path = Path(path).expanduser()
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def value_choice(label, value, key):
    """A white label with its value in grey; the focused row turns cyan as a whole."""
    value = str(value)
    if not value:
        return questionary.Choice(label.rstrip(), key)
    return questionary.Choice([("class:text", label), ("class:muted", value)], key)


def aligned_choices(console, entries):
    """Align labels and values, keeping long rows inside the terminal."""
    width = max(len(label) for _, label, _ in entries)
    room = max(12, console.width - width - 7)
    choices = []
    for key, label, value in entries:
        value = str(value)
        if len(value) > room:
            value = value[: room - 1] + "…"
        choices.append(value_choice(f"{label:<{width}}  ", value, key))
    return choices


def screen_intro(console, title, purpose):
    """Use the prompt's transient instruction slot instead of stacking banners."""
    return title + "\n  " + fill(
        purpose, width=max(30, console.width - 4), subsequent_indent="  "
    )
