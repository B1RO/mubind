"""Browse scored candidates without changing their ranking or saved results."""

from __future__ import annotations

from io import StringIO

from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from rich.console import Console, Group
from rich.table import Table
from rich.text import Text


ACTIONS = (
    ("finish", "Finish and save"),
    ("more", "Generate more sequences for the selected scaffolds"),
)
MUTED = "#808080"
CYAN = "#67e8f9"
ERROR_NOTE = "Errors are absolute deviations from the requested properties."
RANKING_NOTE = "Sorted by 0.5 × binder pLDDT + 0.5 × ipTM. Both metrics use 0–1 units."


def candidate_table(rows, count, *, start=0, limit=10, selected=None, width=120):
    stacked = width < 60
    compact = width < 100
    table = Table(
        title=(f"Ranking · {len(rows)}/{count} scored" if stacked
               else f"Combined ranking · {len(rows)}/{count} candidates scored"),
        caption=(f"Ranks {start + 1}–{min(start + limit, len(rows))} of {len(rows)}"
                 if rows else "No candidates scored"),
        padding=(0, 0) if compact else (0, 1),
        show_lines=stacked,
    )
    error_names = (["Len err", "Chg err", "Hyd err"] if compact else
                   ["Length err", "Charge err", "Hydro err"])
    names = (["Rank", "Candidate", "pLDDT\nipTM", "Errors"] if stacked else
             ["Rank", "Candidate", "pLDDT", "ipTM", *error_names])
    values = []
    for index in range(start, min(start + limit, len(rows))):
        row = rows[index]
        identifier = f"{row['scaffold']}/{row['candidate_id']}"
        if stacked:
            identifier = f"{row['scaffold']}\n{row['candidate_id']}"
        plddt = f"{row['metrics']['binder_plddt']:.3f}"
        iptm = f"{row['metrics']['iptm']:.3f}"
        errors = row["property_errors"]
        length, charge, hydro = (f"{errors['length']:g}", f"{errors['charge']:.3f}",
                                 f"{errors['hydrophobicity']:.3f}")
        cells = ([str(index + 1), identifier, f"{plddt}\n{iptm}",
                  f"Len {length}\nChg {charge}\nHyd {hydro}"] if stacked else
                 [str(index + 1), identifier, plddt, iptm, length, charge, hydro])
        values.append((index, cells))
    # Let identifiers yield space before any confidence or error value does.
    minimums = [
        1 if name == "Candidate" else max(
            len(line) for cell in [name, *(cells[column] for _, cells in values)]
            for line in cell.splitlines()
        )
        for column, name in enumerate(names)
    ]
    candidate_width = (width - sum(minimums) + minimums[1] - len(names) - 1
                       - (0 if compact else 2 * len(names)))
    if not stacked and any(len(cells[1]) > candidate_width for _, cells in values):
        for _, cells in values:
            cells[1] = cells[1].replace("/", "\n", 1)
    for name, minimum in zip(names, minimums):
        table.add_column(name, justify="left" if name == "Candidate" else "right",
                         min_width=minimum, no_wrap=True, overflow="ellipsis")
    for index, cells in values:
        table.add_row(*cells, style=f"bold {CYAN}" if index == selected else None)
    return table


class CandidateBrowser:
    def __init__(self, ranked, console, *, input=None, output=None):
        self.rows = [row for row in ranked if row.get("fold_status") == "complete"]
        self.count = len(ranked)
        self.console = console
        self.selected = 0
        self.page_size = 10
        self.in_actions = not self.rows
        self.action = 0
        self._cache_key = self._cache = None
        bindings = KeyBindings()

        @bindings.add("up")
        def up(event):
            self.move(-1)

        @bindings.add("down")
        def down(event):
            self.move(1)

        @bindings.add("left")
        @bindings.add("pageup")
        def previous_page(event):
            self.move_page(-1)

        @bindings.add("right")
        @bindings.add("pagedown")
        def next_page(event):
            self.move_page(1)

        @bindings.add("home")
        def first(event):
            self.selected = 0
            self.in_actions = not self.rows

        @bindings.add("end")
        def last(event):
            self.selected = max(0, len(self.rows) - 1)
            self.in_actions = not self.rows

        @bindings.add("tab")
        @bindings.add("s-tab")
        def switch(event):
            if self.rows:
                self.in_actions = not self.in_actions

        @bindings.add("enter")
        def enter(event):
            if self.in_actions:
                event.app.exit(result=ACTIONS[self.action][0])
            else:
                self.in_actions = True

        @bindings.add("escape")
        def actions(event):
            self.in_actions = True

        @bindings.add("c-c")
        @bindings.add("c-d")
        def cancel(event):
            event.app.exit(result=None)

        control = FormattedTextControl(self.render, focusable=True, show_cursor=False)
        self.application = Application(
            layout=Layout(Window(control, wrap_lines=False, dont_extend_height=True)),
            key_bindings=bindings, input=input, output=output,
            full_screen=False,
        )

    def move(self, amount):
        if self.in_actions:
            self.action = (self.action + amount) % len(ACTIONS)
        elif self.rows:
            self.selected = max(0, min(len(self.rows) - 1, self.selected + amount))

    def move_page(self, amount):
        if self.rows:
            self.in_actions = False
            self.move(amount * self.page_size)

    def content(self, width):
        start = self.selected // self.page_size * self.page_size
        table = candidate_table(self.rows, self.count, start=start,
                                limit=self.page_size, width=width,
                                selected=None if self.in_actions else self.selected)
        parts = [table]
        if self.rows:
            row = self.rows[self.selected]
            parts.append(Text(f"Rank {self.selected + 1} · {row['scaffold']}/{row['candidate_id']}",
                              style=CYAN if not self.in_actions else MUTED))
        narrow = width < 60
        parts.extend([
            Text("Sort = (pLDDT + ipTM) / 2 · 0–1" if narrow else RANKING_NOTE, style=MUTED),
            Text("Errors = absolute property deviations." if narrow else ERROR_NOTE, style=MUTED),
            Text("↑↓ rows · ←→ pages · Home/End ends" if narrow else
                 "↑↓ rows · ←→/PgUp/PgDn pages · Home/End first/last", style=MUTED),
            Text("Tab next step · Enter continue" if narrow else
                 "Tab switches ranking / next step · Enter continues", style=MUTED),
            Text("› Next step", style="bold"),
        ])
        for index, (_, label) in enumerate(ACTIONS):
            focused = self.in_actions and index == self.action
            parts.append(Text(f"{'❯' if focused else ' '} {label}",
                              style=f"bold {CYAN}" if focused else ""))
        return Group(*parts)

    def render(self):
        size = self.application.output.get_size()
        width, height = max(10, size.columns - 2), max(1, size.rows - 1)
        key = (width, height, self.selected, self.in_actions, self.action)
        if key == self._cache_key:
            return self._cache
        stream = StringIO()
        renderer = Console(file=stream, width=width, height=height,
                           force_terminal=True, color_system=self.console.color_system)
        # Use actual rendered height, including wrapped help, to keep actions visible.
        self.page_size = min(10, max(1, len(self.rows)))
        while True:
            content = self.content(width)
            lines = renderer.render_lines(content, renderer.options, pad=False)
            if len(lines) <= height or self.page_size == 1:
                break
            self.page_size -= 1
        renderer.print(content, end="")
        self._cache_key, self._cache = key, ANSI(stream.getvalue().rstrip("\n"))
        return self._cache

    def run(self):
        return self.application.run()
