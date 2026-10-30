"""Keep Rich's live cursor accounting correct when tmux reflows a narrow pane."""

import math

from rich.cells import cell_len
from rich.live_render import LiveRender
from rich.segment import Segment


class ReflowLiveRender(LiveRender):
    def __init__(self, console, original):
        super().__init__(original.renderable, original.style, original.vertical_overflow)
        self.console = console
        self.width = console.width
        self.line_widths = []

    def position_cursor(self):
        width = self.console.width
        if width < self.width and self._shape is not None:
            # A line which fit before the resize now occupies several terminal
            # rows. Rich otherwise erases only the old row count, leaving ghosts.
            height = sum(max(1, math.ceil(length / width)) for length in self.line_widths)
            self._shape = (width, min(height, self.console.height))
        self.width = width
        return super().position_cursor()

    def __rich_console__(self, console, options):
        segments = list(super().__rich_console__(console, options))
        self.line_widths = [
            cell_len("".join(segment.text for segment in line).rstrip())
            for line in Segment.split_lines(segments)
        ]
        self.width = console.width
        yield from segments


def track_reflow(live):
    # Isolate this Rich integration here; rendering and the live stack stay Rich's.
    live._live_render = ReflowLiveRender(live.console, live._live_render)
    return live
