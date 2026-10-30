"""Immediate provider presentation, using only the Python standard library.

The launcher and module entry point share one title across exec. The transient
status occupies one line; the Rich worker display takes over that line later.
"""

import os
import sys
import threading
import time

TITLE = "mubind · Provider"
TITLE_STYLE = "bold #d787ff"
_HANDOFF = "_MUBIND_PROVIDER_STARTUP"
_screen = None


def provider_requested(arguments):
    arguments = list(arguments)
    while arguments[:1] == ["--bootstrap-yes"]:
        arguments = arguments[1:]
    return arguments[:1] == ["provider"] and "--help" not in arguments


class _ErrorStream:
    """Clear the transient line before Click or an import reports an error."""

    def __init__(self, stream):
        self.stream = stream

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def write(self, text):
        if text:
            finish_provider_startup()
        return self.stream.write(text)


class _Startup:
    def __init__(self, inherited, *, title=TITLE, message="Preparing provider…"):
        self.stream = sys.stdout
        self.terminal = self.stream.isatty() and os.environ.get("TERM") != "dumb"
        self.color = self.terminal and "NO_COLOR" not in os.environ
        self.started = time.monotonic()
        self.message = message
        resumed = False
        try:
            terminal, started = inherited.split(",")
            started = float(started)
            if terminal == str(int(self.terminal)) and 0 <= started <= self.started:
                self.started, resumed = started, True
        except (AttributeError, ValueError):
            pass
        self.stop = threading.Event()
        self.thread = None
        self.error_stream = None
        if not resumed and title:
            self.stream.write(self.paint(title, "1;38;2;215;135;255") + "\n")
        if self.terminal:
            self.draw()
            self.thread = threading.Thread(target=self.refresh, daemon=True)
            self.thread.start()
            self.error_stream = _ErrorStream(sys.stderr)
            sys.stderr = self.error_stream
        elif not resumed:
            self.stream.write(message + "\n")
        self.stream.flush()

    def paint(self, text, style):
        return f"\033[{style}m{text}\033[0m" if self.color else text

    def draw(self):
        elapsed = max(0, time.monotonic() - self.started)
        spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"[int(elapsed * 8) % 10]
        text = f"{spinner} {self.message}"
        if elapsed >= 1:
            text += f" · {int(elapsed)}s"
        try:
            width = os.get_terminal_size(self.stream.fileno()).columns
        except (AttributeError, OSError, ValueError):
            width = 80
        self.stream.write("\r\033[2K" + self.paint(text[:max(1, width - 1)], "90"))
        self.stream.flush()

    def refresh(self):
        while not self.stop.wait(0.125):
            self.draw()

    def finish(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
        if sys.stderr is self.error_stream:
            sys.stderr = self.error_stream.stream
        if self.terminal:
            self.stream.write("\r\033[2K")
            self.stream.flush()


def start_provider_startup():
    global _screen
    if _screen is None:
        _screen = _Startup(os.environ.pop(_HANDOFF, None))


def start_cli_startup(arguments):
    """Show feedback before environment checks and UI imports, including home."""
    global _screen
    if provider_requested(arguments):
        start_provider_startup()
        return
    arguments = list(arguments)
    while arguments[:1] == ["--bootstrap-yes"]:
        arguments = arguments[1:]
    if not arguments and _screen is None and sys.stdin.isatty() and sys.stdout.isatty():
        _screen = _Startup(os.environ.pop(_HANDOFF, None), title=None, message="Opening mubind…")


def startup_message(message):
    if _screen is not None:
        previous = _screen.message
        _screen.message = message
        return previous


def finish_provider_startup(stream=None):
    """Remove the status and report whether this stream already has the title."""
    global _screen
    screen, _screen = _screen, None
    if screen is None:
        return False
    screen.finish()
    return stream is None or stream is screen.stream


def handoff_provider_startup(env):
    """Keep the title and elapsed time when uv starts the application Python."""
    if _screen is not None:
        env[_HANDOFF] = f"{int(_screen.terminal)},{_screen.started}"
        finish_provider_startup()
