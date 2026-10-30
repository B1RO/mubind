"""Terminal presentation for mubind provider: one row per model.

Loading details are measured from the worker process, the GPU and the worker's
log. The spinner runs only while one of those shows activity.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from collections import deque
from contextlib import contextmanager, nullcontext
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from .startup import TITLE, TITLE_STYLE, finish_provider_startup
from .terminal_live import track_reflow

QUIET_AFTER = 30
MUTED = "#808080"
OPERATIONS = {
    "generate": "scaffolds",
    "fold_monomer": "monomer fold",
    "fold_complex": "complex fold",
}
ACTIVE_OPERATIONS = {
    "generate": "generating scaffolds",
    "fold_monomer": "folding monomer",
    "fold_complex": "folding complex",
}
# tqdm bars such as "Loading checkpoint shards:  50%|█████     | 3/6 [...]".
COUNTED_STEP = re.compile(r"([A-Za-z][\w ]*?):\s*\d+%\|[^|\r\n]*\|\s*(\d+)/(\d+)")


def short_duration(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes}m {seconds:02d}s" if minutes else f"{seconds}s"


def gigabytes(mebibytes: float) -> str:
    return f"{mebibytes / 1024:.1f} GiB"


def model_name(service) -> str:
    # The internal label also names sockets; keep that identity unchanged.
    return "Proteina Complexa" if service.label == "Complexa" else service.label


def single_line(value: str) -> str:
    return " ".join(re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", value).split())


def join(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def exit_reason(returncode: int) -> str:
    if returncode < 0:
        try:
            return f"worker was killed by {signal.Signals(-returncode).name}"
        except ValueError:
            return f"worker was killed by signal {-returncode}"
    return f"worker exited with code {returncode}"


def process_usage(pid: int) -> tuple[int, int] | None:
    """CPU ticks and resident memory in MiB, read from /proc."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        ticks = int(fields[11]) + int(fields[12])
        return ticks, int(fields[21]) * os.sysconf("SC_PAGE_SIZE") // 2**20
    except (OSError, ValueError, IndexError):
        return None


class GpuMemory:
    """Per-process and device memory from nvidia-smi; absent when it is unavailable."""

    def __init__(self, device: str) -> None:
        index = device.partition(":")[2] or "0"
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            entries = [entry.strip() for entry in visible.split(",")]
            index = entries[int(index)] if int(index) < len(entries) else None
        self.index = index
        self.available = True
        self.sampled = 0.0
        self.processes: dict[int, int] = {}
        self.device_sampled = 0.0
        self.device_usage = None

    def query(self, *arguments: str) -> list[list[str]] | None:
        if not self.available:
            return None
        try:
            result = subprocess.run(
                ["nvidia-smi", *arguments, "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except FileNotFoundError:
            self.available = False
            return None
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode:
            return None
        return [
            [field.strip() for field in line.split(",")]
            for line in result.stdout.splitlines()
            if line.strip()
        ]

    def process(self, pid: int | None) -> int | None:
        if pid is None or self.index is None:
            return None
        now = time.monotonic()
        if now - self.sampled >= 1:
            self.sampled = now
            rows = self.query(
                "--query-compute-apps=pid,used_memory", "-i", self.index
            ) or []
            self.processes = {
                int(row[0]): int(row[1])
                for row in rows
                if len(row) == 2 and row[0].isdigit() and row[1].isdigit()
            }
        return self.processes.get(pid)

    def device(self) -> tuple[int, int] | None:
        if self.index is None:
            return None
        now = time.monotonic()
        if now - self.device_sampled < 2:
            return self.device_usage
        self.device_sampled = now
        rows = self.query("--query-gpu=memory.used,memory.total", "-i", self.index)
        try:
            used, total = rows[0]
            self.device_usage = int(used), int(total)
        except (TypeError, ValueError, IndexError):
            self.device_usage = None
        return self.device_usage


class ModelRow:
    def __init__(self, service, gpu: GpuMemory) -> None:
        self.service = service
        self.gpu = gpu
        self.state = "waiting"
        self.reason = ""
        self.started = self.finished = 0.0
        self.spinner = Spinner("dots", style="yellow")
        self.pid: int | None = None
        self.elsewhere = False
        self.status_checked = 0.0
        self.log_offset = self.log_position = 0
        self.log_rest = ""
        self.step = ""
        self.sample: tuple | None = None
        self.sampled = 0.0
        self.last_activity = 0.0
        self.ram = self.vram = None
        self.active = None
        self.event_pid = None
        self.last_event = 0
        self.events_supported = None
        self.progress_supported = False

    def begin(self) -> None:
        self.state = "loading"
        self.started = self.last_activity = time.monotonic()
        log = self.service.log
        self.log_offset = self.log_position = log.stat().st_size if log.exists() else 0

    def end(self, state: str, reason: str = "") -> None:
        self.state, self.reason = state, reason
        self.finished = time.monotonic()
        self.status_checked = self.gpu.sampled = 0.0
        self.vram = self.gpu.process(self.worker_pid())

    def worker_pid(self) -> int | None:
        if self.service.process is not None:
            return self.service.process.pid
        # A worker started by another command reports its own process ID.
        now = time.monotonic()
        if self.pid is None and now - self.status_checked >= 2:
            self.status_checked = now
            try:
                status = self.service.status()
            except Exception:  # noqa: BLE001 - display only; the provider reports failures
                status = None
            self.elsewhere = status is not None
            self.pid = (status or {}).get("pid")
        return self.pid

    def read_log(self) -> bool:
        try:
            with self.service.log.open("rb") as stream:
                stream.seek(self.log_position)
                block = stream.read(1 << 20)
        except OSError:
            return False
        if not block:
            return False
        self.log_position += len(block)
        text = self.log_rest + block.decode(errors="replace")
        self.log_rest = text[-200:]
        matches = COUNTED_STEP.findall(text)
        if matches:
            name, done, total = matches[-1]
            name = name.strip().removeprefix("Loading ").lower()
            # A finished count is no longer what the worker is doing.
            self.step = f"{name} {done}/{total}" if done != total else ""
        return True

    def observe(self) -> None:
        """Sample at most once per second; any change counts as activity."""
        now = time.monotonic()
        if now - self.sampled < 1:
            return
        self.sampled = now
        pid = self.worker_pid()
        usage = process_usage(pid) if pid is not None else None
        self.vram = self.gpu.process(pid)
        self.ram = usage[1] if usage is not None else None
        sample = (usage, self.vram)
        if self.read_log() or (self.sample is not None and sample != self.sample):
            self.last_activity = now
        self.sample = sample

    def memory(self) -> str:
        ram = gigabytes(self.ram) if self.ram is not None else "n/a"
        vram = gigabytes(self.vram) if self.vram is not None else "n/a"
        return f"RAM {ram} · VRAM {vram}"

    def progress_detail(self) -> str:
        if not self.active or self.active["operation"] != "generate":
            return ""
        progress = self.active.get("progress")
        if not progress:
            return ("waiting for sampling progress" if self.progress_supported
                    else "detailed progress unavailable in this running worker")
        if progress.get("phase") == "sampling":
            text = (
                f"candidates {progress['candidate_first']}–{progress['candidate_last']}"
                f"/{progress['candidates']} · step {progress['step']}/{progress['steps']}"
            )
            if progress.get("batches"):
                text += f" · batch {progress['batch']}/{progress['batches']}"
        else:
            text = single_line(progress["message"]).removeprefix("Complexa · ")
        quiet = time.time() - progress["updated_at"]
        if quiet >= QUIET_AFTER:
            text += f" · last update {short_duration(quiet)} ago"
        return text

    def log_tail(self, count: int = 6) -> list[str]:
        try:
            with self.service.log.open("rb") as stream:
                stream.seek(
                    max(self.log_offset, self.service.log.stat().st_size - 20000)
                )
                text = stream.read().decode(errors="replace")
        except OSError:
            return []
        lines = [line.strip() for line in re.split(r"[\r\n]+", text)]
        return [
            line
            for line in lines
            # Skip progress bars and traceback underlines.
            if line.strip("^~ ") and not COUNTED_STEP.search(line)
        ][-count:]

    def cells(self) -> tuple:
        now = time.monotonic()
        elapsed = ""
        if self.state == "waiting":
            return Text("·", style="dim"), Text("waiting", style="dim"), elapsed
        if self.state == "loading":
            self.observe()
            elapsed = short_duration(now - self.started)
            details = [self.memory()]
            if self.step:
                details.append(self.step)
            quiet = now - self.last_activity
            if quiet >= QUIET_AFTER:
                text = Text(f"no activity for {short_duration(quiet)}", style="yellow")
                symbol = Text("…", style="yellow")
            else:
                if self.service.process is not None:
                    text = Text("loading weights", style="yellow")
                elif self.elsewhere:
                    text = Text("loading weights in another process", style="yellow")
                else:
                    text = Text("starting worker", style="yellow")
                symbol = self.spinner
            for detail in details:
                text.append(" · " + detail, style=MUTED)
            return symbol, text, elapsed
        if self.state in ("ready", "reused"):
            self.observe()
            label = "ready" if self.state == "ready" else "already loaded"
            text = Text(label, style="green")
            if self.state == "ready":
                elapsed = short_duration(self.finished - self.started)
            if self.active:
                operation = ACTIVE_OPERATIONS.get(self.active["operation"], "processing")
                text = Text(operation, style="bold cyan")
                elapsed = short_duration(max(0, time.time() - self.active["at"]))
            text.append(f" · {self.memory()}", style=MUTED)
            if self.active:
                return Text("▶", style="bold cyan"), text, elapsed
            return Text("✓", style="green"), text, elapsed
        elapsed = short_duration(self.finished - self.started)
        if self.state == "interrupted":
            return (
                Text("■", style="yellow"),
                Text("stopped while loading", style="yellow"),
                elapsed,
            )
        return Text("✗", style="red"), Text(self.reason, style="red"), elapsed


class ProviderDisplay:
    """Live rows in a terminal; one line per change when output is redirected."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self.rows: list[ModelRow] = []
        self.live: Live | None = None
        self.events = deque(maxlen=5)
        self.header = Text()
        self.metadata = Text()
        self.header_shown = False
        self.hint = ""
        self.legend = Text(
            "RAM = worker resident CPU memory · VRAM = worker GPU memory (includes cache)",
            style=MUTED, no_wrap=True, overflow="ellipsis",
        )

    def __rich_console__(self, console, options):
        # Leave spare columns: a terminal echoing ^C at a full-width line would
        # wrap and shift the live region down by one row.
        # Narrow terminals keep the status and drop the weight file names.
        wide = options.max_width >= 120
        narrow = options.max_width < 60
        table = Table.grid(padding=(0, 2, 0, 0))
        table.width = max(options.max_width - 3, 1)
        table.add_column(width=1)
        table.add_column(style="bold", no_wrap=True, overflow="ellipsis",
                         max_width=17 if not narrow else 10)
        if wide:
            table.add_column(
                style="dim", no_wrap=True, overflow="ellipsis", max_width=36
            )
        if not narrow:
            table.add_column(justify="right", min_width=6, style=MUTED, no_wrap=True)
        table.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
        for row in self.rows:
            symbol, text, elapsed = row.cells()
            models = [" + ".join(row.service.models())] if wide else []
            if narrow and elapsed:
                text.append(f" · {elapsed}", style=MUTED)
            table.add_row(symbol, model_name(row.service), *models,
                          *([] if narrow else [elapsed]), text)
        prefix = ([] if self.header_shown else [self.header]) + [self.metadata, self.legend]
        if self.hint:
            prefix.extend([Text("Provider ready", style="bold green"), Text(self.hint)])
        prefix.append(table)
        for row in self.rows:
            detail = row.progress_detail()
            if detail:
                # Give measured work its own full-width line so memory and model
                # names do not push candidate/step counts out of a narrow pane.
                prefix.append(Text(f"{model_name(row.service)} · {detail}", style="cyan"))
        usage = self.rows[0].gpu.device() if self.rows else None
        if usage is not None:
            prefix.append(Text(
                f"Device VRAM {gigabytes(usage[0])} / {gigabytes(usage[1])} · all processes",
                style=MUTED, no_wrap=True, overflow="ellipsis",
            ))
        yield from prefix
        yield Rule(Text("Recent activity · last 5", style=MUTED),
                   style=MUTED, align="left")
        # Fixed row count avoids leaving old frames behind as the history fills.
        # Short panes show the newest rows while retaining all five in memory.
        used = sum(len(console.render_lines(item, options, pad=False)) for item in prefix)
        count = min(5, max(1, options.max_height - used - 1))
        events = list(self.events)[-count:]
        for _ in range(count - len(events)):
            yield Text(" ")
        for at, label, message, style in events:
            stamp = time.strftime("%H:%M:%S", time.localtime(at))
            text = Text(f"{stamp}  ", style=MUTED)
            # Put the action first on a narrow pane so ellipsis keeps it useful.
            text.append(f"{message} · {label}" if narrow else f"{label} · {message}",
                        style=style)
            text.no_wrap, text.overflow = True, "ellipsis"
            text.truncate(max(1, options.max_width - 3), overflow="ellipsis")
            yield text

    def row(self, service) -> ModelRow:
        return next(row for row in self.rows if row.service is service)

    def say(self, text: str) -> None:
        if not self.console.is_terminal:
            self.console.print(Text(text))

    def event(self, service, message, *, at=None, style="") -> None:
        at = time.time() if at is None else at
        entry = (at, model_name(service), single_line(message), style)
        self.events = deque(sorted([*self.events, entry], key=lambda item: item[0])[-5:], maxlen=5)
        self.say(f"{time.strftime('%H:%M:%S', time.localtime(at))}  {entry[1]} · {entry[2]}")

    def update(self, service, status) -> None:
        row = self.row(service)
        pid = status.get("pid")
        if pid != row.event_pid:
            row.event_pid, row.last_event = pid, 0
        row.pid = pid
        row.active = status.get("active")
        row.progress_supported = status.get("progress_supported", False)
        if row.events_supported is None:
            row.events_supported = "events" in status
            if not row.events_supported:
                self.event(service, "job events unavailable in this running worker", style=MUTED)
        for event in status.get("events", []):
            if event["id"] <= row.last_event:
                continue
            row.last_event = event["id"]
            if event["kind"] == "progress":
                message = single_line(event["message"]).removeprefix("Complexa · ")
            else:
                operation = OPERATIONS.get(event["operation"], "job")
                message = f"{operation} #{event['request']} {event['kind']}"
            if "duration" in event:
                message += f" in {short_duration(event['duration'])}"
            if event.get("error"):
                message += f" · {event['error']}"
            self.event(service, message, at=event["at"],
                       style="red" if event["kind"] == "failed" else "")

    def begin(self, services, device: str) -> None:
        self.header_shown = finish_provider_startup(self.console.file)
        self.device = device
        gpu = GpuMemory(device)
        self.rows = [ModelRow(service, gpu) for service in services]
        self.events.clear()
        self.hint = ""
        self.header = Text(TITLE, style=TITLE_STYLE,
                           no_wrap=True, overflow="ellipsis")
        self.metadata = Text(
            f"{device} · models in {services[0].runtime.models_dir}", style=MUTED,
            no_wrap=True, overflow="ellipsis",
        )
        if self.console.is_terminal:
            self.live = track_reflow(Live(self, console=self.console, refresh_per_second=8))
            self.live.start(refresh=True)
        else:
            if not self.header_shown:
                self.console.print(self.header)
            self.console.print(self.metadata)
            self.console.print(self.legend)

    def end_live(self) -> None:
        if self.live is not None:
            self.live.stop()
            self.live = None

    def loading(self, service) -> None:
        self.row(service).begin()
        self.event(service, "loading models")

    def loaded(self, service) -> None:
        row = self.row(service)
        row.end("ready" if service.process is not None else "reused")
        if row.state == "ready":
            self.event(service, f"ready in {short_duration(row.finished - row.started)}")
        else:
            self.event(service, "already loaded by another process")
        status = service.status()
        if status is not None:
            self.update(service, status)

    def interrupted(self, service) -> None:
        self.row(service).end("interrupted")
        self.event(service, "stopped while loading", style="yellow")
        self.end_live()

    def failed(self, service, reason: str) -> None:
        row = self.row(service)
        if row.state == "loading":
            row.end("failed", reason)
        else:
            row.state, row.reason = "failed", reason
        self.event(service, reason, style="red")
        # A live region keeps its final frame, which already shows this reason.
        if self.live is None:
            self.console.print(Text(f"✗ {service.label} {reason}", style="red"))
        self.end_live()
        for line in row.log_tail():
            self.console.print(
                Text("  " + line, style="dim"), no_wrap=True, overflow="ellipsis"
            )
        self.console.print(
            Text(f"  Full log: {service.log}", style="dim"), soft_wrap=True
        )

    def ready(self, services, hint: str) -> None:
        self.hint = hint
        if self.live is not None:
            self.live.refresh()
        else:
            self.console.print(Text("Provider ready", style="bold green"))
            self.console.print(Text(hint))

    @contextmanager
    def stopping(self):
        self.end_live()
        owned = [
            model_name(row.service) for row in self.rows if row.service.process is not None
        ]
        kept = [model_name(row.service) for row in self.rows if row.state == "reused"]
        status = (
            self.console.status(f"Stopping {join(owned)}…")
            if owned and self.console.is_terminal
            else nullcontext()
        )
        with status:
            yield
        parts = []
        if owned:
            parts.append(
                f"{join(owned)} {'worker' if len(owned) == 1 else 'workers'} exited"
            )
        if kept:
            parts.append(
                f"{join(kept)} {'was' if len(kept) == 1 else 'were'} already running "
                "and reused by this provider, so "
                f"{'it remains' if len(kept) == 1 else 'they remain'} running"
            )
        self.console.print(
            Text("Provider stopped." + (f" {'. '.join(parts)}." if parts else ""))
        )
