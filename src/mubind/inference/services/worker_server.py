"""Shared Unix-socket RPC server, request cancellation, and parent-EOF watchdog for isolated workers.

Runs inside the isolated `esmfold2` or `complexa` virtual environments without importing `mubind`.
"""

from __future__ import annotations

import fcntl
import json
import os
import select
import socketserver
import sys
import tempfile
import threading
import time
import traceback
from collections import deque
from contextlib import contextmanager
from pathlib import Path


def watch_owner() -> None:
    """Exit the worker immediately (`os._exit(0)`) if its controller closes stdin."""

    def wait_for_eof():
        # Read the raw descriptor directly so a daemon thread never holds sys.stdin's
        # buffered lock if the worker exits after an exception.
        while os.read(sys.stdin.fileno(), 1):
            pass
        os._exit(0)

    threading.Thread(target=wait_for_eof, daemon=True).start()


def write_output(path: str | Path, text: str) -> None:
    """Atomically write `text` to `path` via a temporary file in the same directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".fold-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class RequestCancelled(Exception):
    """Raised when the client disconnects before its active request completes."""


@contextmanager
def request_cancellation(connection):
    """Monitor `connection` in a background thread and raise `RequestCancelled` on EOF."""
    cancelled, finished = threading.Event(), threading.Event()

    def watch():
        while not finished.is_set():
            try:
                readable, _, _ = select.select([connection], [], [], 0.1)
                if not readable:
                    continue
                # Requests contain one line; EOF or unexpected bytes cancel the job.
                connection.recv(1)
            except OSError:
                pass
            cancelled.set()
            return

    def check():
        if cancelled.is_set():
            raise RequestCancelled("Request cancelled by caller")

    monitor = threading.Thread(target=watch, daemon=True)
    monitor.start()
    try:
        yield check
    finally:
        finished.set()
        monitor.join()


@contextmanager
def cancellable_model_calls(check):
    """Check for client disconnection before every `torch.nn.Module` forward pass."""
    torch = sys.modules.get("torch")
    hook = (
        torch.nn.modules.module.register_module_forward_pre_hook(
            lambda module, inputs: check()
        )
        if torch is not None
        else None
    )
    try:
        check()
        yield
        check()
    finally:
        if hook is not None:
            hook.remove()


class FoldingServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


def serve(
    settings: dict,
    backend_factory,
    *,
    operations: list[str] | None = None,
    name: str = "ESMFold2",
) -> None:
    """Bind the Unix domain socket, initialize `backend_factory(settings)`, and serve JSON-line jobs."""
    if settings.get("owned"):
        watch_owner()
    operations = operations or ["fold_monomer", "fold_complex"]
    socket_path = Path(settings["socket"])
    state = {
        **settings["protocol"],
        "service_id": settings["service_id"],
        "pid": os.getpid(),
        "state": "loading",
        "operations": operations,
        "progress_supported": False,
        "cancel_on_disconnect": True,
    }
    fold_lock = threading.Lock()
    activity_lock = threading.Lock()
    events = deque(maxlen=64)
    event_id = request_id = 0
    active = None
    backend = None
    last_milestone = None
    check_active = lambda: None

    def record(kind, operation, request, **details):
        nonlocal event_id
        event_id += 1
        event = dict(
            id=event_id,
            at=time.time(),
            kind=kind,
            operation=operation,
            request=request,
            **details,
        )
        events.append(event)
        return event

    def report_progress(progress):
        nonlocal active, last_milestone
        check_active()
        with activity_lock:
            if active is None:
                return
            active = {**active, "progress": {**progress, "updated_at": time.time()}}
            phase = progress.get("phase")
            milestone = (active["request"], phase, progress.get("candidate_last"))
            if phase in {"configured", "sampled", "saved"} and milestone != last_milestone:
                record(
                    "progress",
                    active["operation"],
                    active["request"],
                    message=progress["message"],
                )
                last_milestone = milestone

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            nonlocal request_id, active, check_active
            try:
                self.connection.settimeout(30)
                raw = self.rfile.readline(1024 * 1024)
                if not raw.endswith(b"\n"):
                    raise ValueError("Expected one JSON request ending with a newline")
                job = json.loads(raw)
                operation = job.get("operation")
                if operation == "status":
                    with activity_lock:
                        response = {
                            "ok": True,
                            "status": {
                                **state,
                                "events": list(events),
                                "active": active,
                            },
                        }
                elif operation in operations:
                    if state["state"] != "ready":
                        raise RuntimeError(f"{name} is still loading")
                    with request_cancellation(self.connection) as check:
                        while not fold_lock.acquire(timeout=0.1):
                            check()
                        try:
                            check()
                            check_active = check
                            started = time.monotonic()
                            with activity_lock:
                                request_id += 1
                                active = record("started", operation, request_id)
                            try:
                                with cancellable_model_calls(check):
                                    metrics = getattr(backend, operation)(job)
                            except Exception as exc:
                                with activity_lock:
                                    record(
                                        "cancelled"
                                        if isinstance(exc, RequestCancelled)
                                        else "failed",
                                        operation,
                                        request_id,
                                        duration=time.monotonic() - started,
                                        error=f"{type(exc).__name__}: {exc}"[:200],
                                    )
                                raise
                            else:
                                with activity_lock:
                                    record(
                                        "completed",
                                        operation,
                                        request_id,
                                        duration=time.monotonic() - started,
                                    )
                                response = {"ok": True, "metrics": metrics}
                        finally:
                            check_active = lambda: None
                            with activity_lock:
                                active = None
                            fold_lock.release()
                else:
                    raise ValueError(f"Unknown folding operation: {operation!r}")
            except RequestCancelled:
                return
            except Exception as exc:  # noqa: BLE001 - return model errors without killing the service
                traceback.print_exc()
                response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            try:
                self.wfile.write(
                    json.dumps(response, allow_nan=False).encode() + b"\n"
                )
            except (BrokenPipeError, ConnectionResetError):
                pass

    with Path(settings["lock"]).open("a") as lock:
        for attempt in range(20):
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if attempt == 19:
                    raise RuntimeError(
                        f"A {name} worker already owns this socket"
                    ) from None
                time.sleep(0.01)
        socket_path.unlink(missing_ok=True)
        try:
            with FoldingServer(str(socket_path), Handler) as server:
                os.chmod(socket_path, 0o600)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    backend = backend_factory(settings)
                    if hasattr(backend, "set_progress_reporter"):
                        backend.set_progress_reporter(report_progress)
                        state["progress_supported"] = True
                    state["state"] = "ready"
                    print(f"{name} ready", flush=True)
                    thread.join()
                finally:
                    server.shutdown()
                    thread.join()
        finally:
            socket_path.unlink(missing_ok=True)
