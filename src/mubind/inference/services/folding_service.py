"""Start or reuse the persistent ESMFold2 folding worker prepared by `mubind setup`."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path

from ...runtime import get_runtime
from ...setup.models import HF_MODELS

WORKER_SCRIPT = Path(__file__).with_name("esmfold2_worker.py")


class ManagedFoldingService:
    """Unix-socket controller for an isolated model worker (`ESMFold2` by default)."""

    label = "ESMFold2"
    worker_script = WORKER_SCRIPT

    def __init__(self, config, *, training_device: str = "auto") -> None:
        self.runtime = get_runtime("esmfold2", models_dir=config.models_dir)
        self.timeout = config.timeout
        self.protocol = dict(config.protocol)
        for key, expected in (
            ("model", HF_MODELS["model"][0]),
            ("esmc_model", HF_MODELS["esmc_model"][0]),
            ("revision", HF_MODELS["model"][1]),
        ):
            if self.protocol.get(key, expected) != expected:
                raise ValueError(
                    f"Managed ESMFold2 uses {key}={expected!r}. "
                    "Use an explicit folding endpoint for another model."
                )
            self.protocol[key] = expected
        for key in ("num_loops", "num_sampling_steps", "num_diffusion_samples"):
            if type(self.protocol.get(key)) is not int or self.protocol[key] < 1:
                raise ValueError(f"Folding protocol {key} must be a positive integer")
        if self.protocol["num_diffusion_samples"] != 1:
            raise ValueError("Monomer evaluation requires num_diffusion_samples=1")
        device = config.device if config.device != "auto" else training_device
        if device == "auto":
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            device = "cuda:0"
        identity = {
            "version": 1,
            "python": str(self.runtime.python),
            "paths": self.runtime.paths,
            "device": device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "protocol": self.protocol,
        }
        self.configure(identity)

    def configure(self, identity: dict) -> None:
        """Compute the deterministic service key, socket path, and worker settings."""
        self.process = None
        self.key = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()[:20]
        if os.name != "posix":
            raise RuntimeError("Managed model providers require Linux")
        self.directory = self.runtime.models_dir / ".mubind" / "services" / self.key
        self.socket_dir = Path(tempfile.gettempdir()) / f"mubind-{os.getuid()}"
        self.socket_path = self.socket_dir / f"{self.label.lower()}-{self.key}.sock"
        if len(os.fsencode(self.socket_path)) >= 104:
            self.socket_dir = Path("/tmp") / f"mubind-{os.getuid()}"
            self.socket_path = self.socket_dir / f"{self.label.lower()}-{self.key}.sock"
        self.endpoint = "unix://" + str(self.socket_path)
        self.log = self.directory / "worker.log"
        self.settings = {
            **identity,
            "service_id": self.key,
            "socket": str(self.socket_path),
            "lock": str(self.directory / "worker.lock"),
        }

    def status(self) -> dict | None:
        """Query the worker over its Unix socket, returning `None` if it is not running."""
        try:
            with socket.socket(socket.AF_UNIX) as connection:
                connection.settimeout(min(5, self.timeout))
                connection.connect(str(self.socket_path))
                connection.sendall(b'{"operation":"status"}\n')
                with connection.makefile("rb") as stream:
                    raw = stream.readline(1024 * 1024)
        except (
            FileNotFoundError,
            ConnectionRefusedError,
            ConnectionResetError,
            BrokenPipeError,
        ):
            return None
        if not raw:
            return None
        response = json.loads(raw)
        if not response.get("ok"):
            raise RuntimeError(f"{self.label} status failed: {response}")
        status = response["status"]
        if status.get("service_id") != self.key:
            raise RuntimeError(f"Unexpected service at {self.socket_path}")
        for key, value in self.protocol.items():
            if status.get(key) != value:
                raise ValueError(f"{self.label} service protocol differs: {key}")
        if status.get("state") not in {"loading", "ready"}:
            raise RuntimeError(f"{self.label} is unavailable. Details: {self.log}")
        if not status.get("cancel_on_disconnect"):
            raise RuntimeError(
                f"Restart the {self.label} provider to enable Ctrl+C cancellation."
            )
        return status

    def models(self) -> list[str]:
        """Return short model names loaded by this worker for UI status displays."""
        return [
            self.protocol[key].rsplit("/", 1)[-1] for key in ("model", "esmc_model")
        ]

    def check_ready(self) -> None:
        """Inspect the installed runtime and any live worker without spawning a new one."""
        self.status()

    def start(self, on_status=None, *, owned: bool = False) -> None:
        """Reuse a running worker or launch one and wait until its weights are loaded."""
        previous = self.process
        try:
            self._start(on_status, owned=owned)
        except KeyboardInterrupt:
            if self.process is not previous:
                self.close()
            raise
        finally:
            if not owned:
                self.process = previous

    def close(self) -> None:
        """Stop a worker process only when it was started with `owned=True` by this instance."""
        process, self.process = self.process, None
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def _worker_active(self) -> bool:
        from filelock import FileLock, Timeout

        try:
            with FileLock(self.directory / "worker.lock", timeout=0):
                return False
        except Timeout:
            return True

    def _failure(self) -> RuntimeError:
        tail = ""
        if self.log.exists():
            with self.log.open("rb") as stream:
                stream.seek(max(0, self.log.stat().st_size - 4000))
                tail = stream.read().decode(errors="replace").strip()
        return RuntimeError(
            f"{self.label} could not start. Details: {self.log}\n{tail}"
        )

    def _start(self, on_status=None, *, owned: bool = False) -> None:
        notify = on_status or (lambda message: None)
        status = self.status()
        if status is not None and status["state"] == "ready":
            notify(f"Reusing {self.label}…")
            return
        notify(f"Starting {self.label}…")
        deadline = time.monotonic() + self.timeout
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.socket_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        if (
            self.socket_dir.is_symlink()
            or self.socket_dir.stat().st_uid != os.getuid()
            or self.socket_dir.stat().st_mode & 0o077
        ):
            raise RuntimeError(
                f"{self.label} socket directory must be private: {self.socket_dir}"
            )
        process = None
        from filelock import FileLock

        with FileLock(self.directory / "start.lock", timeout=self.timeout):
            if self.status() is None and not self._worker_active():
                settings = self.directory / "worker.json"
                settings.write_text(
                    json.dumps({**self.settings, "owned": owned}, indent=2) + "\n"
                )
                with self.log.open("ab") as stream:
                    process = subprocess.Popen(
                        [
                            str(self.runtime.python),
                            "-u",
                            str(self.worker_script),
                            str(settings),
                        ],
                        stdin=subprocess.PIPE if owned else subprocess.DEVNULL,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        cwd=self.runtime.models_dir,
                        env=self.runtime.environment(),
                        start_new_session=True,
                    )
                self.process = process
                while not self._worker_active():
                    if process.poll() is not None:
                        raise self._failure()
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"{self.label} startup timed out. Details: {self.log}"
                        )
                    time.sleep(0.1)
        notify(f"Loading {self.label} weights…")
        while time.monotonic() < deadline:
            if process is not None and process.poll() is not None:
                raise self._failure()
            status = self.status()
            if status is not None and status["state"] == "ready":
                notify(f"{self.label} ready")
                return
            if process is None and status is None and not self._worker_active():
                raise self._failure()
            time.sleep(0.2)
        raise TimeoutError(f"{self.label} is still starting. Details: {self.log}")
