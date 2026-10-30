"""Discover a provider's Complexa worker and submit serialized scaffold generation jobs."""

from __future__ import annotations

import json
import os
import select
import socket
import time
from pathlib import Path

from .folding_service import ManagedFoldingService


class ManagedComplexaService(ManagedFoldingService):
    """Unix-socket controller for the isolated Complexa scaffold generation worker."""

    label = "Complexa"
    worker_script = Path(__file__).with_name("complexa_worker.py")

    def __init__(self, runtime, device, *, timeout: float = 1800) -> None:
        self.runtime = runtime
        self.timeout = timeout
        self.protocol = {}
        device = "cuda:0" if str(device) == "cuda" else str(device)
        self.configure(
            {
                "version": 1,
                "python": str(runtime.python),
                "paths": runtime.paths,
                "device": device,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "protocol": {},
                "component": "complexa",
            }
        )

    def models(self) -> list[str]:
        paths = self.runtime.paths
        return [
            Path(paths[key]).name
            for key in ("checkpoint", "autoencoder")
            if key in paths
        ]

    def generate(
        self, config_path: Path, progress_path: Path, log_path: Path, on_status, on_progress
    ) -> None:
        """Submit a Complexa job to the warm worker socket and stream progress events."""
        from .complexa import read_progress

        deadline = time.monotonic() + self.timeout
        while True:
            status = self.status()
            if status is None:
                raise RuntimeError(
                    "The Complexa provider stopped before the job started"
                )
            if status["state"] == "ready":
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Complexa is still loading. Details: {self.log}")
            time.sleep(0.2)
        on_status("Reusing Complexa provider…")
        payload = {
            "operation": "generate",
            "config": str(config_path.resolve()),
            "progress": str(progress_path.resolve()),
            "log": str(log_path.resolve()),
        }
        with (
            socket.socket(socket.AF_UNIX) as connection,
            progress_path.open() as events,
        ):
            connection.settimeout(5)
            connection.connect(str(self.socket_path))
            connection.sendall(json.dumps(payload).encode() + b"\n")
            response = bytearray()
            while b"\n" not in response:
                read_progress(events, on_status, on_progress)
                if not select.select([connection], [], [], 0.5)[0]:
                    continue
                block = connection.recv(65536)
                if not block:
                    raise RuntimeError(
                        f"Complexa provider disconnected. Details: {self.log}"
                    )
                response.extend(block)
                if len(response) > 1024 * 1024:
                    raise RuntimeError("Invalid Complexa provider response")
            read_progress(events, on_status, on_progress)
        result = json.loads(response)
        if not result.get("ok"):
            raise RuntimeError(
                f"Complexa failed: {result.get('error')}. Details: {log_path}"
            )
