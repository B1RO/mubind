"""Clients for persistent ESMFold2 services; clients never stop the service."""

from __future__ import annotations

import json
import socket
from pathlib import Path
from urllib.request import Request, urlopen

from .config import FoldingConfig


class FoldingClient:
    def __init__(self, config: FoldingConfig, *, training_device="auto"):
        self.config = config
        self.training_device = training_device
        self._protocol = None
        self._managed = None

    def managed_service(self):
        if self._managed is None:
            from ..inference.services import ManagedFoldingService

            self._managed = ManagedFoldingService(
                self.config, training_device=self.training_device
            )
        return self._managed

    def check_ready(self):
        """Check dependencies without starting a process or loading model weights."""
        if self.config.endpoint == "auto":
            self.managed_service().check_ready()
        else:
            self.protocol()

    def start(self, on_status=None):
        if self.config.endpoint == "auto":
            self.managed_service().start(on_status)
        self.protocol()

    def _request(self, operation, job=None):
        endpoint = (
            self.managed_service().endpoint
            if self.config.endpoint == "auto"
            else self.config.endpoint
        )
        if endpoint.startswith("unix://"):
            path = str(Path(endpoint[7:]).expanduser())
            with socket.socket(socket.AF_UNIX) as connection:
                connection.settimeout(
                    min(15, self.config.timeout)
                    if operation == "status"
                    else self.config.timeout
                )
                connection.connect(path)
                payload = {"operation": operation, **(job or {})}
                connection.sendall(json.dumps(payload).encode() + b"\n")
                with connection.makefile("rb") as stream:
                    data = stream.readline(16 * 1024 * 1024)
            if not data:
                raise RuntimeError("Folding service closed without a response")
            result = json.loads(data)
        elif endpoint.startswith(("http://", "https://")):
            url = endpoint.rstrip("/") + (
                "/health" if operation == "status" else "/" + operation
            )
            request = (
                Request(url)
                if operation == "status"
                else Request(
                    url,
                    data=json.dumps(job or {}).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
            )
            with urlopen(
                request,
                timeout=min(15, self.config.timeout)
                if operation == "status"
                else self.config.timeout,
            ) as response:
                result = json.load(response)
        else:
            raise ValueError(
                "Set evaluation.folding.endpoint to an HTTP URL or unix:///socket/path"
            )
        if not isinstance(result, dict) or result.get("ok") is False:
            raise RuntimeError(f"Folding service failed: {result}")
        return result

    def status(self):
        result = self._request("status")
        return result.get("status", result)

    def protocol(self):
        if self._protocol is None:
            if self.config.endpoint == "auto":
                self.managed_service().start()
            health = self.status()
            if not isinstance(health, dict):
                raise RuntimeError("Folding service returned invalid status")
            reported = dict(health.get("protocol", health))
            # The original persistent service uses these shorter field names.
            for alias, name in (
                ("loops", "num_loops"),
                ("sampling_steps", "num_sampling_steps"),
                ("diffusion_samples", "num_diffusion_samples"),
            ):
                if alias in reported:
                    if name in reported and reported[name] != reported[alias]:
                        raise ValueError(f"Conflicting folding protocol field: {name}")
                    reported[name] = reported[alias]
            configured = self.config.protocol
            for key, value in configured.items():
                if key in reported and reported[key] != value:
                    raise ValueError(
                        f"Folding service {key}={reported[key]!r} differs from configured {value!r}"
                    )
            required = (
                "model",
                "esmc_model",
                "num_loops",
                "num_sampling_steps",
                "num_diffusion_samples",
            )
            combined = configured | reported
            if any(key not in combined for key in required):
                raise ValueError("Folding service protocol is incomplete")
            self._protocol = {key: combined[key] for key in required} | {
                "seed": self.config.seed
            }
            if "revision" in combined:
                self._protocol["revision"] = combined["revision"]
        return dict(self._protocol)

    def _service_request(self, operation, job):
        self.protocol()
        return self._request(operation, job)

    def fold_monomer(self, job):
        result = self._service_request("fold_monomer", job)
        metrics = result.get("metrics")
        if not isinstance(metrics, dict):
            raise TypeError("Folding service returned no metrics")
        return metrics

    def check_complex_ready(self):
        """Inspect compatibility without replacing a worker used by another run."""
        self.check_ready()
        if self.config.endpoint == "auto":
            status = self.managed_service().status()
            if status is not None and "fold_complex" not in status.get(
                "operations", []
            ):
                raise RuntimeError(
                    "The running mubind ESMFold2 worker predates complex folding. "
                    "Restart that worker when its current evaluations finish, or "
                    "choose an external complex-folding endpoint."
                )

    def fold_complex(self, job):
        result = self._service_request("fold_complex", job)
        metrics = result.get("metrics")
        if not isinstance(metrics, dict):
            raise TypeError("Folding service returned no complex metrics")
        return metrics

    def close(self):
        """No owned worker to shut down; the persistent model remains loaded."""
