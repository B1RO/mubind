"""Foreground owner (`mubind provider`) for warm Complexa and ESMFold2 GPU workers."""

from __future__ import annotations

import os
import shlex
import signal
import time

from ...cli.provider_display import exit_reason
from ...evaluation.config import FoldingConfig
from ...runtime import get_runtime
from .complexa_service import ManagedComplexaService
from .folding_service import ManagedFoldingService


class WorkerFailed(RuntimeError):
    """A managed worker failed after the provider display reported the failure."""


def startup_failure(service, error: Exception) -> str:
    """Format a concise one-line startup failure message for `ProviderDisplay`."""
    process = service.process
    if process is not None and process.poll() is not None:
        return exit_reason(process.returncode)
    if isinstance(error, TimeoutError):
        return f"not ready after {service.timeout:g} s; --timeout allows longer"
    return str(error).splitlines()[0].split(" Details: ")[0].rstrip(".")


def generate_hint(models_dir, device: str) -> str:
    """Return the `mubind generate` command string that connects to these workers."""
    command = ["mubind", "generate"]
    if models_dir is not None:
        command += ["--models-dir", str(models_dir)]
    if device != "cuda:0":
        command += ["--device", device]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    prefix = f"CUDA_VISIBLE_DEVICES={shlex.quote(visible)} " if visible else ""
    return (
        f"Run {prefix}{shlex.join(command)} in another terminal to reuse these "
        "models. Ctrl+C stops the provider."
    )


def run_provider(
    *, models_dir=None, device: str = "auto", timeout: float = 1800, display=None
) -> None:
    """Start and monitor owned Complexa and ESMFold2 workers until interrupted."""
    import torch

    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    selected = torch.device(device)
    if selected.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("The folding and Complexa provider requires a CUDA device")
    if selected.index is not None and selected.index >= torch.cuda.device_count():
        raise ValueError(f"Requested device is unavailable: {device}")
    device = f"cuda:{selected.index or 0}"

    services = [
        ManagedComplexaService(
            get_runtime("complexa", models_dir=models_dir), device, timeout=timeout
        ),
        ManagedFoldingService(
            FoldingConfig(models_dir=models_dir, device=device, timeout=timeout)
        ),
    ]
    if display is None:
        from ...cli.provider_display import ProviderDisplay

        display = ProviderDisplay()

    def stop(signum, frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, stop)
    try:
        display.begin(services, device)
        for service in services:
            display.loading(service)
            try:
                service.start(owned=True)
            except KeyboardInterrupt:
                display.interrupted(service)
                raise
            except Exception as exc:
                display.failed(service, startup_failure(service, exc))
                raise WorkerFailed(f"{service.label} could not start") from exc
            display.loaded(service)
        display.ready(services, generate_hint(models_dir, device))
        while True:
            time.sleep(1)
            for service in services:
                if service.process is not None and service.process.poll() is not None:
                    reason = exit_reason(service.process.returncode)
                else:
                    status = service.status()
                    if status is not None:
                        display.update(service, status)
                        continue
                    reason = "worker stopped responding"
                display.failed(service, reason)
                raise WorkerFailed(f"{service.label} {reason}")
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGTERM, previous)
        with display.stopping():
            for service in reversed(services):
                service.close()
