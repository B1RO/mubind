"""Persistent and standalone external model services (`Complexa` and `ESMFold2`)."""

from __future__ import annotations

_EXPORTS = {
    "complexa_config": (".complexa", "complexa_config"),
    "final_complexa_pdbs": (".complexa", "final_complexa_pdbs"),
    "read_progress": (".complexa", "read_progress"),
    "run_complexa": (".complexa", "run_complexa"),
    "ManagedComplexaService": (".complexa_service", "ManagedComplexaService"),
    "ManagedFoldingService": (".folding_service", "ManagedFoldingService"),
    "WorkerFailed": (".provider", "WorkerFailed"),
    "generate_hint": (".provider", "generate_hint"),
    "run_provider": (".provider", "run_provider"),
    "startup_failure": (".provider", "startup_failure"),
}


def __getattr__(name: str):
    if name in _EXPORTS:
        from importlib import import_module

        module_name, attr = _EXPORTS[name]
        return getattr(import_module(module_name, __name__), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = sorted(_EXPORTS)
