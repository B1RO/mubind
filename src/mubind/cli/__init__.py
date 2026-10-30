"""The mubind command-line interface and terminal menus."""

from __future__ import annotations


def __getattr__(name: str):
    if name == "app":
        from .app import app

        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
