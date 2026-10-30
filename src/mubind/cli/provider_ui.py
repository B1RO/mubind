"""Inspect reusable workers without starting them from a menu redraw."""

import questionary
from rich.text import Text

from .terminal_ui import screen_intro, select_prompt


def worker_status(models_dir=None, device="auto"):
    from ..evaluation.config import FoldingConfig
    from ..inference.services import ManagedComplexaService, ManagedFoldingService
    from ..runtime import get_runtime

    if device == "auto":
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    factories = [
        ("Complexa", lambda: ManagedComplexaService(get_runtime("complexa", models_dir=models_dir), device, timeout=0.2)),
        ("ESMFold2", lambda: ManagedFoldingService(FoldingConfig(models_dir=models_dir, device=device, timeout=0.2))),
    ]
    states = []
    for label, factory in factories:
        try:
            status = factory().status()
            state = "warm" if status and status["state"] == "ready" else "loading" if status else "not running"
        except (ValueError, OSError, RuntimeError, KeyError):
            state = "unavailable"
        states.append(f"{label} {state}")
    return " · ".join(states)


def provider_screen(console, style=None, *, models_dir=None, device="auto"):
    from ..inference.services import run_provider
    from .provider_display import ProviderDisplay

    while True:
        purpose = "Keep scaffold and folding models loaded for repeated designs. Start keeps this terminal occupied; Ctrl+C stops the provider."
        choice = select_prompt(
            "", choices=[
                questionary.Choice("Start provider in this terminal", "start"),
                questionary.Choice("Refresh worker status", "refresh"),
                questionary.Choice("Back", "back"),
            ], default="back", style=style, qmark="›", pointer="❯",
            instruction=screen_intro(console, "mubind · Model workers", purpose) + "\n  " + worker_status(models_dir, device),
        ).ask()
        if choice in (None, "back"):
            return
        if choice == "start":
            try:
                run_provider(models_dir=models_dir, device=device, display=ProviderDisplay(console))
            except (ValueError, OSError, RuntimeError) as exc:
                console.print(Text(str(exc), style="red"))
