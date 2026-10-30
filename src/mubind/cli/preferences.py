"""Remember checkpoint choices and target notes for the minibinder generation menu."""

from __future__ import annotations

import json
from pathlib import Path

from filelock import FileLock

from ..paths import existing_artifact
from ..preprocessing.prostt5.steps._io import write_manifest
from ..setup.checkpoint import load_checkpoint_choice as load_checkpoint_choice
from ..setup.config import default_config_path


def save_checkpoint_choice(checkpoint: Path) -> None:
    """Save an absolute path so the choice survives working-directory changes."""
    checkpoint = existing_artifact(checkpoint).resolve()
    if not checkpoint.is_file():
        raise ValueError(f"Checkpoint file not found: {checkpoint}")
    path = default_config_path().with_name("generation.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock", timeout=5):
        write_manifest(path, {"checkpoint": str(checkpoint)})


def target_notes_path() -> Path:
    return default_config_path().with_name("target_notes.json")


def load_target_notes() -> dict[str, str]:
    """Notes keyed by absolute PDB path; a missing or unreadable file means no notes."""
    try:
        notes = json.loads(target_notes_path().read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(notes, dict):
        return {}
    return {str(existing_artifact(Path(key)).resolve()): value for key, value in notes.items() if isinstance(key, str) and isinstance(value, str)}


def target_note(target: Path) -> str:
    return load_target_notes().get(str(Path(target).expanduser().resolve()), "")


def save_target_note(target: Path, note: str) -> None:
    """Store a one-line note beside the settings, never in the PDB; empty removes it."""
    key = str(Path(target).expanduser().resolve())
    note = " ".join(note.split())
    path = target_notes_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock", timeout=5):
        notes = load_target_notes()
        if note:
            notes[key] = note
        else:
            notes.pop(key, None)
        write_manifest(path, notes)
