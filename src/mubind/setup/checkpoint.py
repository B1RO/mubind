"""The released mubind checkpoint, without model imports or GPU initialization."""

from __future__ import annotations

import json
from pathlib import Path

from filelock import FileLock, Timeout

from ..paths import CHECKPOINTS, existing_artifact
from ..preprocessing.prostt5.steps._io import write_manifest
from ..common.progress import ProgressCallback, ProgressEvent
from .config import default_config_path, load_config
from .models import Checkpoint, fingerprints, verify_checkpoint

MUBIND_REPOSITORY = "b1ro/mubind"
MUBIND_REPO_TYPE = "space"
# Full Space commit so later demo updates do not change installed weights.
MUBIND_REVISION = "ac1c3d1d916d1e50277143cda141d9372a1641fc"
MUBIND_FILE = Checkpoint(
    "prostt5-300k.pt",
    62375142,
    "1af8427b2832e3e67bbefe90a02cc06a00d2bd43ea421aad3d1626b39781e2ea",
)


def checkpoint_path(models_dir: Path) -> Path:
    return existing_artifact(models_dir).resolve() / "mubind" / MUBIND_FILE.name


def load_checkpoint_choice() -> Path | None:
    """Read generation preferences without importing the inference package."""
    try:
        settings = json.loads(default_config_path().with_name("generation.json").read_text())
        value = settings.get("checkpoint") if isinstance(settings, dict) else None
        if isinstance(value, str) and value:
            path = existing_artifact(Path(value))
            if path.is_absolute() and path.is_file():
                return path
    except (OSError, ValueError):
        pass
    return None


def default_checkpoint(models_dir: Path | None = None) -> Path | None:
    """Respect the saved choice before looking for available 300k checkpoints."""
    remembered = load_checkpoint_choice()
    if remembered is not None:
        return remembered
    root = existing_artifact(models_dir or load_config().models_dir)
    candidates = (
        CHECKPOINTS / MUBIND_FILE.name,
        checkpoint_path(root),
        CHECKPOINTS / "checkpoint_step300000.pt",
        root / "mubind" / "checkpoint_step300000.pt",
    )
    return next((path.resolve() for path in candidates if path.is_file()), None)


def checkpoint_identity() -> dict:
    return {
        "repository": MUBIND_REPOSITORY,
        "repo_type": MUBIND_REPO_TYPE,
        "revision": MUBIND_REVISION,
        "filename": MUBIND_FILE.name,
        "bytes": MUBIND_FILE.size,
        "sha256": MUBIND_FILE.sha256,
    }


def checkpoint_ready(models_dir: Path) -> bool:
    """Read the verified-file marker only; menu redraws never hash or download."""
    root = models_dir.expanduser().resolve()
    target = checkpoint_path(root)
    try:
        report = json.loads((root / ".mubind/mubind.json").read_text())
        return (
            report["state"] == "complete"
            and report["identity"] == checkpoint_identity()
            and target.is_file()
            and target.stat().st_size == MUBIND_FILE.size
            and report["files"] == fingerprints([target])
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def setup_checkpoint(models_dir: Path, emit: ProgressCallback) -> dict[str, str]:
    """Verify existing weights or download to staging before publishing the file."""
    root = models_dir.expanduser().resolve()
    managed = root / ".mubind"
    if managed.is_symlink():
        raise ValueError(f"Setup directory must not be a symlink: {managed}")
    managed.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(managed / "mubind.lock", timeout=0):
            target = checkpoint_path(root)
            if target.is_symlink() or target.parent.is_symlink():
                raise ValueError(f"Checkpoint destination must not be a symlink: {target}")
            if target.exists():
                # A user-supplied file is never replaced by a download.
                verify_checkpoint(target, MUBIND_FILE, emit)
                emit(ProgressEvent("reuse", message="mubind checkpoint verified"))
            else:
                from huggingface_hub import hf_hub_download
                from huggingface_hub.utils import HfHubHTTPError

                staging = managed / "downloads" / "mubind" / MUBIND_REVISION
                options = {
                    "repo_id": MUBIND_REPOSITORY,
                    "repo_type": MUBIND_REPO_TYPE,
                    "revision": MUBIND_REVISION,
                    "filename": MUBIND_FILE.name,
                    "local_dir": staging,
                    # This release is public; saved or expired credentials must
                    # not affect installation or hide a visibility regression.
                    "token": False,
                }
                emit(ProgressEvent("download", message="mubind checkpoint (62.38 MB)"))
                try:
                    downloaded = Path(hf_hub_download(**options))
                    try:
                        verify_checkpoint(downloaded, MUBIND_FILE, emit)
                    except ValueError:
                        downloaded = Path(hf_hub_download(**options, force_download=True))
                        verify_checkpoint(downloaded, MUBIND_FILE, emit)
                except HfHubHTTPError as exc:
                    status = getattr(exc.response, "status_code", None)
                    if status in (401, 403, 404):
                        raise RuntimeError(
                            f"Cannot download the public mubind checkpoint (HTTP {status}). "
                            f"Expected {MUBIND_FILE.name} in the {MUBIND_REPOSITORY} Space "
                            f"at revision {MUBIND_REVISION}. The publisher must keep this "
                            "release publicly accessible; users should not need to log in."
                        ) from exc
                    raise
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() or target.is_symlink():
                    raise FileExistsError(f"Checkpoint appeared during download; keeping it: {target}")
                downloaded.replace(target)
            paths = {"checkpoint": str(target)}
            write_manifest(managed / "mubind.json", {
                "state": "complete",
                "identity": checkpoint_identity(),
                "paths": paths,
                "files": fingerprints([target]),
            })
            return paths
    except Timeout as exc:
        raise RuntimeError("Another setup is already preparing the mubind checkpoint.") from exc
