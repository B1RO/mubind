"""Install and locate command-line tools used by evaluation."""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

from filelock import FileLock

from ..preprocessing.prostt5.steps._io import write_manifest
from ..common.progress import ProgressCallback, ProgressEvent
from .config import load_config

FOLDSEEK_VERSION = "10-941cd33"
FOLDSEEK_RELEASE = (
    f"https://github.com/steineggerlab/foldseek/releases/download/{FOLDSEEK_VERSION}"
)
FOLDSEEK_ASSETS = {
    "linux-avx2": (
        11941946,
        "e6a3e3860e76b0b7b858736f627fe680d4d7f0fc7298301d8a92b9f711d74ac7",
    ),
    "linux-arm64": (
        10973583,
        "c71907a7ea06cb6e93d8eb000b0c3689e5e25f5edeb3969e2553cdb8f42982c1",
    ),
    "osx-universal": (
        17021264,
        "6d8b07e188d443044f0c98db30aa3d4cca6750e97b50679372b54b45a664cae5",
    ),
}


def foldseek_executable(root: Path) -> Path:
    return root / ".mubind" / "tools" / "foldseek" / "bin" / "foldseek"


def find_foldseek(
    value: str | None = "foldseek", *, models_dir: Path | None = None
) -> str | None:
    """Honor explicit overrides; otherwise prefer setup's copy, then PATH."""
    if value and value != "foldseek":
        return shutil.which(os.path.expanduser(value))
    root = models_dir if models_dir is not None else load_config().models_dir
    managed = shutil.which(str(foldseek_executable(root.expanduser().resolve())))
    return managed or shutil.which("foldseek")


def _version(executable: str | Path) -> str:
    result = subprocess.run(
        [str(executable), "version"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    version = result.stdout.strip()
    if not version:
        raise ValueError(f"Foldseek did not report a version: {executable}")
    return version


def foldseek_ready(root: Path) -> bool:
    executable = find_foldseek(models_dir=root)
    if executable is None:
        return False
    try:
        _version(executable)
        return True
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


def _asset() -> str:
    system, architecture = platform.system(), platform.machine().lower()
    if system == "Darwin":
        return "osx-universal"
    if system == "Linux":
        if architecture in ("aarch64", "arm64"):
            return "linux-arm64"
        if architecture in ("x86_64", "amd64"):
            if "avx2" not in Path("/proc/cpuinfo").read_text().split():
                raise RuntimeError("This Foldseek build needs an AVX2-capable CPU.")
            return "linux-avx2"
    raise RuntimeError("Automatic Foldseek installation supports Linux and macOS.")


def ensure_foldseek(root: Path, emit: ProgressCallback) -> Path:
    """Download a pinned release, check its bytes, and verify the executable."""
    import httpx

    executable = find_foldseek(models_dir=root)
    if executable is not None:
        try:
            version = _version(executable)
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        else:
            emit(ProgressEvent("reuse", message=f"Foldseek {version}"))
            return Path(executable)

    asset = _asset()
    size, checksum = FOLDSEEK_ASSETS[asset]
    target = foldseek_executable(root)
    directory = target.parent.parent
    directory.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(directory.parent / "foldseek.lock", timeout=0):
        marker = directory / "install.json"
        if directory.is_symlink() or (directory.exists() and not marker.is_file()):
            raise FileExistsError(f"Unrecognised Foldseek directory: {directory}")
        directory.mkdir(exist_ok=True)
        report = {
            "state": "installing",
            "version": FOLDSEEK_VERSION,
            "sha256": checksum,
        }
        write_manifest(marker, report)
        try:
            with tempfile.TemporaryDirectory(
                prefix=".foldseek-", dir=directory.parent
            ) as temporary:
                temporary = Path(temporary)
                archive_path = temporary / "foldseek.tar.gz"
                digest, completed = hashlib.sha256(), 0
                with httpx.Client(follow_redirects=True, timeout=60) as client:  # noqa: SIM117
                    with client.stream(
                        "GET", f"{FOLDSEEK_RELEASE}/foldseek-{asset}.tar.gz"
                    ) as response:
                        response.raise_for_status()
                        with archive_path.open("wb") as stream:
                            for block in response.iter_bytes(1024 * 1024):
                                completed += len(block)
                                if completed > size:
                                    raise ValueError(
                                        "The Foldseek download exceeds its expected size."
                                    )
                                digest.update(block)
                                stream.write(block)
                                emit(
                                    ProgressEvent(
                                        "download",
                                        completed,
                                        size,
                                        unit="bytes",
                                        message="Foldseek",
                                    )
                                )
                if completed != size or digest.hexdigest() != checksum:
                    raise ValueError(
                        "The Foldseek download failed checksum verification."
                    )
                binary = temporary / "foldseek"
                with tarfile.open(archive_path) as archive:
                    # Extract only the known regular files; never follow archive links.
                    for name, destination in (
                        ("foldseek/bin/foldseek", binary),
                        ("foldseek/README.md", directory / "README.md"),
                    ):
                        member = archive.getmember(name)
                        if not member.isfile():
                            raise ValueError(
                                f"Unexpected Foldseek archive member: {name}"
                            )
                        with (
                            archive.extractfile(member) as source,
                            destination.open("wb") as output,
                        ):
                            shutil.copyfileobj(source, output)
                binary.chmod(0o755)
                emit(ProgressEvent("verify", message="Checking Foldseek"))
                report["detected_version"] = _version(binary)
                target.parent.mkdir(exist_ok=True)
                binary.replace(target)
            report["state"] = "complete"
            write_manifest(marker, report)
        except BaseException as exc:
            report.update(state="failed", error=str(exc))
            write_manifest(marker, report)
            raise
    return target
