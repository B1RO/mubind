"""Make packaged reference inputs available in the working artifacts directory."""

from functools import lru_cache
from pathlib import Path
import shutil

from filelock import FileLock

from .paths import ARTIFACTS

GROUPS = ("benchmarks", "validation", "examples")


def bundled_artifacts() -> Path:
    package = Path(__file__).resolve().parent
    checkout = package.parent.parent
    if package == checkout / "src/mubind" and (checkout / "pyproject.toml").is_file():
        return checkout / "artifacts"
    return package / "artifacts"


@lru_cache(maxsize=16)
def _materialize(source: Path, destination: Path) -> Path:
    if source.resolve() == destination.resolve():
        return destination
    if not source.is_dir():
        raise FileNotFoundError(f"Bundled mubind inputs are missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination) + ".lock", timeout=30):
        for path in source.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            target = destination / path.relative_to(source)
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                with target.open("xb") as output, path.open("rb") as contents:
                    shutil.copyfileobj(contents, output)
            except FileExistsError:
                pass
    return destination


def resource_directory(group: str) -> Path:
    if group not in GROUPS:
        raise ValueError(f"Unknown input group: {group}")
    return _materialize(bundled_artifacts() / group, (ARTIFACTS / group).absolute())
