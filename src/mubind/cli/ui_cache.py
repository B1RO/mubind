"""Keep small UI libraries local when the Python environment is on another volume.

The environment remains authoritative. Package installs change site-packages'
directory stamp and select a fresh cache; no models, native libraries, editable
packages or .pth files are copied. This also works for the run-local entry point.
"""

import hashlib
import json
import os
import shutil
import sys
import sysconfig
import tempfile
from pathlib import Path

from .startup import startup_message

UI_PACKAGES = (
    "questionary", "prompt_toolkit", "rich", "pygments", "typer", "click",
    "filelock", "wcwidth", "shellingham", "annotated_doc", "typing_extensions.py",
)


def _stamp(source):
    stat = source.stat()
    return (str(source), stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns)


def use_local_ui():
    """Cache installed UI files once, then import them without remote file walks.

    Failure to populate this optional cache always falls back to the environment.
    MUBIND_UI_CACHE=0 is available for diagnosis and custom dependency development.
    """
    if os.environ.get("MUBIND_UI_CACHE") == "0":
        return
    previous_message = None
    try:
        source = Path(sysconfig.get_path("purelib"))
        if str(source) not in sys.path:
            return
        root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")).expanduser()
        existing = root
        while not existing.exists():
            existing = existing.parent
        # A local environment needs no second copy. In particular, this avoids
        # copying packages back onto the same slow filesystem as the environment.
        signature = _stamp(source)
        if signature[1] == existing.stat().st_dev:
            return
        key = hashlib.sha256(json.dumps(
            [1, sys.implementation.cache_tag, signature, UI_PACKAGES]
        ).encode()).hexdigest()[:24]
        directory = root / "mubind" / "ui" / key
        receipt = directory / "complete.json"
        if not receipt.is_file():
            previous_message = startup_message("Preparing local menu libraries (once per environment update)…")
            directory.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".preparing-", dir=directory.parent) as temporary:
                staged = Path(temporary) / "packages"
                staged.mkdir()
                for name in UI_PACKAGES:
                    path = source / name
                    if path.is_dir():
                        shutil.copytree(path, staged / name)
                    elif path.is_file():
                        shutil.copy2(path, staged / name)
                if _stamp(source) != signature:
                    return  # An installer changed the environment while copying.
                (staged / "complete.json").write_text(json.dumps({"source": signature}))
                try:
                    staged.rename(directory)
                except OSError:
                    if not receipt.is_file():
                        raise
                    # Another launcher published the same complete cache first.
        sys.path.insert(sys.path.index(str(source)), str(directory))
    except (OSError, ValueError):
        pass
    finally:
        if previous_message is not None:
            startup_message(previous_message)
