"""Small bundled inputs for a first run."""

import json
from pathlib import Path
from .resources import resource_directory


def ensure_example_target(directory: Path) -> Path:
    """Make bundled targets available without replacing user files."""
    directory = directory.expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    bundled = resource_directory("examples") / "targets"
    for source in sorted(bundled.iterdir(), key=lambda item: item.name):
        if not source.name.endswith(".pdb"):
            continue
        target = directory / source.name
        if target.exists():
            continue
        contents = source.read_bytes()
        try:
            with target.open("xb") as stream:
                stream.write(contents)
        except FileExistsError:
            pass
    return directory / "ubiquitin_1UBQ.pdb"


def example_target_options(path: Path) -> dict:
    """Apply example settings only to an unchanged bundled structure."""
    directory = resource_directory("examples") / "targets"
    manifest = json.loads(directory.joinpath("targets.json").read_text(encoding="utf-8"))
    options = manifest["targets"].get(path.name)
    if options is None:
        return {}
    bundled = directory.joinpath(path.name).read_bytes()
    if path.read_bytes() != bundled:
        return {}
    return {
        "target_spec": options["target_spec"],
        "hotspots": tuple(options["hotspots"]),
        "binder_length": options["binder_length"],
    }
