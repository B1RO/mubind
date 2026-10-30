"""Versioned model assets. Downloading never loads a model onto a GPU."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from ..preprocessing.prostt5.steps._io import write_manifest
from ..common.progress import ProgressCallback, ProgressEvent

NGC_BASE = "https://api.ngc.nvidia.com/v2/models/nvidia/clara/proteina_complexa/versions/1.0/files"


@dataclass(frozen=True)
class Checkpoint:
    name: str
    size: int
    sha256: str


COMPLEXA_FILES = (
    Checkpoint(
        "complexa.ckpt",
        2934289381,
        base64.b64decode("WJ2xdB8pg4x5YThva4cwhyOMcmguVhibieCuAmEMGek=").hex(),
    ),
    Checkpoint(
        "complexa_ae.ckpt",
        4100101779,
        base64.b64decode("NfiGXv0mmZXurxZw4cEIWs/imIxAq97ajgmg4V60CBY=").hex(),
    ),
)
HF_MODELS = {
    # Pre-bundling releases matching the original worker's Transformers API.
    "model": ("biohub/ESMFold2-Fast", "c6c7958d63f5f2f1f0fed0bb9462316f8ccceea6"),
    "esmc_model": ("biohub/ESMC-6B", "45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a"),
}
# The input builder needs CCD data from the full model's repository, even when
# using Fast weights. Pin the May 2026 snapshot independently of those weights.
CCD_REPOSITORY = "biohub/ESMFold2"
CCD_REVISION = "884915755baba1ac8aa6485f8ce3e5f7680a64a7"
CCD_FILE = Checkpoint(
    "ccd.pkl",
    417306584,
    "9ff44b1927c6b9198e38ffe0928706827a09a350c15530beeeabebfa88038fc5",
)


def verify_checkpoint(path: Path, asset: Checkpoint, emit: ProgressCallback) -> None:
    if path.stat().st_size != asset.size:
        raise ValueError(f"Wrong size for {path}; expected {asset.size:,} bytes.")
    digest = hashlib.sha256()
    completed = 0
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
            completed += len(block)
            emit(
                ProgressEvent(
                    "verify", completed, asset.size, unit="bytes", message=asset.name
                )
            )
    if digest.hexdigest() != asset.sha256:
        raise ValueError(f"Checksum mismatch: {path}")


def download_checkpoint(
    directory: Path, asset: Checkpoint, emit: ProgressCallback
) -> Path:
    """Resume a managed partial file and publish only verified bytes.

    A conflicting final file is never overwritten. Range responses are checked
    before appending, and servers that ignore Range cause a clean restart.
    """
    import httpx

    directory.mkdir(parents=True, exist_ok=True)
    target = directory / asset.name
    if target.exists():
        verify_checkpoint(target, asset, emit)
        return target
    partial = directory / (asset.name + ".mubind-part")
    if target.is_symlink() or partial.is_symlink():
        raise ValueError(f"Download destination must not be a dangling link: {target}")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > asset.size:
        partial.unlink()
        offset = 0
    if offset < asset.size:
        headers = {"Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        with httpx.Client(follow_redirects=True, timeout=60) as client:  # noqa: SIM117
            with client.stream(
                "GET", f"{NGC_BASE}/{asset.name}", headers=headers
            ) as response:
                response.raise_for_status()
                if response.status_code == 206:
                    value = response.headers.get("Content-Range", "")
                    if not value.startswith(f"bytes {offset}-") or not value.endswith(
                        f"/{asset.size}"
                    ):
                        raise ValueError(
                            f"Unexpected download range for {asset.name}: {value}"
                        )
                elif response.status_code == 200:
                    offset = 0
                else:
                    raise ValueError(
                        f"Unexpected download status: {response.status_code}"
                    )
                with partial.open("ab" if offset else "wb") as stream:
                    for block in response.iter_bytes(1024 * 1024):
                        if offset + len(block) > asset.size:
                            raise ValueError(
                                f"Download exceeds expected size: {asset.name}"
                            )
                        stream.write(block)
                        offset += len(block)
                        emit(
                            ProgressEvent(
                                "download",
                                offset,
                                asset.size,
                                unit="bytes",
                                message=asset.name,
                            )
                        )
    try:
        verify_checkpoint(partial, asset, emit)
    except ValueError:
        partial.unlink(missing_ok=True)
        raise
    partial.rename(target)
    return target


def snapshot_files(path: Path, *, tokenizer: bool = False) -> list[Path]:
    """Check every sharded weight referenced by the downloaded model index."""
    index = path / "model.safetensors.index.json"
    if index.exists():
        names = set(json.loads(index.read_text())["weight_map"].values())
        if not names:
            raise ValueError(f"Model index contains no weights: {path}")
        names.add(index.name)
    else:
        names = {"model.safetensors"}
    names.add("config.json")
    if tokenizer:
        names.update(
            ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
        )
    files = []
    for name in sorted(names):
        file = path / name
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"Invalid model filename: {name}")
        if not file.is_file() or not file.stat().st_size:
            raise ValueError(f"Missing or empty model file: {file}")
        files.append(file)
    return files


def weights_available(component: str, root: Path) -> bool:
    """Check local download progress without fetching or loading weights."""
    try:
        if component == "complexa":
            return all(
                (root / "Proteina-Complexa" / asset.name).stat().st_size == asset.size
                for asset in COMPLEXA_FILES
            )
        from huggingface_hub import try_to_load_from_cache

        for key, (repository, revision) in HF_MODELS.items():
            config = try_to_load_from_cache(
                repository,
                "config.json",
                revision=revision,
                cache_dir=root / "huggingface" / "hub",
            )
            if not isinstance(config, str):
                return False
            snapshot_files(Path(config).parent, tokenizer=key == "esmc_model")
        return True
    except (ImportError, OSError, ValueError, KeyError, TypeError):
        return False


def local_fold_bundle(root: Path, fold: Path, esmc: Path) -> Path:
    """Give the old loader an explicit local backbone without changing HF files.

    Its from_pretrained() loads config.esmc_id internally. A small local config
    and links to immutable weight files remove that implicit request for main.
    """
    identity = {"fold": str(fold), "esmc": str(esmc)}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    directory = root / ".mubind" / "bundles" / f"esmfold2-{key}"
    marker = directory / "bundle.json"
    if directory.exists():
        if not marker.is_file() or json.loads(marker.read_text()) != identity:
            raise FileExistsError(f"Unrecognised local model directory: {directory}")
    else:
        directory.mkdir(parents=True)
        write_manifest(marker, identity)
    config = json.loads((fold / "config.json").read_text())
    config["esmc_id"] = str(esmc)
    write_manifest(directory / "config.json", config)
    for file in snapshot_files(fold):
        if file.name == "config.json":
            continue
        link = directory / file.name
        if link.is_symlink() and link.resolve() == file.resolve():
            continue
        if link.exists() or link.is_symlink():
            raise FileExistsError(f"Conflicting model file: {link}")
        link.symlink_to(file)
    return directory


def download_ccd(root: Path, emit: ProgressCallback) -> Path:
    """Fetch and verify the input builder's data before any offline inference."""
    from inspect import signature

    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import LocalEntryNotFoundError
    from tqdm.auto import tqdm

    class DownloadProgress(tqdm):
        def display(self, msg=None, pos=None):
            emit(
                ProgressEvent(
                    "download",
                    int(self.n),
                    self.total,
                    unit="bytes",
                    message="ESMFold2 CCD data",
                )
            )

    options = {
        "repo_id": CCD_REPOSITORY,
        "revision": CCD_REVISION,
        "filename": CCD_FILE.name,
        "cache_dir": root / "huggingface" / "hub",
    }
    # Older supported Hub clients expose this only on snapshot_download().
    if "tqdm_class" in signature(hf_hub_download).parameters:
        options["tqdm_class"] = DownloadProgress
    force = False
    try:
        path = Path(hf_hub_download(**options, local_files_only=True))
        verify_checkpoint(path, CCD_FILE, emit)
        return path
    except LocalEntryNotFoundError:
        pass
    except (OSError, ValueError):
        force = True  # Repair a corrupt cached copy rather than reusing it.
    emit(
        ProgressEvent(
            "download", 0, CCD_FILE.size, unit="bytes", message="ESMFold2 CCD data"
        )
    )
    path = Path(hf_hub_download(**options, force_download=force))
    verify_checkpoint(path, CCD_FILE, emit)
    return path


def download_models(
    component: str, root: Path, emit: ProgressCallback
) -> dict[str, str]:
    if component == "complexa":
        directory = root / "Proteina-Complexa"
        paths = [
            download_checkpoint(directory, asset, emit) for asset in COMPLEXA_FILES
        ]
        return {"checkpoint": str(paths[0]), "autoencoder": str(paths[1])}

    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import LocalEntryNotFoundError
    from tqdm.auto import tqdm

    class DownloadProgress(tqdm):
        def display(self, msg=None, pos=None):
            if getattr(self, "unit", None) != "B":
                emit(
                    ProgressEvent(
                        "download",
                        int(self.n),
                        self.total,
                        unit="files",
                        message=repository,
                    )
                )

    paths = {}
    for key, (repository, revision) in HF_MODELS.items():
        emit(ProgressEvent("download", unit="files", message=repository))
        options = {
            "repo_id": repository,
            "revision": revision,
            "cache_dir": root / "huggingface" / "hub",
            "allow_patterns": ["*.json", "*.safetensors", "*.py"],
            "tqdm_class": DownloadProgress,
        }
        try:
            path = Path(snapshot_download(**options, local_files_only=True))
            snapshot_files(path, tokenizer=key == "esmc_model")
        except (LocalEntryNotFoundError, OSError, ValueError, KeyError):
            path = Path(snapshot_download(**options))
            snapshot_files(path, tokenizer=key == "esmc_model")
        paths[key] = str(path)
    paths["model"] = str(
        local_fold_bundle(root, Path(paths["model"]), Path(paths["esmc_model"]))
    )
    paths["ccd"] = str(download_ccd(root, emit))
    return paths


def asset_files(component: str, paths: dict[str, str]) -> list[Path]:
    if component == "complexa":
        return [Path(paths["checkpoint"]), Path(paths["autoencoder"])]
    ccd = Path(paths["ccd"])
    if not ccd.is_file() or ccd.stat().st_size != CCD_FILE.size:
        raise ValueError(f"Missing or incomplete ESMFold2 CCD data: {ccd}")
    return [
        file
        for key in HF_MODELS
        for file in snapshot_files(Path(paths[key]), tokenizer=key == "esmc_model")
    ] + [ccd]


def fingerprints(files: list[Path]) -> dict[str, dict[str, int]]:
    return {
        str(file): {"size": file.stat().st_size, "mtime_ns": file.stat().st_mtime_ns}
        for file in files
    }
