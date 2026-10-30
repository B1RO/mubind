"""Download the pinned ProstT5 release used by the original training pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ....common.progress import ProgressCallback, ProgressEvent
from ._io import SPLITS, write_manifest

REPOSITORY = "Rostlab/ProstT5Dataset"
REVISION = "a35fbb7dc26c2e00225098b55153044d76ebff4e"


def download_prostt5(
    out: Path | str, *, workers: int = 4, on_progress: ProgressCallback | None = None
) -> Path:
    """Fetch raw Parquet shards and metadata; resume an interrupted download.

    Hugging Face handles concurrent transfers and its local download cache.
    This downloads the full raw dataset, before any sequence-length filtering.
    Existing directories must be empty or belong to this same download.
    """
    if workers < 1:
        raise ValueError("workers must be positive")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "Install preprocessing dependencies: pip install -e '.[preprocessing]'"
        ) from exc

    out = Path(out)
    manifest = out / "download_manifest.json"
    identity = {"repository": REPOSITORY, "revision": REVISION}
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError("Existing download has a different repository or revision")
    elif out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Download directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    report = {"state": "downloading", **identity}
    write_manifest(manifest, report)
    progress_options = {}
    if on_progress:
        from tqdm.auto import tqdm

        class DownloadProgress(tqdm):
            # Keep tqdm's counters and locking, replacing only its rendering.
            def display(self, msg=None, pos=None):
                if getattr(self, "unit", None) != "B":
                    on_progress(
                        ProgressEvent("download", int(self.n), self.total, unit="files")
                    )

        progress_options["tqdm_class"] = DownloadProgress
        on_progress(
            ProgressEvent(
                "download", unit="files", message="Connecting to Hugging Face"
            )
        )
    try:
        snapshot_download(
            repo_id=REPOSITORY,
            repo_type="dataset",
            revision=REVISION,
            local_dir=out,
            allow_patterns=["data/*.parquet", "README.md", "*.json"],
            max_workers=workers,
            **progress_options,
        )
        files = {
            split: sorted((out / "data").glob(f"{split}-*.parquet")) for split in SPLITS
        }
        if any(not paths for paths in files.values()):
            raise ValueError("Download is missing one or more ProstT5 splits")
        report.update(
            state="complete",
            files=[
                str(path.relative_to(out)) for paths in files.values() for path in paths
            ],
        )
        write_manifest(manifest, report)
        if on_progress:
            count = len(report["files"])
            on_progress(ProgressEvent("download", count, count, unit="files"))
    except Exception as exc:
        report.update(state="failed", error=str(exc))
        write_manifest(manifest, report)
        raise
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(download_prostt5(args.out, workers=args.workers))


if __name__ == "__main__":
    main()
