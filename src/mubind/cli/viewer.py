"""Portable structure previews. This module needs only the Python standard library."""

from __future__ import annotations

import json
import math
import os
import re
import sys
import tempfile
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import urlsplit

from ..paths import GENERATIONS, existing_artifact


@dataclass
class Preview:
    directory: Path
    candidates: list[dict]
    skipped: int = 0


def finite_number(value):
    return (
        float(value)
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        else None
    )


def structure_path(directory: Path, row: dict) -> Path | None:
    """Prefer the run's own files, including after moving a run to another machine."""
    paths = []
    if row.get("scaffold") and row.get("candidate_id"):
        paths.append(
            directory / "folds" / str(row["scaffold"]) / str(row["candidate_id"])
            / "complex.cif"
        )
    if isinstance(row.get("structure"), str):
        stored = Path(row["structure"])
        paths.append(stored if stored.is_absolute() else directory / stored)
    for path in paths:
        path = path.resolve()
        if path.is_relative_to(directory) and path.is_file():
            return path
    return None


def read_run(directory: Path) -> Preview:
    directory = existing_artifact(directory).resolve()
    if directory.name == "results.json" and directory.is_file():
        directory = directory.parent
    results = directory / "results.json"
    try:
        rows = json.loads(results.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"No results.json in {directory}. Choose a generation run.") from exc
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(f"Could not read {results}. Wait for generation to finish saving.") from exc
    if not isinstance(rows, list):
        raise ValueError(f"Expected a candidate list in {results}.")
    candidates = []
    skipped = 0
    for row in rows:
        if not isinstance(row, dict) or row.get("fold_status") != "complete":
            skipped += 1
            continue
        path = structure_path(directory, row)
        if path is None:
            skipped += 1
            continue
        try:
            structure = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            skipped += 1
            continue
        if not structure.strip():
            skipped += 1
            continue
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}
        candidates.append(
            {
                "id": f"{row.get('scaffold', '')}/{row.get('candidate_id', path.parent.name)}",
                "score": finite_number(row.get("selection_score")),
                "plddt": finite_number(metrics.get("binder_plddt")),
                "iptm": finite_number(metrics.get("iptm")),
                "structure": structure,
                "format": "pdb" if path.suffix.lower() == ".pdb" else "cif",
            }
        )
    if not candidates:
        raise ValueError(
            f"No completed structures to view in {directory}. "
            "Finish folding at least one candidate first."
        )
    candidates.sort(
        key=lambda c: c["score"] if c["score"] is not None else -math.inf,
        reverse=True,
    )
    return Preview(directory, candidates, skipped)


def load_preview(run: Path | None = None) -> Preview:
    if run is not None:
        return read_run(run)
    # Viewing/exporting does not change results.json, so it cannot make an old
    # run appear newest. Ignore newer failed or sequence-only runs.
    results = sorted(
        GENERATIONS.glob("*/results.json"),
        key=lambda p: p.stat().st_mtime_ns,
        reverse=True,
    )
    for result in results:
        try:
            return read_run(result.parent)
        except (OSError, ValueError):
            continue
    raise ValueError(
        "No viewable runs in artifacts/generations/. Run mubind generate first, "
        "or pass a run folder to mubind view."
    )


def render_html(preview: Preview) -> str:
    assets = files("mubind.cli").joinpath("viewer_assets")
    data = json.dumps(
        {"run": preview.directory.name, "candidates": preview.candidates},
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    ).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    library = assets.joinpath("3Dmol-min.js").read_text(encoding="utf-8")
    # Keep downloaded reports offline, including when developer tools are open.
    library = re.sub(r"//# sourceMappingURL=[^\r\n]*", "", library)
    library = re.sub(r"</script", r"<\\/script", library, flags=re.IGNORECASE)
    license_text = assets.joinpath("3Dmol-LICENSE.txt").read_text(encoding="utf-8")
    license_text = re.sub(r"</script", r"<\\/script", license_text, flags=re.IGNORECASE)
    replacements = {"DATA": data, "LIBRARY": library, "LICENSE": license_text}
    template = assets.joinpath("view.html").read_text(encoding="utf-8")
    return re.sub(r"@@(DATA|LIBRARY|LICENSE)@@", lambda m: replacements[m[1]], template)


def save_html(path: Path, document: str) -> Path:
    path = path.expanduser().resolve()
    if path.suffix.lower() != ".html":
        raise ValueError("The export path must end in .html.")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".view-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(document)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


def viewer_server(document: str, port: int) -> ThreadingHTTPServer:
    """Serve only this preview, never the run directory or other server files."""
    payload = document.encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.respond(body=True)

        def do_HEAD(self):
            self.respond(body=False)

        def respond(self, *, body):
            path = urlsplit(self.path).path
            if path not in ("/", "/index.html", "/view.html"):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if path == "/view.html":
                self.send_header("Content-Disposition", 'attachment; filename="view.html"')
            self.end_headers()
            if body:
                try:
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def log_message(self, *_):
            pass

    try:
        return ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError as exc:
        raise ValueError(
            f"Could not listen on 127.0.0.1:{port}: {exc}. "
            "Choose another port with --port and forward the same port over SSH."
        ) from exc


def open_browser(url: str) -> bool:
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"):
        return False
    if os.name != "nt" and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        or sys.platform == "darwin"
    ):
        return False
    try:
        return webbrowser.open(url)
    except (OSError, webbrowser.Error):
        return False
