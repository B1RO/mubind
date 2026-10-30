"""Install model dependencies outside the mubind interpreter and source tree."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sysconfig
from pathlib import Path

from ..preprocessing.prostt5.steps._io import write_manifest
from ..common.progress import ProgressCallback, ProgressEvent

COMPLEXA_REVISION = "acf55a89a9a7206d5ce9abc0bf2246c3b3829f92"
ESM_REVISION = "c94ed8d763bbd7088b296949e5b401e8ea12073a"
TRANSFORMERS_REVISION = "3a8956fb4d4ea16b0ec8e71deef2c2909b6a5cbf"
COMPLEXA_PACKAGE = f"proteinfoundation @ https://github.com/NVIDIA-BioNeMo/Proteina-Complexa/archive/{COMPLEXA_REVISION}.tar.gz"
ESM_PACKAGE = f"esm @ https://github.com/Biohub/esm/archive/{ESM_REVISION}.tar.gz"

# Use CUDA 12.8 builds, including Blackwell, while keeping incompatible Transformers
# installations in separate environments. No inference, downloads of weights,
# or service lifecycle operations happen in these probes.
PROBES = {
    "complexa": "from proteinfoundation.generate import load_ckpt_n_configure_inference; from proteinfoundation.proteina import Proteina",
    "esmfold2": "from esm.models.esmfold2 import ESMFold2InputBuilder; from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model; from transformers.models.esmc.modeling_esmc import ESMCForMaskedLM",
}


def install_steps(component: str) -> list[tuple[str, list[str]]]:
    if component == "esmfold2":
        return [
            (
                "Installing PyTorch",
                [
                    "torch",
                    "--upgrade",
                ],
            ),
            (
                "Installing ESMFold2",
                [ESM_PACKAGE, "hf_transfer>=0.1.9,<0.2"],
            ),
        ]
    return [
        (
            "Installing PyTorch",
            [
                "torch",
                "torchvision",
                "torchaudio",
                "--upgrade",
            ],
        ),
        ("Installing Proteina-Complexa", [COMPLEXA_PACKAGE]),
        (
            "Installing geometric libraries",
            [
                "torch_geometric",
                "torch_scatter",
                "torch_sparse",
                "torch_cluster",
                "--only-binary",
                "torch-scatter,torch-sparse,torch-cluster",
                "--reinstall-package",
                "torch-scatter",
                "--reinstall-package",
                "torch-sparse",
                "--reinstall-package",
                "torch-cluster",
            ],
        ),
        (
            "Installing structure libraries",
            [
                "atomworks[ml,openbabel]==2.2.1",
                "biotite==1.6.0",
                "colabdesign==1.1.1",
                "alphafold-colabfold==2.3.7",
                "jax==0.4.29",
                "jaxlib==0.4.29",
                "flax==0.9.0",
            ],
        ),
        ("Installing Graphein", ["graphein==1.7.8"]),
    ]


def dependency_overrides(component: str) -> list[str]:
    if component == "esmfold2":
        # Biohub's former transformers repository is no longer public. The
        # exact pinned commit remains accessible through the upstream repo.
        return [
            f"transformers @ https://github.com/huggingface/transformers/archive/{TRANSFORMERS_REVISION}.tar.gz",
            "torch",
        ]
    # Upstream's base metadata pins scipy/einops below Atomworks' minimum,
    # and its installer subsequently replaces Biotite with 1.6. Keep those
    # compatibility choices explicit instead of relying on install order.
    return [
        "scipy==1.16.3",
        "einops==0.8.2",
        "biotite==1.6.0",
        "torch",
        "torchvision",
        "torchaudio",
    ]


def dependency_constraints(component: str) -> list[str]:
    if component == "esmfold2":
        return []
    return [
        "numpy==1.26.4",
        "jax==0.4.29",
        "jaxlib==0.4.29",
        "flax==0.9.0",
        "chex==0.1.86",
        "optax==0.2.3",
        "dm-haiku==0.0.12",
    ]


def runtime_identity(component: str) -> dict:
    return {
        "version": 2,
        "python": "3.12",
        "steps": install_steps(component),
        "probe": PROBES[component],
        "overrides": dependency_overrides(component),
        "constraints": dependency_constraints(component),
        "torch_backend": "cu128",
        "geometric_wheels": "installed-torch" if component == "complexa" else None,
    }


def runtime_directory(component: str, root: Path) -> Path:
    names = {"complexa": "proteina-complexa", "esmfold2": "esmfold2"}
    if component not in names:
        raise ValueError(f"Unknown model environment: {component}")
    return root / ".mubind" / "runtimes" / names[component]


def runtime_identity_matches(component: str, recorded) -> bool:
    """Keep working CUDA 12.8 ESMFold2 installs usable as version pins are removed."""
    expected = json.loads(json.dumps(runtime_identity(component)))
    if recorded == expected:
        return True
    if component != "esmfold2" or not isinstance(recorded, dict):
        return False
    constraints = recorded.get("constraints", [])
    if (
        not isinstance(constraints, list)
        or len(constraints) != 1
        or not isinstance(constraints[0], str)
    ):
        return False
    previous_torch = constraints[0]
    if not previous_torch.startswith("torch==") or not previous_torch.endswith(
        "+cu128"
    ):
        return False
    legacy = {
        "version": 1,
        "python": "3.12",
        "steps": [
            [
                "Installing PyTorch",
                [
                    previous_torch,
                    "--index-url",
                    "https://download.pytorch.org/whl/cu128",
                ],
            ],
            [
                "Installing ESMFold2",
                [ESM_PACKAGE, "hf_transfer>=0.1.9,<0.2", previous_torch],
            ],
        ],
        "probe": PROBES[component],
        "overrides": [
            value for value in dependency_overrides(component) if value != "torch"
        ],
        "constraints": [previous_torch],
    }
    return recorded == legacy


def runtime_ready(component: str, root: Path) -> bool:
    """Read the last successful environment check without starting Python."""
    directory = runtime_directory(component, root)
    try:
        report = json.loads((directory / "runtime.json").read_text())
        return (
            report["state"] == "complete"
            and runtime_identity_matches(component, report["identity"])
            and (directory / "venv" / "bin" / "python").is_file()
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def runtime_environment(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(name, None)
    env.update(
        UV_CACHE_DIR=str(root / ".mubind" / "cache" / "uv"),
        UV_PYTHON_INSTALL_DIR=str(root / ".mubind" / "python"),
        UV_LINK_MODE="copy",
        HF_HOME=str(root / "huggingface"),
        HF_HUB_CACHE=str(root / "huggingface" / "hub"),
        PYTHONUNBUFFERED="1",
    )
    return env


INSTALLER_COUNT = re.compile(r"(Resolved|Installed) (\d+ packages?) in ")


def installer_activity(line: str) -> str | None:
    """Summarize uv's milestone lines. Other output only shows the step is producing output."""
    text = line.strip()
    if match := INSTALLER_COUNT.match(text):
        return f"{match[1]} {match[2]}"
    if text.startswith("Prepared "):
        # uv prints nothing while it installs, which can take minutes.
        return "Installing packages"
    for prefix in ("Downloading ", "Downloaded ", "Building ", "Built "):
        if text.startswith(prefix):
            return prefix + text[len(prefix) :].split(" @ ", 1)[0]
    return None


def _run(
    command: list[str],
    *,
    env: dict[str, str],
    log: Path,
    emit: ProgressCallback | None = None,
    installer: bool = False,
) -> None:
    with log.open("a") as stream:
        stream.write("\n" + repr(command) + "\n")
        stream.flush()
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )
        try:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                if emit is not None:
                    activity = installer_activity(line) if installer else None
                    emit(ProgressEvent("output", activity=activity or ""))
            returncode = process.wait()
        except BaseException:
            process.kill()
            process.wait()
            raise
        finally:
            process.stdout.close()
    if returncode:
        raise RuntimeError(
            f"Runtime installation failed (exit {returncode}). Details: {log}"
        )


def geometric_wheel_index(python: Path, env: dict[str, str]) -> str:
    """Select extension wheels using the installed PyTorch ABI and CUDA build."""
    result = subprocess.run(
        [
            str(python),
            "-c",
            "import json,torch; print(json.dumps([torch.__version__,torch.version.cuda]))",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    version, cuda = json.loads(result.stdout.strip().splitlines()[-1])
    if cuda is None:
        raise RuntimeError("Complexa requires a CUDA-enabled PyTorch installation")
    major, minor, *_ = version.split("+", 1)[0].split(".")
    return f"https://data.pyg.org/whl/torch-{int(major)}.{int(minor)}.0+cu{cuda.replace('.', '')}.html"


def verify_cuda(
    python: Path,
    component: str,
    env: dict[str, str],
    log: Path,
    emit: ProgressCallback | None = None,
):
    """Exercise CUDA kernels when a GPU is present, without loading any models."""
    script = """import torch
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        device = torch.device('cuda', index)
        x = torch.ones((4, 4), device=device)
        assert (x @ x).sum().item() == 64
        if COMPONENT == 'complexa':
            import torch_scatter, torch_sparse, torch_cluster
            indices = torch.tensor([0, 1, 0, 1], device=device)
            torch_scatter.scatter_add(x[:, 0], indices)
            edge = torch.tensor([[0, 1], [1, 0]], device=device)
            torch_sparse.spmm(edge, x[:2, 0], 2, 2, x[:2])
            torch_cluster.knn_graph(torch.arange(12, device=device, dtype=torch.float32).reshape(4, 3), k=1)
        torch.cuda.synchronize(device)
        print('CUDA verified:', torch.cuda.get_device_name(index), torch.__version__)
""".replace("COMPONENT", repr(component))
    _run([str(python), "-c", script], env=env, log=log, emit=emit)


def runtime_users(python: Path) -> list[int]:
    """Find processes using this environment before changing its packages."""
    users = []
    for process in Path("/proc").iterdir():
        if not process.name.isdecimal() or int(process.name) == os.getpid():
            continue
        try:
            arguments = (process / "cmdline").read_bytes().split(b"\0")
            if arguments and os.fsdecode(arguments[0]) == str(python):
                users.append(int(process.name))
        except (OSError, ProcessLookupError):
            continue
    return users


def verify_esmfold2_inputs(
    python: Path, root: Path, paths: dict[str, str], emit: ProgressCallback
) -> None:
    """Exercise the real input builder offline without loading model weights."""
    emit(ProgressEvent("verify", message="Checking ESMFold2 input data offline"))
    env = runtime_environment(root)
    env.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        ESMCFOLD_CCD_PATH=paths["ccd"],
    )
    log = root / ".mubind" / "logs" / "esmfold2.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            str(python),
            "-c",
            "from esm.models.esmfold2 import ESMFold2InputBuilder; ESMFold2InputBuilder()",
        ],
        env=env,
        log=log,
        emit=emit,
    )


def find_uv() -> str:
    """Prefer this interpreter's uv installation, even without activation."""
    name = "uv.exe" if os.name == "nt" else "uv"
    executable = shutil.which(str(Path(sysconfig.get_path("scripts")) / name))
    if executable is not None:
        return executable
    try:
        from uv import find_uv_bin

        executable = shutil.which(find_uv_bin())
    except (ImportError, FileNotFoundError):
        executable = None
    if executable is None:
        executable = shutil.which("uv")
    if executable is None:
        raise RuntimeError(
            "The uv installer is missing. Install mubind's dependencies with python -m pip install -e ."
        )
    return executable


def ensure_runtime(component: str, root: Path, emit: ProgressCallback) -> Path:
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        raise RuntimeError(
            "Model runtimes currently support Linux x86_64 with NVIDIA CUDA. Data preparation works independently."
        )
    uv = find_uv()
    directory = runtime_directory(component, root)
    marker = directory / "runtime.json"
    identity = json.loads(json.dumps(runtime_identity(component)))
    if directory.is_symlink():
        raise ValueError(f"Runtime directory must not be a symlink: {directory}")
    previous = json.loads(marker.read_text()) if marker.exists() else {}
    if directory.exists() and not previous:
        raise FileExistsError(f"Unrecognised runtime directory: {directory}")
    upgrading = bool(
        previous and not runtime_identity_matches(component, previous.get("identity"))
    )
    if upgrading:
        old = previous.get("identity", {})
        if (
            not isinstance(old, dict)
            or old.get("python") != "3.12"
            or old.get("probe") != PROBES[component]
        ):
            raise ValueError(f"Runtime identity mismatch: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    log = root / ".mubind" / "logs" / f"{component}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    python = directory / "venv" / "bin" / "python"
    env = runtime_environment(root)
    if not upgrading and previous.get("state") == "complete" and python.is_file():
        emit(ProgressEvent("runtime", message="Checking existing environment"))
        try:
            _run([str(python), "-c", PROBES[component]], env=env, log=log, emit=emit)
            verify_cuda(python, component, env, log, emit)
        except (RuntimeError, OSError) as exc:
            previous.update(state="failed", error=str(exc))
            write_manifest(marker, previous)
            raise
        return python
    if python.is_file():
        users = runtime_users(python)
        if users:
            raise RuntimeError(
                f"Cannot update {component} while its environment is in use by processes {users}. "
                "Finish those runs before retrying setup."
            )
    if upgrading:
        emit(ProgressEvent("runtime", message="Updating the existing environment"))
    report = {"state": "installing", "identity": identity}
    if upgrading:
        report["previous_identity"] = previous["identity"]
    write_manifest(marker, report)
    try:
        if not python.is_file():
            emit(ProgressEvent("runtime", message="Creating Python 3.12 environment"))
            _run(
                [uv, "venv", "--python", "3.12", str(directory / "venv")],
                env=env,
                log=log,
                emit=emit,
            )
        # Keep non-PyTorch compatibility constraints separate. PyTorch versions
        # are selected by the resolver from the CUDA 12.8 package index.
        constraints = directory / "constraints.txt"
        constraints.write_text("\n".join(dependency_constraints(component)) + "\n")
        overrides = directory / "overrides.txt"
        overrides.write_text("\n".join(dependency_overrides(component)) + "\n")
        for label, requirements in install_steps(component):
            emit(
                ProgressEvent(
                    "runtime", message=label, activity="Resolving dependencies"
                )
            )
            if label == "Installing geometric libraries":
                requirements = [
                    *requirements,
                    "--find-links",
                    geometric_wheel_index(python, env),
                ]
            _run(
                [
                    uv,
                    "pip",
                    "install",
                    "--python",
                    str(python),
                    "--constraint",
                    str(constraints),
                    "--override",
                    str(overrides),
                    "--torch-backend",
                    "cu128",
                    *requirements,
                ],
                env=env,
                log=log,
                emit=emit,
                installer=True,
            )
        emit(ProgressEvent("runtime", message="Checking environment imports"))
        _run([str(python), "-c", PROBES[component]], env=env, log=log, emit=emit)
        emit(ProgressEvent("verify", message="Checking CUDA compatibility"))
        verify_cuda(python, component, env, log, emit)
        report["state"] = "complete"
        write_manifest(marker, report)
    except BaseException as exc:
        report.update(state="failed", error=str(exc))
        write_manifest(marker, report)
        raise
    return python
