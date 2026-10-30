"""Build Complexa Hydra configs, run Complexa (standalone or via provider) with OOM backoff, and read final scaffold PDBs."""

from __future__ import annotations

import csv
import json
import math
import os
import signal
import subprocess
from dataclasses import replace
from pathlib import Path


def final_complexa_pdbs(directory: Path) -> list[Path]:
    """Read `rewards_*.csv` manifests in `directory` and return sorted `sample_type == 'final'` PDB paths."""
    root = directory.resolve()
    manifests = sorted(root.glob("rewards_*.csv"))
    if not manifests:
        raise ValueError(f"No Complexa rewards_*.csv metadata in {root}")
    paths = set()
    for manifest in manifests:
        with manifest.open(newline="") as stream:
            rows = csv.DictReader(stream)
            if not {"pdb_path", "sample_type"}.issubset(rows.fieldnames or []):
                raise ValueError(f"Missing final-sample metadata in {manifest}")
            for row in rows:
                if row["sample_type"] != "final":
                    continue
                path = Path(row["pdb_path"])
                path = (
                    (root / path).resolve()
                    if not path.is_absolute()
                    else path.resolve()
                )
                if root not in path.parents or not path.is_file():
                    raise ValueError(
                        f"Missing or out-of-folder Complexa final output: {path}"
                    )
                paths.add(path)
    if not paths:
        raise ValueError("Complexa produced no final scaffolds")
    return sorted(paths)


def complexa_config(settings, target, target_path: Path, out: Path, runtime, seed: int) -> dict:
    """Build the Hydra/OmegaConf JSON configuration for a Complexa scaffold generation run."""
    simulation = {
        "sampling_mode": "sc",
        "sc_scale_noise": 0.1,
        "sc_scale_score": 1.0,
        "t_lim_ode": 0.98,
        "t_lim_ode_below": 0.02,
        "tsr_k": 1.0,
        "tsr_sigma": 1.0,
        "center_every_step": False,
    }
    length = settings.binder_length
    return {
        **runtime.complexa_config(),
        "seed": seed,
        "root_path": str(out / "complexa_scaffolds"),
        "gen_njobs": 1,
        "run_name": "mubind",
        "base_config_name": "mubind",
        "generation": {
            "target_dict_cfg": {
                "mubind_target": {
                    "target_path": str(target_path),
                    "target_input": target.spec,
                    "hotspot_residues": list(target.hotspots),
                    "binder_length": [length, length],
                    "pdb_id": "target",
                }
            },
            "dataloader": {
                "_target_": "torch.utils.data.DataLoader",
                "batch_size": max(
                    4, math.ceil(settings.complexa_batch_size / settings.replicas)
                ),
                "shuffle": False,
                "collate_fn": {
                    "_target_": "proteinfoundation.datasets.gen_dataset.collate_fn",
                    "_partial_": True,
                    "padding_values": {"target_chains": -1, "chains": -1},
                },
                "dataset": {
                    "_target_": "proteinfoundation.datasets.gen_dataset.GenDataset",
                    "task_name": "mubind_target",
                    "nres": {
                        "_target_": "proteinfoundation.datasets.gen_dataset.UniformInt",
                        "low": length,
                        "high": length,
                        "nsamples": math.ceil(
                            settings.num_complexa / settings.replicas
                        ),
                        "endpoint": True,
                    },
                    "nrepeat_per_sample": 1,
                    "conditional_features": [
                        {
                            "_target_": "proteinfoundation.datasets.gen_dataset.TargetFeatures",
                            "task_name": "mubind_target",
                            "binder_gen_only": True,
                            "pdb_path": str(target_path),
                            "input_spec": target.spec,
                            "target_hotspots": list(target.hotspots),
                            "binder_center": None,
                            "pdb_id": "target",
                        }
                    ],
                    "transforms": [
                        {
                            "_target_": "proteinfoundation.datasets.transforms.CoordsTensorCenteringTransform",
                            "tensor_name": "x_target",
                            "mask_name": "target_mask",
                            "data_mode": "all-atom",
                        }
                    ],
                },
            },
            "args": {
                "nsteps": settings.complexa_steps,
                "self_cond": True,
                "guidance_w": 1.0,
                "ag_ratio": 0.0,
                "ag_ckpt_path": None,
                "save_trajectory_every": 0,
                "fold_cond": False,
            },
            "model": {
                "bb_ca": {
                    "schedule": {"mode": "log", "p": 2.0},
                    "gt": {"mode": "1/t", "p": 1.0, "clamp_val": None},
                    "simulation_step_params": simulation,
                },
                "local_latents": {
                    "schedule": {"mode": "power", "p": 2.0},
                    "gt": {"mode": "tan", "p": 1.0, "clamp_val": None},
                    "simulation_step_params": dict(simulation),
                },
            },
            "n_recycle": 0,
            "task_name": "mubind_target",
            "search": {
                "algorithm": "best-of-n",
                "max_batch_size": settings.complexa_batch_size,
                "reward_threshold": None,
                "step_checkpoints": sorted(
                    {0, *(round(settings.complexa_steps * i / 4) for i in range(1, 5))}
                ),
                "best_of_n": {"replicas": settings.replicas},
            },
            "refinement": {
                "algorithm": None,
                "refine_targets": "final",
                "save_pre_refinement": "none",
            },
            "filter": {
                "filter_samples_limit": 1000,
                "delete_non_top_n_samples": False,
                "dedup_sequence": True,
                "reward_threshold": None,
            },
            "reward_model": None,
        },
    }


def read_progress(stream, on_status, on_progress=None) -> None:
    """Read complete JSONL progress events from `stream`, leaving partial lines for the next poll."""
    while True:
        position = stream.tell()
        line = stream.readline()
        if not line.endswith("\n"):
            stream.seek(position)
            return
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict) or not isinstance(event.get("message"), str):
            continue
        if on_progress is not None and "completed" in event and "total" in event:
            on_progress(event)
        else:
            on_status(event["message"])


def run_complexa(
    settings,
    target,
    target_path: Path,
    out: Path,
    runtime,
    seed: int,
    device,
    on_status,
    *,
    on_progress=None,
) -> list[Path]:
    """Run Complexa, halving `complexa_batch_size` and retrying if CUDA runs out of memory."""
    while True:
        try:
            return _run_complexa_once(
                settings,
                target,
                target_path,
                out,
                runtime,
                seed,
                device,
                on_status,
                on_progress=on_progress,
            )
        except RuntimeError as exc:
            log = out / "complexa.log"
            detail = str(exc).lower()
            if log.exists():
                detail += log.read_text(errors="replace").lower()
            if "cuda out of memory" not in detail or settings.complexa_batch_size <= 1:
                raise
            settings = replace(
                settings, complexa_batch_size=max(1, settings.complexa_batch_size // 2)
            )
            on_status(
                f"Complexa · retrying with {settings.complexa_batch_size} candidates per batch to fit GPU memory…"
            )


def _run_complexa_once(
    settings,
    target,
    target_path: Path,
    out: Path,
    runtime,
    seed: int,
    device,
    on_status,
    *,
    on_progress=None,
) -> list[Path]:
    config = complexa_config(settings, target, target_path, out, runtime, seed)
    config_path = out / "complexa.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    environment = runtime.environment()
    environment.update(
        OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false"
    )
    worker = Path(__file__).with_name("complexa_worker.py")
    log = out / "complexa.log"
    progress_file = out / "complexa.progress.jsonl"
    progress_file.write_text("")
    from .complexa_service import ManagedComplexaService

    service = ManagedComplexaService(runtime, device)
    if service.status() is not None:
        service.generate(config_path, progress_file, log, on_status, on_progress)
        return final_complexa_pdbs(out / "complexa_scaffolds")[: settings.num_complexa]

    with log.open("w") as stream, progress_file.open() as events:
        process = subprocess.Popen(
            [
                str(runtime.python),
                "-u",
                str(worker),
                str(config_path),
                str(device),
                str(progress_file),
            ],
            cwd=out,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            while True:
                try:
                    process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    read_progress(events, on_status, on_progress)
                else:
                    read_progress(events, on_status, on_progress)
                    break
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
    if process.returncode:
        raise RuntimeError(
            f"Complexa failed (exit {process.returncode}). Details: {log}"
        )
    # The last replica group can exceed the requested count by up to replicas - 1.
    return final_complexa_pdbs(out / "complexa_scaffolds")[: settings.num_complexa]
