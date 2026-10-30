"""Standalone or persistent Complexa scaffold generator running in its isolated runtime.

Kept free of `mubind` package imports so it runs directly inside `.mubind/runtimes/complexa`.
"""

from __future__ import annotations

import json
import os
import sys
import time
from functools import wraps
from pathlib import Path


def load_model_on_cpu(cfg):
    """Restore Complexa checkpoints on CPU before moving the model to the target GPU."""
    import torch
    from proteinfoundation.generate import load_ckpt_n_configure_inference

    original_load = torch.load

    @wraps(original_load)
    def load_on_cpu(file, map_location=None, *args, **kwargs):
        return original_load(file, "cpu", *args, **kwargs)

    torch.load = load_on_cpu
    try:
        return load_ckpt_n_configure_inference(cfg)
    finally:
        torch.load = original_load


class SamplingProgress:
    """Observe Complexa's flow-matching sampler without altering its numerical work."""

    def __init__(self, report, total_samples: int) -> None:
        self.report = report
        self.total_samples = total_samples
        self.completed_samples = 0
        self.last_report = 0.0
        self.batch = self.batches = None

    def emit(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_report < 0.5:
            return
        self.last_report = now
        first = self.completed_samples + 1
        last = self.completed_samples + self.samples
        self.report(
            f"Complexa · candidates {first}–{last}/{self.total_samples} · step {self.step}/{self.steps}",
            completed=self.completed_samples * self.steps + self.samples * self.step,
            total=self.total_samples * self.steps,
            at=now,
            phase="sampling",
            candidate_first=first,
            candidate_last=last,
            candidates=self.total_samples,
            step=self.step,
            steps=self.steps,
            batch=self.batch,
            batches=self.batches,
        )

    def attach(self, model) -> None:
        simulate = model.fm.full_simulation
        simulation_step = model.fm.simulation_step
        self.model = model
        self.original_simulate = simulate
        self.original_step = simulation_step

        @wraps(simulate)
        def observed_simulation(*args, **kwargs):
            self.samples = int(kwargs["nsamples"])
            self.steps = int(kwargs["nsteps"])
            self.step = 0
            self.emit(force=True)
            result = simulate(*args, **kwargs)
            first = self.completed_samples + 1
            self.completed_samples += self.samples
            self.report(
                f"Complexa · sampled candidates {first}–{self.completed_samples}/{self.total_samples}",
                phase="sampled",
                candidate_first=first,
                candidate_last=self.completed_samples,
                candidates=self.total_samples,
                completed=self.completed_samples * self.steps,
                total=self.total_samples * self.steps,
                at=time.monotonic(),
                batch=self.batch,
                batches=self.batches,
            )
            return result

        @wraps(simulation_step)
        def observed_step(*args, **kwargs):
            result = simulation_step(*args, **kwargs)
            self.step += 1
            self.emit(force=self.step == self.steps)
            return result

        model.fm.full_simulation = observed_simulation
        model.fm.simulation_step = observed_step

    def detach(self) -> None:
        self.model.fm.full_simulation = self.original_simulate
        self.model.fm.simulation_step = self.original_step


def run(config_path, device, progress_path, *, model=None, on_progress=None) -> None:
    """Run one Complexa scaffold generation job and save predicted PDBs and `rewards_*.csv`."""
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    import hydra
    import lightning as L
    import torch
    from omegaconf import OmegaConf
    from proteinfoundation.generate import (
        check_cfg_validity,
        save_predictions,
        save_rewards_to_csv,
        validate_checkpoint_paths,
    )

    def report(message, **progress):
        event = {"message": message, **progress}
        with Path(progress_path).open("a") as stream:
            stream.write(json.dumps(event) + "\n")
        if on_progress is not None:
            on_progress(event)

    class Progress(L.Callback):
        def on_predict_batch_start(
            self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
        ):
            progress.batch, progress.batches = batch_idx + 1, len(dataloader)
            report(
                f"Complexa · sampling batch {batch_idx + 1}/{len(dataloader)}…",
                phase="batch_started",
                batch=batch_idx + 1,
                batches=len(dataloader),
            )

        def on_predict_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
        ):
            report(
                f"Complexa · completed batch {batch_idx + 1}",
                phase="batch_completed",
                batch=batch_idx + 1,
                batches=len(dataloader),
            )

    cfg = OmegaConf.create(json.loads(Path(config_path).read_text()))
    validate_checkpoint_paths(cfg)
    check_cfg_validity(cfg.generation.dataloader.dataset, cfg.generation.args)
    L.seed_everything(int(cfg.seed))
    torch.set_num_threads(4)
    if model is None:
        report("Complexa · loading models…")
        model = load_model_on_cpu(cfg)
    else:
        report("Complexa · reusing loaded models…")
        if cfg.generation.args.ag_ratio != 0:
            raise ValueError("The provider supports generation without autoguidance")
        model.configure_inference(cfg.generation, nn_ag=None)
        model._search_instance = None
        model._refinement_instance = None
        model.reward_model = None
    L.seed_everything(int(cfg.seed))
    torch.set_float32_matmul_precision("high")
    dataloader = hydra.utils.instantiate(cfg.generation.dataloader)
    progress = SamplingProgress(
        report, len(dataloader.dataset) * int(cfg.generation.search.best_of_n.replicas)
    )
    report(
        f"Complexa · generating {progress.total_samples} scaffold candidates",
        phase="configured",
        candidates=progress.total_samples,
    )
    gpu = torch.device(device)
    trainer = L.Trainer(
        accelerator="gpu" if gpu.type == "cuda" else "cpu",
        devices=[gpu.index or 0] if gpu.type == "cuda" else 1,
        logger=False,
        enable_checkpointing=False,
        inference_mode=False,
        enable_progress_bar=False,
        callbacks=[Progress()],
    )
    progress.attach(model)
    try:
        predictions = trainer.predict(model, dataloader)
    finally:
        progress.detach()
        model.to(gpu)
    report("Complexa · saving scaffolds…", phase="saving")
    Path(cfg.root_path).mkdir(parents=True, exist_ok=True)
    _, rewards = save_predictions(
        str(cfg.root_path), predictions, job_id=0, cath_codes=None
    )
    save_rewards_to_csv(rewards, str(cfg.root_path), config_name="mubind", job_id=0)
    report("Complexa · scaffolds saved", phase="saved")


class ComplexaBackend:
    """Persistent Complexa worker backend managed by `worker_server.serve`."""

    def __init__(self, settings: dict) -> None:
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        import torch
        from omegaconf import OmegaConf
        from proteinfoundation.generate import validate_checkpoint_paths

        self.device = settings["device"]
        checkpoint = Path(settings["paths"]["checkpoint"])
        self.checkpoints = {
            "ckpt_path": str(checkpoint.parent),
            "ckpt_name": checkpoint.name,
            "autoencoder_ckpt_path": settings["paths"]["autoencoder"],
        }
        cfg = OmegaConf.create(
            {
                **self.checkpoints,
                "generation": {"args": {"ag_ratio": 0.0, "guidance_w": 1.0}},
            }
        )
        torch.set_num_threads(4)
        validate_checkpoint_paths(cfg)
        self.model = load_model_on_cpu(cfg).to(self.device).eval()
        self.on_progress = None

    def set_progress_reporter(self, reporter) -> None:
        self.on_progress = reporter

    def generate(self, job: dict) -> dict:
        from contextlib import redirect_stderr, redirect_stdout

        config = json.loads(Path(job["config"]).read_text())
        if any(config.get(key) != value for key, value in self.checkpoints.items()):
            raise ValueError(
                "The request uses different checkpoints from this provider"
            )
        with (
            Path(job["log"]).open("w") as stream,
            redirect_stdout(stream),
            redirect_stderr(stream),
        ):
            try:
                run(
                    job["config"],
                    self.device,
                    job["progress"],
                    model=self.model,
                    on_progress=self.on_progress,
                )
            except Exception:
                import traceback

                traceback.print_exc()
                raise
        return {}


if __name__ == "__main__":
    if len(sys.argv) == 2:
        try:
            from .worker_server import serve
        except ImportError:
            from worker_server import serve

        serve(
            json.loads(Path(sys.argv[1]).read_text()),
            ComplexaBackend,
            operations=["generate"],
            name="Complexa",
        )
    else:
        run(*sys.argv[1:])
