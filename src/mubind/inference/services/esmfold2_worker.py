"""Persistent monomer and complex folding worker running in the isolated ESMFold2 runtime.

Kept free of `mubind` package imports so it runs directly inside `.mubind/runtimes/esmfold2`.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

try:
    from .worker_server import (
        FoldingServer,
        RequestCancelled,
        cancellable_model_calls,
        request_cancellation,
        serve,
        write_output,
    )
except ImportError:
    from worker_server import (
        FoldingServer,
        RequestCancelled,
        cancellable_model_calls,
        request_cancellation,
        serve,
        write_output,
    )


class Esmfold2Backend:
    """Loads ESMFold2 once and executes `fold_monomer` and `fold_complex` jobs."""

    def __init__(self, settings: dict) -> None:
        ccd = Path(settings["paths"]["ccd"])
        if not ccd.is_file() or not ccd.stat().st_size:
            raise FileNotFoundError(
                "ESMFold2 CCD data is missing. Run mubind setup --only esmfold2 --yes."
            )
        # Upstream reads this at import time; pin it before importing esmfold2.
        os.environ["ESMCFOLD_CCD_PATH"] = str(ccd)
        import torch
        from esm.models.esmfold2 import ESMFold2InputBuilder
        from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

        self.settings = settings
        self.builder = ESMFold2InputBuilder(ccd_cache=ccd.parent)
        self.model = (
            ESMFold2Model.from_pretrained(settings["paths"]["model"])
            .to(torch.device(settings["device"]))
            .eval()
        )

    def fold_monomer(self, job: dict) -> dict:
        import torch
        from esm.models.esmfold2 import ProteinInput, StructurePredictionInput

        sequence = job["sequence"].strip().upper()
        if not sequence:
            raise ValueError("Empty monomer sequence")
        protocol = self.settings["protocol"]
        inputs = StructurePredictionInput(
            sequences=[ProteinInput(id="B", sequence=sequence)]
        )
        with torch.inference_mode():
            result = self.builder.fold(
                self.model,
                inputs,
                num_loops=protocol["num_loops"],
                num_sampling_steps=protocol["num_sampling_steps"],
                num_diffusion_samples=1,
                seed=int(job.get("seed", 0)),
            )
        metrics = {
            "sequence_length": len(sequence),
            "plddt": None
            if result.plddt is None
            else float(result.plddt.float().mean().item()),
            "ptm": None if result.ptm is None else float(result.ptm),
        }
        metrics_json = json.dumps(metrics, allow_nan=False, indent=2) + "\n"
        write_output(job["out"], result.complex.to_mmcif())
        if job.get("metrics_json"):
            write_output(job["metrics_json"], metrics_json)
        return metrics

    def fold_complex(self, job: dict) -> dict:
        import torch
        from esm.models.esmfold2 import ProteinInput, StructurePredictionInput

        target, binder = job["target"].strip().upper(), job["binder"].strip().upper()
        if not target or not binder:
            raise ValueError(
                "Complex folding needs nonempty target and binder sequences"
            )
        protocol = self.settings["protocol"]
        inputs = StructurePredictionInput(
            sequences=[
                ProteinInput(id="A", sequence=target),
                ProteinInput(id="B", sequence=binder),
            ]
        )
        with torch.inference_mode():
            result = self.builder.fold(
                self.model,
                inputs,
                num_loops=protocol["num_loops"],
                num_sampling_steps=protocol["num_sampling_steps"],
                num_diffusion_samples=1,
                seed=int(job.get("seed", 0)),
            )
        metrics = complex_metrics(result, len(target), len(binder))
        metrics_json = json.dumps(metrics, allow_nan=False, indent=2) + "\n"
        write_output(job["out"], result.complex.to_mmcif())
        if job.get("metrics_json"):
            write_output(job["metrics_json"], metrics_json)
        return metrics


def complex_metrics(result, target_length: int, binder_length: int) -> dict:
    """Compute per-chain pLDDT, pTM, ipTM, and cross-chain interface PAE in `0..1` units."""
    import torch

    def mean(value):
        return None if value is None else float(value.float().mean().item())

    plddt, pae = result.plddt, result.pae
    if plddt is not None and plddt.numel() != target_length + binder_length:
        raise ValueError("ESMFold2 confidence length does not match the complex")
    interface = None
    if pae is not None:
        forward = pae[:target_length, target_length : target_length + binder_length]
        reverse = pae[target_length : target_length + binder_length, :target_length]
        interface = mean(torch.cat([forward.flatten(), reverse.flatten()]))
    return {
        "target_length": target_length,
        "binder_length": binder_length,
        "plddt": mean(plddt),
        "target_plddt": mean(None if plddt is None else plddt[:target_length]),
        "binder_plddt": mean(
            None
            if plddt is None
            else plddt[target_length : target_length + binder_length]
        ),
        "ptm": None if result.ptm is None else float(result.ptm),
        "iptm": None if result.iptm is None else float(result.iptm),
        "interface_pae": interface,
    }


if __name__ == "__main__":
    serve(json.loads(Path(sys.argv[1]).read_text()), Esmfold2Backend)
