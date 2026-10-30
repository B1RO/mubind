"""Compare generated sequences with fixed references using isolated ESMC scoring."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

from .common import (
    DATA,
    ROOT,
    prostt5_targets,
    new_run,
    read,
    require,
    save,
)

ESMC_REVISION = "45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a"


def build_cases(run, *, baselines=True, shuffles=True):
    require(baselines or shuffles, "Choose ESMC pseudo-perplexity or shuffle controls")
    require(run is not None, "ESMC needs --run with generated unconditional samples")
    samples = read(run / "samples.json")
    require(
        samples["mode"] == "unconditional",
        "ESMC --run must contain unconditional samples",
    )
    from .metrics import valid_canvas

    require(len(samples["records"]) == 1024, "Unconditional protocol requires all 1,024 attempts")
    require(len({r["case_id"] for r in samples["records"]}) == 1024, "Repeated unconditional candidate ID")
    generated = [
        {"cohort": "generated", "case_id": r["case_id"], "sequence": r["generated"]}
        for r in samples["records"]
        if valid_canvas(r)
    ]
    generated = sorted(generated, key=lambda r: r["case_id"])
    references = sorted(
        [{"cohort": "test", "case_id": r["esmc_case_id"], "sequence": r["native_sequence"]}
         for r in prostt5_targets()], key=lambda r: r["case_id"]
    )
    ppiref = [{"cohort": "ppiref", **r} for r in read(DATA / "ppiref/sequences.json")]
    require(generated, "ESMC needs at least one valid generated sequence")
    if shuffles:
        require(
            len(generated) >= 128,
            "The shuffle comparison needs 128 valid generated sequences",
        )
        selected = random.Random(202610071).sample(generated, 128)
    # Controls alone need scores only for the fixed 128 generated parents,
    # plus the same reference parents. Their identities and shuffles are unchanged.
    cases = [
        {key: row[key] for key in ("cohort", "case_id", "sequence")}
        for row in (generated if baselines else selected) + references + ppiref
    ]
    if not shuffles:
        return cases
    for cohort, rows in (
        ("generated", selected),
        ("prostt5", references),
        ("ppiref", ppiref),
    ):
        for row in rows:
            seed = int(
                hashlib.sha256(
                    (cohort + row["case_id"] + "20261007").encode()
                ).hexdigest()[:16],
                16,
            )
            middle = list(row["sequence"][1:-1])
            random.Random(seed).shuffle(middle)
            sequence = row["sequence"][0] + "".join(middle) + row["sequence"][-1]
            cases.append(
                {
                    "cohort": cohort + "_shuffled",
                    "case_id": cohort + "_shuffled_" + row["case_id"],
                    "sequence": sequence,
                    "parent_case_id": row["case_id"],
                    "shuffle_seed": seed,
                }
            )
    require(len({r["case_id"] for r in cases}) == len(cases), "Duplicate ESMC case IDs")
    return cases


def run_worker_with_progress(command, out, environment, on_progress):
    """Keep worker diagnostics in a log and stream its measured progress events."""
    progress_path, log_path = out / "progress.jsonl", out / "worker.log"
    progress_path.write_text("")
    with log_path.open("a") as log, progress_path.open() as events:
        process = subprocess.Popen(
            [*command, "--progress", str(progress_path)],
            env=environment, stdout=log, stderr=subprocess.STDOUT,
        )

        def drain():
            while True:
                offset = events.tell()
                line = events.readline()
                if not line:
                    return
                if not line.endswith("\n"):
                    events.seek(offset)
                    return
                on_progress(json.loads(line))

        try:
            while process.poll() is None:
                drain()
                time.sleep(0.1)
            drain()
        finally:
            # This worker belongs to this evaluation; keep no orphan after Ctrl+C.
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if process.returncode:
            with log_path.open("rb") as failure:
                failure.seek(max(0, log_path.stat().st_size - 4000))
                detail = failure.read().decode(errors="replace").strip()
            raise RuntimeError(f"ESMC scoring failed · see {log_path}\n{detail}")


def run_esmc(out, models_dir=None, run=None, device="cuda", *, baselines=True, shuffles=True, on_progress=None, on_status=None):
    from ..runtime import get_runtime

    cases = build_cases(run, baselines=baselines, shuffles=shuffles)
    runtime = get_runtime("esmfold2", models_dir=models_dir)
    model = Path(runtime.paths["esmc_model"])
    require(
        model.name == ESMC_REVISION,
        "ESMC runtime does not point to the paper's pinned snapshot",
    )
    out = new_run(
        out,
        {
            "operation": "esmc",
            "baselines": baselines,
            "shuffles": shuffles,
            "samples": str((run / "samples.json").resolve()),
            "revision": ESMC_REVISION,
            "device": device,
        },
    )
    save(out / "cases.json", cases)
    command = [
            str(runtime.python),
            str(ROOT / "esmc_worker.py"),
            "--model",
            str(model),
            "--cases",
            str(out / "cases.json"),
            "--out",
            str(out),
            "--device",
            device,
        ]
    if on_progress is not None:
        if on_status is not None:
            on_status("Starting ESMC sequence scoring…")
        run_worker_with_progress(command, out, runtime.environment(), on_progress)
    else:
        subprocess.run(command, env=runtime.environment(), check=True)
    return out
