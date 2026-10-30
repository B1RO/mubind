"""Leave-one-residue-out ESMC scoring; runs inside the prepared ESMFold2 runtime."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def summarize_scores(records):
    """Summarize selected cohorts and the matching unshuffled control parents."""
    by_id = {row["case_id"]: row for row in records}
    groups = defaultdict(list)
    for row in records:
        groups[row["cohort"]].append(row["pseudo_perplexity"])
        if "parent_case_id" in row:
            parent = by_id[row["parent_case_id"]]
            groups[row["cohort"] + "_parents"].append(parent["pseudo_perplexity"])
    return {
        name: {
            "count": len(values),
            "median": statistics.median(values),
            "mean": statistics.mean(values),
        }
        for name, values in groups.items()
    }


def score_sequence(model, tokenizer, sequence, device, batch_size=64):
    import torch

    require(
        bool(sequence) and set(sequence) <= set("ACDEFGHIKLMNPQRSTVWY"),
        "Invalid protein sequence",
    )
    with torch.inference_mode():
        encoded = tokenizer(
            sequence, return_tensors="pt", return_special_tokens_mask=True
        )
        ids = encoded["input_ids"].to(device)
        attention = encoded["attention_mask"].to(device)
        special = encoded["special_tokens_mask"].to(device).bool()
        positions = ((attention > 0) & ~special)[0].nonzero(as_tuple=False).flatten()
        require(
            positions.tolist() == list(range(1, len(sequence) + 1)),
            "Unexpected tokenizer residue positions",
        )
        log_probabilities = []
        for start in range(0, len(sequence), batch_size):
            chunk = positions[start : start + batch_size]
            masked = ids.expand(len(chunk), -1).clone()
            row = torch.arange(len(chunk), device=device)
            masked[row, chunk] = tokenizer.mask_token_id
            require(
                bool((masked.eq(tokenizer.mask_token_id).sum(dim=1) == 1).all()),
                "Each input must mask exactly one residue",
            )
            logits = model(
                input_ids=masked,
                attention_mask=attention.expand(len(chunk), -1),
                compute_sae=False,
            ).logits
            require(
                logits.shape[-1] == 64,
                "The paper scores the complete 64-token ESMC vocabulary",
            )
            logp = torch.log_softmax(logits[row, chunk].float(), dim=-1)
            log_probabilities.extend(logp[row, ids[0, chunk]].cpu().tolist())
        mean = math.fsum(log_probabilities) / len(sequence)
        require(
            math.isfinite(mean) and mean <= 0,
            "Invalid masked-language-model probabilities",
        )
        return {
            "sequence_length": len(sequence),
            "mean_log_likelihood": mean,
            "pseudo_perplexity": math.exp(-mean),
            "per_residue_log_probability": log_probabilities,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress", type=Path)
    args = parser.parse_args()
    import torch
    import transformers
    from transformers import ESMCForMaskedLM, ESMCTokenizer

    if args.model.name != "45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a":
        raise ValueError("Wrong ESMC revision")
    torch.set_num_threads(8)
    torch.manual_seed(20261006)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable for ESMC scoring")
        torch.cuda.manual_seed_all(20261006)
    cases = json.loads(args.cases.read_text())

    def report(description, completed=None):
        if args.progress is not None:
            event = {"stage": "esmc", "description": description}
            if completed is not None:
                event.update(completed=completed, total=len(cases),
                             detail=f"{completed:,}/{len(cases):,} sequences", minimum=1)
            with args.progress.open("a") as events:
                events.write(json.dumps(event) + "\n")

    if len({r["case_id"] for r in cases}) != len(cases):
        raise ValueError("Duplicate input cases")
    protocol = {
        "model": "biohub/ESMC-6B",
        "revision": args.model.name,
        "precision": "bfloat16",
        "mask_batch_size": 64,
        "attention_implementation": "sdpa",
        "probability_vocabulary": "All model output tokens, without amino-acid renormalization",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": args.device,
    }
    records = args.out / "scores.jsonl"
    scored = []
    # The parent command supplies a fresh folder. Exclusive creation also keeps
    # direct worker invocations from replacing previous scores.
    with records.open("x", buffering=1) as log:
        save(args.out / "protocol.json", protocol)
        report("Loading ESMC-6B…")
        print("Loading ESMC-6B", flush=True)
        model, loading = ESMCForMaskedLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
            output_loading_info=True,
        )
        if any(
            loading.get(key)
            for key in (
                "missing_keys",
                "unexpected_keys",
                "mismatched_keys",
                "error_msgs",
            )
        ):
            raise ValueError(f"Incomplete ESMC weight loading: {loading}")
        expected_parameters = json.loads(
            (args.model / "model.safetensors.index.json").read_text()
        )["metadata"]["total_parameters"]
        if sum(p.numel() for p in model.parameters()) != expected_parameters:
            raise ValueError("ESMC parameter count mismatch")
        model = model.to(torch.device(args.device)).eval().requires_grad_(False)
        tokenizer = ESMCTokenizer()
        if (
            tokenizer.cls_token_id,
            tokenizer.pad_token_id,
            tokenizer.eos_token_id,
            tokenizer.mask_token_id,
        ) != (0, 1, 2, 32):
            raise ValueError("Unexpected ESMC tokenizer")
        random.Random(20261006).shuffle(cases)
        report("Scoring sequences", 0)
        for number, case in enumerate(cases, 1):
            started = time.perf_counter()
            row = {
                **case,
                **score_sequence(model, tokenizer, case["sequence"], torch.device(args.device)),
                "scoring_seconds": time.perf_counter() - started,
            }
            log.write(json.dumps(row, allow_nan=False) + "\n")
            scored.append(row)
            report("Scoring sequences", number)
            print(f"ESMC {number}/{len(cases)} {case['case_id']}: {row['pseudo_perplexity']:.4f}", flush=True)
    report("Saving ESMC comparisons…")
    save(args.out / "summary.json", summarize_scores(scored))


if __name__ == "__main__":
    main()
