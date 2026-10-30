"""Evaluate a selected checkpoint and process its generated candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..paths import existing_artifact
from ..benchmarks.common import MODES, verify_bundle


def build_parser():
    parser = argparse.ArgumentParser(prog="mubind evaluate", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run", help="Evaluate your checkpoint through generation, folding and scoring"
    )
    run.add_argument("mode", choices=MODES)
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True, help="New run folder; existing paths get a numbered sibling")
    run.add_argument(
        '--complexes', type=Path,
        help='PDB folder for full-design or inpainting; first protein chain is binder, second is target (default: 45 bundled PPIRef complexes)',
    )
    run.add_argument(
        "--dataset", type=Path,
        help="Prepared training dataset for property control (default: setup dataset), or original paper corpus for unconditional novelty"
    )
    run.add_argument("--seed", type=int, help="Property-control sampling and generation seed; each run chooses a random seed when omitted")
    run.add_argument("--device", default="auto")
    run.add_argument("--endpoint", default="auto")
    run.add_argument("--models-dir", type=Path)
    run.add_argument("--foldseek", default="foldseek")
    run.add_argument(
        "--figures", action="store_true", help="Save property-control figures"
    )
    metrics = run.add_mutually_exclusive_group()
    metrics.add_argument(
        "--esmc",
        action="store_true",
        help="Also compare unconditional sequences using ESMC",
    )
    from ..benchmarks.unconditional import COMPUTATIONS

    metrics.add_argument(
        "--metrics", nargs="+", choices=tuple(COMPUTATIONS),
        help="Run only these section 4.1 computations; shuffle controls include parent scores",
    )
    verify = commands.add_parser(
        "verify", help="Check bundled benchmark input hashes"
    )
    fold = commands.add_parser(
        "fold", help="Fold and score a regenerated structural candidate pool"
    )
    fold.add_argument("--run", type=Path, required=True)
    fold.add_argument("--out", type=Path, help="Fresh folding output (default: RUN/folding, numbered if it exists)")
    fold.add_argument("--endpoint", default="auto")
    fold.add_argument("--models-dir", type=Path)
    fold.add_argument("--foldseek", default="foldseek")
    fold.add_argument("--device", default="auto")
    score = commands.add_parser("score", help="Summarize a regenerated candidate pool")
    score.add_argument("--run", type=Path, required=True)
    score.add_argument("--out", type=Path, required=True, help="New summary folder; existing paths get a numbered sibling")
    score.add_argument("--folds", type=Path, help="Structural scores folder returned by evaluate fold")
    score.add_argument("--dataset", type=Path)
    esmc = commands.add_parser(
        "esmc", help="Score generated sequences and fixed references with ESMC-6B"
    )
    esmc.add_argument("--out", type=Path, required=True, help="New ESMC output folder; existing paths get a numbered sibling")
    esmc.add_argument("--models-dir", type=Path)
    esmc.add_argument(
        "--run",
        type=Path,
        required=True,
        help="Generated unconditional samples, compared with the fixed references",
    )
    esmc.add_argument("--device", default="cuda")
    return parser, {
        "run": run,
        "verify": verify,
        "fold": fold,
        "score": score,
        "esmc": esmc,
    }


def main(argv=None):
    parser, _ = build_parser()
    args = parser.parse_args(argv)
    for name in ("checkpoint", "run", "folds"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(args, name, existing_artifact(value))
    if args.command == "run":
        from rich.console import Console
        from ..benchmarks.workflow import EvaluationConfig
        from .evaluate_progress import run_with_progress

        return run_with_progress(
            EvaluationConfig(
                **{key: value for key, value in vars(args).items() if key != "command"}
            ),
            Console(),
        )
    return execute(args)


def execute(args):
    if args.command == "verify":
        result = verify_bundle()
    elif args.command == "fold":
        from ..benchmarks.folding import fold_run

        result = fold_run(
            args.run, args.endpoint, args.models_dir, args.foldseek, args.device, out=args.out
        )
        print(f"Folds saved to {result['directory']}")
    elif args.command == "score":
        from ..benchmarks.reports import score_run

        result = score_run(args.run, args.out, args.dataset, folds=args.folds)
    elif args.command == "esmc":
        from ..benchmarks.esmc import run_esmc

        result = run_esmc(args.out, args.models_dir, args.run, args.device)
        print(f"ESMC scores saved to {result}")
    if args.command == "verify":
        print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
