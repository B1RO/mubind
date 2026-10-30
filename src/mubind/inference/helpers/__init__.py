"""Checkpoint loading, structural guidance, inpainting, targets, and binder-design stage helpers."""

from __future__ import annotations

_EXPORTS = {
    "InferenceCheckpoint": (".checkpoint", "InferenceCheckpoint"),
    "load_inference_checkpoint": (".checkpoint", "load_inference_checkpoint"),
    "validate_design_checkpoint": (".checkpoint", "validate_design_checkpoint"),
    "check_inpainting": (".inpainting", "check_inpainting"),
    "inpainting_seed": (".inpainting", "inpainting_seed"),
    "inpainting_spec": (".inpainting", "inpainting_spec"),
    "parse_edit_residues": (".inpainting", "parse_edit_residues"),
    "export_results": (".redesign", "export_results"),
    "fold_and_rank_candidates": (".redesign", "fold_and_rank_candidates"),
    "redesign_batches": (".redesign", "redesign_batches"),
    "redesign_seed": (".redesign", "redesign_seed"),
    "redesign_tasks": (".redesign", "redesign_tasks"),
    "run_redesign_round": (".redesign", "run_redesign_round"),
    "confidence": (".scaffolds", "confidence"),
    "describe_scaffolds": (".scaffolds", "describe_scaffolds"),
    "encode_scaffolds": (".scaffolds", "encode_scaffolds"),
    "fold_pair": (".scaffolds", "fold_pair"),
    "interactive_select_scaffolds": (".scaffolds", "interactive_select_scaffolds"),
    "rank_scaffolds": (".scaffolds", "rank_scaffolds"),
    "ranking_score": (".scaffolds", "ranking_score"),
    "requested_properties": (".scaffolds", "requested_properties"),
    "select_scaffolds": (".scaffolds", "select_scaffolds"),
    "stage_progress": (".scaffolds", "stage_progress"),
    "write_json": (".scaffolds", "write_json"),
    "StructuralGuidance": (".structural_guidance", "StructuralGuidance"),
    "Target": (".targets", "Target"),
    "chain_sequence": (".targets", "chain_sequence"),
    "discover_targets": (".targets", "discover_targets"),
    "first_model": (".targets", "first_model"),
    "inspect_target": (".targets", "inspect_target"),
    "write_chain": (".targets", "write_chain"),
}


def __getattr__(name: str):
    if name in _EXPORTS:
        from importlib import import_module

        module_name, attr = _EXPORTS[name]
        return getattr(import_module(module_name, __name__), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = sorted(_EXPORTS)
