"""Binder edit-region parsing, seed masking, and post-decoding verification for inpainting."""

from __future__ import annotations

import re


def parse_edit_residues(value: str, *, length: int = 70) -> tuple[int, ...]:
    """Parse 1-based inclusive binder residue ranges (such as `'12-20,35-40'`), merging overlaps."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("--edit-residues needs binder positions such as 12-20,35-40")
    positions = set()
    for part in value.split(","):
        match = re.fullmatch(r"([0-9]+)(?:\s*-\s*([0-9]+))?", part.strip())
        if match is None:
            raise ValueError(
                "--edit-residues accepts positions and ranges such as 12-20,35-40"
            )
        start = int(match[1])
        end = int(match[2]) if match[2] is not None else start
        if not 1 <= start <= end <= length:
            raise ValueError(
                f"--edit-residues must use ascending ranges within binder positions 1-{length}"
            )
        positions.update(range(start, end + 1))
    return tuple(sorted(positions))


def inpainting_spec(sequence: str, edit_residues: str) -> dict:
    """Build a binary edit mask string (`'1'` at editable positions, `'0'` at fixed positions)."""
    positions = set(parse_edit_residues(edit_residues, length=len(sequence)))
    return {
        "source_sequence": sequence,
        "edit_mask": "".join(
            "1" if i in positions else "0" for i in range(1, len(sequence) + 1)
        ),
    }


def inpainting_seed(spec: dict, threedi: str, count: int, *, normalized_embeddings: bool) -> dict:
    """Embed `source_sequence` and zero out all seed channels at editable positions."""
    import torch

    from ...dataset.embed import embed_diffusion_sequence

    source = embed_diffusion_sequence(
        spec["source_sequence"],
        threedi,
        pad_to=70,
        normalized=normalized_embeddings,
        joint_3di=True,
    ).T.contiguous()
    edit_mask = torch.zeros(70, dtype=torch.bool)
    edit_mask[: len(spec["edit_mask"])] = torch.tensor(
        [c == "1" for c in spec["edit_mask"]]
    )
    source[:, edit_mask] = 0
    return {
        "seed_x": source.expand(count, -1, -1).clone(),
        "edit_mask": edit_mask.expand(count, -1).clone(),
    }


def check_inpainting(decoded: dict, spec: dict) -> dict:
    """Verify that unedited residues and padding layout were preserved without patching outputs."""
    sequence, mask = spec["source_sequence"], spec["edit_mask"]
    canvas = decoded["decoded_canvas"]
    observed = canvas[: len(sequence)].ljust(len(sequence), "-")
    retained_changes = [
        i + 1
        for i, (before, after, edit) in enumerate(
            zip(sequence, observed, mask, strict=True)
        )
        if edit == "0" and before != after
    ]
    edited_changes = sum(
        before != after
        for before, after, edit in zip(sequence, observed, mask, strict=True)
        if edit == "1"
    )
    layout = (
        len(canvas) == 70
        and "-" not in canvas[: len(sequence)]
        and set(canvas[len(sequence) :]) <= {"-"}
    )
    errors = []
    if retained_changes:
        errors.append(
            "Changed retained binder positions " + ",".join(map(str, retained_changes))
        )
    if not layout:
        errors.append("Changed the required sequence length or padding layout")
    return {
        **spec,
        "fixed_residues_preserved": not retained_changes,
        "fixed_layout_preserved": layout,
        "completion_specification_preserved": not retained_changes and layout,
        "retained_changes_count": len(retained_changes),
        "edited_changes_count": edited_changes,
        "inpainting_error": "; ".join(errors),
    }
