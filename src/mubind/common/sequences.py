"""Validate amino-acid text without loading tensor dependencies."""

AA_ORDER = "ARNDCQEGHILKMFPSTWYV"


def normalize_sequence(sequence: str) -> str:
    seq = sequence.strip().upper()
    for pos, aa in enumerate(seq, start=1):
        if aa not in AA_ORDER:
            raise ValueError(f"unsupported amino acid {aa!r} at position {pos}")
    return seq
