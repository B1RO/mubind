"""Selectable computations from paper section 4.1 and their prerequisites."""

from .common import require


# Order follows the sequence analyses, distribution comparisons and model scores.
COMPUTATIONS = {
    "validity": ("Sequence validity", "Checks for empty outputs or gaps inside a sequence."),
    "distinctness": ("Distinct sequences", "Counts unique sequences; identical copies count only once."),
    "diversity": ("Pairwise diversity", "Measures how different the generated sequences are from one another."),
    "exact-novelty": ("Exact novelty", "Counts sequences that do not appear in the training data. Requires the training dataset."),
    "nearest-training": ("Nearest-training distance", "Measures how much each sequence differs from its closest training example. Requires the training dataset."),
    "length": ("Length distribution", "Compares the lengths of generated sequences with those of real proteins."),
    "charge": ("Net-charge distribution", "Compares the charges of generated sequences with those of real proteins."),
    "hydrophobicity": ("Hydrophobicity distribution", "Compares hydrophobicity values between generated sequences and real proteins."),
    "composition": ("Amino-acid composition", "Compares how often each amino acid appears in generated sequences and real proteins."),
    "esmc": ("ESMC sequence plausibility", "Uses the ESMC protein language model to assess how plausible the sequences are."),
    "esmc-shuffle": ("ESMC shuffle comparison", "Compares ESMC scores before and after shuffling the amino-acid order."),
}
CORPUS_METRICS = frozenset({"exact-novelty", "nearest-training"})
ESMC_METRICS = frozenset({"esmc", "esmc-shuffle"})
DEFAULT_METRICS = tuple(key for key in COMPUTATIONS if key not in ESMC_METRICS)


def selected_metrics(metrics=None, esmc=False):
    """Keep legacy CLI defaults; an explicit selection is authoritative."""
    chosen = set(metrics if metrics is not None else (
        tuple(COMPUTATIONS) if esmc else DEFAULT_METRICS
    ))
    require(not chosen.difference(COMPUTATIONS), f"Unknown unconditional metrics: {sorted(chosen.difference(COMPUTATIONS))}")
    return tuple(key for key in COMPUTATIONS if key in chosen)
