"""Download and prepare the ProstT5 dataset for diffusion training.

Install the optional dependencies with ``pip install -e '.[preprocessing]'``.
The Python workflow for the 50–70-residue ProstT5 subset is::

    from mubind.preprocessing.prostt5.steps.download import download_prostt5
    from mubind.preprocessing.prostt5.steps.subset import subset_prostt5
    from mubind.preprocessing.prostt5.steps.prepare import prepare_prostt5

    download_prostt5("artifacts/datasets/prostt5_raw")
    subset_prostt5(
        "artifacts/datasets/prostt5_raw", "artifacts/datasets/prostt5_len50_70",
        min_length=50, max_length=70,
    )
    prepare_prostt5("artifacts/datasets/prostt5_len50_70", "artifacts/datasets/prostt5_prepared")

The same operations are available as module commands::

    python -m mubind.preprocessing.prostt5.steps.download --out datasets/prostt5_raw
    python -m mubind.preprocessing.prostt5.steps.subset \
        --source datasets/prostt5_raw --out datasets/prostt5_len50_70 \
        --min-length 50 --max-length 70
    python -m mubind.preprocessing.prostt5.steps.prepare \
        --source datasets/prostt5_len50_70 --out datasets/prostt5_prepared

Preparation pads to 128 positions by default, matching the existing format.
``mubind.dataset.ProstT5Dataset`` can load the resulting directory with
``max_length=70``. Sequence filtering never truncates a protein.

Downloads can be resumed in the same directory. Subsetting and preparation
require new output directories. No work runs on import. Reference PDB recovery
and reference-index remapping are separate and are not included here.
"""
