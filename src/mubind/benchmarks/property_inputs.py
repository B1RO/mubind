"""Sample property requests from a prepared training split."""

from pathlib import Path

from .common import PROPERTIES, read, require, sha256


SETTINGS = (("length_only", "100"), ("charge_only", "010"),
            ("hydrophobicity_only", "001"), ("all_properties", "111"))


def has_property_dataset(path):
    return path is not None and all((Path(path).expanduser() / name).is_file() for name in (
        "prostt5_diffusion.json", "train/properties.npy",
    ))


def default_property_dataset():
    """Use the prepared subset selected by setup, without preparing new data."""
    from ..preprocessing.prostt5.steps.validate import discover_datasets, is_ready_for_training
    from ..setup.config import load_config

    try:
        setup = load_config()
        source = setup.source.expanduser().resolve() if setup.source else None
        for info in discover_datasets(setup.data_dir.expanduser()):
            if is_ready_for_training(info, setup.min_length, setup.max_length, source):
                return info.path
    except (OSError, ValueError):
        pass
    return None


def inspect_property_dataset(path):
    import numpy as np

    require(has_property_dataset(path),
            "Property control needs a prepared training dataset containing prostt5_diffusion.json and train/properties.npy. Use --dataset or run mubind setup --only prostt5.")
    path = Path(path).expanduser().resolve()
    manifest = read(path / "prostt5_diffusion.json")
    require(manifest.get("state") == "complete", "Property training dataset is incomplete")
    require(manifest.get("property_definition") == "omegamp_modlamp_v1",
            "Property training dataset must use omegamp_modlamp_v1")
    train = manifest.get("splits", {}).get("train", {})
    count = train.get("rows")
    require(isinstance(count, int) and count > 0, "Property training split has no usable rows")
    values = np.load(path / "train/properties.npy", mmap_mode="r", allow_pickle=False)
    require(values.ndim == 2 and values.shape[1] == 3 and len(values) >= count
            and np.issubdtype(values.dtype, np.floating),
            "Training property array must contain length, charge and hydrophobicity for every manifest row")
    values = values[:count]  # Prepared arrays can reserve unused trailing rows.
    require(np.isfinite(values).all(), "Training properties must be finite")
    lengths = values[:, 0]
    require(np.all((lengths >= 1) & (lengths <= 70) & (lengths == np.floor(lengths))),
            "Property control requires training proteins of 1–70 residues")
    std = np.asarray(train.get("property_std"), dtype=float)
    require(std.shape == (3,) and np.isfinite(std).all() and np.all(std > 0),
            "Training property standard deviations must be three positive finite values")
    return {
        "dataset": str(path), "split": "train", "training_rows": count,
        "property_definition": manifest["property_definition"],
        "training_std": std.tolist(),
        "manifest_sha256": sha256(path / "prostt5_diffusion.json"),
        "properties_sha256": sha256(path / "train/properties.npy"),
        "sampling": "uniform_with_replacement", "requests_per_setting": 1000,
    }


def sample_property_requests(inputs):
    """Keep joint targets together by drawing complete training property rows."""
    import numpy as np

    path = Path(inputs["dataset"]) / "train/properties.npy"
    require(sha256(path) == inputs["properties_sha256"], "Training properties changed before sampling")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    rng = np.random.Generator(np.random.PCG64(inputs["seed"]))
    records = []
    for name, mask in SETTINGS:
        indices = rng.integers(inputs["training_rows"], size=inputs["requests_per_setting"])
        for number, index in enumerate(indices):
            records.append({
                "case_id": f"{name}_{number:04d}_row_{index}",
                "dataset_index": int(index), "property_conditioning": name,
                "condition_mask": mask,
                **{f"requested_{prop}": float(value)
                   for prop, value in zip(PROPERTIES, values[index])},
            })
    return records
