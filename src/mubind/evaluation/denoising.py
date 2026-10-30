"""Repeated denoising and conditioning diagnostics from the 300k run."""

from __future__ import annotations

import itertools
import math
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..common.properties import (
    DEFAULT_PROPERTY_DEFINITION,
    residue_property_scales,
)
from ..dataset.decode import (
    biochemical_embedding_dim,
    biochemical_features,
    nearest_foldseek_3di,
)
from ..dataset.embed import (
    AA_ORDER,
    EMBEDDING_DIM,
    JOINT_EMBEDDING_DIM,
    OMEGAMP_AA_SCALE,
    normalize_residue_vectors,
)
from ..diffusion import (
    masked_mse_loss,
    model_output_to_x0,
    residue_pad_balanced_mse_loss,
    training_target,
)
from ..inference.predict import autocast_context, inference_context, model_inputs
from ..model import PropertyInput, SeedInput, StructureInput
from ..training.tasks import TaskCollator, TaskConfig
from ..training.tasks.config import LAYOUT_NAMES
from .config import DenoisingConfig

PROPERTY_NAMES = ("length", "charge", "hydrophobicity")
LENGTH_STAT_NAMES = (
    "true_mean",
    "generated_mean",
    "valid_fraction",
    "too_short_fraction",
    "too_long_fraction",
    "exact_fraction",
    "zero_fraction",
    "max_fraction",
)


def evaluation_tasks():
    return TaskConfig(
        layout_probabilities=(0, 0, 0, 0, 1),
        redesign_conditioning_probabilities=(0.05, 0.15, 0.2, 0.6),
        terminal_pad_min=0,
        terminal_pad_max=0,
        global_offset_probability=0,
        left_align_threedi_redesign=True,
    )


@torch.inference_mode()
def evaluate_denoising(
    model,
    diffusion,
    dataset,
    training_config,
    settings: DenoisingConfig,
    *,
    device,
    utilization_batches=None,
):
    """Use a separate validation task mixture and repeat corruptions exactly."""
    loader = DataLoader(
        dataset,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=TaskCollator(evaluation_tasks(), training=False),
    )
    losses, gains = [], []
    buckets = torch.zeros(settings.noise_bins, dtype=torch.float64, device=device)
    denominators = torch.zeros_like(buckets)
    errors = dict.fromkeys(PROPERTY_NAMES, 0.0)
    lengths = dict.fromkeys(LENGTH_STAT_NAMES, 0.0)
    joint_errors = dict.fromkeys(("biochemical", "threedi"), 0.0)
    joint_denoms = dict(joint_errors)
    correct = tokens = count = 0
    layouts = torch.zeros(len(LAYOUT_NAMES), dtype=torch.long)
    tables = validation_decode_tables(
        embedding_dim=training_config.backbone.x_channels,
        normalized_embeddings=training_config.data.normalize_embeddings,
        device=device,
        property_definition=training_config.data.property_definition,
    )
    with inference_context(model, device, seed=settings.seed):
        for _ in range(settings.noise_samples):
            for batch in itertools.islice(loader, settings.batches):
                b = move_batch(batch, device)
                layouts += torch.bincount(
                    batch["layout_id"], minlength=len(LAYOUT_NAMES)
                )
                t = diffusion.sample_timesteps(len(b["x"]), device)
                noisy, noise = diffusion.add_noise(b["x"], t)
                inputs = model_inputs(b, device)
                with autocast_context(device, training_config.precision):
                    hint = None
                    if training_config.self_condition.enabled:
                        first = model(noisy, t, **inputs)
                        hint = model_output_to_x0(diffusion, noisy, t, first)
                    output = model(noisy, t, **inputs, self_cond_x=hint)
                    target = training_target(diffusion, b["x"], noise, t)
                    reconstruction = training_config.reconstruction
                    loss = residue_pad_balanced_mse_loss(
                        diffusion,
                        output,
                        target,
                        b["edit_mask"],
                        b["residue_mask"],
                        timesteps=t,
                        pad_weighting=reconstruction.pad_weighting,
                        pad_weight=reconstruction.pad_weight,
                        example_weights=b["target_length"]
                        if reconstruction.example_weighting == "target_length"
                        else None,
                    )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non-finite denoising validation loss")
                pred = model_output_to_x0(diffusion, noisy, t, output)
                losses.append(float(loss))
                clean_loss = masked_mse_loss(diffusion, pred, b["x"], b["edit_mask"])
                gains.append(
                    float(
                        1
                        - clean_loss
                        / masked_mse(noisy, b["x"], b["edit_mask"]).clamp_min(1e-8)
                    )
                )
                sums, denoms = noise_bucket_loss_sums(
                    output,
                    target,
                    torch.ones_like(b["edit_mask"]),
                    t,
                    total_timesteps=diffusion.timesteps,
                    num_buckets=settings.noise_bins,
                )
                buckets += sums
                denominators += denoms
                e, l, n = decoded_validation_sums(
                    pred,
                    b["c"],
                    normalized_embeddings=training_config.data.normalize_embeddings,
                    property_definition=training_config.data.property_definition,
                    decode_tables=tables,
                )
                count += n
                for key in errors:
                    errors[key] += e[key]
                for key in lengths:
                    lengths[key] += l[key]
                if pred.shape[1] == JOINT_EMBEDDING_DIM:
                    for name, channels in (
                        ("biochemical", slice(0, 6)),
                        ("threedi", slice(6, 8)),
                    ):
                        sums, denoms = masked_mse_sample_sums(
                            pred[:, channels].float(),
                            b["x"][:, channels].float(),
                            b["edit_mask"],
                        )
                        joint_errors[name] += float(sums.sum())
                        joint_denoms[name] += float(denoms.sum())
                    selected = b["edit_mask"] & b["residue_mask"].bool()
                    predicted = nearest_foldseek_3di(pred[:, 6:].transpose(1, 2))
                    actual = nearest_foldseek_3di(b["x"][:, 6:].transpose(1, 2))
                    correct += int(((predicted == actual) & selected).sum())
                    tokens += int(selected.sum())
        if not losses:
            raise ValueError("Denoising validation has no examples")
        result = {
            "loss": sum(losses) / len(losses),
            "denoising_gain": sum(gains) / len(gains),
        }
        result.update(
            {f"condition_{key}_mae": value / count for key, value in errors.items()}
        )
        result.update(
            {f"length/{key}": value / count for key, value in lengths.items()}
        )
        result.update(
            noise_bucket_loss_metrics(
                buckets, denominators, noise_bucket_labels(settings.noise_bins)
            )
        )
        result.update(
            {
                f"layout/{name}_fraction": float(layouts[i]) / int(layouts.sum())
                for i, name in enumerate(LAYOUT_NAMES)
            }
        )
        if tokens:
            result.update(
                {
                    f"joint/{name}_x0_mse": joint_errors[name]
                    / max(joint_denoms[name], 1)
                    for name in joint_errors
                }
            )
            result["joint/threedi_token_accuracy"] = correct / tokens
            result["joint/threedi_token_count"] = tokens
        if utilization_batches is not None:
            result.update(
                validate_threedi_utilization(
                    model,
                    diffusion,
                    utilization_batches,
                    device,
                    training_config.precision,
                    noise_seed=settings.utilization_noise_seed,
                    high_noise_start=settings.high_noise_start,
                )
            )
    return result


@torch.inference_mode()
def inpainting_denoising(
    model, diffusion, batches, *, device, seed, repeats, self_condition, precision
):
    """Continuous x0 error in the missing and retained regions of fixed cases."""
    totals = {name: [0.0, 0] for name in ("missing", "retained")}
    with inference_context(model, device, seed=seed):
        for _ in range(repeats):
            for source in batches:
                batch = move_batch(source, device)
                t = diffusion.sample_timesteps(len(batch["x"]), device)
                noisy, _ = diffusion.add_noise(batch["x"], t)
                inputs = model_inputs(batch, device)
                with autocast_context(device, precision):
                    prediction = model(noisy, t, **inputs)
                    if self_condition:
                        hint = model_output_to_x0(diffusion, noisy, t, prediction)
                        prediction = model(noisy, t, **inputs, self_cond_x=hint)
                    clean = model_output_to_x0(diffusion, noisy, t, prediction)
                errors = (clean.float() - batch["x"].float()).square().mean(1)
                for name, selected in (
                    ("missing", batch["residue_mask"].bool() & batch["edit_mask"]),
                    ("retained", batch["residue_mask"].bool() & ~batch["edit_mask"]),
                ):
                    totals[name][0] += float(errors[selected].sum())
                    totals[name][1] += int(selected.sum())
    return {
        name + "_residue_mse": total / count if count else None
        for name, (total, count) in totals.items()
    }


def move_batch(batch, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    edit_mask: torch.Tensor,
) -> torch.Tensor:
    sq = (pred - target) ** 2
    mask_f = edit_mask[:, None, :].to(dtype=sq.dtype)
    return (sq * mask_f).sum() / (mask_f.sum() * sq.shape[1]).clamp_min(1.0)


def masked_mse_sample_sums(
    pred: torch.Tensor,
    target: torch.Tensor,
    edit_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    sq = (pred - target) ** 2
    mask_f = edit_mask[:, None, :].to(dtype=sq.dtype)
    loss_sums = (sq * mask_f).sum(dim=(1, 2))
    denoms = mask_f.sum(dim=(1, 2)) * sq.shape[1]
    return loss_sums, denoms


def noise_bucket_labels(num_buckets: int) -> tuple[str, ...]:
    if num_buckets <= 0:
        raise ValueError("noise_loss_bins must be positive")
    return tuple(
        f"noise_{round(idx * 100 / num_buckets):02d}_{round((idx + 1) * 100 / num_buckets):02d}"
        for idx in range(num_buckets)
    )


def noise_bucket_loss_sums(
    pred: torch.Tensor,
    target: torch.Tensor,
    edit_mask: torch.Tensor,
    timesteps: torch.Tensor,
    *,
    total_timesteps: int,
    num_buckets: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    loss_sums, denoms = masked_mse_sample_sums(pred, target, edit_mask)
    bucket_idx = (timesteps * num_buckets // max(total_timesteps, 1)).clamp(
        max=num_buckets - 1
    )

    # Autocast can produce BF16 predictions against an FP32 diffusion target;
    # the subtraction promotes the squared error and its reductions to FP32.
    # scatter_add_ requires its source and destination dtypes to agree.
    bucket_loss_sums = torch.zeros(
        num_buckets,
        device=pred.device,
        dtype=loss_sums.dtype,
    )
    bucket_denoms = torch.zeros(
        num_buckets,
        device=pred.device,
        dtype=denoms.dtype,
    )
    bucket_loss_sums.scatter_add_(0, bucket_idx, loss_sums)
    bucket_denoms.scatter_add_(0, bucket_idx, denoms)
    # Keep these tiny accumulators on the prediction device. Copying them to
    # CPU here forced the entire CUDA stream to synchronize on every step.
    return (
        bucket_loss_sums.detach().to(dtype=torch.float64),
        bucket_denoms.detach().to(dtype=torch.float64),
    )


def noise_bucket_loss_metrics(
    loss_sums: torch.Tensor,
    denoms: torch.Tensor,
    labels: tuple[str, ...],
) -> dict[str, float]:
    # One transfer at the logging boundary instead of one per training step.
    loss_sums = loss_sums.detach().cpu()
    denoms = denoms.detach().cpu()
    metrics: dict[str, float] = {}
    for idx, label in enumerate(labels):
        if float(denoms[idx].item()) > 0.0:
            metrics[f"loss_{label}"] = float((loss_sums[idx] / denoms[idx]).item())
    return metrics


def validation_decode_tables(
    *,
    embedding_dim: int,
    normalized_embeddings: bool,
    device: torch.device,
    property_definition: str = DEFAULT_PROPERTY_DEFINITION,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    embedding_dim = biochemical_embedding_dim(embedding_dim)

    residue_vectors = torch.tensor(
        [OMEGAMP_AA_SCALE[aa] for aa in AA_ORDER],
        dtype=torch.float32,
        device=device,
    )
    if normalized_embeddings:
        residue_vectors = normalize_residue_vectors(residue_vectors)
    if embedding_dim == EMBEDDING_DIM:
        residue_vectors = torch.cat(
            [
                residue_vectors,
                torch.ones((len(AA_ORDER), 1), dtype=torch.float32, device=device),
            ],
            dim=1,
        )
    residue_vectors = torch.cat(
        [
            residue_vectors,
            torch.zeros((1, embedding_dim), dtype=torch.float32, device=device),
        ],
        dim=0,
    )

    charge_scale, hydrophobicity_scale, terminal_charge = residue_property_scales(
        property_definition
    )
    charges = torch.tensor(
        [charge_scale[aa] for aa in AA_ORDER] + [0.0],
        dtype=torch.float32,
        device=device,
    )
    hydrophobicities = torch.tensor(
        [hydrophobicity_scale[aa] for aa in AA_ORDER] + [0.0],
        dtype=torch.float32,
        device=device,
    )
    terminal_charge_tensor = torch.tensor(
        terminal_charge,
        dtype=torch.float32,
        device=device,
    )
    return residue_vectors, charges, hydrophobicities, terminal_charge_tensor


def decoded_validation_sums(
    pred: torch.Tensor,
    targets: torch.Tensor,
    *,
    normalized_embeddings: bool,
    property_definition: str = DEFAULT_PROPERTY_DEFINITION,
    decode_tables: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
    | None = None,
) -> tuple[dict[str, float], dict[str, float], int]:
    pred_float = biochemical_features(pred.detach().float())
    targets_float = targets.detach().float()
    if decode_tables is None:
        decode_tables = validation_decode_tables(
            embedding_dim=int(pred_float.shape[1]),
            normalized_embeddings=normalized_embeddings,
            device=pred_float.device,
            property_definition=property_definition,
        )
    residue_vectors, charges, hydrophobicities, terminal_charge = decode_tables

    # Decode every residue in the batch at once. Squared Euclidean distance has
    # the same nearest neighbour as torch.cdist, without copying to CPU or
    # launching one distance calculation per sequence.
    residues = pred_float.transpose(1, 2)
    squared_distances = (
        residues.square().sum(dim=-1, keepdim=True)
        + residue_vectors.square().sum(dim=-1)
        - 2.0 * torch.matmul(residues, residue_vectors.transpose(0, 1))
    )
    residue_ids = squared_distances.argmin(dim=-1)

    pad_id = len(AA_ORDER)
    # Tasks may place a protein anywhere on the canvas. Fixed outer PAD is
    # restored before this metric, so count residues without assuming offset 0.
    generated_residues = residue_ids != pad_id
    generated_lengths = generated_residues.sum(dim=1).to(dtype=torch.float32)
    # PAD* AA+ PAD* has exactly one transition into a residue block.
    block_starts = generated_residues[:, 0].to(dtype=torch.int64)
    if generated_residues.shape[1] > 1:
        block_starts = block_starts + (
            generated_residues[:, 1:] & ~generated_residues[:, :-1]
        ).sum(dim=1)
    valid_sequences = block_starts == 1
    predicted_charges = (charges[residue_ids] * generated_residues).sum(dim=1)
    predicted_charges += terminal_charge * (generated_lengths > 0)
    predicted_charges = torch.round(predicted_charges * 1000.0) / 1000.0
    hydrophobicity_sums = (hydrophobicities[residue_ids] * generated_residues).sum(
        dim=1
    )
    predicted_hydrophobicities = torch.where(
        generated_lengths > 0,
        hydrophobicity_sums / generated_lengths.clamp_min(1.0),
        torch.zeros_like(hydrophobicity_sums),
    )
    predicted_properties = torch.stack(
        [generated_lengths, predicted_charges, predicted_hydrophobicities],
        dim=1,
    )
    property_errors = (predicted_properties - targets_float).abs().sum(dim=0)

    true_lengths = targets_float[:, 0]
    max_length = pred_float.shape[-1]
    length_stats = {
        "true_mean": float(true_lengths.sum().item()),
        "generated_mean": float(generated_lengths.sum().item()),
        "valid_fraction": float(valid_sequences.sum().item()),
        "too_short_fraction": float((generated_lengths < true_lengths).sum().item()),
        "too_long_fraction": float((generated_lengths > true_lengths).sum().item()),
        "exact_fraction": float((generated_lengths == true_lengths).sum().item()),
        "zero_fraction": float((generated_lengths == 0).sum().item()),
        "max_fraction": float((generated_lengths == max_length).sum().item()),
    }
    errors = {
        name: float(property_errors[prop_idx].item())
        for prop_idx, name in enumerate(PROPERTY_NAMES)
    }
    return errors, length_stats, int(pred_float.shape[0])


def shuffle_threedi_condition(
    seed_3di: torch.Tensor,
    available_mask: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Shuffle visible 3Di tokens within each sequence."""
    if seed_3di.ndim != 2 or available_mask.shape != seed_3di.shape:
        raise ValueError("3Di condition tensors must have shape [B,L]")

    shuffled_3di = seed_3di.clone()
    for row_idx in range(seed_3di.shape[0]):
        positions = torch.nonzero(
            available_mask[row_idx].bool() & (seed_3di[row_idx] != 0),
            as_tuple=False,
        ).flatten()
        token_count = int(positions.numel())
        if token_count < 2:
            continue
        permutation = torch.randperm(token_count, generator=generator)
        identity = torch.arange(token_count)
        if torch.equal(permutation, identity):
            permutation = identity.roll(1)
        permutation = permutation.to(device=positions.device)
        source_positions = positions[permutation]
        shuffled_3di[row_idx, positions] = seed_3di[row_idx, source_positions]
    return shuffled_3di


def _relative_conditioning_gain(
    correct_loss: float,
    control_loss: float,
) -> float:
    if control_loss <= 0:
        return 0.0 if correct_loss <= 0 else float("-inf")
    return (control_loss - correct_loss) / control_loss


@torch.inference_mode()
def validate_threedi_utilization(
    model: torch.nn.Module,
    diffusion,
    task_batches: list[dict[str, Any]],
    device: torch.device,
    precision: str,
    *,
    noise_seed: int,
    high_noise_start: float,
) -> dict[str, float]:
    """Measure paired denoising loss with correct, zero, and shuffled 3Di.

    The panel is deliberately evaluated without property conditioning or
    self-conditioning. All three variants see the same examples, timesteps, and
    diffusion noise. Loss is unweighted MSE on native residue positions so the
    score measures use of the positional 3Di signal rather than PAD prediction.
    """
    if not task_batches:
        raise ValueError("3Di utilization validation requires at least one batch")
    if not 0 <= high_noise_start < 1:
        raise ValueError("high_noise_start must be in [0, 1)")

    generator_device = device if device.type == "cuda" else torch.device("cpu")
    noise_generator = torch.Generator(device=generator_device).manual_seed(
        int(noise_seed)
    )
    shuffle_generator = torch.Generator().manual_seed(int(noise_seed) + 1)
    variants = ("correct", "zero", "shuffled")
    scopes = ("all", "high_noise")
    loss_sums = {scope: {variant: 0.0 for variant in variants} for scope in scopes}
    denoms = {scope: 0.0 for scope in scopes}
    example_counts = {scope: 0 for scope in scopes}
    high_noise_timestep = math.ceil(high_noise_start * diffusion.timesteps)

    was_training = model.training
    model.eval()
    try:
        for host_batch in task_batches:
            batch = move_batch(host_batch, device)
            batch_size = int(batch["x"].shape[0])
            timesteps = torch.randint(
                0,
                diffusion.timesteps,
                (batch_size,),
                device=device,
                generator=noise_generator,
                dtype=torch.long,
            )
            noise = torch.randn(
                batch["x"].shape,
                device=device,
                dtype=batch["x"].dtype,
                generator=noise_generator,
            )
            noisy_x, noise = diffusion.add_noise(
                batch["x"],
                timesteps,
                noise=noise,
            )
            target = training_target(diffusion, batch["x"], noise, timesteps)
            residue_mask = batch["residue_mask"].bool()
            correct_3di = batch["seed_3di"]
            shuffled_condition = shuffle_threedi_condition(
                correct_3di,
                residue_mask,
                generator=shuffle_generator,
            )
            conditions = {
                "correct": correct_3di,
                "zero": torch.zeros_like(correct_3di),
                "shuffled": shuffled_condition,
            }
            condition_mask = batch.get("condition_mask")
            if condition_mask is None:
                condition_mask = torch.zeros_like(batch["c"])

            predictions: dict[str, torch.Tensor] = {}
            with autocast_context(device, precision):
                for name, seed_3di in conditions.items():
                    predictions[name] = model(
                        noisy_x,
                        timesteps,
                        StructureInput(seed_3di),
                        PropertyInput(batch["c"], condition_mask),
                        seed=SeedInput(batch["seed_x"], batch["edit_mask"]),
                    )

            scope_masks = {
                "all": torch.ones_like(timesteps, dtype=torch.bool),
                "high_noise": timesteps >= high_noise_timestep,
            }
            for name, prediction in predictions.items():
                sample_loss_sums, sample_denoms = masked_mse_sample_sums(
                    biochemical_features(prediction),
                    biochemical_features(target),
                    residue_mask,
                )
                for scope, scope_mask in scope_masks.items():
                    loss_sums[scope][name] += float(
                        sample_loss_sums[scope_mask].sum().item()
                    )
                    if name == "correct":
                        denoms[scope] += float(sample_denoms[scope_mask].sum().item())
            for scope, scope_mask in scope_masks.items():
                example_counts[scope] += int(scope_mask.sum().item())
    finally:
        model.train(was_training)

    metrics: dict[str, float] = {}
    for scope in scopes:
        prefix = "threedi_utilization"
        if scope == "high_noise":
            prefix += "/high_noise"
        losses = {
            name: loss_sums[scope][name] / max(denoms[scope], 1.0) for name in variants
        }
        metrics.update(
            {
                f"{prefix}/correct_loss": losses["correct"],
                f"{prefix}/zero_loss": losses["zero"],
                f"{prefix}/shuffled_loss": losses["shuffled"],
                f"{prefix}/gain_vs_zero": _relative_conditioning_gain(
                    losses["correct"], losses["zero"]
                ),
                f"{prefix}/gain_vs_shuffled": _relative_conditioning_gain(
                    losses["correct"], losses["shuffled"]
                ),
                f"{prefix}/example_count": float(example_counts[scope]),
            }
        )
    return metrics
