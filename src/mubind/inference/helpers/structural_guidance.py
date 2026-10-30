"""Classifier-free AA/PAD structural guidance with separate null self-conditioning history."""

from __future__ import annotations

import math

import torch

from ...model import StructureInput


class StructuralGuidance(torch.nn.Module):
    """Guide the first 6 channels (AA + PAD) toward the 3Di structure condition.

    At each sampling step, runs both a conditional pass (with `structure.tokens`)
    and a null pass (with all non-padding 3Di tokens masked out and its own
    `null_self_cond` trajectory), then blends the first 6 channels using `scale`.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        scale: float = 5.0,
        *,
        self_condition: bool = True,
        cache_conditioning: bool = False,
    ) -> None:
        super().__init__()
        if not math.isfinite(scale) or scale < 0:
            raise ValueError("Structural guidance must be finite and nonnegative")
        self.model = model
        self.scale = float(scale)
        self.self_condition = self_condition
        self.cache_conditioning = cache_conditioning
        self.reset_sampling_state()

    def reset_sampling_state(self) -> None:
        """Clear per-prediction conditioning caches and null self-conditioning history."""
        self.null_self_cond = None
        self._condition_key = None
        self._conditions = {}

    @property
    def property_config(self):
        return self.model.property_config

    def forward(self, x, t, structure, properties, seed=None, *, self_cond_x=None):
        # Never cache a training or gradient-enabled call; predict() clears the cache on exit.
        self._use_cache = (
            self.cache_conditioning
            and not self.training
            and not torch.is_grad_enabled()
        )
        key = (
            structure.tokens,
            structure.hidden_mask,
            properties.values,
            properties.mask,
        )
        if self._condition_key is None or any(
            a is not b for a, b in zip(key, self._condition_key)
        ):
            self._conditions.clear()
            self._condition_key = key

        # 1. Run the conditional pass, saving RNG state so dropout (if any) matches in the null pass.
        cpu_state = torch.get_rng_state()
        devices = [x.device.index] if x.is_cuda else []
        cuda_state = torch.cuda.get_rng_state(x.device) if x.is_cuda else None
        conditional = self._predict(
            "conditional", x, t, structure, properties, seed, self_cond_x
        )
        if self.scale == 1:
            return conditional

        # 2. Run the unconditional (null-3Di) pass with its own null self-conditioning history.
        hidden = structure.tokens.ne(0)
        if structure.hidden_mask is not None:
            hidden = hidden | structure.hidden_mask
        null_structure = StructureInput(
            torch.zeros_like(structure.tokens),
            hidden,
        )
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state(cuda_state, x.device)
            unconditional = self._predict(
                "null",
                x,
                t,
                null_structure,
                properties,
                seed,
                self.null_self_cond,
            )
        self.null_self_cond = unconditional.detach() if self.self_condition else None

        # 3. Extrapolate the AA and PAD channels (:6) while keeping 3Di channels (6:) unscaled.
        guided = conditional.clone()
        guided[:, :6] = unconditional[:, :6] + self.scale * (
            conditional[:, :6] - unconditional[:, :6]
        )
        return guided

    def _predict(self, branch, x, t, structure, properties, seed, self_cond_x):
        if not self._use_cache:
            return self.model(
                x, t, structure, properties, seed, self_cond_x=self_cond_x
            )
        if branch not in self._conditions:
            self._conditions[branch] = self.model.prepare_conditioning(
                structure, properties
            )
        return self.model.forward_prepared(
            x, t, structure, self._conditions[branch], seed, self_cond_x
        )
