"""Conditioned 1D diffusion U-Net for peptide generation.

ProteinDiffusionUNet takes a noisy peptide tensor [B, 6, 70] at diffusion
timestep t and predicts the denoised peptide (or noise), guided by
  1. Per-residue 3Di structure tokens (ThreeDiConditioningEncoder)
  2. Global biochemical targets for length, charge, and hydrophobicity
     (PropertyConditioningEncoder)
  3. Optional fixed seed residues for motif scaffolding or partial editing
"""

from __future__ import annotations

import torch
from torch import nn

from .blocks import DownBlocks, MiddleBlock, UpBlocks
from .conditioning import (
    FoldseekThreeDiEmbedding, PropertyConditioningEncoder, ThreeDiConditioningEncoder,
)
from .conditioning.biochem_encoder import _property_input, _property_statistics
from .config import BackboneConfig, PropertyConfig, ThreeDiConfig
from .inputs import PropertyInput, SeedInput, StructureInput
from .layers import (
    ResidualBlock1d, SinusoidalTimeEmbedding, ThreeDiCrossAttention1d, _match_length,
)


class ProteinDiffusionUNet(nn.Module):
    """Denoise a peptide sequence across three U-Net resolutions.

      Inputs per position (length 70)
        noisy_x [6] + self_cond_x [6] + kept seed [6] + threedi [128] + edit [1]
                                          │
                                          ▼  self.input (Conv1d, k=7)
        initial_features  [B, 32, 70] ────┼───────────────────────────────────┐
                                          │                                   │
                    ┌─────────────────────▼─────────────────────┐             │
        maps[0] ───►│ DownBlock 0  [B, 32, 70] ──► skips ───────┼──────┐      │
        [70]        │      │ stride-2 down (70 -> 35)           │      │      │
        maps[1] ───►│ DownBlock 1  [B, 32, 35] ──► skips ───────┼───┐  │      │
        [35]        │      │ stride-2 down (35 -> 17, 32 -> 64) │   │  │      │
        maps[2] ───►│ DownBlock 2  [B, 64, 17] ──► skips ───────┼┐  │  │      │
        [17]        │      │ stride-1 conv (64 -> 128)          ││  │  │      │
                    └──────┼────────────────────────────────────┘│  │  │      │
                           ▼                                     │  │  │      │
        maps[3] ───►  MiddleBlock  [B, 128, 17]  ◄── threedi     │  │  │      │
        [17]               │          (bottleneck)               │  │  │      │
                    ┌──────▼────────────────────────────────────┐│  │  │      │
        maps[2] ───►│ UpBlock 0    [B, 64, 17] ◄─────────────────┘  │  │      │
        [17]        │      │ stride-2 up (17 -> 35)                 │  │      │
        maps[1] ───►│ UpBlock 1    [B, 32, 35] ◄────────────────────┘  │      │
        [35]        │      │ stride-2 up (35 -> 70)                    │      │
        maps[0] ───►│ UpBlock 2    [B, 32, 70] ◄───────────────────────┘      │
        [70]        │      │ 1x1 conv                                         │
                    └──────┼──────────────────────────────────────────────────┘
                           ▼
        final_res_block   [B, 32, 70]  (merges decoder output + initial_features)
                           │
                           ▼  optional threed_cross_attention + self.output (1x1 conv)
        prediction        [B,  6, 70]



        (ASCII art is remarkable ins't it) 
        (And sufficiently advanced technology is indistinguhisable from art :))  
        (And I can't type :))
    """

    def __init__(
        self,
        backbone: BackboneConfig,
        threedi: ThreeDiConfig | None = None,
        properties: PropertyConfig | None = None,
    ) -> None:
        super().__init__()
        threedi = ThreeDiConfig() if threedi is None else threedi
        properties = PropertyConfig() if properties is None else properties
        self.backbone_config = backbone
        self.threedi_config = threedi
        self.property_config = properties
        self.x_channels = backbone.x_channels
        self.max_length = backbone.max_length
        hidden_channels = backbone.hidden_channels

        # 1. 3Di token embedding and initial sequence attention.
        #    Built before the U-Net backbone so seeded weight init stays identical.
        if threedi.representation == "foldseek_2d":
            self.threed_embedding = FoldseekThreeDiEmbedding(threedi.dim)
        else:
            self.threed_embedding = nn.Embedding(threedi.vocab_size, threedi.dim, padding_idx=0)
        context = ThreeDiConditioningEncoder.make_context(
            threedi.context, channels=threedi.dim, max_length=backbone.max_length,
        )

        # 2. Diffusion timestep embedding and global property normalization stats.
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_channels),
            nn.Linear(hidden_channels, backbone.cond_dim),
            nn.GELU(),
            nn.Linear(backbone.cond_dim, backbone.cond_dim),
        )
        property_mean, property_std = _property_statistics(properties.mean, properties.std)
        self.register_buffer("property_mean", property_mean)
        self.register_buffer("property_std", property_std)

        # 3. Input stem.
        #    Stacks noisy_x (6), self_cond_x (6), kept seed residues (6),
        #    full-length 3Di embedding (128), and the binary edit mask (1).
        input_channels = 3 * backbone.x_channels + threedi.dim + 1
        self.input = nn.Conv1d(input_channels, hidden_channels, kernel_size=7, padding=3)

        # 4. U-Net encoder, conditioning pyramids, bottleneck, and decoder.
        #    Channel widths per level default to [32, 32, 64] with bottleneck 128.
        dimensions = [hidden_channels, *[hidden_channels * mult for mult in backbone.channel_mults]]
        skip_channels = dimensions[:-1]
        bottleneck_channels = dimensions[-1]
        condition_multiplier = 2 if threedi.multiscale else 1
        block_options = dict(
            time_dim=backbone.cond_dim, attention_mode=backbone.attention_mode,
            attention_heads=backbone.attention_heads,
            blocks_per_level=backbone.blocks_per_level,
        )

        self.down_blocks = DownBlocks(dimensions, condition_multiplier, **block_options)
        self.property_conditioning = PropertyConditioningEncoder(
            input_dim=6 if properties.mask_enabled else 3, max_length=backbone.max_length,
            level_channels=skip_channels, bottleneck_channels=bottleneck_channels,
            attention_heads=backbone.attention_heads,
        )
        self.threed_conditioning = ThreeDiConditioningEncoder(
            input_channels=threedi.dim, level_channels=skip_channels,
            bottleneck_channels=bottleneck_channels,
            attention_heads=backbone.attention_heads,
            context=context, multiscale=threedi.multiscale,
        )
        self.threed_condition_scales = (
            nn.Parameter(torch.ones(len(skip_channels) + 1))
            if threedi.multiscale else None
        )
        self.middle_block = MiddleBlock(
            bottleneck_channels, bottleneck_channels * condition_multiplier, **block_options,
        )
        self.up_blocks = UpBlocks(
            dimensions, condition_multiplier, backbone.max_length, **block_options,
        )

        # 5. Output head and optional full-sequence 3Di cross-attention.
        self.final_res_block = ResidualBlock1d(
            dimensions[0] + hidden_channels, hidden_channels, backbone.cond_dim,
        )
        self.output = nn.Conv1d(hidden_channels, backbone.x_channels, kernel_size=1)
        cross_attention = threedi.cross_attention
        self.threed_cross_attention = (
            ThreeDiCrossAttention1d(
                hidden_channels, threedi.dim, heads=cross_attention.heads,
                dim_head=cross_attention.dim_head,
            ) if cross_attention is not None else None
        )
        if cross_attention is not None and cross_attention.bottleneck:
            with torch.random.fork_rng(devices=[]):
                self.middle_block.structure_attention = ThreeDiCrossAttention1d(
                    bottleneck_channels, threedi.dim, heads=cross_attention.heads,
                    dim_head=cross_attention.dim_head,
                )

    # -------------------------------------------------------------------------
    # Public forward pass
    # -------------------------------------------------------------------------

    def forward(
        self,
        noisy_x: torch.Tensor,
        timesteps: torch.Tensor,
        structure: StructureInput,
        properties: PropertyInput,
        seed: SeedInput | None = None,
        self_cond_x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode conditioning and run one denoising step."""
        return self.forward_prepared(
            noisy_x, timesteps, structure,
            self.prepare_conditioning(structure, properties), seed, self_cond_x,
        )

    def prepare_conditioning(self, structure: StructureInput, properties: PropertyInput):
        """Encode 3Di and biochemical conditions once (can be cached across sampling steps)."""
        # 1. Embed 3Di tokens, run sequence attention, and build multiscale 3Di maps.
        threedi, context_mask, structure_maps = self._prepare_threedi(structure)
        # 2. Standardize and mask global biochemical properties [length, charge, hydrophobicity].
        property_input = _property_input(
            properties.values, self.property_mean, self.property_std,
            condition_mask_enabled=self.property_config.mask_enabled,
            condition_mask=properties.mask,
        )
        # 3. Build multiscale property maps and concatenate them with the 3Di maps per level.
        condition_maps = self._condition_maps(property_input, structure_maps)
        return threedi, context_mask, condition_maps

    def forward_prepared(
        self, noisy_x, timesteps, structure, conditioning, seed=None, self_cond_x=None,
    ):
        """Run one denoising step using precomputed (threedi, context_mask, condition_maps)."""
        threedi, context_mask, condition_maps = conditioning
        time_condition = self.time_mlp(timesteps)

        # Input stem projects [noisy_x, self_cond_x, kept_seed, threedi, edit] to hidden_channels.
        features = self._input_features(noisy_x, self_cond_x, seed, structure.tokens, threedi)
        initial_features = features.clone()

        # Encoder steps down across resolutions (70 -> 35 -> 17) and saves skip connections.
        features, skips = self.down_blocks(features, time_condition, condition_maps[:-1])

        # Bottleneck processes the coarsest resolution (17) with full self-attention.
        features = self.middle_block(
            features, time_condition, condition_maps[-1], threedi, context_mask,
        )

        # Decoder steps back up (17 -> 35 -> 70), merging skip connections at each level.
        features = self.up_blocks(features, skips, time_condition, condition_maps[:-1])

        # Final full-resolution refinement and projection back to 6 physicochemical channels.
        features = _match_length(features, initial_features.shape[-1])
        features = torch.cat([features, initial_features], dim=1)
        features = self.final_res_block(features, time_condition)
        if self.threed_cross_attention is not None:
            features = self.threed_cross_attention(features, threedi, context_mask)
        return _match_length(self.output(features), noisy_x.shape[-1])

    # -------------------------------------------------------------------------
    # Conditioning and input helpers
    # -------------------------------------------------------------------------

    # Embed 3Di letters, hiding masked positions, then run ThreeDiConditioningEncoder.
    #
    #   tokens        [B, L]       A   C   D   E   0   0    (0 is padding)
    #   hidden_mask   [B, L]       .   X   .   .   .   .
    #   threedi       [B, 128, L]  a'  x'  d'  e'  0   0    (contextualized)
    #   context_mask  [B, L]       1   1   1   1   0   0
    def _prepare_threedi(
        self, structure: StructureInput,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor] | None]:
        seed_3di = structure.tokens
        if seed_3di.ndim != 2:
            raise ValueError("structure tokens must have shape [B,L]")
        if self.threedi_config.context is not None and seed_3di.shape[1] > self.max_length:
            raise ValueError("3Di sequence exceeds contextual encoder length")
        threedi_hidden_mask = structure.hidden_mask
        if threedi_hidden_mask is not None:
            if threedi_hidden_mask.shape != seed_3di.shape:
                raise ValueError("threedi_hidden_mask must match seed_3di [B,L]")
            threedi_hidden_mask = threedi_hidden_mask.bool()
            seed_3di = seed_3di.masked_fill(threedi_hidden_mask, 0)
        threedi_available = seed_3di != 0
        context_mask = threedi_available
        if self.threedi_config.infer_missing and threedi_hidden_mask is not None:
            # Explicitly marked unknown tokens are distinct from padding.
            # Do not derive this from the target AA/residue mask: that could
            # expose lengths that were not part of the supplied condition.
            context_mask = (
                (threedi_available | threedi_hidden_mask)
                & threedi_available.any(dim=1, keepdim=True)
            )
        threedi = self.threed_embedding(seed_3di).transpose(1, 2)
        threedi, structure_maps = self.threed_conditioning(
            threedi, threedi_available, context_mask=context_mask,
        )
        return threedi, context_mask, structure_maps

    # Combine per-resolution property maps and 3Di maps into one list of conditioning
    # tensors [maps[0], maps[1], maps[2], maps[3]] for the 3 U-Net levels + bottleneck.
    def _condition_maps(
        self, property_input: torch.Tensor, structure_maps: list[torch.Tensor] | None,
    ) -> list[torch.Tensor]:
        property_maps = self.property_conditioning(property_input)
        if structure_maps is None:
            return property_maps
        return [
            torch.cat([properties, structure * self.threed_condition_scales[index]], dim=1)
            for index, (properties, structure) in enumerate(zip(property_maps, structure_maps))
        ]

    # Stack per-position inputs and project to hidden_channels (default 32).
    #   noisy_x [6] + self_cond_x [6] + kept_seed [6] + threedi [128] + edit [1] -> [B, 147, L]
    def _input_features(
        self, noisy_x: torch.Tensor, self_cond_x: torch.Tensor | None,
        seed: SeedInput | None, seed_3di: torch.Tensor, threedi: torch.Tensor,
    ) -> torch.Tensor:
        if self_cond_x is None:
            self_cond_x = torch.zeros_like(noisy_x)
        if seed is None:
            seed_x = torch.zeros_like(noisy_x)
            edit_mask = torch.ones_like(seed_3di, dtype=torch.bool)
        else:
            if seed.x.shape != noisy_x.shape:
                raise ValueError("seed.x must have the same shape as noisy_x")
            seed_x, edit_mask = seed.x, seed.edit_mask
        edit = edit_mask[:, None, :].to(device=noisy_x.device, dtype=noisy_x.dtype)
        seed_condition = seed_x * (1.0 - edit)
        return self.input(
            torch.cat([noisy_x, self_cond_x, seed_condition, threedi, edit], dim=1)
        )

    # Remap legacy checkpoints saved before ContextualThreeDiEncoder moved into ThreeDiConditioningEncoder.
    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys,
        unexpected_keys, error_msgs,
    ):
        old_prefix = prefix + "threed_context."
        new_prefix = prefix + "threed_conditioning.context."
        for key in list(state_dict):
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix):]
                if new_key in state_dict:
                    error_msgs.append(f"Checkpoint contains both {key} and {new_key}")
                else:
                    state_dict[new_key] = state_dict[key]
                del state_dict[key]
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs,
        )
