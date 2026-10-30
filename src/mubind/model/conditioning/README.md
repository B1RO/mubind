# Conditioning

Encodes global properties and 3Di structure for `ProteinDiffusionUNet`; the 3Di encoder returns the full-length contextual embedding and optional per-level maps.

```text
conditioning/
├── biochem_encoder.py           Standardizes and masks properties, then builds per-level feature maps
├── threedi_encoder.py           Owns ContextualThreeDiEncoder and builds optional per-level maps
├── threedi_2d_representation.py Fixed Foldseek 2D codebook and learned projection
└── _pyramid.py                  Shared pyramid builder for property and 3Di feature maps
```

![Biochemical encoder showing the responsibilities of biochem_encoder.py, _pyramid.py, and unet.py](assets/biochem_encoder.svg)

![3Di encoder showing token embedding, contextual encoding, shared pyramid construction, and U-Net routing in their source files](assets/threedi_encoder.svg)
