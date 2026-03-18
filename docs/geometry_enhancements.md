# PatchAlign3D Geometric Enhancements

## Motivation
The original PatchAlign3D backbone is strong and simple, but its local modeling has two clear limits:

1. Inside each patch, the encoder is purely PointNet-style. This is efficient, but it treats the patch mostly as an unordered set and does not explicitly model local geometric relations between nearby points.
2. Across patches, the model relies on a global transformer alone. This captures context well, but it does not explicitly emphasize short-range geometric relations between neighboring patch centers before global reasoning starts.

For a thesis-style upgrade, this makes PatchAlign3D a good target for a "local geometry + patch geometry" enhancement while keeping the original two-stage training framework intact.

## Added Modules
### 1. HybridPatchEncoder
We keep the original PointNet patch encoder and add a parallel multi-scale EdgeConv branch.

- The PointNet branch preserves the original PatchAlign3D patch-token pipeline.
- The multi-scale EdgeConv branch is inspired by DGCNN and PointNet++.
- It processes the same patch at three scales: 8, 16, and 32 points.
- Each scale builds a patch-internal kNN graph and applies two lightweight EdgeConv units.
- The three scale descriptors are fused and then combined with the PointNet descriptor.

This design lets the encoder keep the stability of the original PointNet path while adding stronger local geometric awareness for fine part boundaries.

### 2. PatchGeometryRefiner
Before patch tokens enter the original global transformer, we add a small patch-level geometric refinement module.

- It builds a local graph over patch centers.
- Each block uses relative-position encoding between neighboring patches.
- Messages are aggregated locally before the global transformer starts.

This module is inspired by Point Transformer and Point Transformer V2, where relative geometry is important for local attention.

## Literature Basis
The upgrade borrows design ideas from popular point-cloud literature:

- PointNet++: multi-scale local structure modeling.
- DGCNN / EdgeConv: explicit local geometric aggregation on point neighborhoods.
- Point Transformer / Point Transformer V2: relative-position-aware local relation modeling.

The implementation is intentionally lightweight so it can fit the current PatchAlign3D codebase and training flow without adding new custom CUDA dependencies.

## What Stays Unchanged
The following core properties of PatchAlign3D are preserved:

- The two-stage training design stays the same.
- Stage 1 still uses cosine-similarity regression against cached DINO patch features.
- Stage 2 still aligns patch tokens with text features.
- The training data format and `patch_dino/patch_features.pt` cache format stay unchanged.
- `forward_patches()` still returns patch embeddings, patch centers, and patch membership indices with the same shapes.

## Ablation Variants
The implementation supports clean ablation settings for thesis writing and defense:

- Original PointNet patch encoder only.
- PointNet + PatchGeometryRefiner.
- HybridPatchEncoder only.
- HybridPatchEncoder + PatchGeometryRefiner.

These variants can be toggled from the command line through the backbone options.
