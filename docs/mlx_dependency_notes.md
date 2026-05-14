# MLX dependency notes for TRIBE v2

This note records the dependency lane decision for MLX-assisted TRIBE work.

## Recommendation

Do **not** add `mlx-image` or `mlx-vision` to core dependencies yet.

- `mlx-image` is the practical DINOv2 hidden-state candidate (`mlxim.model.create_model("vit_large_patch14_518.dinov2")`), but current package metadata conflicts with TRIBE's pinned `numpy==2.2.6` and pins an older `mlx` version.
- `mlx-vision` installs more cleanly with modern MLX/Numpy, but it is an image preprocessing / Apple Vision interop package and does not provide DINOv2 hidden states.
- Bare `mlx` is sufficient for backend probes and the repo-local proof adapter. The notebook now also installs optional `mlx-lm` so the `text_mlx_lm` lane can run Llama final-hidden-state extraction on Apple GPU when explicitly enabled.

## Consequence

The first committed adapter is a **proof adapter** (`MLXProofFeatureExtractor`) that verifies MLX-generated tensors can cross the neuralset/TRIBE PyTorch boundary. A semantically equivalent DINOv2 adapter should be prototyped in an isolated environment or after dependency bounds are resolved.

## Current first checkpoint-active target

`tools/inspect_tribe_feature_contracts.py` shows the released checkpoint uses `text`, `audio`, and `video` features. DINOv2/image is not checkpoint-active in `model_build_args`, so a DINOv2 MLX adapter would currently be backend proof only. Text is checkpoint-active and has MLX Llama candidates. This repository now provides two opt-in text adapters: `text_proof` (deterministic shape/contract proof) and `text_mlx_lm` (real MLX-LM final hidden states). `text_mlx_lm` is still labelled non-parity because neuralset aggregates selected intermediate Hugging Face layers, while the public MLX-LM path used here exposes final hidden states.
