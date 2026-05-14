# TRIBE v2 MLX Feature Contracts

Repo: `facebook/tribev2`
Checkpoint: `best.ckpt`
Features to use: `['text', 'audio', 'video']`

## Feature dimensions

- `text`: `[2, 3072]`
- `audio`: `[2, 1024]`
- `video`: `[2, 1408]`

## Modality gate

- `text`: **candidate**; active=True; dims=`[2, 3072]`; candidates=`['mlx-community/Llama-3.2-3B-bf16', 'mlx-community/Llama-3.2-3B-8bit']`; Checkpoint-active modality with at least one known MLX candidate.
- `audio`: **blocked**; active=True; dims=`[2, 1024]`; candidates=`[]`; Checkpoint-active but no known MLX candidate.
- `video`: **blocked**; active=True; dims=`[2, 1408]`; candidates=`[]`; Checkpoint-active but no known MLX candidate.
- `image`: **backend_proof_only**; active=False; dims=`None`; candidates=`['mlx-image/mlxim vit_large_patch14_518.dinov2']`; DINOv2 MLX candidates exist, but released model_build_args has no image feature_dims entry.

## Interpretation

The first MLX-assisted inference proof should target a checkpoint-active modality. For the released checkpoint, text is the only active modality with an obvious MLX candidate from the current inventory; video and audio remain blocked without V-JEPA2/Wav2Vec-BERT MLX equivalents. DINOv2/image is useful only as a backend proof unless the model/config is changed.
