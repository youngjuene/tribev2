"""MLX-assisted feature extraction adapters for TRIBE experiments.

These adapters are intentionally repo-local and do not patch neuralset.  They
produce torch tensors at the TRIBE boundary because the released brain model and
SegmentData loader are PyTorch-based, while the feature-generation proof itself
runs through MLX.
"""

from .vjepa2_mlx import (
    TinyVJEPA2Config,
    rotate_queries_or_keys_mlx,
    safe_scaled_dot_product_attention,
    tiny_vjepa2_forward,
    torch_state_to_mlx,
    vjepa2_encoder_forward_mlx,
)
from .precomputed import (
    MLXLmTextFeatureExtractor,
    MLXProofFeatureExtractor,
    PrecomputedFeatureExtractor,
)

__all__ = [
    "TinyVJEPA2Config",
    "tiny_vjepa2_forward",
    "rotate_queries_or_keys_mlx",
    "safe_scaled_dot_product_attention",
    "vjepa2_encoder_forward_mlx",
    "torch_state_to_mlx",
    "MLXLmTextFeatureExtractor",
    "MLXProofFeatureExtractor",
    "PrecomputedFeatureExtractor",
]
