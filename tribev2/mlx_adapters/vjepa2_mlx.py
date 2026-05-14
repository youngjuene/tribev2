"""Tiny MLX video-transformer mechanics for V-JEPA2 feasibility.

This module is deliberately small and test-focused.  It proves that the core
building blocks needed by a V-JEPA2-style video transformer can run in MLX and
match an equivalent PyTorch fixture after explicit weight copying.  It is **not**
a full V-JEPA2 implementation and must not be used as evidence of semantic
V-JEPA2 parity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class TinyVJEPA2Config:
    frames: int = 4
    image_size: int = 8
    patch_size: int = 4
    tubelet_size: int = 2
    in_chans: int = 3
    hidden_size: int = 8
    num_attention_heads: int = 2
    mlp_ratio: float = 2.0
    layer_norm_eps: float = 1e-5

    @property
    def patch_dim(self) -> int:
        return self.tubelet_size * self.patch_size * self.patch_size * self.in_chans

    @property
    def intermediate_size(self) -> int:
        return int(round(self.hidden_size * self.mlp_ratio))

    @property
    def num_patches(self) -> int:
        temporal = self.frames // self.tubelet_size
        spatial = self.image_size // self.patch_size
        return temporal * spatial * spatial


def torch_state_to_mlx(state: Mapping[str, object]) -> dict[str, object]:
    """Copy a PyTorch/NumPy state dict to MLX arrays."""
    import mlx.core as mx

    copied: dict[str, object] = {}
    for key, value in state.items():
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        copied[key] = mx.array(np.asarray(value, dtype=np.float32))
    return copied


def _linear(x, weight, bias=None):
    import mlx.core as mx

    y = mx.matmul(x, mx.swapaxes(weight, -1, -2))
    if bias is not None:
        y = y + bias
    return y


def _layer_norm(x, weight, bias, eps: float):
    import mlx.core as mx

    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.mean((x - mean) * (x - mean), axis=-1, keepdims=True)
    return (x - mean) * mx.rsqrt(var + eps) * weight + bias


def _gelu(x):
    import mlx.core as mx

    return 0.5 * x * (1.0 + mx.erf(x / mx.sqrt(mx.array(2.0, dtype=x.dtype))))


def patchify_video(video, config: TinyVJEPA2Config):
    """Return flattened tubelet/patch tokens from ``(B,T,H,W,C)`` video."""
    import mlx.core as mx

    b, t, h, w, c = video.shape
    if (t, h, w, c) != (config.frames, config.image_size, config.image_size, config.in_chans):
        raise ValueError(
            f"expected video shape (*,{config.frames},{config.image_size},{config.image_size},{config.in_chans}), got {video.shape}"
        )
    tube = config.tubelet_size
    patch = config.patch_size
    patches = mx.reshape(
        video,
        (b, t // tube, tube, h // patch, patch, w // patch, patch, c),
    )
    patches = mx.transpose(patches, (0, 1, 3, 5, 2, 4, 6, 7))
    return mx.reshape(patches, (b, config.num_patches, config.patch_dim))


def tiny_vjepa2_forward(video, weights: Mapping[str, object], config: TinyVJEPA2Config):
    """Run one tiny V-JEPA2-like transformer block in MLX.

    Parameters
    ----------
    video:
        MLX array with shape ``(batch, frames, height, width, channels)``.
    weights:
        MLX arrays matching the names produced by the test fixture.
    config:
        Tiny architecture configuration.
    """
    import mlx.core as mx

    tokens = patchify_video(video, config)
    x = _linear(tokens, weights["patch_embed.weight"], weights["patch_embed.bias"])
    x = x + weights["pos_embed"]

    residual = x
    y = _layer_norm(x, weights["norm1.weight"], weights["norm1.bias"], config.layer_norm_eps)
    qkv = _linear(y, weights["attn.qkv.weight"], weights["attn.qkv.bias"])
    b, n, _ = qkv.shape
    heads = config.num_attention_heads
    head_dim = config.hidden_size // heads
    qkv = mx.reshape(qkv, (b, n, 3, heads, head_dim))
    qkv = mx.transpose(qkv, (2, 0, 3, 1, 4))
    q, k, v = qkv[0], qkv[1], qkv[2]
    scores = mx.matmul(q, mx.swapaxes(k, -1, -2)) / mx.sqrt(mx.array(head_dim, dtype=q.dtype))
    probs = mx.softmax(scores, axis=-1)
    attn = mx.matmul(probs, v)
    attn = mx.transpose(attn, (0, 2, 1, 3))
    attn = mx.reshape(attn, (b, n, config.hidden_size))
    x = residual + _linear(attn, weights["attn.proj.weight"], weights["attn.proj.bias"])

    residual = x
    y = _layer_norm(x, weights["norm2.weight"], weights["norm2.bias"], config.layer_norm_eps)
    y = _linear(y, weights["mlp.fc1.weight"], weights["mlp.fc1.bias"])
    y = _gelu(y)
    y = _linear(y, weights["mlp.fc2.weight"], weights["mlp.fc2.bias"])
    return residual + y


def _to_mx(value, dtype=None):
    """Convert torch/NumPy values to MLX arrays without retaining torch tensors."""
    import mlx.core as mx

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = mx.array(np.asarray(value))
    return arr.astype(dtype) if dtype is not None else arr


def _linear_state(x, state: Mapping[str, object], weight_key: str, bias_key: str | None = None):
    import mlx.core as mx

    weight = _to_mx(state[weight_key], dtype=x.dtype)
    bias = _to_mx(state[bias_key], dtype=x.dtype) if bias_key and bias_key in state else None
    return _linear(x, weight, bias)


def _layer_norm_state(x, state: Mapping[str, object], prefix: str, eps: float):
    return _layer_norm(
        x,
        _to_mx(state[f"{prefix}.weight"], dtype=x.dtype),
        _to_mx(state[f"{prefix}.bias"], dtype=x.dtype),
        eps,
    )


def vjepa2_patch_embed(video, state: Mapping[str, object], config):
    """HF-compatible V-JEPA2 Conv3d patch embedding in MLX.

    Parameters
    ----------
    video:
        MLX array shaped ``(batch, frames, channels, height, width)`` matching
        Transformers ``pixel_values_videos``.
    state:
        HF state dict containing ``encoder.embeddings.patch_embeddings.proj.*``.
    config:
        ``VJEPA2Config`` or an object exposing the same patch/tubelet fields.
    """
    import mlx.core as mx

    b, t, c, h, w = video.shape
    tube = int(config.tubelet_size)
    patch = int(config.patch_size)
    if t < tube:
        reps = [1, int(np.ceil(tube / t)), 1, 1, 1]
        video = mx.tile(video, reps)[:, :tube]
        b, t, c, h, w = video.shape
    if t % tube or h % patch or w % patch:
        raise ValueError(f"Video shape {video.shape} is not divisible by tubelet/patch {(tube, patch)}")
    patches = mx.reshape(video, (b, t // tube, tube, c, h // patch, patch, w // patch, patch))
    patches = mx.transpose(patches, (0, 1, 4, 6, 3, 2, 5, 7))
    patches = mx.reshape(patches, (b, (t // tube) * (h // patch) * (w // patch), c * tube * patch * patch))
    weight = _to_mx(state["encoder.embeddings.patch_embeddings.proj.weight"], dtype=video.dtype)
    weight = mx.reshape(weight, (weight.shape[0], -1))
    bias = _to_mx(state["encoder.embeddings.patch_embeddings.proj.bias"], dtype=video.dtype)
    return _linear(patches, weight, bias)


def _position_ids(token_count: int, *, grid_size: int, grid_depth: int, heads: int):
    import mlx.core as mx

    ids = mx.arange(token_count)
    tokens_per_frame = grid_size * grid_size
    frame = ids // tokens_per_frame
    within = ids - tokens_per_frame * frame
    height = within // grid_size
    width = within - grid_size * height
    # Shape [1, heads, tokens] to match HF broadcast shape for unmasked inputs.
    return (
        mx.broadcast_to(frame[None, None, :], (1, heads, token_count)),
        mx.broadcast_to(height[None, None, :], (1, heads, token_count)),
        mx.broadcast_to(width[None, None, :], (1, heads, token_count)),
    )


def rotate_queries_or_keys_mlx(x, pos):
    """MLX port of HF ``rotate_queries_or_keys`` for V-JEPA2 3D RoPE."""
    import mlx.core as mx

    dim = x.shape[-1]
    omega = mx.arange(dim // 2, dtype=x.dtype) / (dim / 2.0)
    omega = 1.0 / (10000 ** omega)
    freq = pos[..., None].astype(x.dtype) * omega
    emb_sin = mx.repeat(mx.sin(freq), 2, axis=-1)
    emb_cos = mx.repeat(mx.cos(freq), 2, axis=-1)
    y = mx.reshape(x, (*x.shape[:-1], -1, 2))
    y1 = y[..., 0]
    y2 = y[..., 1]
    rotated = mx.reshape(mx.stack([-y2, y1], axis=-1), x.shape)
    return (x * emb_cos) + (rotated * emb_sin)


def _apply_vjepa2_rope(q_or_k, pos_ids, head_dim: int):
    import mlx.core as mx

    d_dim = int(2 * ((head_dim // 3) // 2))
    h_dim = d_dim
    w_dim = d_dim
    s = 0
    parts = []
    for width, pos in [(d_dim, pos_ids[0]), (h_dim, pos_ids[1]), (w_dim, pos_ids[2])]:
        parts.append(rotate_queries_or_keys_mlx(q_or_k[..., s : s + width], pos=pos))
        s += width
    if s < head_dim:
        parts.append(q_or_k[..., s:])
    return mx.concatenate(parts, axis=-1)


def safe_scaled_dot_product_attention(q, k, v, *, scale: float, chunk_size: int | None = None):
    """MLX attention with optional query chunking for large sequence probes."""
    import mlx.core as mx

    if chunk_size is None or q.shape[2] <= chunk_size:
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
    chunks = []
    for start in range(0, q.shape[2], chunk_size):
        q_chunk = q[:, :, start : start + chunk_size, :]
        chunks.append(mx.fast.scaled_dot_product_attention(q_chunk, k, v, scale=scale))
    return mx.concatenate(chunks, axis=2)


def vjepa2_encoder_layer_forward(hidden_states, state: Mapping[str, object], config, layer_idx: int, *, chunk_size: int | None = None):
    """One HF-compatible V-JEPA2 encoder layer in MLX."""
    import mlx.core as mx

    prefix = f"encoder.layer.{layer_idx}"
    residual = hidden_states
    y = _layer_norm_state(hidden_states, state, f"{prefix}.norm1", float(config.layer_norm_eps))
    b, n, _ = y.shape
    heads = int(config.num_attention_heads)
    head_dim = int(config.hidden_size) // heads
    q = _linear_state(y, state, f"{prefix}.attention.query.weight", f"{prefix}.attention.query.bias")
    k = _linear_state(y, state, f"{prefix}.attention.key.weight", f"{prefix}.attention.key.bias")
    v = _linear_state(y, state, f"{prefix}.attention.value.weight", f"{prefix}.attention.value.bias")
    q = mx.transpose(mx.reshape(q, (b, n, heads, head_dim)), (0, 2, 1, 3))
    k = mx.transpose(mx.reshape(k, (b, n, heads, head_dim)), (0, 2, 1, 3))
    v = mx.transpose(mx.reshape(v, (b, n, heads, head_dim)), (0, 2, 1, 3))
    grid_size = int(config.crop_size) // int(config.patch_size)
    grid_depth = int(config.frames_per_clip) // int(config.tubelet_size)
    pos_ids = _position_ids(n, grid_size=grid_size, grid_depth=grid_depth, heads=heads)
    q = _apply_vjepa2_rope(q, pos_ids, head_dim)
    k = _apply_vjepa2_rope(k, pos_ids, head_dim)
    attn = safe_scaled_dot_product_attention(q, k, v, scale=head_dim**-0.5, chunk_size=chunk_size)
    attn = mx.reshape(mx.transpose(attn, (0, 2, 1, 3)), (b, n, int(config.hidden_size)))
    hidden_states = residual + _linear_state(attn, state, f"{prefix}.attention.proj.weight", f"{prefix}.attention.proj.bias")
    residual = hidden_states
    y = _layer_norm_state(hidden_states, state, f"{prefix}.norm2", float(config.layer_norm_eps))
    y = _linear_state(y, state, f"{prefix}.mlp.fc1.weight", f"{prefix}.mlp.fc1.bias")
    y = _gelu(y)
    y = _linear_state(y, state, f"{prefix}.mlp.fc2.weight", f"{prefix}.mlp.fc2.bias")
    return residual + y


def vjepa2_encoder_forward_mlx(video, state: Mapping[str, object], config, *, output_hidden_states: bool = False, chunk_size: int | None = None):
    """Run the V-JEPA2 encoder path in MLX from HF-style state tensors."""
    import mlx.core as mx

    hidden_states = vjepa2_patch_embed(video, state, config)
    hidden_states_trace = [hidden_states] if output_hidden_states else None
    for layer_idx in range(int(config.num_hidden_layers)):
        hidden_states = vjepa2_encoder_layer_forward(hidden_states, state, config, layer_idx, chunk_size=chunk_size)
        if hidden_states_trace is not None:
            hidden_states_trace.append(hidden_states)
    hidden_states = _layer_norm_state(hidden_states, state, "encoder.layernorm", float(config.layer_norm_eps))
    mx.eval(hidden_states)
    if hidden_states_trace is not None:
        hidden_states_trace[-1] = hidden_states
        return hidden_states, hidden_states_trace
    return hidden_states
