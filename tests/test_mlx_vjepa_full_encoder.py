from __future__ import annotations

import numpy as np
import pytest
import torch
from transformers import VJEPA2Config, VJEPA2Model

from tribev2.mlx_adapters.vjepa2_mlx import (
    rotate_queries_or_keys_mlx,
    safe_scaled_dot_product_attention,
    vjepa2_encoder_forward_mlx,
)

mx = pytest.importorskip("mlx.core")


def _small_config() -> VJEPA2Config:
    return VJEPA2Config(
        patch_size=4,
        crop_size=8,
        frames_per_clip=4,
        tubelet_size=2,
        hidden_size=12,
        in_chans=3,
        num_attention_heads=3,
        num_hidden_layers=2,
        mlp_ratio=2.0,
        layer_norm_eps=1e-6,
        qkv_bias=True,
        hidden_act="gelu",
        pred_hidden_size=6,
        pred_num_attention_heads=3,
        pred_num_hidden_layers=1,
    )


def test_full_encoder_mlx_matches_small_hf_vjepa2_encoder():
    torch.manual_seed(11)
    config = _small_config()
    model = VJEPA2Model(config).eval()
    video = torch.randn(1, config.frames_per_clip, config.in_chans, config.crop_size, config.crop_size)
    with torch.no_grad():
        expected = model(pixel_values_videos=video, skip_predictor=True).last_hidden_state.cpu().numpy()
    actual_mx = vjepa2_encoder_forward_mlx(mx.array(video.numpy()), model.state_dict(), config, chunk_size=3)
    mx.eval(actual_mx)
    actual = np.array(actual_mx)
    assert actual.shape == expected.shape
    max_abs = float(np.max(np.abs(expected - actual)))
    cosine = float(
        np.dot(expected.reshape(-1), actual.reshape(-1))
        / (np.linalg.norm(expected.reshape(-1)) * np.linalg.norm(actual.reshape(-1)))
    )
    assert max_abs < 5e-4
    assert cosine > 0.99999


def test_safe_attention_chunked_matches_unchunked():
    rng = np.random.default_rng(5)
    q = mx.array(rng.normal(size=(1, 2, 9, 4)).astype(np.float32))
    k = mx.array(rng.normal(size=(1, 2, 9, 4)).astype(np.float32))
    v = mx.array(rng.normal(size=(1, 2, 9, 4)).astype(np.float32))
    full = safe_scaled_dot_product_attention(q, k, v, scale=4**-0.5, chunk_size=None)
    chunked = safe_scaled_dot_product_attention(q, k, v, scale=4**-0.5, chunk_size=3)
    mx.eval(full, chunked)
    assert np.max(np.abs(np.array(full) - np.array(chunked))) < 1e-5


def test_rope_preserves_shape_with_leftover_dims():
    x = mx.ones((1, 3, 8, 4), dtype=mx.float32)
    pos = mx.broadcast_to(mx.arange(8)[None, None, :], (1, 3, 8))
    out = rotate_queries_or_keys_mlx(x, pos)
    mx.eval(out)
    assert tuple(out.shape) == (1, 3, 8, 4)
