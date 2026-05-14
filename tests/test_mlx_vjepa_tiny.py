from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from tribev2.mlx_adapters.vjepa2_mlx import (
    TinyVJEPA2Config,
    tiny_vjepa2_forward,
    torch_state_to_mlx,
)

mx = pytest.importorskip("mlx.core")


class TinyTorchVJEPA2(nn.Module):
    def __init__(self, config: TinyVJEPA2Config):
        super().__init__()
        self.config = config
        self.patch_embed = nn.Linear(config.patch_dim, config.hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.num_patches, config.hidden_size))
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def patchify(self, video: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        b, t, h, w, c = video.shape
        patches = video.reshape(
            b,
            t // cfg.tubelet_size,
            cfg.tubelet_size,
            h // cfg.patch_size,
            cfg.patch_size,
            w // cfg.patch_size,
            cfg.patch_size,
            c,
        )
        patches = patches.permute(0, 1, 3, 5, 2, 4, 6, 7)
        return patches.reshape(b, cfg.num_patches, cfg.patch_dim)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        x = self.patch_embed(self.patchify(video)) + self.pos_embed
        residual = x
        y = self.norm1(x)
        qkv = self.qkv(y).reshape(video.shape[0], cfg.num_patches, 3, cfg.num_attention_heads, -1)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = q @ k.transpose(-1, -2) / (q.shape[-1] ** 0.5)
        probs = torch.softmax(scores, dim=-1)
        attn = (probs @ v).permute(0, 2, 1, 3).reshape(video.shape[0], cfg.num_patches, cfg.hidden_size)
        x = residual + self.proj(attn)
        residual = x
        y = self.norm2(x)
        y = torch.nn.functional.gelu(self.fc1(y), approximate="none")
        return residual + self.fc2(y)

    def mlx_named_weights(self) -> dict[str, torch.Tensor]:
        return {
            "patch_embed.weight": self.patch_embed.weight,
            "patch_embed.bias": self.patch_embed.bias,
            "pos_embed": self.pos_embed,
            "norm1.weight": self.norm1.weight,
            "norm1.bias": self.norm1.bias,
            "attn.qkv.weight": self.qkv.weight,
            "attn.qkv.bias": self.qkv.bias,
            "attn.proj.weight": self.proj.weight,
            "attn.proj.bias": self.proj.bias,
            "norm2.weight": self.norm2.weight,
            "norm2.bias": self.norm2.bias,
            "mlp.fc1.weight": self.fc1.weight,
            "mlp.fc1.bias": self.fc1.bias,
            "mlp.fc2.weight": self.fc2.weight,
            "mlp.fc2.bias": self.fc2.bias,
        }


def test_tiny_mlx_video_transformer_matches_pytorch_fixture():
    torch.manual_seed(7)
    config = TinyVJEPA2Config()
    model = TinyTorchVJEPA2(config).eval()
    video = torch.randn(2, config.frames, config.image_size, config.image_size, config.in_chans)

    with torch.no_grad():
        expected = model(video).numpy()

    weights = torch_state_to_mlx(model.mlx_named_weights())
    actual_mx = tiny_vjepa2_forward(mx.array(video.numpy()), weights, config)
    mx.eval(actual_mx)
    actual = np.array(actual_mx)

    max_abs = float(np.max(np.abs(expected - actual)))
    cosine = float(
        np.dot(expected.reshape(-1), actual.reshape(-1))
        / (np.linalg.norm(expected.reshape(-1)) * np.linalg.norm(actual.reshape(-1)))
    )
    assert max_abs < 2e-4
    assert cosine >= 0.99999
