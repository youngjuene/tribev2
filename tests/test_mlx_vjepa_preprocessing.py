from pathlib import Path

import pytest

from tools.run_mlx_vjepa2_feature_probe import _frames_to_tiny_video
from tribev2.mlx_adapters.vjepa2_mlx import TinyVJEPA2Config

mx = pytest.importorskip("mlx.core")


def test_tiny_video_preprocessing_shape_from_local_mp4():
    path = Path("data/c3_video/ll_video_bangkok_313.mp4")
    config = TinyVJEPA2Config(frames=4, image_size=8, patch_size=4, tubelet_size=2)
    video = _frames_to_tiny_video(path, seconds=1.0, width=32, config=config)
    assert tuple(video.shape) == (1, 4, 8, 8, 3)
    assert float(video.min()) >= 0.0
    assert float(video.max()) <= 1.0
