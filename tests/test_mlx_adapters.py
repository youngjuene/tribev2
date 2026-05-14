from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tribev2.mlx_adapters import (
    MLXLmTextFeatureExtractor,
    MLXProofFeatureExtractor,
    PrecomputedFeatureExtractor,
)


def test_mlx_proof_extractor_generates_expected_shape(tmp_path: Path):
    feature_path = tmp_path / "proof.npy"
    events = pd.DataFrame(
        [
            {
                "type": "Word",
                "text": "hello",
                "start": 0.0,
                "duration": 1.0,
                "timeline": "default",
                "subject": "default",
            }
        ]
    )
    extractor = MLXProofFeatureExtractor(
        feature_path=feature_path,
        layers_count=2,
        feature_dim=8,
        time_steps=6,
        frequency=2.0,
    )
    extractor.prepare(events)
    out = extractor(events, start=0.0, duration=2.0)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 8, 4)
    assert torch.isfinite(out).all()
    assert feature_path.exists()
    assert feature_path.with_suffix(".json").exists()


def test_precomputed_extractor_slices_and_pads(tmp_path: Path):
    feature_path = tmp_path / "features.npy"
    np.save(feature_path, np.ones((2, 3, 3), dtype=np.float32))
    extractor = PrecomputedFeatureExtractor(feature_path=feature_path, frequency=2.0)
    extractor.prepare(pd.DataFrame())
    out = extractor(pd.DataFrame(), start=1.0, duration=2.0)
    assert out.shape == (2, 3, 4)
    assert torch.allclose(out[..., :1], torch.ones((2, 3, 1)))
    assert torch.allclose(out[..., 1:], torch.zeros((2, 3, 3)))


def test_mlx_lm_text_extractor_is_lazy_and_labels_non_parity(tmp_path: Path):
    extractor = MLXLmTextFeatureExtractor(
        feature_path=tmp_path / "mlx_lm.npy",
        model_name="mlx-community/Llama-3.2-3B-Instruct-4bit",
    )
    assert extractor.model_name.startswith("mlx-community/")
    assert extractor.feature_dim == 3072
    assert extractor.metadata == {}
