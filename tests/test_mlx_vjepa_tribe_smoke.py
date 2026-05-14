import json
from pathlib import Path

import numpy as np

from tools.run_mlx_vjepa2_tribe_smoke import validate_mlx_vjepa_features


def _metadata(path: Path, *, shape=(2, 1408, 1), parity_status="passed") -> None:
    path.write_text(
        json.dumps(
            {
                "status": "ok",
                "feature_shape": list(shape),
                "finite": True,
                "tribe_compatible": True,
                "reference_parity": {"status": parity_status},
            }
        )
    )


def test_validate_mlx_vjepa_features_accepts_finite_probe_artifact(tmp_path: Path):
    features = tmp_path / "features.npy"
    metadata = tmp_path / "metadata.json"
    np.save(features, np.ones((2, 1408, 1), dtype=np.float32))
    _metadata(metadata)

    report = validate_mlx_vjepa_features(features, metadata)

    assert report["status"] == "ok"
    assert report["feature_shape"] == [2, 1408, 1]
    assert report["finite"] is True


def test_validate_mlx_vjepa_features_accepts_temporal_artifact_frequency(tmp_path: Path):
    features = tmp_path / "features.npy"
    metadata = tmp_path / "metadata.json"
    np.save(features, np.ones((2, 1408, 32), dtype=np.float32))
    metadata.write_text(
        json.dumps(
            {
                "status": "ok",
                "feature_shape": [2, 1408, 32],
                "finite": True,
                "tribe_compatible": True,
                "temporal_pooling": True,
                "feature_frequency_hz": 3.2,
                "feature_duration_seconds": 10.0,
                "reference_parity": {"status": "passed"},
            }
        )
    )

    report = validate_mlx_vjepa_features(features, metadata)

    assert report["status"] == "ok"
    assert report["feature_shape"] == [2, 1408, 32]
    assert report["feature_frequency_hz"] == 3.2
    assert report["temporal_pooling"] is True


def test_validate_mlx_vjepa_features_accepts_contract_reused_parity(tmp_path: Path):
    features = tmp_path / "features.npy"
    metadata = tmp_path / "metadata.json"
    np.save(features, np.ones((2, 1408, 32), dtype=np.float32))
    metadata.write_text(
        json.dumps(
            {
                "status": "ok",
                "feature_shape": [2, 1408, 32],
                "finite": True,
                "tribe_compatible": True,
                "temporal_pooling": True,
                "feature_frequency_hz": 3.2,
                "reference_parity": {
                    "status": "contract_reused",
                    "message": "canonical parity fixture reused",
                },
            }
        )
    )

    report = validate_mlx_vjepa_features(features, metadata)

    assert report["status"] == "ok"
    assert report["metadata_parity"]["status"] == "contract_reused"


def test_validate_mlx_vjepa_features_fails_closed_for_bad_shape(tmp_path: Path):
    features = tmp_path / "features.npy"
    metadata = tmp_path / "metadata.json"
    np.save(features, np.ones((1, 8, 1), dtype=np.float32))
    _metadata(metadata, shape=(1, 8, 1))

    report = validate_mlx_vjepa_features(features, metadata)

    assert report["status"] == "blocked"
    blocker_types = {item["type"] for item in report["blockers"]}
    assert "feature_shape_prefix_mismatch" in blocker_types


def test_validate_mlx_vjepa_features_requires_passed_reference_parity(tmp_path: Path):
    features = tmp_path / "features.npy"
    metadata = tmp_path / "metadata.json"
    np.save(features, np.ones((2, 1408, 1), dtype=np.float32))
    _metadata(metadata, parity_status="not_available")

    report = validate_mlx_vjepa_features(features, metadata)

    assert report["status"] == "blocked"
    blocker_types = {item["type"] for item in report["blockers"]}
    assert "metadata_reference_parity_not_passed" in blocker_types


def test_build_events_c1_video_audio_creates_video_and_audio_rows(tmp_path: Path):
    from tools import run_mlx_vjepa2_tribe_smoke as smoke

    video = Path("data/c1_video_audio/hh_video_bangkok_021.mp4")
    events = smoke.build_events("c1_video_audio", video, tmp_path / "cache")

    assert set(events["type"]) == {"Video", "Audio"}
    assert events["duration"].min() > 0


def test_build_events_c1_requires_audio_stream(tmp_path: Path, monkeypatch):
    from tools import run_mlx_vjepa2_tribe_smoke as smoke

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not-real-video")
    monkeypatch.setattr(smoke, "ffprobe_streams", lambda path: (3.0, True, False))

    try:
        smoke.build_events("c1_video_audio", video, tmp_path / "cache")
    except ValueError as exc:
        assert "audio stream" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("c1_video_audio accepted a video without audio")
