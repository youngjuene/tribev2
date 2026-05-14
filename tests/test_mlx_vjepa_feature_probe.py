import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _write_clean_vitg_gates(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    preflight = tmp_path / "preflight.json"
    mapping = tmp_path / "mapping.json"
    aggregation = tmp_path / "vjepa_reference_float32" / "aggregation_report.json"
    reference = tmp_path / "vjepa_reference_float32" / "aggregated_features.npy"
    aggregation.parent.mkdir(parents=True, exist_ok=True)
    preflight.write_text(
        json.dumps(
            {
                "decision": "allow_full_load",
                "model_id": "local-test/nonexistent-vjepa2",
                "correctness_policy": "float32",
                "runtime_dtype": "float32",
                "selected_policy": {
                    "dtype": "float32",
                    "runtime_dtype": "float32",
                    "correctness_policy": "float32",
                    "policy": "correctness",
                },
            }
        )
    )
    mapping.write_text(
        json.dumps(
            {
                "status": "ok",
                "blockers": [],
                "dtype_policy": "float32",
                "correctness_policy": "float32",
                "runtime_dtype": "float32",
                "artifact_role": "correctness",
            }
        )
    )
    aggregation.write_text(
        json.dumps(
            {
                "status": "ok",
                "output_shape": [2, 1408, 1],
                "source_hidden_states_dtype": "float32",
                "reported_hidden_states_dtype": "float32",
                "required_hidden_states_dtype": "float32",
                "artifact": str(reference),
            }
        )
    )
    np.save(reference, np.zeros((2, 1408, 1), dtype=np.float32))
    return preflight, mapping, aggregation, reference


def test_tiny_mlx_vjepa_probe_outputs_backend_feasibility_features(tmp_path: Path):
    output_dir = tmp_path / "tiny"
    subprocess.check_call(
        [
            sys.executable,
            "tools/run_mlx_vjepa2_feature_probe.py",
            "--model-size",
            "tiny",
            "--input",
            "data/c3_video/ll_video_bangkok_313.mp4",
            "--seconds",
            "1",
            "--width",
            "32",
            "--output-dir",
            str(output_dir),
        ]
    )
    report = json.loads((output_dir / "metadata.json").read_text())
    assert report["status"] == "ok"
    assert report["semantic_equivalence"] == "backend_feasibility_only"
    assert report["tribe_compatible"] is False
    arr = np.load(output_dir / "features.npy")
    assert arr.shape == (1, 8, 1)
    assert np.isfinite(arr).all()


def test_vitg_probe_blocks_when_full_runtime_not_enabled(tmp_path: Path):
    output_dir = tmp_path / "vitg"
    preflight, mapping, aggregation, reference = _write_clean_vitg_gates(tmp_path)
    subprocess.check_call(
        [
            sys.executable,
            "tools/run_mlx_vjepa2_feature_probe.py",
            "--model-size",
            "vitg",
            "--input",
            "data/c3_video/ll_video_bangkok_313.mp4",
            "--preflight",
            str(preflight),
            "--mapping",
            str(mapping),
            "--aggregation",
            str(aggregation),
            "--reference-features",
            str(reference),
            "--output-dir",
            str(output_dir),
        ]
    )
    report = json.loads((output_dir / "metadata.json").read_text())
    assert report["status"] == "blocked"
    assert report["feature_shape"] is None
    assert report["tribe_compatible"] is False
    blocker_types = {item["type"] for item in report["blockers"]}
    assert "full_runtime_not_enabled" in blocker_types
    assert report["reference_aggregation_status"] == "ok"
    assert report["correctness_provenance"]["status"] == "ok"


def test_vitg_probe_enable_full_runtime_reports_checkpoint_cache_blocker(tmp_path: Path):
    output_dir = tmp_path / "vitg-runtime"
    preflight, mapping, aggregation, reference = _write_clean_vitg_gates(tmp_path)

    subprocess.check_call(
        [
            sys.executable,
            "tools/run_mlx_vjepa2_feature_probe.py",
            "--model-size",
            "vitg",
            "--input",
            "data/c3_video/ll_video_bangkok_313.mp4",
            "--preflight",
            str(preflight),
            "--mapping",
            str(mapping),
            "--aggregation",
            str(aggregation),
            "--reference-features",
            str(reference),
            "--output-dir",
            str(output_dir),
            "--enable-full-runtime",
            "--timeout-seconds",
            "1",
        ]
    )
    report = json.loads((output_dir / "metadata.json").read_text())
    assert report["status"] == "blocked"
    assert report["runtime"]["type"] == "checkpoint_not_cached"
    blocker_types = {item["type"] for item in report["blockers"]}
    assert "full_runtime_attempt_blocked" in blocker_types
    assert "full_runtime_not_enabled" not in blocker_types


def test_vitg_probe_blocks_fp16_mapping_provenance(tmp_path: Path):
    output_dir = tmp_path / "vitg-fp16-mapping"
    preflight, mapping, aggregation, reference = _write_clean_vitg_gates(tmp_path)
    mapping.write_text(
        json.dumps(
            {
                "status": "ok",
                "blockers": [],
                "dtype_policy": "float16",
                "correctness_policy": "float32",
                "runtime_dtype": "float16",
                "artifact_role": "optimization",
            }
        )
    )
    subprocess.check_call(
        [
            sys.executable,
            "tools/run_mlx_vjepa2_feature_probe.py",
            "--model-size",
            "vitg",
            "--input",
            "data/c3_video/ll_video_bangkok_313.mp4",
            "--preflight",
            str(preflight),
            "--mapping",
            str(mapping),
            "--aggregation",
            str(aggregation),
            "--reference-features",
            str(reference),
            "--output-dir",
            str(output_dir),
            "--enable-full-runtime",
            "--timeout-seconds",
            "1",
        ]
    )
    report = json.loads((output_dir / "metadata.json").read_text())
    blocker_types = {item["type"] for item in report["blockers"]}
    assert "mapping_not_float32_correctness" in blocker_types
    assert report["runtime"] is None


def test_reference_feature_parity_helper_gates_shape_and_thresholds(tmp_path: Path):
    from tools.run_mlx_vjepa2_feature_probe import _compare_reference_features

    actual = tmp_path / "actual.npy"
    reference_dir = tmp_path / "vjepa_reference_float32"
    reference_dir.mkdir()
    reference = reference_dir / "reference.npy"
    np.save(actual, np.ones((2, 4, 1), dtype=np.float32))
    np.save(reference, np.ones((2, 4, 1), dtype=np.float32))
    passed = _compare_reference_features(actual, reference, cosine_threshold=0.99, mean_abs_threshold=0.01)
    assert passed["status"] == "passed"
    assert passed["cosine"] > 0.999999
    assert passed["reference_dtype_policy"] == "float32_correctness"

    np.save(reference, np.ones((2, 5, 1), dtype=np.float32))
    failed = _compare_reference_features(actual, reference, cosine_threshold=0.99, mean_abs_threshold=0.01)
    assert failed["status"] == "failed"
    assert failed["reason"] == "shape_mismatch"


def test_reference_feature_parity_helper_missing_reference_is_not_available(tmp_path: Path):
    from tools.run_mlx_vjepa2_feature_probe import _compare_reference_features

    actual = tmp_path / "actual.npy"
    missing = tmp_path / "missing.npy"
    np.save(actual, np.ones((2, 4, 1), dtype=np.float32))
    result = _compare_reference_features(actual, missing, cosine_threshold=0.99, mean_abs_threshold=0.01)
    assert result["status"] == "not_available"
    assert result["status"] != "passed"


def test_temporal_runtime_parity_compares_mean_over_time(tmp_path: Path):
    from tools.run_mlx_vjepa2_feature_probe import _compare_reference_features_for_runtime

    actual = tmp_path / "features.npy"
    reference_dir = tmp_path / "vjepa_reference_float32"
    reference_dir.mkdir()
    reference = reference_dir / "aggregated_features.npy"
    temporal = np.stack(
        [
            np.full((4, 3), 1.0, dtype=np.float32),
            np.full((4, 3), 2.0, dtype=np.float32),
        ]
    )
    np.save(actual, temporal)
    np.save(reference, temporal.mean(axis=-1, keepdims=True))

    result = _compare_reference_features_for_runtime(
        actual,
        reference,
        temporal_pooling=True,
        cosine_threshold=0.99,
        mean_abs_threshold=0.01,
    )

    assert result["status"] == "passed"
    assert result["comparison_pooling"] == "mean_over_temporal_bins"
    assert result["actual_temporal_shape"] == [2, 4, 3]


def test_contract_reused_parity_requires_canonical_pass(tmp_path: Path):
    from argparse import Namespace
    from tools.run_mlx_vjepa2_feature_probe import _contract_reused_parity

    metadata = tmp_path / "canonical.json"
    metadata.write_text(
        json.dumps(
            {
                "status": "ok",
                "input": "data/c3_video/ll_video_bangkok_313.mp4",
                "reference_parity": {"status": "passed", "cosine": 0.99},
            }
        )
    )

    result = _contract_reused_parity(Namespace(canonical_parity_metadata=metadata))

    assert result["status"] == "contract_reused"
    assert result["canonical_reference_parity"]["status"] == "passed"
