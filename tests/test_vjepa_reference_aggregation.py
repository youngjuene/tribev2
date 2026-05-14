import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _contract(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "vjepa2": {"model_id": "facebook/vjepa2-vitg-fpc64-256"},
                "selected_feature_contract": {
                    "target_shape_prefix": [2, 1408],
                    "target_tensor_order": ["layers", "dim", "time"],
                    "layers": [0.5, 0.75, 1.0],
                    "cache_n_layers": 20,
                    "layer_aggregation": "group_mean",
                    "token_pooling": "mean",
                    "frequency_hz": 2.0,
                },
            }
        )
    )


def test_reference_aggregation_blocks_last_hidden_only(tmp_path: Path):
    contract = tmp_path / "contract.json"
    reference = tmp_path / "reference.npz"
    out_dir = tmp_path / "out"
    _contract(contract)
    np.savez_compressed(reference, last_hidden_state=np.zeros((1, 8192, 1408), dtype=np.float32))
    subprocess.check_call(
        [
            sys.executable,
            "tools/aggregate_vjepa_reference_features.py",
            "--contract",
            str(contract),
            "--reference",
            str(reference),
            "--output-dir",
            str(out_dir),
        ]
    )
    report = json.loads((out_dir / "aggregation_report.json").read_text())
    assert report["status"] == "blocked"
    assert report["error_type"] == "InsufficientReferenceLayers"
    assert "output_hidden_states=True" in report["next_action"]
    assert not (out_dir / "aggregated_features.npy").exists()


def test_reference_aggregation_matches_neuralset_shape(tmp_path: Path):
    contract = tmp_path / "contract.json"
    reference = tmp_path / "reference.npz"
    out_dir = tmp_path / "out"
    _contract(contract)
    hidden = np.arange(41 * 8 * 1408, dtype=np.float32).reshape(41, 8, 1408)
    np.savez_compressed(reference, hidden_states=hidden)
    subprocess.check_call(
        [
            sys.executable,
            "tools/aggregate_vjepa_reference_features.py",
            "--contract",
            str(contract),
            "--reference",
            str(reference),
            "--output-dir",
            str(out_dir),
        ]
    )
    report = json.loads((out_dir / "aggregation_report.json").read_text())
    assert report["status"] == "ok"
    assert report["output_shape"] == [2, 1408, 1]
    assert report["source_hidden_states_dtype"] == "float32"
    assert report["required_hidden_states_dtype"] == "float32"
    assert report["source_dtype_status"] == "ok"
    arr = np.load(out_dir / "aggregated_features.npy")
    assert arr.shape == (2, 1408, 1)
    assert report["selected_cache_indices"][0] == 0
    assert report["selected_cache_indices"][-1] == 40
    assert report["layer_indices_after_cache_subselection"] == [9, 14, 19]


def test_reference_aggregation_blocks_float16_correctness_source(tmp_path: Path):
    contract = tmp_path / "contract.json"
    reference = tmp_path / "reference.npz"
    out_dir = tmp_path / "out"
    _contract(contract)
    hidden = np.zeros((41, 8, 1408), dtype=np.float16)
    np.savez_compressed(reference, hidden_states=hidden)
    subprocess.check_call(
        [
            sys.executable,
            "tools/aggregate_vjepa_reference_features.py",
            "--contract",
            str(contract),
            "--reference",
            str(reference),
            "--output-dir",
            str(out_dir),
        ]
    )
    report = json.loads((out_dir / "aggregation_report.json").read_text())
    assert report["status"] == "blocked"
    assert "dtype must be float32" in report["error_message"]
    assert not (out_dir / "aggregated_features.npy").exists()
