import json
import subprocess
import sys


def test_apple_backend_probe_json_shape():
    out = subprocess.check_output(
        [sys.executable, "tools/apple_backend_probe.py", "--json"],
        text=True,
    )
    report = json.loads(out)
    assert "mlx" in report
    assert "torch" in report
    assert "capability_matrix" in report
    assert report["torch"]["device_mlx"]["accepted"] is False
    assert any(row["component"] == "V-JEPA2 video extractor" for row in report["capability_matrix"])


def test_feature_contracts_identify_active_text_candidate():
    out = subprocess.check_output(
        [sys.executable, "tools/inspect_tribe_feature_contracts.py", "--json"],
        text=True,
    )
    report = json.loads(out)
    assert report["features_to_use"] == ["text", "audio", "video"]
    assert report["feature_dims"]["text"] == [2, 3072]
    gate = {item["modality"]: item for item in report["modality_gate"]}
    assert gate["text"]["gate_status"] == "candidate"
    assert gate["video"]["gate_status"] == "blocked"
    assert gate["image"]["gate_status"] == "backend_proof_only"
