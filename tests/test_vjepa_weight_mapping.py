import json
import subprocess
import sys
from pathlib import Path


def test_vjepa_weight_mapping_dry_run_covers_encoder_keys(tmp_path: Path):
    output = tmp_path / "mapping.json"
    subprocess.check_call(
        [
            sys.executable,
            "tools/convert_vjepa2_hf_to_mlx.py",
            "--model-id",
            "facebook/vjepa2-vitg-fpc64-256",
            "--dry-run",
            "--output",
            str(output),
        ]
    )
    report = json.loads(output.read_text())
    assert report["status"] == "ok"
    assert report["full_weight_load_performed"] is False
    assert report["dtype_policy"] == "float32"
    assert report["correctness_policy"] == "float32"
    assert report["runtime_dtype"] == "float32"
    assert report["artifact_role"] == "correctness"
    assert report["config"]["hidden_size"] == 1408
    assert report["checkpoint_inspection"]["expected_critical_encoder_key_count"] == 644
    assert report["mapping"]["mapped_key_count"] == 644
    assert report["mapping"]["missing_keys"] == []
    assert report["mapping"]["shape_mismatches"] == []
    assert report["architecture_features"]["patch_embedding"]["kernel_size"] == [2, 16, 16]
    assert report["architecture_features"]["patch_embedding"]["input_transform"] == "(B,T,C,H,W)->(B,C,T,H,W)"
    assert report["architecture_features"]["rope_3d"]["resolved"] is True
    assert report["architecture_features"]["rope_3d"]["applies_to"] == ["query", "key"]
    assert report["architecture_features"]["rope_3d"]["leftover_unrotated_dims"] >= 0
    assert report["architecture_features"]["mlp_activation"]["resolved"] is True
    assert report["architecture_features"]["mlp_activation"]["hf_hidden_act"] == "gelu"
    assert report["architecture_features"]["mlp_activation"]["mlx_required_activation"] == "gelu"
    assert report["architecture_features"]["predictor"]["skip_predictor"] is True


def test_vjepa_weight_mapping_marks_fp16_as_optimization(tmp_path: Path):
    output = tmp_path / "mapping-fp16.json"
    subprocess.check_call(
        [
            sys.executable,
            "tools/convert_vjepa2_hf_to_mlx.py",
            "--model-id",
            "facebook/vjepa2-vitg-fpc64-256",
            "--dry-run",
            "--dtype",
            "float16",
            "--output",
            str(output),
        ]
    )
    report = json.loads(output.read_text())
    assert report["status"] == "ok"
    assert report["dtype_policy"] == "float16"
    assert report["correctness_policy"] == "float32"
    assert report["artifact_role"] == "optimization"


def test_vjepa_weight_mapping_refuses_non_dry_run():
    proc = subprocess.run(
        [sys.executable, "tools/convert_vjepa2_hf_to_mlx.py"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.returncode != 0
    assert "--dry-run" in (proc.stderr + proc.stdout)
