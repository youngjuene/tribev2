import json
import subprocess
import sys
from pathlib import Path


def test_vjepa_memory_preflight_emits_decision(tmp_path: Path):
    output = tmp_path / "preflight.json"
    subprocess.check_call(
        [
            sys.executable,
            "tools/estimate_vjepa2_vitg_memory.py",
            "--model-id",
            "facebook/vjepa2-vitg-fpc64-256",
            "--seconds",
            "1",
            "--width",
            "256",
            "--num-frames",
            "64",
            "--output",
            str(output),
        ]
    )
    report = json.loads(output.read_text())
    assert report["model_id"] == "facebook/vjepa2-vitg-fpc64-256"
    assert report["config"]["hidden_size"] == 1408
    assert report["activation_estimate"]["tokens"] == 8192
    assert report["parameter_estimate"]["total_parameters"] > 1_000_000_000
    assert report["selected_policy"]["dtype"] == "float32"
    assert report["selected_policy"]["policy"] == "correctness"
    assert report["selected_policy"]["correctness_policy"] == "float32"
    assert report["runtime_dtype"] == "float32"
    assert report["artifact_role"] == "correctness"
    assert report["decision"] in {
        "allow_full_load",
        "require_chunking",
        "require_quantization",
        "blocked_on_memory",
        "require_measurement",
    }
    assert report["selected_policy"]["selected_weight_memory_bytes"] > 0


def test_vjepa_memory_preflight_blocks_tiny_budget(tmp_path: Path):
    output = tmp_path / "blocked.json"
    subprocess.check_call(
        [
            sys.executable,
            "tools/estimate_vjepa2_vitg_memory.py",
            "--model-id",
            "facebook/vjepa2-vitg-fpc64-256",
            "--max-process-memory-gb",
            "1",
            "--output",
            str(output),
        ]
    )
    report = json.loads(output.read_text())
    assert report["decision"] in {"blocked_on_memory", "require_quantization"}
    assert "Do not attempt full ViT-g" in report["stop_rule"]


def test_vjepa_memory_preflight_marks_lower_precision_as_optimization(tmp_path: Path):
    output = tmp_path / "fp16.json"
    subprocess.check_call(
        [
            sys.executable,
            "tools/estimate_vjepa2_vitg_memory.py",
            "--model-id",
            "facebook/vjepa2-vitg-fpc64-256",
            "--dtype",
            "float16",
            "--output",
            str(output),
        ]
    )
    report = json.loads(output.read_text())
    assert report["selected_policy"]["dtype"] == "float16"
    assert report["selected_policy"]["policy"] == "optimization"
    assert report["selected_policy"]["correctness_policy"] == "float32"
    assert report["artifact_role"] == "optimization"
