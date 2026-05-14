import json
import subprocess
import sys
from pathlib import Path


def test_vjepa_contract_reports_released_video_dims(tmp_path: Path):
    output = tmp_path / "contract.json"
    doc = tmp_path / "contract.md"
    out = subprocess.check_output(
        [
            sys.executable,
            "tools/inspect_vjepa_video_contract.py",
            "--json",
            "--output",
            str(output),
            "--write-doc",
            str(doc),
        ],
        text=True,
    )
    report = json.loads(out)
    assert report["vjepa2"]["model_id"] == "facebook/vjepa2-vitg-fpc64-256"
    assert report["tribe"]["video_feature_dims"] == [2, 1408]
    assert report["vjepa2"]["hidden_size"] == 1408
    assert report["vjepa2"]["frames_per_clip"] == 64
    assert report["selected_feature_contract"]["target_tensor_order"] == ["layers", "dim", "time"]
    assert output.exists()
    assert "Released TRIBE video feature dims" in doc.read_text()
