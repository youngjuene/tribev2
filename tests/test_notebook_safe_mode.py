import json
from pathlib import Path


def test_notebook_documents_safe_backend_flags():
    nb = json.loads(Path("tribe_demo_mac_m2.ipynb").read_text())
    source = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])
    runner = Path("tools/run_notebook_safe.py").read_text()

    assert 'RUN_LOCAL_VIDEO_PREDICTION = os.environ.get("TRIBEV2_RUN_LOCAL_VIDEO_PREDICTION", "0") == "1"' in source
    assert 'BACKEND_MODE = os.environ.get("TRIBEV2_NOTEBOOK_MODE", "safe")' in source
    assert 'RUN_FULL_MLX_BACKEND_ATTEMPT = NOTEBOOK_BACKEND_MODE == "mlx_probe"' in source
    assert 'RUN_MLX_VJEPA_REPORT = NOTEBOOK_BACKEND_MODE in {"mlx_vjepa_probe", "mlx_vjepa_predict"}' in source
    assert 'RUN_MLX_VJEPA_PREDICT = BACKEND_MODE == "mlx_vjepa_predict"' in source
    assert 'MLX_VJEPA_PREDICT_OPT_IN = os.environ.get("TRIBEV2_MLX_VJEPA_PREDICT_OPT_IN", "0") == "1"' in source
    assert 'TRIBEV2_MLX_VJEPA_PREDICT_OPT_IN=1 is required for mlx_vjepa_predict' in source
    assert "cache_mac_m2_verification/mlx_vjepa_vitg/metadata.json" in source
    assert "cache_mac_m2_verification/mlx_vjepa_tribe_c3.json" in source
    assert "cache_mac_m2_verification/mlx_vjepa_tribe_c1_video_audio.json" in source
    assert "tools/run_mlx_vjepa2_tribe_smoke.py" in source
    assert "tiny/non-ViT-g substitute" in source
    assert "text_mlx_lm" in source
    assert "mlx_probe" in source
    assert "mlx_hybrid" in source
    assert "mps_experimental" in source
    assert "mlx_vjepa_probe" in source
    assert "mlx_vjepa_predict" in source
    assert 'choices=["safe", "mlx_probe", "mlx_hybrid", "mps_experimental", "mlx_vjepa_probe", "mlx_vjepa_predict"]' in runner
    # Original full extractor prediction remains opt-in and separate from report-backed MLX V-JEPA predict.
    assert "if RUN_MLX_VJEPA_PREDICT:" in source
    assert "elif RUN_LOCAL_VIDEO_DEMO and RUN_LOCAL_VIDEO_PREDICTION" in source
    assert "VIDEO_PREDICTION_TIMEOUT_SECONDS" in source
    assert "local_video_prediction_report.json" in source
    assert "tools/run_local_video_prediction_case.py" in source
    assert 'TRIBEV2_PREFER_MPS' in source
    assert 'TRIBEV2_SMOKE_TEST_SECONDS' in source
