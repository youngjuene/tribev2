#!/usr/bin/env python3
"""Probe Apple Silicon backend support for TRIBE v2.

This script is intentionally evidence-oriented: it verifies MLX itself, checks
PyTorch MPS, confirms why a literal `mlx` device cannot be passed through the
released PyTorch/neuralset pipeline, and records available MLX model candidates.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _safe_call(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except Exception:
        return None


def _hf_search(query: str, limit: int = 5) -> dict[str, Any]:
    try:
        from huggingface_hub import list_models

        return {
            "status": "ok",
            "models": [
                {
                    "model_id": m.modelId,
                    "library_name": getattr(m, "library_name", None),
                    "pipeline_tag": getattr(m, "pipeline_tag", None),
                }
                for m in list_models(search=query, limit=limit)
            ],
        }
    except Exception as exc:  # network/offline safe
        return {"status": "error", "type": type(exc).__name__, "message": str(exc)}


def run_probe() -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": {
            "cpu": _safe_call(["sysctl", "-n", "machdep.cpu.brand_string"]),
            "mem_bytes": _safe_call(["sysctl", "-n", "hw.memsize"]),
            "macos": _safe_call(["sw_vers", "-productVersion"]),
        },
        "imports": {},
        "mlx": {},
        "torch": {},
        "neuralset": {},
        "hf_search": {},
        "capability_matrix": [],
    }

    try:
        import mlx.core as mx

        report["imports"]["mlx"] = True
        report["mlx"]["default_device"] = str(mx.default_device())
        a = mx.random.normal((1024, 1024))
        b = mx.random.normal((1024, 1024))
        start = time.perf_counter()
        c = a @ b
        mx.eval(c)
        report["mlx"]["matmul_1024_seconds"] = time.perf_counter() - start
        report["mlx"]["matmul_shape"] = list(c.shape)
    except Exception as exc:
        report["imports"]["mlx"] = False
        report["mlx"]["error"] = {"type": type(exc).__name__, "message": str(exc)}

    try:
        import torch

        report["imports"]["torch"] = True
        report["torch"]["version"] = torch.__version__
        report["torch"]["mps_available"] = bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        try:
            torch.nn.Linear(2, 2).to("mlx")
            report["torch"]["device_mlx"] = {"accepted": True}
        except Exception as exc:
            report["torch"]["device_mlx"] = {
                "accepted": False,
                "type": type(exc).__name__,
                "message": str(exc),
            }
    except Exception as exc:
        report["imports"]["torch"] = False
        report["torch"]["error"] = {"type": type(exc).__name__, "message": str(exc)}

    try:
        from neuralset.extractors.base import HuggingFaceExtractor

        field = HuggingFaceExtractor.model_fields["device"]
        report["imports"]["neuralset"] = True
        report["neuralset"]["huggingface_extractor_device_annotation"] = str(
            field.annotation
        )
        report["neuralset"]["accepted_device_literals"] = [
            "auto",
            "cpu",
            "cuda",
            "accelerate",
        ]
    except Exception as exc:
        report["imports"]["neuralset"] = False
        report["neuralset"]["error"] = {"type": type(exc).__name__, "message": str(exc)}

    searches = {
        "vjepa2": "mlx vjepa2",
        "dinov2": "mlx dinov2",
        "wav2vec_bert": "mlx wav2vec bert",
        "llama_3_2_3b": "mlx llama 3.2 3b",
    }
    for key, query in searches.items():
        report["hf_search"][key] = _hf_search(query)

    report["capability_matrix"] = [
        {
            "component": "TRIBE brain model",
            "current_implementation": "PyTorch checkpoint loaded with torch.load",
            "mlx_candidate": None,
            "status": "not_ported",
            "note": "Released checkpoint executes as a torch.nn.Module; not MLX-native.",
        },
        {
            "component": "V-JEPA2 video extractor",
            "current_implementation": "Hugging Face PyTorch model facebook/vjepa2-vitg-fpc64-256",
            "mlx_candidate": report["hf_search"].get("vjepa2", {}).get("models", []),
            "status": "blocked",
            "note": "No MLX V-JEPA2 candidate found by default search.",
        },
        {
            "component": "DINOv2 image extractor",
            "current_implementation": "Hugging Face PyTorch model facebook/dinov2-large",
            "mlx_candidate": report["hf_search"].get("dinov2", {}).get("models", []),
            "status": "portable",
            "note": "MLX DINOv2 candidates exist, but active released feature contract must be checked first.",
        },
        {
            "component": "Wav2Vec-BERT audio extractor",
            "current_implementation": "Hugging Face PyTorch model facebook/w2v-bert-2.0",
            "mlx_candidate": report["hf_search"].get("wav2vec_bert", {}).get("models", []),
            "status": "blocked",
            "note": "No MLX Wav2Vec-BERT candidate found by default search.",
        },
        {
            "component": "Llama text extractor",
            "current_implementation": "Hugging Face PyTorch Llama-compatible hidden states",
            "mlx_candidate": report["hf_search"].get("llama_3_2_3b", {}).get("models", []),
            "status": "portable",
            "note": "MLX Llama candidates exist; hidden-state parity still required.",
        },
        {
            "component": "WhisperX/audio transcription",
            "current_implementation": "uvx whisperx using cuda/cpu selection",
            "mlx_candidate": None,
            "status": "not_in_scope",
            "note": "Not in the no-text local smoke path; not MLX-enabled in current code.",
        },
        {
            "component": "Plotting",
            "current_implementation": "PyVista/matplotlib/nilearn stack",
            "mlx_candidate": None,
            "status": "native",
            "note": "Not a neural inference bottleneck.",
        },
    ]
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Print JSON report")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    report = run_probe()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"MLX device: {report.get('mlx', {}).get('default_device')}")
        print(f"PyTorch MPS available: {report.get('torch', {}).get('mps_available')}")
        print(
            "torch device='mlx' accepted:",
            report.get("torch", {}).get("device_mlx", {}).get("accepted"),
        )


if __name__ == "__main__":
    main()
