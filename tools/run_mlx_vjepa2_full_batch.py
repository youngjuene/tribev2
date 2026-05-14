#!/usr/bin/env python3
"""Run the corrected per-video MLX V-JEPA2 -> TRIBE batch.

The older notebook smoke path reused one validated ``features.npy`` for every
video.  This batch runner fixes that by generating a temporal MLX ViT-g feature
artifact for each input video, then passing that per-video artifact into the
TRIBE smoke/predict script and plotting every prediction timestep.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from run_mlx_vjepa2_tribe_smoke import ffprobe_streams
except ModuleNotFoundError:  # imported as tools.run_mlx_vjepa2_full_batch
    from tools.run_mlx_vjepa2_tribe_smoke import ffprobe_streams


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _case_inputs(data_root: Path) -> list[tuple[str, Path]]:
    cases: list[tuple[str, Path]] = []
    for case, folder in [("c3_video", data_root / "c3_video"), ("c1_video_audio", data_root / "c1_video_audio")]:
        cases.extend((case, path) for path in sorted(folder.glob("*.mp4")))
    return cases


def _feature_artifact_ok(feature_dir: Path, input_path: Path) -> bool:
    features = feature_dir / "features.npy"
    metadata = feature_dir / "metadata.json"
    if not features.exists() or not metadata.exists():
        return False
    try:
        meta = _load_json(metadata)
        arr = np.load(features, mmap_mode="r")
    except Exception:
        return False
    return (
        meta.get("status") == "ok"
        and meta.get("input") == str(input_path)
        and meta.get("temporal_pooling") is True
        and list(arr.shape[:2]) == [2, 1408]
        and int(arr.shape[-1]) > 1
        and bool(meta.get("finite")) is True
        and (meta.get("reference_parity") or {}).get("status") in {"passed", "contract_reused"}
    )


def _run_checked(cmd: list[str], *, timeout: int) -> dict[str, Any]:
    started = time.perf_counter()
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def _prediction_hash(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return hashlib.sha256(np.asarray(np.load(path), dtype=np.float32).tobytes()).hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    root: Path = args.output_root
    dirs = {
        "features": args.feature_root,
        "reports": root / "reports",
        "predictions": root / "predictions",
        "segments": root / "segments",
        "plots": root / "plots",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    items: list[dict[str, Any]] = []
    for idx, (case, input_path) in enumerate(_case_inputs(args.data_root), start=1):
        slug = f"{idx:02d}_{case}_{input_path.stem}"
        feature_dir = args.feature_root / slug
        report_path = dirs["reports"] / f"{slug}.json"
        predictions_path = dirs["predictions"] / f"{slug}.npy"
        segments_path = dirs["segments"] / f"{slug}.csv"
        plot_path = dirs["plots"] / f"{slug}.png"
        duration, _, _ = ffprobe_streams(input_path)

        item: dict[str, Any] = {
            "slug": slug,
            "case": case,
            "input": str(input_path),
            "duration_seconds": duration,
            "feature_dir": str(feature_dir),
            "report": str(report_path),
            "predictions_output": str(predictions_path),
            "segments_output": str(segments_path),
            "plot_output": str(plot_path),
        }

        if not args.skip_feature_generation and (args.force or not _feature_artifact_ok(feature_dir, input_path)):
            feature_cmd = [
                sys.executable,
                "tools/run_mlx_vjepa2_feature_probe.py",
                "--model-size",
                "vitg",
                "--input",
                str(input_path),
                "--seconds",
                str(duration),
                "--width",
                str(args.width),
                "--num-frames",
                str(args.num_frames),
                "--output-dir",
                str(feature_dir),
                "--preflight",
                str(args.preflight),
                "--mapping",
                str(args.mapping),
                "--aggregation",
                str(args.aggregation),
                "--reference-features",
                str(args.reference_features),
                "--parity-mode",
                "contract-reused",
                "--canonical-parity-metadata",
                str(args.canonical_parity_metadata),
                "--timeout-seconds",
                str(args.feature_timeout_seconds),
                "--attention-chunk-size",
                str(args.attention_chunk_size),
                "--enable-full-runtime",
                "--temporal-pooling",
            ]
            item["feature_generation"] = _run_checked(feature_cmd, timeout=args.feature_timeout_seconds + 30)

        feature_ok = _feature_artifact_ok(feature_dir, input_path)
        item["feature_ok"] = feature_ok
        if feature_ok:
            metadata = _load_json(feature_dir / "metadata.json")
            item["feature_shape"] = metadata.get("feature_shape")
            item["feature_frequency_hz"] = metadata.get("feature_frequency_hz")
            item["feature_sha256"] = _sha256(feature_dir / "features.npy")
        else:
            item["status"] = "blocked"
            item["blocked_phase"] = "feature_generation"
            items.append(item)
            continue

        if args.force or not report_path.exists():
            smoke_cmd = [
                sys.executable,
                "tools/run_mlx_vjepa2_tribe_smoke.py",
                "--case",
                case,
                "--input",
                str(input_path),
                "--features",
                str(feature_dir / "features.npy"),
                "--metadata",
                str(feature_dir / "metadata.json"),
                "--output",
                str(report_path),
                "--predictions-output",
                str(predictions_path),
                "--segments-output",
                str(segments_path),
                "--plot-output",
                str(plot_path),
                "--plot-timesteps",
                "0",
                "--feature-frequency",
                str(args.smoke_feature_frequency),
                "--device",
                args.device,
                "--audio-device",
                args.audio_device,
                "--cache-folder",
                str(args.cache_folder),
                "--timeout-seconds",
                str(args.smoke_timeout_seconds),
                "--json",
            ]
            item["prediction_run"] = _run_checked(smoke_cmd, timeout=args.smoke_timeout_seconds + 30)

        report = _load_json(report_path) if report_path.exists() else {"status": "missing"}
        item["status"] = report.get("status")
        item["predictions_shape"] = report.get("predictions_shape")
        item["predictions_finite"] = report.get("predictions_finite")
        item["segments"] = report.get("segments")
        item["plot_timesteps"] = report.get("plot_timesteps")
        item["video_feature_frequency_hz"] = report.get("video_feature_frequency_hz")
        item["prediction_sha256"] = _prediction_hash(predictions_path)
        items.append(item)

    completed = [item for item in items if item.get("status") == "completed"]
    prediction_hashes = [item["prediction_sha256"] for item in completed if item.get("prediction_sha256")]
    feature_hashes = [item["feature_sha256"] for item in items if item.get("feature_sha256")]
    failures = [item for item in items if item.get("status") != "completed"]
    validation = {
        "status": "passed" if not failures and len(set(feature_hashes)) == len(feature_hashes) and len(set(prediction_hashes)) == len(prediction_hashes) else "failed",
        "items": len(items),
        "completed": len(completed),
        "failed": failures,
        "feature_hashes_unique": len(set(feature_hashes)),
        "feature_hashes_total": len(feature_hashes),
        "prediction_hashes_unique": len(set(prediction_hashes)),
        "prediction_hashes_total": len(prediction_hashes),
        "shape_11_20484": sum(1 for item in completed if item.get("predictions_shape") == [11, 20484]),
        "finite_predictions": sum(1 for item in completed if item.get("predictions_finite") is True),
        "full_timeline_plots": sum(1 for item in completed if item.get("plot_timesteps") == item.get("segments")),
    }
    summary = {
        "status": "completed" if validation["status"] == "passed" else "blocked",
        "phase": "mlx_vjepa2_full_per_video_temporal_batch",
        "root": str(root),
        "directories": {key: str(value) for key, value in dirs.items()},
        "items": items,
        "validation": validation,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    _write_json(root / "summary.json", summary)
    _write_json(root / "validation.json", validation)
    _write_json(
        root / "index.json",
        {
            "root": str(root),
            "summary": str(root / "summary.json"),
            "validation": str(root / "validation.json"),
            "count": len(items),
            "directories": {key: str(value) for key, value in dirs.items()},
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--feature-root", type=Path, default=Path("cache_mac_m2_verification/mlx_vjepa_vitg_per_video_temporal"))
    parser.add_argument("--output-root", type=Path, default=Path("cache_mac_m2/mlx_vjepa_predict_reports/all24_c1_c3_per_video_temporal"))
    parser.add_argument("--cache-folder", type=Path, default=Path("cache_mac_m2/tribe_smoke_cache_per_video_temporal"))
    parser.add_argument("--preflight", type=Path, default=Path("cache_mac_m2_verification/vjepa_vitg_memory_preflight.json"))
    parser.add_argument("--mapping", type=Path, default=Path("cache_mac_m2_verification/vjepa_mlx_weight_mapping_float32.json"))
    parser.add_argument("--aggregation", type=Path, default=Path("cache_mac_m2_verification/vjepa_reference_float32/aggregation_report.json"))
    parser.add_argument("--reference-features", type=Path, default=Path("cache_mac_m2_verification/vjepa_reference_float32/aggregated_features.npy"))
    parser.add_argument("--canonical-parity-metadata", type=Path, default=Path("cache_mac_m2_verification/mlx_vjepa_vitg/metadata.json"))
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--attention-chunk-size", type=int, default=128)
    parser.add_argument("--feature-timeout-seconds", type=int, default=600)
    parser.add_argument("--smoke-timeout-seconds", type=int, default=300)
    parser.add_argument(
        "--smoke-feature-frequency",
        type=float,
        default=2.0,
        help="Feature frequency passed to TRIBE for precomputed video slicing; keep 2.0 to match the released video/audio contract.",
    )
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--audio-device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--skip-feature-generation", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
