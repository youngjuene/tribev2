#!/usr/bin/env python3
"""Run a bounded TRIBE smoke using precomputed MLX V-JEPA2 video features.

This is the integration gate after the full MLX ViT-g probe.  It does not patch
neuralset or replace installed packages: the released TRIBE model is loaded as
usual, then its video extractor is replaced in-process with the repo-local
``PrecomputedFeatureExtractor`` backed by the already-validated MLX feature
artifact.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from neuralset.events.utils import standardize_events

from tribev2.demo_utils import TribeModel
from tribev2.mlx_adapters import PrecomputedFeatureExtractor


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text())


class SmokeTimeout(RuntimeError):
    pass


def _alarm_handler(signum: int, frame: Any) -> None:  # noqa: ARG001
    raise SmokeTimeout("TRIBE smoke exceeded timeout")


def ffprobe_streams(path: Path) -> tuple[float, bool, bool]:
    meta = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration:stream=codec_type",
                "-of",
                "json",
                str(path),
            ],
            text=True,
        )
    )
    duration = float(meta.get("format", {}).get("duration", 0.0))
    stream_types = {stream.get("codec_type") for stream in meta.get("streams", [])}
    return duration, "video" in stream_types, "audio" in stream_types


def _video_event(path: Path, duration: float) -> dict[str, Any]:
    return {
        "type": "Video",
        "filepath": str(path),
        "start": 0.0,
        "duration": duration,
        "timeline": "default",
        "subject": "default",
    }


def _extract_audio(path: Path, cache_folder: Path) -> Path:
    audio_dir = cache_folder / "local_extracted_audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path = audio_dir / f"{path.stem}.wav"
    if not audio_path.exists():
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(path),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "44100",
                "-ac",
                "2",
                str(audio_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return audio_path


def build_events(case: str, path: Path, cache_folder: Path) -> pd.DataFrame:
    duration, has_video, has_audio = ffprobe_streams(path)
    if duration <= 0:
        raise ValueError(f"Could not determine positive video duration for {path}")
    if not has_video:
        raise ValueError(f"Expected a video stream for {path}")
    rows: list[dict[str, Any]] = [_video_event(path, duration)]
    if case == "c1_video_audio":
        if not has_audio:
            raise ValueError(f"Expected an audio stream for {path}")
        audio_path = _extract_audio(path, cache_folder)
        rows.append(
            {
                "type": "Audio",
                "filepath": str(audio_path),
                "start": 0.0,
                "duration": duration,
                "timeline": "default",
                "subject": "default",
            }
        )
    elif case != "c3_video":
        raise ValueError(f"Unsupported case: {case}")
    return standardize_events(pd.DataFrame(rows))


def validate_mlx_vjepa_features(features_path: Path, metadata_path: Path | None = None) -> dict[str, Any]:
    if not features_path.exists():
        return {"status": "blocked", "type": "features_missing", "message": f"missing {features_path}"}
    features = np.asarray(np.load(features_path), dtype=np.float32)
    report: dict[str, Any] = {
        "features": str(features_path),
        "feature_shape": list(features.shape),
        "finite": bool(np.isfinite(features).all()),
    }
    blockers: list[dict[str, Any]] = []
    if features.ndim != 3:
        blockers.append({"type": "feature_rank_mismatch", "expected": 3, "actual": int(features.ndim)})
    if list(features.shape[:2]) != [2, 1408]:
        blockers.append({"type": "feature_shape_prefix_mismatch", "expected": [2, 1408], "actual": list(features.shape[:2])})
    if not report["finite"]:
        blockers.append({"type": "nonfinite_features", "nonfinite_count": int((~np.isfinite(features)).sum())})

    metadata = _load_json(metadata_path)
    if metadata:
        report["metadata"] = str(metadata_path)
        report["metadata_status"] = metadata.get("status")
        report["metadata_feature_shape"] = metadata.get("feature_shape")
        report["metadata_parity"] = metadata.get("reference_parity")
        report["feature_frequency_hz"] = metadata.get("feature_frequency_hz")
        report["feature_duration_seconds"] = metadata.get("feature_duration_seconds")
        report["temporal_pooling"] = metadata.get("temporal_pooling")
        if metadata.get("status") != "ok":
            blockers.append({"type": "metadata_status_not_ok", "status": metadata.get("status")})
        if metadata.get("finite") is not True:
            blockers.append({"type": "metadata_not_finite", "finite": metadata.get("finite")})
        if metadata.get("tribe_compatible") is not True:
            blockers.append({"type": "metadata_not_tribe_compatible", "tribe_compatible": metadata.get("tribe_compatible")})
        if metadata.get("feature_shape") != list(features.shape):
            blockers.append({"type": "metadata_feature_shape_mismatch", "metadata": metadata.get("feature_shape"), "actual": list(features.shape)})
        parity = metadata.get("reference_parity") or {}
        if parity.get("status") not in {"passed", "contract_reused"}:
            blockers.append({"type": "metadata_reference_parity_not_passed", "parity_status": parity.get("status")})
    else:
        blockers.append({"type": "metadata_missing", "metadata": str(metadata_path) if metadata_path else None})

    if blockers:
        return {**report, "status": "blocked", "blockers": blockers}
    return {**report, "status": "ok"}


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    report: dict[str, Any] = {
        "status": "started",
        "phase": "mlx_vjepa2_tribe_smoke",
        "case": args.case,
        "backend": "mlx-vjepa2",
        "input": str(args.input),
        "features": str(args.features),
        "metadata": str(args.metadata),
        "device": args.device,
    }
    try:
        validation = validate_mlx_vjepa_features(args.features, args.metadata)
        report["feature_validation"] = validation
        if validation.get("status") != "ok":
            return {
                **report,
                "status": "blocked",
                "blocked_phase": "feature_validation",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }

        events = build_events(args.case, args.input, args.cache_folder)
        event_types = sorted(events["type"].dropna().unique().tolist())
        report["event_types"] = event_types
        report["event_duration_seconds"] = float(events["duration"].max())
        features_to_use = ["video"] if args.case == "c3_video" else ["audio", "video"]
        config_update = {
            "data.features_to_use": features_to_use,
            "data.num_workers": 0,
            "data.batch_size": 1,
            "data.duration_trs": 100,
            "data.overlap_trs_train": 0,
        }
        if "audio" in features_to_use:
            config_update["data.audio_feature.device"] = args.audio_device
        model = TribeModel.from_pretrained(
            "facebook/tribev2",
            cache_folder=args.cache_folder,
            device=args.device,
            config_update=config_update,
        )
        model.data.video_feature = PrecomputedFeatureExtractor(
            feature_path=args.features,
            event_types="Video",
            frequency=(
                float(args.feature_frequency)
                if args.feature_frequency is not None
                else float(validation.get("feature_frequency_hz") or 2.0)
            ),
            aggregation="sum",
            layers=[0.5, 0.75, 1.0],
            layer_aggregation="group_mean",
            allow_missing=False,
            metadata={"backend": "mlx-vjepa2", "source_metadata": str(args.metadata)},
        )
        preds, segments = model.predict(events=events, verbose=False)
        if args.predictions_output:
            args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.predictions_output, np.asarray(preds, dtype=np.float32))
        if args.segments_output:
            args.segments_output.parent.mkdir(parents=True, exist_ok=True)
            if hasattr(segments, "to_csv"):
                segments.to_csv(args.segments_output, index=False)
            elif args.segments_output.suffix.lower() == ".json":
                args.segments_output.write_text(
                    json.dumps([str(item) for item in segments], indent=2)
                )
            else:
                pd.DataFrame({"segment": [str(item) for item in segments]}).to_csv(
                    args.segments_output,
                    index=False,
                )
        if args.plot_output:
            os.environ.setdefault("MPLBACKEND", "Agg")
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
            from tribev2.plotting import PlotBrain

            n_timesteps = len(preds) if args.plot_timesteps <= 0 else min(max(1, args.plot_timesteps), len(preds))
            plotter = PlotBrain(mesh="fsaverage5")
            fig = plotter.plot_timesteps(
                preds[:n_timesteps],
                segments=segments[:n_timesteps],
                cmap="fire",
                norm_percentile=99,
                vmin=0.6,
                alpha_cmap=(0, 0.2),
                show_stimuli=args.plot_show_stimuli,
            )
            args.plot_output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.plot_output, dpi=160, bbox_inches="tight")
            plt.close(fig)
        report.update(
            {
                "status": "completed",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "predictions_shape": list(preds.shape),
                "predictions_finite": bool(np.isfinite(preds).all()),
                "segments": len(segments),
                "video_feature_shape": validation["feature_shape"],
                "features_to_use": features_to_use,
                "video_blocker": False,
                "required_output_shape": ["n_segments", 20484],
                "predictions_output": str(args.predictions_output) if args.predictions_output else None,
                "segments_output": str(args.segments_output) if args.segments_output else None,
                "plot_output": str(args.plot_output) if args.plot_output else None,
                "plot_timesteps": (len(preds) if args.plot_timesteps <= 0 else min(max(1, args.plot_timesteps), len(preds))) if args.plot_output else None,
                "video_feature_frequency_hz": (
                    float(args.feature_frequency)
                    if args.feature_frequency is not None
                    else float(validation.get("feature_frequency_hz") or 2.0)
                ),
            }
        )
        if preds.ndim != 2 or preds.shape[1] != 20484 or not report["predictions_finite"]:
            report.update(
                {
                    "status": "blocked",
                    "blocked_phase": "prediction_validation",
                    "error_type": "PredictionShapeOrFinitenessMismatch",
                    "video_blocker": False,
                    "blocker_domain": "prediction_validation",
                }
            )
        return report
    except Exception as exc:
        report.update(
            {
                "status": "blocked",
                "blocked_phase": "exception",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback_tail": traceback.format_exc(limit=5),
                "video_blocker": False if report.get("feature_validation", {}).get("status") == "ok" else None,
                "blocker_domain": "audio_or_released_tribe_audio_path"
                if args.case == "c1_video_audio" and report.get("feature_validation", {}).get("status") == "ok"
                else "video_feature_validation_or_general_exception",
            }
        )
        return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["c3_video", "c1_video_audio"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=Path("cache_mac_m2_verification/mlx_vjepa_vitg/metadata.json"))
    parser.add_argument("--cache-folder", type=Path, default=Path("cache_mac_m2_verification/tribe_smoke_cache"))
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--audio-device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--predictions-output", type=Path, default=None)
    parser.add_argument("--segments-output", type=Path, default=None)
    parser.add_argument("--plot-output", type=Path, default=None)
    parser.add_argument("--plot-timesteps", type=int, default=3, help="Number of prediction timesteps to plot; <=0 plots all timesteps.")
    parser.add_argument("--feature-frequency", type=float, default=None, help="Override precomputed video feature frequency in Hz; default reads metadata feature_frequency_hz or falls back to 2.0.")
    parser.add_argument("--plot-show-stimuli", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    previous_handler = None
    if hasattr(signal, "SIGALRM"):
        previous_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(max(1, int(args.timeout_seconds)))
    try:
        result = run(args)
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
            if previous_handler is not None:
                signal.signal(signal.SIGALRM, previous_handler)
    _write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
