#!/usr/bin/env python3
"""Run one local no-text video prediction case in an isolated process.

The Mac notebook calls this helper with a subprocess timeout so expensive
V-JEPA/PyTorch video extraction can be killed cleanly without wedging the
notebook kernel.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import pandas as pd
from neuralset.events.utils import standardize_events

from tribev2.demo_utils import TribeModel


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
    stream_types = {s.get("codec_type") for s in meta.get("streams", [])}
    return duration, "video" in stream_types, "audio" in stream_types


def build_events(path: Path, mode: str, cache_folder: Path) -> pd.DataFrame:
    duration, has_video, has_audio = ffprobe_streams(path)
    if mode == "video_audio_no_text":
        if not has_video or not has_audio:
            raise ValueError(f"Expected audio+video streams for {path}")
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
        rows = [
            {
                "type": "Video",
                "filepath": str(path),
                "start": 0,
                "duration": duration,
                "timeline": "default",
                "subject": "default",
            },
            {
                "type": "Audio",
                "filepath": str(audio_path),
                "start": 0,
                "duration": duration,
                "timeline": "default",
                "subject": "default",
            },
        ]
    else:
        if not has_video or has_audio:
            raise ValueError(f"Expected video-only stream for {path}")
        rows = [
            {
                "type": "Video",
                "filepath": str(path),
                "start": 0,
                "duration": duration,
                "timeline": "default",
                "subject": "default",
            }
        ]
    events = standardize_events(pd.DataFrame(rows))
    if "Word" in set(events["type"].dropna()):
        raise AssertionError("No-text video case unexpectedly produced Word events")
    return events


def enable_experimental_mps_extractors(model: Any, device: str, enabled: bool) -> bool:
    if device != "mps" or not enabled:
        return False
    patched: list[str] = []

    def force_device(obj: Any, attr: str = "device") -> None:
        if obj is not None and hasattr(obj, attr):
            object.__setattr__(obj, attr, "mps")
            patched.append(f"{type(obj).__name__}.{attr}")

    for attr in ("text_feature", "audio_feature"):
        force_device(getattr(model.data, attr, None))
    for attr in ("video_feature", "image_feature"):
        extractor = getattr(model.data, attr, None)
        force_device(getattr(extractor, "image", None))
    return bool(patched)


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    args.cache_folder.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    report: dict[str, Any] = {
        "case": args.case,
        "path": str(args.path),
        "mode": args.mode,
        "device": args.device,
        "mps_extractors_requested": args.use_mps_extractors,
        "status": "started",
    }
    try:
        events = build_events(args.path, args.mode, args.cache_folder)
        report["event_types"] = sorted(events["type"].dropna().unique().tolist())
        model = TribeModel.from_pretrained(
            "facebook/tribev2",
            cache_folder=args.cache_folder,
            device=args.device,
            config_update={
                "data.text_feature.model_name": "unsloth/Llama-3.2-3B",
                "data.text_feature.device": "cpu" if args.device == "mps" else args.device,
                "data.audio_feature.device": "cpu" if args.device == "mps" else args.device,
                "data.video_feature.image.device": "cpu" if args.device == "mps" else args.device,
                "data.image_feature.image.device": "cpu" if args.device == "mps" else args.device,
            },
        )
        report["mps_extractors_enabled"] = enable_experimental_mps_extractors(
            model, args.device, args.use_mps_extractors
        )
        preds, segments = model.predict(events=events, verbose=False)
        report.update(
            {
                "status": "completed",
                "elapsed_seconds": time.perf_counter() - t0,
                "preds_shape": list(preds.shape),
                "segments": len(segments),
            }
        )
    except Exception as exc:
        report.update(
            {
                "status": "failed",
                "elapsed_seconds": time.perf_counter() - t0,
                "error_type": type(exc).__name__,
                "error": str(exc)[:2000],
            }
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--mode", choices=["video_audio_no_text", "video_only_no_text"], required=True)
    parser.add_argument("--cache-folder", type=Path, default=Path("cache_mac_m2"))
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--use-mps-extractors", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
