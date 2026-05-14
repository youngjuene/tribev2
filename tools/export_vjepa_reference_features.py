#!/usr/bin/env python3
"""Bounded PyTorch/Hugging Face V-JEPA2 reference feature exporter.

Foundation mode permits a controlled blocker report when local weights or runtime
capacity are unavailable.  To avoid accidental multi-GB downloads during Ralph
foundation verification, model weights are loaded with ``local_files_only=True``
unless ``--allow-download`` is explicitly supplied.
"""
from __future__ import annotations

import argparse
import json
import signal
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from transformers import VJEPA2Config, VJEPA2Model, VJEPA2VideoProcessor

DEFAULT_MODEL_ID = "facebook/vjepa2-vitg-fpc64-256"


class TimeoutExpired(RuntimeError):
    pass


def _alarm_handler(signum: int, frame: Any) -> None:  # noqa: ARG001
    raise TimeoutExpired("reference export timed out")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _blocker(
    output_dir: Path,
    *,
    phase: str,
    model_id: str,
    input_path: Path,
    started: float,
    exc: BaseException,
    num_frames: int,
    width: int,
    device: str,
) -> dict[str, Any]:
    return {
        "status": "blocked",
        "phase": phase,
        "model_id": model_id,
        "input": str(input_path),
        "input_shape": {"num_frames": num_frames, "width": width},
        "device": device,
        "elapsed_seconds": round(time.time() - started, 3),
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback_tail": traceback.format_exc(limit=4),
        "invalidates_full_tribe_compatibility": True,
        "invalidates_current_machine_feasibility_only": False,
        "foundation_semantics": "Tiny parity may continue; semantic V-JEPA2 parity remains unverified.",
    }


def sample_video_frames(path: Path, *, seconds: float, width: int, num_frames: int) -> list[Image.Image]:
    """Sample RGB frames with moviepy; caller records any failure as evidence."""
    from moviepy import VideoFileClip

    frames: list[Image.Image] = []
    with VideoFileClip(str(path)) as clip:
        duration = min(float(seconds), float(clip.duration or seconds))
        times = np.linspace(0.0, max(0.0, duration - 1e-3), num_frames)
        for t in times:
            arr = clip.get_frame(float(t))
            img = Image.fromarray(arr).convert("RGB")
            if width:
                ratio = width / max(1, min(img.size))
                new_size = (max(1, int(round(img.size[0] * ratio))), max(1, int(round(img.size[1] * ratio))))
                img = img.resize(new_size)
                left = max(0, (img.size[0] - width) // 2)
                top = max(0, (img.size[1] - width) // 2)
                img = img.crop((left, top, left + width, top + width))
            frames.append(img)
    return frames


def export_reference(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cpu"
    try:
        cfg = VJEPA2Config.from_pretrained(args.model_id)
        frames = sample_video_frames(
            args.input,
            seconds=args.seconds,
            width=args.width,
            num_frames=args.num_frames,
        )
        processor = VJEPA2VideoProcessor.from_pretrained(args.model_id)
        inputs = processor(videos=frames, return_tensors="pt")
        model = VJEPA2Model.from_pretrained(
            args.model_id,
            local_files_only=not args.allow_download,
        )
        model.eval()
        import torch

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        last = out.last_hidden_state.detach().cpu().float().numpy()
        npz_path = output_dir / "reference_features.npz"
        npz_payload: dict[str, np.ndarray] = {"last_hidden_state": last}
        hidden_states_shape: list[int] | None = None
        hidden_states_dtype: str | None = None
        if args.save_hidden_states:
            if out.hidden_states is None:
                raise RuntimeError(
                    "V-JEPA2 did not return hidden_states; cannot build aggregation reference"
                )
            hidden_states = np.stack(
                [item.detach().cpu().to(dtype=torch.float32).numpy()[0] for item in out.hidden_states],
                axis=0,
            )
            if args.hidden_states_dtype == "float16":
                hidden_states = hidden_states.astype(np.float16)
            hidden_states_shape = list(hidden_states.shape)
            hidden_states_dtype = str(hidden_states.dtype)
            npz_payload["hidden_states"] = hidden_states
        np.savez_compressed(npz_path, **npz_payload)
        report = {
            "status": "ok",
            "phase": "foundation_reference_export",
            "model_id": args.model_id,
            "config": {
                "hidden_size": cfg.hidden_size,
                "frames_per_clip": cfg.frames_per_clip,
                "tubelet_size": cfg.tubelet_size,
                "patch_size": cfg.patch_size,
                "crop_size": cfg.crop_size,
            },
            "input": str(args.input),
            "input_shape": {"num_frames": len(frames), "width": args.width},
            "raw_output_shape": list(last.shape),
            "hidden_states_shape": hidden_states_shape,
            "aggregated_output_shape": "not_aggregated_foundation_reference",
            "dtype": str(last.dtype),
            "last_hidden_state_dtype": str(last.dtype),
            "hidden_states_dtype": hidden_states_dtype,
            "reference_role": "correctness" if hidden_states_dtype == "float32" else "compact_or_foundation",
            "correctness_policy": "float32",
            "device": device,
            "elapsed_seconds": round(time.time() - started, 3),
            "artifact": str(npz_path),
        }
        _write_json(output_dir / "reference_report.json", report)
        return report
    except Exception as exc:  # controlled foundation blocker
        report = _blocker(
            output_dir,
            phase="foundation_reference_export",
            model_id=args.model_id,
            input_path=args.input,
            started=started,
            exc=exc,
            num_frames=args.num_frames,
            width=args.width,
            device=device,
        )
        _write_json(output_dir / "reference_blocker.json", report)
        if not args.timeout_report:
            raise
        return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-report", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--save-hidden-states", action="store_true")
    parser.add_argument("--hidden-states-dtype", choices=["float32", "float16"], default="float32")
    args = parser.parse_args()

    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(max(1, int(args.timeout_seconds)))
    try:
        report = export_reference(args)
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
