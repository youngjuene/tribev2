#!/usr/bin/env python3
"""Run a tiny MLX-assisted text-feature proof through released TRIBE.

This is intentionally labelled proof/non-equivalent: the text features are
deterministic MLX-generated tensors with the released text feature shape, not
Llama hidden states.  The purpose is to verify that an MLX-generated artifact can
cross the neuralset/TRIBE boundary through a repo-local shim and produce a brain
prediction without invoking the original text extractor.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
from neuralset.events.utils import standardize_events

from tribev2.demo_utils import TribeModel
from tribev2.mlx_adapters import MLXLmTextFeatureExtractor, MLXProofFeatureExtractor


def build_word_events() -> pd.DataFrame:
    return standardize_events(
        pd.DataFrame(
            [
                {
                    "type": "Word",
                    "text": "mlx",
                    "context": "mlx proof feature",
                    "sentence": "mlx proof feature",
                    "sequence_id": 0,
                    "filepath": "mlx_proof_text",
                    "start": 0.0,
                    "duration": 1.0,
                    "timeline": "default",
                    "subject": "default",
                },
                {
                    "type": "Word",
                    "text": "proof",
                    "context": "mlx proof feature",
                    "sentence": "mlx proof feature",
                    "sequence_id": 0,
                    "filepath": "mlx_proof_text",
                    "start": 1.0,
                    "duration": 1.0,
                    "timeline": "default",
                    "subject": "default",
                },
            ]
        )
    )


def run(cache_folder: Path, device: str, adapter: str, model_name: str) -> dict[str, Any]:
    cache_folder.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    model = TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder=cache_folder,
        device=device,
        config_update={
            "data.text_feature.model_name": "unsloth/Llama-3.2-3B",
            "data.text_feature.device": "cpu",
            "data.audio_feature.device": "cpu",
            "data.video_feature.image.device": "cpu",
        },
    )
    feature_path = cache_folder / "mlx_features" / (
        "text_mlx_lm.npy" if adapter == "mlx_lm" else "text_proof.npy"
    )
    model.data.num_workers = 0
    model.data.batch_size = 1
    if adapter == "proof":
        text_feature = MLXProofFeatureExtractor(
            feature_path=feature_path,
            event_types="Word",
            layers_count=2,
            feature_dim=3072,
            time_steps=80,
            frequency=2.0,
        )
        semantic_equivalence = "backend_proof_only_not_lm_hidden_states"
    elif adapter == "mlx_lm":
        text_feature = MLXLmTextFeatureExtractor(
            feature_path=feature_path,
            event_types="Word",
            model_name=model_name,
            feature_dim=3072,
            frequency=2.0,
        )
        semantic_equivalence = "mlx_lm_final_state_not_neuralset_layer_parity"
    else:
        raise ValueError(f"Unknown adapter: {adapter}")
    model.data.text_feature = text_feature
    events = build_word_events()
    preds, segments = model.predict(events=events, verbose=False)
    elapsed = time.perf_counter() - t0
    metadata_path = feature_path.with_suffix(".json")
    feature_metadata = (
        json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    )
    return {
        "status": "completed",
        "adapter": adapter,
        "model_name": model_name if adapter == "mlx_lm" else None,
        "semantic_equivalence": semantic_equivalence,
        "device": device,
        "elapsed_seconds": elapsed,
        "preds_shape": list(preds.shape),
        "segments": len(segments),
        "feature_path": str(feature_path),
        "metadata_path": str(metadata_path),
        "feature_metadata": feature_metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-folder", type=Path, default=Path("cache_mac_m2/mlx_text_proof"))
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--adapter", default="proof", choices=["proof", "mlx_lm"])
    parser.add_argument("--model-name", default="mlx-community/Llama-3.2-3B-Instruct-4bit")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = run(args.cache_folder, args.device, args.adapter, args.model_name)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(report)


if __name__ == "__main__":
    main()
