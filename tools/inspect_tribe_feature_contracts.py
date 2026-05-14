#!/usr/bin/env python3
"""Inspect released TRIBE v2 feature contracts without running extractors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from huggingface_hub import hf_hub_download


def _nested_get(data: dict[str, Any], dotted: str) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _summarize_extractor(config: dict[str, Any] | None) -> dict[str, Any] | None:
    if config is None:
        return None
    image = config.get("image") if isinstance(config.get("image"), dict) else None
    src = image or config
    return {
        "name": config.get("name"),
        "event_types": config.get("event_types"),
        "model_name": src.get("model_name"),
        "device": src.get("device"),
        "layers": src.get("layers"),
        "cache_n_layers": src.get("cache_n_layers"),
        "layer_aggregation": src.get("layer_aggregation"),
        "token_aggregation": src.get("token_aggregation"),
        "frequency": config.get("frequency"),
        "aggregation": config.get("aggregation"),
        "image_wrapped": bool(image),
    }


def inspect_contracts(repo_id: str, checkpoint_name: str) -> dict[str, Any]:
    config_path = hf_hub_download(repo_id, "config.yaml")
    ckpt_path = hf_hub_download(repo_id, checkpoint_name)
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.UnsafeLoader)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)
    build_args = ckpt["model_build_args"]
    feature_dims = {
        key: list(value) if value is not None else None
        for key, value in build_args["feature_dims"].items()
    }
    features_to_use = _nested_get(config, "data.features_to_use") or []
    extractors = {
        name: _summarize_extractor(_nested_get(config, f"data.{name}_feature"))
        for name in ["text", "audio", "video", "image"]
    }
    modality_gate = []
    mlx_candidates = {
        "text": ["mlx-community/Llama-3.2-3B-bf16", "mlx-community/Llama-3.2-3B-8bit"],
        "audio": [],
        "video": [],
        "image": ["mlx-image/mlxim vit_large_patch14_518.dinov2"],
    }
    for modality, dims in feature_dims.items():
        active = modality in features_to_use
        candidates = mlx_candidates.get(modality, [])
        modality_gate.append(
            {
                "modality": modality,
                "active_in_features_to_use": active,
                "feature_dims": dims,
                "extractor": extractors.get(modality),
                "mlx_candidates": candidates,
                "gate_status": "candidate" if active and candidates else "blocked" if active else "inactive",
                "note": (
                    "Checkpoint-active modality with at least one known MLX candidate."
                    if active and candidates
                    else "Checkpoint-active but no known MLX candidate."
                    if active
                    else "Not active in released checkpoint feature_dims/features_to_use."
                ),
            }
        )
    # Include inactive image/DINO explicitly because it is often tempting but not checkpoint-active.
    if "image" not in feature_dims:
        modality_gate.append(
            {
                "modality": "image",
                "active_in_features_to_use": "image" in features_to_use,
                "feature_dims": None,
                "extractor": extractors.get("image"),
                "mlx_candidates": mlx_candidates["image"],
                "gate_status": "backend_proof_only",
                "note": "DINOv2 MLX candidates exist, but released model_build_args has no image feature_dims entry.",
            }
        )
    return {
        "repo_id": repo_id,
        "checkpoint_name": checkpoint_name,
        "config_path": config_path,
        "checkpoint_path": ckpt_path,
        "features_to_use": features_to_use,
        "feature_dims": feature_dims,
        "n_outputs": build_args["n_outputs"],
        "n_output_timesteps": build_args["n_output_timesteps"],
        "extractors": extractors,
        "modality_gate": modality_gate,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# TRIBE v2 MLX Feature Contracts",
        "",
        f"Repo: `{report['repo_id']}`",
        f"Checkpoint: `{report['checkpoint_name']}`",
        f"Features to use: `{report['features_to_use']}`",
        "",
        "## Feature dimensions",
        "",
    ]
    for name, dims in report["feature_dims"].items():
        lines.append(f"- `{name}`: `{dims}`")
    lines += ["", "## Modality gate", ""]
    for item in report["modality_gate"]:
        lines.append(
            f"- `{item['modality']}`: **{item['gate_status']}**; "
            f"active={item['active_in_features_to_use']}; dims=`{item['feature_dims']}`; "
            f"candidates=`{item['mlx_candidates']}`; {item['note']}"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "The first MLX-assisted inference proof should target a checkpoint-active modality. "
        "For the released checkpoint, text is the only active modality with an obvious MLX candidate from the current inventory; video and audio remain blocked without V-JEPA2/Wav2Vec-BERT MLX equivalents. DINOv2/image is useful only as a backend proof unless the model/config is changed.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="facebook/tribev2")
    parser.add_argument("--checkpoint-name", default="best.ckpt")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--write-doc", type=Path, default=None)
    args = parser.parse_args()
    report = inspect_contracts(args.repo_id, args.checkpoint_name)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
    if args.write_doc:
        write_markdown(report, args.write_doc)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("features_to_use:", report["features_to_use"])
        print("feature_dims:", report["feature_dims"])
        for item in report["modality_gate"]:
            print(item["modality"], item["gate_status"], item["note"])


if __name__ == "__main__":
    main()
