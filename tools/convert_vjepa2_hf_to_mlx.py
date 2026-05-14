#!/usr/bin/env python3
"""Dry-run HF V-JEPA2 to MLX mapping manifest.

The mapping milestone intentionally does not materialize MLX weights by default.
This script establishes the architecture/key contract that a later conversion
must obey and emits blockers instead of silently skipping critical tensors.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from huggingface_hub import try_to_load_from_cache
from transformers import VJEPA2Config

DEFAULT_MODEL_ID = "facebook/vjepa2-vitg-fpc64-256"


def _expected_encoder_keys(cfg: VJEPA2Config) -> dict[str, list[int]]:
    keys: dict[str, list[int]] = {
        "encoder.embeddings.patch_embeddings.proj.weight": [cfg.hidden_size, cfg.in_chans, cfg.tubelet_size, cfg.patch_size, cfg.patch_size],
        "encoder.embeddings.patch_embeddings.proj.bias": [cfg.hidden_size],
        "encoder.layernorm.weight": [cfg.hidden_size],
        "encoder.layernorm.bias": [cfg.hidden_size],
    }
    intermediate = int(round(cfg.hidden_size * cfg.mlp_ratio))
    for i in range(cfg.num_hidden_layers):
        p = f"encoder.layer.{i}"
        keys.update(
            {
                f"{p}.norm1.weight": [cfg.hidden_size],
                f"{p}.norm1.bias": [cfg.hidden_size],
                f"{p}.attention.query.weight": [cfg.hidden_size, cfg.hidden_size],
                f"{p}.attention.query.bias": [cfg.hidden_size],
                f"{p}.attention.key.weight": [cfg.hidden_size, cfg.hidden_size],
                f"{p}.attention.key.bias": [cfg.hidden_size],
                f"{p}.attention.value.weight": [cfg.hidden_size, cfg.hidden_size],
                f"{p}.attention.value.bias": [cfg.hidden_size],
                f"{p}.attention.proj.weight": [cfg.hidden_size, cfg.hidden_size],
                f"{p}.attention.proj.bias": [cfg.hidden_size],
                f"{p}.norm2.weight": [cfg.hidden_size],
                f"{p}.norm2.bias": [cfg.hidden_size],
                f"{p}.mlp.fc1.weight": [intermediate, cfg.hidden_size],
                f"{p}.mlp.fc1.bias": [intermediate],
                f"{p}.mlp.fc2.weight": [cfg.hidden_size, intermediate],
                f"{p}.mlp.fc2.bias": [cfg.hidden_size],
            }
        )
    return keys


def _actual_shapes(model_id: str) -> tuple[dict[str, list[int]], str | None]:
    path = try_to_load_from_cache(model_id, "model.safetensors")
    if not path or not Path(path).exists():
        return {}, "model.safetensors is not cached locally; dry-run used expected key patterns only"
    from safetensors import safe_open

    shapes: dict[str, list[int]] = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            shapes[key] = list(handle.get_slice(key).get_shape())
    return shapes, None


def _mlx_target_name(hf_key: str) -> str:
    key = hf_key
    key = key.replace("encoder.embeddings.patch_embeddings.proj", "encoder.patch_embed.conv3d")
    key = key.replace("encoder.layernorm", "encoder.norm")
    key = key.replace("encoder.layer.", "encoder.layers.")
    key = key.replace(".attention.query", ".attention.q_proj")
    key = key.replace(".attention.key", ".attention.k_proj")
    key = key.replace(".attention.value", ".attention.v_proj")
    key = key.replace(".attention.proj", ".attention.out_proj")
    key = key.replace(".mlp.fc1", ".mlp.fc1")
    key = key.replace(".mlp.fc2", ".mlp.fc2")
    return key


def build_mapping_report(args: argparse.Namespace) -> dict[str, Any]:
    cfg = VJEPA2Config.from_pretrained(args.model_id)
    artifact_role = "correctness" if args.dtype == "float32" else "optimization"
    correctness_policy = "float32"
    expected = _expected_encoder_keys(cfg)
    actual, actual_note = _actual_shapes(args.model_id)
    actual_keys = set(actual)
    expected_keys = set(expected)
    predictor_keys = sorted(k for k in actual_keys if k.startswith("predictor."))
    critical_unexpected = sorted(k for k in actual_keys - expected_keys if not k.startswith("predictor."))
    missing = sorted(expected_keys - actual_keys) if actual else []
    shape_mismatches = []
    if actual:
        for key in sorted(expected_keys & actual_keys):
            if list(expected[key]) != list(actual[key]):
                shape_mismatches.append({"key": key, "expected": expected[key], "actual": actual[key]})

    mapped = [
        {
            "hf_key": key,
            "mlx_key": _mlx_target_name(key),
            "shape": actual.get(key, expected[key]),
            "dtype_policy": args.dtype,
            "status": "mapped" if key not in missing and not any(item["key"] == key for item in shape_mismatches) else "blocked",
        }
        for key in sorted(expected_keys)
    ]
    blockers = []
    if missing:
        blockers.append({"type": "missing_critical_encoder_keys", "count": len(missing), "keys": missing[:50]})
    if shape_mismatches:
        blockers.append({"type": "shape_mismatch", "count": len(shape_mismatches), "items": shape_mismatches[:50]})
    if critical_unexpected:
        blockers.append({"type": "unexpected_non_predictor_keys", "count": len(critical_unexpected), "keys": critical_unexpected[:50]})

    attention_head_size = cfg.hidden_size // cfg.num_attention_heads
    rope_each = int(2 * ((attention_head_size // 3) // 2))
    rope_leftover = attention_head_size - 3 * rope_each
    arch = {
        "hf_classes_to_port": [
            "VJEPA2PatchEmbeddings3D",
            "VJEPA2Embeddings",
            "rotate_queries_or_keys",
            "VJEPA2RopeAttention",
            "VJEPA2Layer",
            "VJEPA2Encoder",
            "VJEPA2Model.get_vision_features",
        ],
        "patch_embedding": {
            "operation": "Conv3d",
            "kernel_size": [cfg.tubelet_size, cfg.patch_size, cfg.patch_size],
            "stride": [cfg.tubelet_size, cfg.patch_size, cfg.patch_size],
            "input_transform": "(B,T,C,H,W)->(B,C,T,H,W)",
            "resolved": True,
        },
        "rope_3d": {
            "resolved": True,
            "applies_to": ["query", "key"],
            "position_axes": ["frame", "height", "width"],
            "attention_head_size": attention_head_size,
            "rotary_dim_each_axis": rope_each,
            "leftover_unrotated_dims": rope_leftover,
            "source_behavior": "HF VJEPA2RopeAttention.apply_rotary_embeddings splits q/k dims into frame, height, width rotary blocks and preserves leftover dims unrotated.",
        },
        "mlp_activation": {
            "resolved": True,
            "hf_hidden_act": cfg.hidden_act,
            "mlx_required_activation": cfg.hidden_act,
            "source_behavior": "HF VJEPA2MLP applies the configured hidden_act between fc1 and fc2.",
        },
        "predictor": {
            "default_policy": "excluded_for_TRIBE_encoder_features",
            "skip_predictor": True,
            "excluded_keys": len(predictor_keys),
            "note": "TRIBE video path uses encoder features unless later evidence proves predictor outputs are required.",
        },
    }
    status = "ok" if not blockers else "blocked"
    return {
        "status": status,
        "phase": "mapping_dry_run",
        "model_id": args.model_id,
        "dry_run": args.dry_run,
        "full_weight_load_performed": False,
        "dtype_policy": args.dtype,
        "correctness_policy": correctness_policy,
        "runtime_dtype": args.dtype,
        "artifact_role": artifact_role,
        "optimization_dtype": None if artifact_role == "correctness" else args.dtype,
        "config": {
            "hidden_size": cfg.hidden_size,
            "num_hidden_layers": cfg.num_hidden_layers,
            "num_attention_heads": cfg.num_attention_heads,
            "mlp_ratio": cfg.mlp_ratio,
            "intermediate_size": int(round(cfg.hidden_size * cfg.mlp_ratio)),
            "frames_per_clip": cfg.frames_per_clip,
            "tubelet_size": cfg.tubelet_size,
            "patch_size": cfg.patch_size,
            "crop_size": cfg.crop_size,
            "model_type": cfg.model_type,
        },
        "architecture_features": arch,
        "checkpoint_inspection": {
            "source": "local_safetensors_metadata" if actual else "expected_from_config_only",
            "note": actual_note,
            "actual_key_count": len(actual_keys) if actual else None,
            "expected_critical_encoder_key_count": len(expected_keys),
            "predictor_keys_excluded_by_policy": len(predictor_keys),
        },
        "mapping": {
            "mapped_key_count": len([m for m in mapped if m["status"] == "mapped"]),
            "expected_keys": sorted(expected_keys),
            "mapped_keys": mapped,
            "missing_keys": missing,
            "unexpected_keys": critical_unexpected,
            "excluded_predictor_keys_sample": predictor_keys[:30],
            "shape_mismatches": shape_mismatches,
        },
        "blockers": blockers,
        "semantic_parity_status": "mapping_dry_run_only_no_feature_claim",
        "stop_rule": "Proceed to MLX feature probe only when status is ok, memory preflight allows the selected policy, mapping provenance matches the correctness policy, and reference aggregation parity exists.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not args.dry_run:
        raise SystemExit("Only --dry-run is implemented in the mapping milestone; full conversion belongs to a later gated step.")
    report = build_mapping_report(args)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
    if args.json or not args.output:
        print(json.dumps(report, indent=2))
    else:
        print(f"status={report['status']} mapped={report['mapping']['mapped_key_count']}")


if __name__ == "__main__":
    main()
