#!/usr/bin/env python3
"""Memory preflight for a full V-JEPA2 ViT-g MLX attempt.

This is a static estimator.  It loads only configuration metadata by default and
never instantiates the full model.  When a local safetensors checkpoint is
already cached, it may inspect tensor shapes through safetensors metadata without
materializing full tensors.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
from pathlib import Path
from typing import Any

from huggingface_hub import try_to_load_from_cache
from transformers import VJEPA2Config

DEFAULT_MODEL_ID = "facebook/vjepa2-vitg-fpc64-256"
DTYPE_BYTES = {
    "float32": 4,
    "bfloat16": 2,
    "float16": 2,
    "int8": 1,
    "int4": 0.5,
}


def _product(values: list[int] | tuple[int, ...]) -> int:
    total = 1
    for value in values:
        total *= int(value)
    return int(total)


def _system_memory_bytes() -> int | None:
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        return int(out)
    except Exception:
        return None


def _parameter_shapes_from_cache(model_id: str) -> dict[str, list[int]] | None:
    path = try_to_load_from_cache(model_id, "model.safetensors")
    if not path or not Path(path).exists():
        return None
    try:
        from safetensors import safe_open

        shapes: dict[str, list[int]] = {}
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                shapes[key] = list(handle.get_slice(key).get_shape())
        return shapes
    except Exception:
        return None


def _formula_parameter_count(cfg: VJEPA2Config, include_predictor: bool) -> dict[str, int]:
    hidden = int(cfg.hidden_size)
    intermediate = int(round(cfg.hidden_size * cfg.mlp_ratio))
    patch = hidden * int(cfg.in_chans) * int(cfg.tubelet_size) * int(cfg.patch_size) * int(cfg.patch_size) + hidden
    attn = 4 * hidden * hidden + (4 * hidden if cfg.qkv_bias else hidden)
    mlp = 2 * hidden * intermediate + intermediate + hidden
    norms = 4 * hidden
    encoder_layers = int(cfg.num_hidden_layers) * (attn + mlp + norms)
    final_norm = 2 * hidden
    counts = {
        "encoder_patch_embed": patch,
        "encoder_layers": encoder_layers,
        "encoder_final_norm": final_norm,
    }
    if include_predictor:
        pred_hidden = int(cfg.pred_hidden_size)
        pred_intermediate = int(round(cfg.pred_hidden_size * cfg.pred_mlp_ratio))
        pred_proj = hidden * pred_hidden + pred_hidden
        pred_attn = 4 * pred_hidden * pred_hidden + (4 * pred_hidden if cfg.qkv_bias else pred_hidden)
        pred_mlp = 2 * pred_hidden * pred_intermediate + pred_intermediate + pred_hidden
        pred_norms = 4 * pred_hidden
        pred_layers = int(cfg.pred_num_hidden_layers) * (pred_attn + pred_mlp + pred_norms)
        pred_final_norm = 2 * pred_hidden
        pred_mask = int(cfg.pred_num_mask_tokens) * pred_hidden
        counts.update(
            {
                "predictor_projection": pred_proj,
                "predictor_layers": pred_layers,
                "predictor_final_norm": pred_final_norm,
                "predictor_mask_tokens": pred_mask,
            }
        )
    return counts


def _parameter_count(cfg: VJEPA2Config, model_id: str, include_predictor: bool) -> tuple[dict[str, int], str]:
    shapes = _parameter_shapes_from_cache(model_id)
    if not shapes:
        return _formula_parameter_count(cfg, include_predictor), "formula_from_config"
    groups = {
        "encoder_patch_embed": 0,
        "encoder_layers": 0,
        "encoder_final_norm": 0,
        "predictor": 0,
    }
    for key, shape in shapes.items():
        count = _product(shape)
        if key.startswith("encoder.embeddings.patch_embeddings"):
            groups["encoder_patch_embed"] += count
        elif key.startswith("encoder.layer."):
            groups["encoder_layers"] += count
        elif key.startswith("encoder.layernorm"):
            groups["encoder_final_norm"] += count
        elif key.startswith("predictor."):
            groups["predictor"] += count
    if not include_predictor:
        groups.pop("predictor", None)
    return {k: v for k, v in groups.items() if v}, "local_safetensors_metadata"


def _activation_estimate(cfg: VJEPA2Config, *, batch_size: int, width: int, num_frames: int, dtype_bytes: float) -> dict[str, Any]:
    tokens = int(math.ceil(num_frames / cfg.tubelet_size)) * (width // cfg.patch_size) * (width // cfg.patch_size)
    hidden = int(cfg.hidden_size)
    heads = int(cfg.num_attention_heads)
    head_dim = hidden // heads
    intermediate = int(round(hidden * cfg.mlp_ratio))
    hidden_bytes = batch_size * tokens * hidden * dtype_bytes
    qkv_bytes = 3 * hidden_bytes
    mlp_bytes = batch_size * tokens * intermediate * dtype_bytes
    attention_scores_bytes = batch_size * heads * tokens * tokens * dtype_bytes
    conservative_layer_peak = hidden_bytes + qkv_bytes + mlp_bytes + attention_scores_bytes
    efficient_layer_peak = hidden_bytes + qkv_bytes + mlp_bytes
    return {
        "batch_size": batch_size,
        "tokens": tokens,
        "hidden_size": hidden,
        "num_attention_heads": heads,
        "head_dim": head_dim,
        "intermediate_size": intermediate,
        "hidden_bytes": int(hidden_bytes),
        "qkv_bytes": int(qkv_bytes),
        "mlp_bytes": int(mlp_bytes),
        "attention_scores_materialized_bytes": int(attention_scores_bytes),
        "conservative_layer_peak_bytes": int(conservative_layer_peak),
        "efficient_attention_layer_peak_bytes": int(efficient_layer_peak),
        "note": "Conservative peak assumes materialized attention scores for one layer during inference; efficient peak assumes attention does not persist the full score matrix beyond the op.",
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    cfg = VJEPA2Config.from_pretrained(args.model_id)
    dtype_bytes = DTYPE_BYTES[args.dtype]
    artifact_role = "correctness" if args.dtype == "float32" else "optimization"
    correctness_policy = "float32"
    param_groups, source = _parameter_count(cfg, args.model_id, args.include_predictor)
    param_count = int(sum(param_groups.values()))
    weight_memory = {name: int(param_count * bytes_) for name, bytes_ in DTYPE_BYTES.items()}
    activations = _activation_estimate(
        cfg,
        batch_size=args.batch_size,
        width=args.width,
        num_frames=args.num_frames,
        dtype_bytes=dtype_bytes,
    )
    total_mem = _system_memory_bytes()
    budget = int(args.max_process_memory_gb * (1024**3)) if args.max_process_memory_gb else int((total_mem or 0) * args.budget_fraction)
    selected_weight_bytes = int(weight_memory[args.dtype])
    conservative_peak = selected_weight_bytes + activations["conservative_layer_peak_bytes"]
    efficient_peak = selected_weight_bytes + activations["efficient_attention_layer_peak_bytes"]

    if not budget:
        decision = "require_measurement"
        reason = "system memory unavailable; run with --max-process-memory-gb"
    elif conservative_peak <= budget:
        decision = "allow_full_load"
        reason = "conservative one-layer inference peak is within budget"
    elif efficient_peak <= budget:
        decision = "allow_full_load"
        reason = "selected weights plus efficient-attention peak fit budget; monitor attention implementation at probe time"
    elif weight_memory["float16"] + activations["efficient_attention_layer_peak_bytes"] <= budget and args.dtype == "float32":
        decision = "require_quantization"
        reason = "float32 is high, but float16/bfloat16 policy may fit"
    else:
        decision = "blocked_on_memory"
        reason = "estimated weights plus activation peak exceed budget"

    if activations["attention_scores_materialized_bytes"] > max(budget // 2, 1) and decision == "allow_full_load":
        reason += "; materialized attention scores are large, so probe must remain timeout/metadata gated"

    return {
        "status": "ok",
        "phase": "mapping_memory_preflight",
        "model_id": args.model_id,
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
            "qkv_bias": cfg.qkv_bias,
            "skip_predictor_default": True,
        },
        "parameter_estimate": {
            "source": source,
            "include_predictor": args.include_predictor,
            "groups": param_groups,
            "total_parameters": param_count,
            "weight_memory_bytes_by_dtype": weight_memory,
        },
        "activation_estimate": activations,
        "machine": {
            "system": platform.platform(),
            "machine": platform.machine(),
            "system_memory_bytes": total_mem,
            "max_allowed_process_memory_bytes": budget,
            "budget_fraction": args.budget_fraction,
        },
        "selected_policy": {
            "dtype": args.dtype,
            "policy": artifact_role,
            "correctness_policy": correctness_policy,
            "runtime_dtype": args.dtype,
            "optimization_dtype": None if artifact_role == "correctness" else args.dtype,
            "selected_weight_memory_bytes": selected_weight_bytes,
            "efficient_attention_peak_bytes": int(efficient_peak),
            "conservative_peak_bytes": int(conservative_peak),
            "quantization": "none",
        },
        "correctness_policy": correctness_policy,
        "runtime_dtype": args.dtype,
        "artifact_role": artifact_role,
        "decision": decision,
        "decision_reason": reason,
        "stop_rule": "Do not attempt full ViT-g MLX load unless this decision is allow_full_load for the selected dtype/policy.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", choices=sorted(DTYPE_BYTES), default="float32")
    parser.add_argument("--budget-fraction", type=float, default=0.75)
    parser.add_argument("--max-process-memory-gb", type=float, default=None)
    parser.add_argument("--include-predictor", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = build_report(args)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
    if args.json or not args.output:
        print(json.dumps(report, indent=2))
    else:
        print(f"decision={report['decision']} parameters={report['parameter_estimate']['total_parameters']}")


if __name__ == "__main__":
    main()
