#!/usr/bin/env python3
"""Bounded MLX V-JEPA2 feature probe.

Tiny mode is a backend/mechanics probe using the repo's tiny MLX transformer.
ViT-g mode is strictly gated by mapping, memory, and reference aggregation
artifacts.  Until a full MLX V-JEPA2 encoder with safe 8192-token attention is
implemented, ViT-g emits a controlled blocker rather than a false success.
"""
from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from export_vjepa_reference_features import sample_video_frames
except ModuleNotFoundError:  # imported as tools.run_mlx_vjepa2_feature_probe
    from tools.export_vjepa_reference_features import sample_video_frames
from tribev2.mlx_adapters.vjepa2_mlx import (
    TinyVJEPA2Config,
    tiny_vjepa2_forward,
    vjepa2_encoder_layer_forward,
    vjepa2_patch_embed,
    _layer_norm_state,
)


CORRECTNESS_DTYPE = "float32"
DEFAULT_FLOAT32_MAPPING = Path("cache_mac_m2_verification/vjepa_mlx_weight_mapping_float32.json")
DEFAULT_FLOAT32_AGGREGATION = Path("cache_mac_m2_verification/vjepa_reference_float32/aggregation_report.json")
DEFAULT_FLOAT32_REFERENCE_FEATURES = Path("cache_mac_m2_verification/vjepa_reference_float32/aggregated_features.npy")


class ProbeTimeout(RuntimeError):
    pass


def _alarm_handler(signum, frame):  # noqa: ARG001
    raise ProbeTimeout("full ViT-g MLX runtime exceeded watchdog")


class StreamingSafeTensorsState:
    """On-demand safetensors state mapping for HF keys.

    Each key access loads one tensor from the local safetensors checkpoint and
    lets the caller cast it to the active MLX dtype.  This avoids constructing a
    full in-memory Python state dict before the probe starts.
    """

    def __init__(self, path: Path):
        from safetensors import safe_open

        self.path = Path(path)
        self._handle = safe_open(self.path, framework="pt", device="cpu")
        self.keys = set(self._handle.keys())

    def __contains__(self, key: str) -> bool:
        return key in self.keys

    def __getitem__(self, key: str):
        return self._handle.get_tensor(key)


def _local_safetensors_path(model_id: str) -> Path | None:
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(model_id, "model.safetensors")
    return Path(path) if path and Path(path).exists() else None


def _selected_cache_indices(n_layers: int, cache_n_layers: int | None) -> list[int]:
    if cache_n_layers is None or cache_n_layers >= n_layers:
        return list(range(n_layers))
    return [int(round(x)) for x in np.linspace(0, n_layers - 1, cache_n_layers)]


def _aggregation_plan(aggregation: dict[str, Any]) -> dict[str, Any]:
    raw_layers = 41  # V-JEPA2 encoder hidden states: embedding + 40 layers for ViT-g.
    cache_n_layers = aggregation.get("cache_n_layers")
    selected = _selected_cache_indices(raw_layers, cache_n_layers)
    n_model_layers = len(selected)
    layers = aggregation.get("layers")
    layer_indices = np.unique([int(float(i) * (n_model_layers - 1)) for i in layers]).tolist()
    boundaries = list(layer_indices)
    boundaries[-1] += 1
    groups = []
    for l1, l2 in zip(boundaries[:-1], boundaries[1:]):
        groups.append(selected[l1:l2])
    needed = sorted(set(i for group in groups for i in group))
    return {"selected_cache_indices": selected, "layer_indices": layer_indices, "groups": groups, "needed_raw_indices": needed}


def _temporal_token_pool(hidden: Any, config: Any) -> np.ndarray:
    """Pool V-JEPA tokens spatially while preserving tubelet time bins.

    The MLX patch embedder emits tokens in temporal-major order:
    ``time, patch_y, patch_x``.  Reshaping through that contract turns a single
    64-frame ViT-g forward pass into a TRIBE-compatible temporal feature field
    instead of the old one-bin all-token mean.
    """
    import mlx.core as mx

    grid_size = int(config.crop_size) // int(config.patch_size)
    spatial_tokens = grid_size * grid_size
    token_count = int(hidden.shape[1])
    if token_count % spatial_tokens:
        raise ValueError(
            f"token_count={token_count} is not divisible by spatial grid tokens={spatial_tokens}"
        )
    grid_depth = token_count // spatial_tokens
    pooled = mx.mean(mx.reshape(hidden[0], (grid_depth, spatial_tokens, int(config.hidden_size))), axis=1)
    mx.eval(pooled)
    return np.asarray(pooled, dtype=np.float32)  # [time, dim]


def _global_token_pool(hidden: Any) -> np.ndarray:
    import mlx.core as mx

    pooled = mx.mean(hidden, axis=1)[0]
    mx.eval(pooled)
    return np.asarray(pooled, dtype=np.float32)  # [dim]


def _preprocess_vitg_video(input_path: Path, *, seconds: float, width: int, num_frames: int):
    import mlx.core as mx
    from transformers import VJEPA2VideoProcessor

    frames = sample_video_frames(input_path, seconds=seconds, width=width, num_frames=num_frames)
    processor = VJEPA2VideoProcessor.from_pretrained("facebook/vjepa2-vitg-fpc64-256")
    pixel_values = processor(videos=frames, return_tensors="np")["pixel_values_videos"]
    # Keep the full runtime in float32 by default.  The first watchdoged ViT-g
    # attempt showed fp16 accumulation can complete shape-wise while producing
    # NaNs, which must never be reported as a usable TRIBE feature artifact.
    return mx.array(pixel_values.astype(np.float32))


def _run_full_vitg_runtime(args: argparse.Namespace, *, preflight: dict[str, Any], aggregation: dict[str, Any]) -> dict[str, Any]:
    import mlx.core as mx
    from transformers import VJEPA2Config

    model_id = preflight.get("model_id") or "facebook/vjepa2-vitg-fpc64-256"
    checkpoint_path = _local_safetensors_path(model_id)
    if checkpoint_path is None:
        return {"status": "blocked", "type": "checkpoint_not_cached", "message": "model.safetensors is not available in local HF cache"}
    config = VJEPA2Config.from_pretrained(model_id)
    state = StreamingSafeTensorsState(checkpoint_path)
    plan = _aggregation_plan(aggregation)
    output_dir: Path = args.output_dir
    started = time.time()
    token_means: dict[int, np.ndarray] = {}
    progress_path = output_dir / "runtime_progress.jsonl"

    def record(event: dict[str, Any]) -> None:
        with progress_path.open("a") as fh:
            fh.write(json.dumps({**event, "elapsed_seconds": round(time.time() - started, 3)}) + "\n")

    previous_handler = None
    if hasattr(signal, "SIGALRM"):
        previous_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(max(1, int(args.timeout_seconds)))
    try:
        video = _preprocess_vitg_video(args.input, seconds=args.seconds, width=args.width, num_frames=args.num_frames)
        record({"event": "preprocessed", "shape": list(video.shape), "dtype": str(video.dtype)})
        hidden = vjepa2_patch_embed(video, state, config)
        mx.eval(hidden)
        if 0 in plan["needed_raw_indices"]:
            token_means[0] = (
                _temporal_token_pool(hidden, config)
                if args.temporal_pooling
                else _global_token_pool(hidden)
            )
        record({"event": "patch_embed", "hidden_shape": list(hidden.shape)})
        chunk_size = args.attention_chunk_size
        for layer_idx in range(int(config.num_hidden_layers)):
            hidden = vjepa2_encoder_layer_forward(hidden, state, config, layer_idx, chunk_size=chunk_size)
            mx.eval(hidden)
            raw_idx = layer_idx + 1
            if raw_idx in plan["needed_raw_indices"]:
                pooled_np = (
                    _temporal_token_pool(hidden, config)
                    if args.temporal_pooling
                    else _global_token_pool(hidden)
                )
                if not np.isfinite(pooled_np).all():
                    record({"event": "nonfinite_collected_layer", "layer_idx": layer_idx, "raw_idx": raw_idx})
                    return {
                        "status": "blocked",
                        "type": "nonfinite_layer_features",
                        "message": f"non-finite pooled activations at raw hidden-state index {raw_idx}",
                        "raw_idx": raw_idx,
                        "progress": str(progress_path),
                        "runtime_elapsed_seconds": round(time.time() - started, 3),
                        "aggregation_plan": plan,
                    }
                token_means[raw_idx] = pooled_np
            record({"event": "layer", "layer_idx": layer_idx, "raw_idx": raw_idx, "collected": raw_idx in token_means})
        hidden = _layer_norm_state(hidden, state, "encoder.layernorm", float(config.layer_norm_eps))
        mx.eval(hidden)
        if int(config.num_hidden_layers) in plan["needed_raw_indices"]:
            token_means[int(config.num_hidden_layers)] = (
                _temporal_token_pool(hidden, config)
                if args.temporal_pooling
                else _global_token_pool(hidden)
            )
        grouped = []
        for group in plan["groups"]:
            missing = [idx for idx in group if idx not in token_means]
            if missing:
                return {"status": "blocked", "type": "missing_collected_layers", "missing": missing, "progress": str(progress_path)}
            grouped.append(np.stack([token_means[idx] for idx in group]).mean(axis=0))
        if args.temporal_pooling:
            # grouped: list[[time, dim]] -> [layers, dim, time]
            features = np.stack([group.T for group in grouped]).astype(np.float32)
        else:
            features = np.stack(grouped).astype(np.float32)[..., None]
        features_path = output_dir / "features.npy"
        np.save(features_path, features)
        finite = bool(np.isfinite(features).all())
        if not finite:
            return {
                "status": "blocked",
                "type": "nonfinite_final_features",
                "message": "full ViT-g MLX runtime produced non-finite aggregated features",
                "features": str(features_path),
                "feature_shape": list(features.shape),
                "finite": False,
                "nonfinite_count": int((~np.isfinite(features)).sum()),
                "progress": str(progress_path),
                "runtime_elapsed_seconds": round(time.time() - started, 3),
                "aggregation_plan": plan,
            }
        return {
            "status": "ok",
            "features": str(features_path),
            "feature_shape": list(features.shape),
            "temporal_pooling": bool(args.temporal_pooling),
            "feature_frequency_hz": float(features.shape[-1] / args.seconds) if args.seconds > 0 else None,
            "feature_duration_seconds": float(args.seconds),
            "finite": True,
            "progress": str(progress_path),
            "runtime_elapsed_seconds": round(time.time() - started, 3),
            "aggregation_plan": plan,
        }
    except ProbeTimeout as exc:
        return {"status": "blocked", "type": "full_runtime_timeout", "message": str(exc), "progress": str(progress_path), "runtime_elapsed_seconds": round(time.time() - started, 3), "aggregation_plan": plan}
    except Exception as exc:
        return {"status": "blocked", "type": type(exc).__name__, "message": str(exc), "progress": str(progress_path), "runtime_elapsed_seconds": round(time.time() - started, 3), "aggregation_plan": plan}
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
            if previous_handler is not None:
                signal.signal(signal.SIGALRM, previous_handler)



def _compare_reference_features(features_path: Path, reference_path: Path, *, cosine_threshold: float, mean_abs_threshold: float) -> dict[str, Any]:
    if not reference_path.exists():
        return {"status": "not_available", "reference": str(reference_path), "message": "reference aggregation artifact not found"}
    actual = np.asarray(np.load(features_path), dtype=np.float32)
    expected = np.asarray(np.load(reference_path), dtype=np.float32)
    if actual.shape != expected.shape:
        return {
            "status": "failed",
            "reference": str(reference_path),
            "reason": "shape_mismatch",
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
        }
    diff = actual.astype(np.float64) - expected.astype(np.float64)
    actual_flat = actual.reshape(-1).astype(np.float64)
    expected_flat = expected.reshape(-1).astype(np.float64)
    denom = float(np.linalg.norm(actual_flat) * np.linalg.norm(expected_flat))
    cosine = float(np.dot(actual_flat, expected_flat) / denom) if denom else 0.0
    mean_abs = float(np.mean(np.abs(diff)))
    max_abs = float(np.max(np.abs(diff)))
    passed = cosine >= cosine_threshold and mean_abs <= mean_abs_threshold
    reference_note = (
        "Reference hidden states are persisted as float32 for the correctness lane; this gate records cosine/mean-absolute tolerance because MLX and HF kernels are not bit-identical."
        if "vjepa_reference_float32" in reference_path.as_posix()
        else "Reference hidden-state dtype is artifact-dependent; compact float16 references are not primary correctness evidence."
    )
    return {
        "status": "passed" if passed else "failed",
        "reference": str(reference_path),
        "cosine": cosine,
        "mean_abs": mean_abs,
        "max_abs": max_abs,
        "cosine_threshold": cosine_threshold,
        "mean_abs_threshold": mean_abs_threshold,
        "reference_dtype_policy": "float32_correctness" if "vjepa_reference_float32" in reference_path.as_posix() else "non_primary_or_legacy",
        "note": reference_note,
    }


def _compare_reference_features_for_runtime(
    features_path: Path,
    reference_path: Path,
    *,
    temporal_pooling: bool,
    cosine_threshold: float,
    mean_abs_threshold: float,
) -> dict[str, Any]:
    """Compare runtime features to the one-bin float32 reference.

    Temporal-pooling mode preserves V-JEPA tubelet bins for TRIBE slicing.  The
    existing correctness reference is still a global token mean, so compare the
    mean over temporal bins to retain the same all-token parity gate.
    """
    if not temporal_pooling:
        return _compare_reference_features(
            features_path,
            reference_path,
            cosine_threshold=cosine_threshold,
            mean_abs_threshold=mean_abs_threshold,
        )
    actual = np.asarray(np.load(features_path), dtype=np.float32)
    pooled = actual.mean(axis=-1, keepdims=True)
    pooled_path = features_path.with_name("features_global_mean_for_parity.npy")
    np.save(pooled_path, pooled)
    result = _compare_reference_features(
        pooled_path,
        reference_path,
        cosine_threshold=cosine_threshold,
        mean_abs_threshold=mean_abs_threshold,
    )
    return {
        **result,
        "actual_temporal_features": str(features_path),
        "actual_temporal_shape": list(actual.shape),
        "actual_compared_features": str(pooled_path),
        "actual_compared_shape": list(pooled.shape),
        "comparison_pooling": "mean_over_temporal_bins",
    }


def _contract_reused_parity(args: argparse.Namespace) -> dict[str, Any]:
    """Record that this per-video artifact reuses the canonical encoder parity.

    Running a full PyTorch hidden-state reference for every batch video is
    prohibitively heavy and the existing float32 reference artifact is tied to a
    single canonical video.  For per-video production artifacts we therefore do
    not compare against the canonical video's activations; we require the same
    MLX encoder/provenance gates and record that parity is inherited from the
    canonical metadata.
    """
    canonical = _load_json(args.canonical_parity_metadata)
    canonical_parity = canonical.get("reference_parity") or {}
    if canonical.get("status") != "ok" or canonical_parity.get("status") != "passed":
        return {
            "status": "failed",
            "mode": "contract_reused",
            "canonical_metadata": str(args.canonical_parity_metadata),
            "canonical_status": canonical.get("status"),
            "canonical_parity_status": canonical_parity.get("status"),
            "message": "canonical MLX/HF parity metadata is missing or not passed",
        }
    return {
        "status": "contract_reused",
        "mode": "per_video_temporal_batch",
        "canonical_metadata": str(args.canonical_parity_metadata),
        "canonical_input": canonical.get("input"),
        "canonical_reference_parity": canonical_parity,
        "message": (
            "Per-video temporal features reuse the canonical float32 MLX/HF "
            "encoder parity gate; no invalid cross-video activation comparison "
            "was performed for this input."
        ),
    }

def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _provenance_dtype(report: dict[str, Any], *, kind: str) -> str | None:
    if kind == "preflight":
        selected = report.get("selected_policy", {})
        return selected.get("dtype") or report.get("runtime_dtype")
    if kind == "mapping":
        return report.get("dtype_policy") or report.get("runtime_dtype")
    if kind == "aggregation":
        return report.get("source_hidden_states_dtype") or report.get("reported_hidden_states_dtype")
    return None


def _correctness_provenance(preflight: dict[str, Any], mapping: dict[str, Any], aggregation: dict[str, Any], *, mapping_path: Path, aggregation_path: Path, reference_features: Path) -> dict[str, Any]:
    preflight_dtype = _provenance_dtype(preflight, kind="preflight")
    mapping_dtype = _provenance_dtype(mapping, kind="mapping")
    aggregation_dtype = _provenance_dtype(aggregation, kind="aggregation")
    reference_path = reference_features.as_posix()
    details = {
        "required_dtype": CORRECTNESS_DTYPE,
        "preflight_dtype": preflight_dtype,
        "preflight_runtime_dtype": preflight.get("runtime_dtype") or preflight.get("selected_policy", {}).get("runtime_dtype"),
        "preflight_correctness_policy": preflight.get("correctness_policy") or preflight.get("selected_policy", {}).get("correctness_policy"),
        "mapping": str(mapping_path),
        "mapping_dtype": mapping_dtype,
        "mapping_correctness_policy": mapping.get("correctness_policy"),
        "mapping_artifact_role": mapping.get("artifact_role"),
        "aggregation": str(aggregation_path),
        "aggregation_source_hidden_states_dtype": aggregation_dtype,
        "aggregation_reported_hidden_states_dtype": aggregation.get("reported_hidden_states_dtype"),
        "reference_features": reference_path,
        "reference_features_role": "correctness" if "vjepa_reference_float32" in reference_path else "non_primary",
    }
    blockers: list[dict[str, Any]] = []
    if preflight_dtype != CORRECTNESS_DTYPE or details["preflight_correctness_policy"] not in (None, CORRECTNESS_DTYPE):
        blockers.append({"type": "preflight_not_float32_correctness", **details})
    if mapping_dtype != CORRECTNESS_DTYPE or details["mapping_correctness_policy"] not in (None, CORRECTNESS_DTYPE):
        blockers.append({"type": "mapping_not_float32_correctness", **details})
    if aggregation_dtype != CORRECTNESS_DTYPE or details["aggregation_reported_hidden_states_dtype"] not in (None, CORRECTNESS_DTYPE):
        blockers.append({"type": "reference_aggregation_not_float32_correctness", **details})
    if "vjepa_reference_float32" not in reference_path:
        blockers.append({"type": "reference_features_not_float32_correctness_artifact", **details})
    details["status"] = "ok" if not blockers else "blocked"
    return {"details": details, "blockers": blockers}


def _frames_to_tiny_video(input_path: Path, seconds: float, width: int, config: TinyVJEPA2Config):
    import mlx.core as mx

    frames = sample_video_frames(input_path, seconds=seconds, width=width, num_frames=config.frames)
    arrays = []
    for frame in frames:
        img = frame.resize((config.image_size, config.image_size))
        arrays.append(np.asarray(img, dtype=np.float32) / 255.0)
    arr = np.stack(arrays, axis=0)[None, ...]  # [B,T,H,W,C]
    return mx.array(arr)


def _tiny_weights(config: TinyVJEPA2Config) -> dict[str, Any]:
    import mlx.core as mx

    def zeros(shape):
        return mx.zeros(shape, dtype=mx.float32)

    def ones(shape):
        return mx.ones(shape, dtype=mx.float32)

    return {
        "patch_embed.weight": zeros((config.hidden_size, config.patch_dim)),
        "patch_embed.bias": zeros((config.hidden_size,)),
        "pos_embed": zeros((1, config.num_patches, config.hidden_size)),
        "norm1.weight": ones((config.hidden_size,)),
        "norm1.bias": zeros((config.hidden_size,)),
        "attn.qkv.weight": zeros((config.hidden_size * 3, config.hidden_size)),
        "attn.qkv.bias": zeros((config.hidden_size * 3,)),
        "attn.proj.weight": zeros((config.hidden_size, config.hidden_size)),
        "attn.proj.bias": zeros((config.hidden_size,)),
        "norm2.weight": ones((config.hidden_size,)),
        "norm2.bias": zeros((config.hidden_size,)),
        "mlp.fc1.weight": zeros((config.intermediate_size, config.hidden_size)),
        "mlp.fc1.bias": zeros((config.intermediate_size,)),
        "mlp.fc2.weight": zeros((config.hidden_size, config.intermediate_size)),
        "mlp.fc2.bias": zeros((config.hidden_size,)),
    }


def run_tiny(args: argparse.Namespace) -> dict[str, Any]:
    import mlx.core as mx

    started = time.time()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config = TinyVJEPA2Config(frames=4, image_size=8, patch_size=4, tubelet_size=2, hidden_size=8, num_attention_heads=2)
    video = _frames_to_tiny_video(args.input, seconds=args.seconds, width=args.width, config=config)
    out = tiny_vjepa2_forward(video, _tiny_weights(config), config)
    mx.eval(out)
    pooled = np.array(out, dtype=np.float32).mean(axis=1)[0]  # [hidden]
    features = pooled[None, :, None]  # [layers=1, dim=8, time=1]
    features_path = output_dir / "features.npy"
    np.save(features_path, features)
    report = {
        "status": "ok",
        "phase": "mlx_vjepa2_feature_probe_tiny",
        "model_size": "tiny",
        "backend": "mlx",
        "device": str(mx.default_device()),
        "input": str(args.input),
        "feature_shape": list(features.shape),
        "features": str(features_path),
        "finite": bool(np.isfinite(features).all()),
        "semantic_equivalence": "backend_feasibility_only",
        "tribe_compatible": False,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    _write_json(output_dir / "metadata.json", report)
    return report


def run_vitg(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight = _load_json(args.preflight)
    mapping_path = args.mapping
    aggregation_path = args.aggregation
    mapping = _load_json(mapping_path)
    aggregation = _load_json(aggregation_path)

    blockers: list[dict[str, Any]] = []
    provenance = _correctness_provenance(
        preflight,
        mapping,
        aggregation,
        mapping_path=mapping_path,
        aggregation_path=aggregation_path,
        reference_features=args.reference_features,
    )
    blockers.extend(provenance["blockers"])
    if preflight.get("decision") != "allow_full_load":
        blockers.append({"type": "memory_preflight_not_allow_full_load", "decision": preflight.get("decision")})
    if mapping.get("status") != "ok" or mapping.get("blockers"):
        blockers.append({"type": "mapping_not_clean", "status": mapping.get("status"), "blockers": mapping.get("blockers")})
    if aggregation.get("status") != "ok" or aggregation.get("output_shape", [])[:2] != [2, 1408]:
        blockers.append({"type": "reference_aggregation_not_ready", "status": aggregation.get("status"), "output_shape": aggregation.get("output_shape")})

    runtime_result: dict[str, Any] | None = None
    parity: dict[str, Any] | None = None
    if not blockers:
        if args.enable_full_runtime:
            runtime_result = _run_full_vitg_runtime(args, preflight=preflight, aggregation=aggregation)
            if runtime_result.get("status") == "ok":
                if args.parity_mode == "contract-reused":
                    parity = _contract_reused_parity(args)
                else:
                    parity = _compare_reference_features_for_runtime(
                        Path(runtime_result["features"]),
                        args.reference_features,
                        temporal_pooling=args.temporal_pooling,
                        cosine_threshold=args.parity_cosine_threshold,
                        mean_abs_threshold=args.parity_mean_abs_threshold,
                    )
                runtime_with_parity = {**runtime_result, "parity": parity}
                if parity.get("status") not in {"passed", "contract_reused"}:
                    runtime_result = runtime_with_parity
                    blockers.append(
                        {
                            "type": "reference_parity_not_passed",
                            "parity_status": parity.get("status"),
                            "reason": parity.get("reason") or parity.get("message"),
                            "cosine": parity.get("cosine"),
                            "mean_abs": parity.get("mean_abs"),
                            "max_abs": parity.get("max_abs"),
                            "reference": parity.get("reference"),
                        }
                    )
                else:
                    report = {
                        "status": "ok",
                        "phase": "mlx_vjepa2_feature_probe_vitg",
                        "model_size": "vitg",
                        "model_id": preflight.get("model_id"),
                        "input": str(args.input),
                        "preflight": str(args.preflight),
                        "mapping": str(mapping_path),
                        "aggregation": str(aggregation_path),
                        "preflight_decision": preflight.get("decision"),
                        "mapping_status": mapping.get("status"),
                        "reference_aggregation_status": aggregation.get("status"),
                        "correctness_provenance": provenance["details"],
                        "required_output_shape": [2, 1408, "T"],
                        "feature_shape": runtime_result.get("feature_shape"),
                        "temporal_pooling": runtime_result.get("temporal_pooling"),
                        "feature_frequency_hz": runtime_result.get("feature_frequency_hz"),
                        "feature_duration_seconds": runtime_result.get("feature_duration_seconds"),
                        "features": runtime_result.get("features"),
                        "finite": runtime_result.get("finite"),
                        "semantic_equivalence": "mlx_vitg_encoder_reference_parity_recorded",
                        "reference_parity": parity,
                        "tribe_compatible": runtime_result.get("feature_shape", [])[:2] == [2, 1408]
                        and runtime_result.get("finite") is True,
                        "runtime": runtime_with_parity,
                        "elapsed_seconds": round(time.time() - started, 3),
                    }
                    _write_json(output_dir / "metadata.json", report)
                    return report
            blockers.append(
                {
                    "type": "full_runtime_attempt_blocked",
                    "runtime_status": runtime_result.get("status"),
                    "runtime_blocker_type": runtime_result.get("type"),
                    "message": runtime_result.get("message"),
                    "progress": runtime_result.get("progress"),
                    "runtime_elapsed_seconds": runtime_result.get("runtime_elapsed_seconds"),
                }
            )
        else:
            blockers.append(
                {
                    "type": "full_runtime_not_enabled",
                    "implemented_components": [
                        "streaming safetensors-to-MLX state mapping",
                        "HF-compatible Conv3d patch/tubelet embedding in MLX",
                        "VJEPA2 encoder layer path with q/k/v/proj, MLP/GELU, final norm",
                        "3D RoPE frame/height/width q/k rotation",
                        "chunked/fast scaled-dot-product attention helper with small parity test",
                    ],
                    "missing_components": [
                        "watchdoged full runtime was not requested; pass --enable-full-runtime",
                        "HF-to-MLX full-model parity metrics against aggregated reference after runtime succeeds",
                    ],
                }
            )
    attention_bytes = preflight.get("activation_estimate", {}).get("attention_scores_materialized_bytes")
    if attention_bytes:
        blockers.append(
            {
                "type": "attention_strategy_required_before_full_probe",
                "materialized_attention_scores_bytes_per_layer": attention_bytes,
                "note": "Full ViT-g probe uses chunked MLX fast attention when --enable-full-runtime is set; runtime must still prove it under watchdog.",
            }
        )

    report = {
        "status": "blocked",
        "phase": "mlx_vjepa2_feature_probe_vitg",
        "model_size": "vitg",
        "model_id": preflight.get("model_id"),
        "input": str(args.input),
        "preflight": str(args.preflight),
        "mapping": str(mapping_path),
        "aggregation": str(aggregation_path),
        "preflight_decision": preflight.get("decision"),
        "mapping_status": mapping.get("status"),
        "reference_aggregation_status": aggregation.get("status"),
        "correctness_provenance": provenance["details"],
        "required_output_shape": [2, 1408, "T"],
        "feature_shape": None,
        "features": runtime_result.get("features") if runtime_result else None,
        "finite": runtime_result.get("finite") if runtime_result else None,
        "semantic_equivalence": "blocked_no_mlx_vitg_features_produced" if not runtime_result else "blocked_mlx_vitg_runtime_or_parity_failed",
        "reference_parity": parity,
        "tribe_compatible": False,
        "invalidates_full_tribe_compatibility": True,
        "invalidates_current_machine_feasibility_only": False,
        "blockers": blockers,
        "runtime": runtime_result,
        "next_action": (
            "Resolve the reported full-runtime/parity blocker, rerun with --enable-full-runtime, "
            "and only integrate finite [2,1408,T] features after parity is recorded within tolerance."
            if runtime_result
            else "Rerun with --enable-full-runtime after upstream gates are clean to attempt finite [2,1408,T] MLX ViT-g features."
        ),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    _write_json(output_dir / "metadata.json", report)
    return report

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=["tiny", "vitg"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, default=Path("cache_mac_m2_verification/vjepa_vitg_memory_preflight.json"))
    parser.add_argument("--mapping", type=Path, default=DEFAULT_FLOAT32_MAPPING)
    parser.add_argument("--aggregation", type=Path, default=DEFAULT_FLOAT32_AGGREGATION)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--enable-full-runtime", action="store_true")
    parser.add_argument(
        "--temporal-pooling",
        action="store_true",
        help=(
            "Preserve V-JEPA tubelet time bins by spatially pooling tokens per "
            "tubelet. Output remains [2,1408,T] and parity is checked against "
            "the float32 one-bin reference after mean-over-time pooling."
        ),
    )
    parser.add_argument("--attention-chunk-size", type=int, default=128)
    parser.add_argument("--reference-features", type=Path, default=DEFAULT_FLOAT32_REFERENCE_FEATURES)
    parser.add_argument(
        "--parity-mode",
        choices=["reference", "contract-reused"],
        default="reference",
        help=(
            "reference compares this run to --reference-features. "
            "contract-reused is for per-video batch artifacts and records reuse "
            "of a canonical passed MLX/HF parity fixture instead of comparing "
            "different videos' activations."
        ),
    )
    parser.add_argument("--canonical-parity-metadata", type=Path, default=Path("cache_mac_m2_verification/mlx_vjepa_vitg/metadata.json"))
    parser.add_argument("--parity-cosine-threshold", type=float, default=0.98)
    parser.add_argument("--parity-mean-abs-threshold", type=float, default=1.0)
    args = parser.parse_args()
    if args.model_size == "tiny":
        report = run_tiny(args)
    else:
        report = run_vitg(args)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
