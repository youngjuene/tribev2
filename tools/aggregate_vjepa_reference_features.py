#!/usr/bin/env python3
"""Aggregate HF V-JEPA2 reference outputs into the TRIBE video feature contract.

This tool implements the neuralset-style reference aggregation gate for Ralph's
probe milestone.  It can only produce TRIBE-ready `[layers, dim, time]` features
when the reference NPZ contains per-layer hidden states.  A last-hidden-state-only
reference is reported as a controlled blocker because group-mean layer semantics
cannot be reconstructed truthfully.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _layer_subselection(latents: np.ndarray, cache_n_layers: int | None) -> tuple[np.ndarray, list[int]]:
    n_layers = int(latents.shape[0])
    if cache_n_layers is None or cache_n_layers >= n_layers:
        selected = list(range(n_layers))
        return latents, selected
    selected = [int(round(x)) for x in np.linspace(0, n_layers - 1, cache_n_layers)]
    return latents[selected, ...], selected


def _aggregate_layers(latents: np.ndarray, layers: list[float] | str, layer_aggregation: str | None) -> tuple[np.ndarray, list[int]]:
    n_model_layers = int(latents.shape[0])
    if layers == "all":
        layer_indices = list(range(n_model_layers))
    else:
        values = layers if isinstance(layers, list) else [layers]
        layer_indices = np.unique([int(float(i) * (n_model_layers - 1)) for i in values]).tolist()
    if len(layer_indices) == 1:
        if layer_aggregation is None:
            return latents[layer_indices[0]][None, :], layer_indices
        return latents[layer_indices[0]], layer_indices
    latents = np.asarray(latents)
    if layer_aggregation == "mean":
        return latents[layer_indices].mean(0), layer_indices
    if layer_aggregation == "sum":
        return latents[layer_indices].sum(0), layer_indices
    if layer_aggregation == "group_mean":
        groups = []
        boundaries = list(layer_indices)
        boundaries[-1] += 1
        for l1, l2 in zip(boundaries[:-1], boundaries[1:]):
            groups.append(latents[l1:l2].mean(0))
        return np.stack(groups), layer_indices
    if layer_aggregation is None:
        return latents[layer_indices], layer_indices
    raise ValueError(f"Unknown layer aggregation: {layer_aggregation}")


def _extract_per_layer_hidden(npz: np.lib.npyio.NpzFile) -> tuple[np.ndarray | None, str, str | None]:
    for key in ("hidden_states", "all_hidden_states", "layer_hidden_states"):
        if key in npz.files:
            raw = npz[key]
            source_dtype = str(raw.dtype)
            arr = np.asarray(raw, dtype=np.float32)
            if arr.ndim == 4 and arr.shape[1] == 1:
                arr = arr[:, 0, :, :]
            if arr.ndim == 3:
                return arr, key, source_dtype
            return None, f"{key} has unsupported shape {list(arr.shape)}; expected [layers,tokens,dim] or [layers,1,tokens,dim]", source_dtype
    return None, "reference NPZ has no per-layer hidden state array; found keys " + ", ".join(npz.files), None


def _reference_report_dtype(reference_path: Path) -> str | None:
    report_path = reference_path.with_name("reference_report.json")
    if not report_path.exists():
        return None
    try:
        report = json.loads(report_path.read_text())
    except Exception:
        return None
    dtype = report.get("hidden_states_dtype")
    return str(dtype) if dtype is not None else None


def _blocker(*, contract: dict[str, Any], reference_path: Path, output_dir: Path, started: float, reason: str) -> dict[str, Any]:
    report = {
        "status": "blocked",
        "phase": "reference_aggregation_parity",
        "model_id": contract.get("vjepa2", {}).get("model_id"),
        "reference": str(reference_path),
        "elapsed_seconds": round(time.time() - started, 3),
        "error_type": "InsufficientReferenceLayers",
        "error_message": reason,
        "required_output_shape": [2, 1408, "T"],
        "invalidates_full_tribe_compatibility": True,
        "invalidates_current_machine_feasibility_only": False,
        "next_action": "Regenerate the PyTorch reference artifact with output_hidden_states=True and save per-layer hidden states before semantic MLX probe claims.",
    }
    _write_json(output_dir / "aggregation_report.json", report)
    return report


def aggregate_reference(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    contract = json.loads(args.contract.read_text())
    selected = contract["selected_feature_contract"]
    target_prefix = selected.get("target_shape_prefix")
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.reference) as npz:
        hidden, source_key_or_reason, source_dtype = _extract_per_layer_hidden(npz)
        if hidden is None:
            return _blocker(
                contract=contract,
                reference_path=args.reference,
                output_dir=output_dir,
                started=started,
                reason=source_key_or_reason,
            )
    reported_dtype = _reference_report_dtype(args.reference)
    required_dtype = args.required_hidden_states_dtype
    if required_dtype != "any" and (source_dtype != required_dtype or (reported_dtype is not None and reported_dtype != required_dtype)):
        return _blocker(
            contract=contract,
            reference_path=args.reference,
            output_dir=output_dir,
            started=started,
            reason=(
                f"reference hidden_states dtype must be {required_dtype} for correctness mode; "
                f"npz source dtype={source_dtype}, report dtype={reported_dtype}"
            ),
        )

    cache_n_layers = selected.get("cache_n_layers")
    layers = selected.get("layers")
    layer_aggregation = selected.get("layer_aggregation")
    token_pooling = selected.get("token_pooling")
    if token_pooling != "mean":
        raise ValueError(f"Only token_pooling='mean' is implemented for this gate, got {token_pooling!r}")

    layer_subset, selected_cache_indices = _layer_subselection(hidden, cache_n_layers)
    token_pooled = layer_subset.mean(axis=1)  # [cache_layers, dim]
    with_time = token_pooled[..., None]  # [cache_layers, dim, time=1]
    aggregated, layer_indices = _aggregate_layers(with_time, layers, layer_aggregation)
    aggregated = np.asarray(aggregated, dtype=np.float32)
    artifact = output_dir / "aggregated_features.npy"
    np.save(artifact, aggregated)
    status = "ok" if list(aggregated.shape[:2]) == list(target_prefix) else "blocked"
    report: dict[str, Any] = {
        "status": status,
        "phase": "reference_aggregation_parity",
        "model_id": contract.get("vjepa2", {}).get("model_id"),
        "reference": str(args.reference),
        "source_key": source_key_or_reason,
        "source_hidden_states_dtype": source_dtype,
        "reported_hidden_states_dtype": reported_dtype,
        "required_hidden_states_dtype": required_dtype,
        "source_dtype_status": "ok" if required_dtype == "any" or source_dtype == required_dtype else "blocked",
        "raw_hidden_shape": list(hidden.shape),
        "cache_n_layers": cache_n_layers,
        "selected_cache_indices": selected_cache_indices,
        "layers": layers,
        "layer_indices_after_cache_subselection": layer_indices,
        "layer_aggregation": layer_aggregation,
        "token_pooling": token_pooling,
        "frequency_hz": selected.get("frequency_hz"),
        "target_tensor_order": selected.get("target_tensor_order"),
        "output_shape": list(aggregated.shape),
        "artifact": str(artifact),
        "elapsed_seconds": round(time.time() - started, 3),
        "semantic_parity_status": "reference_aggregation_only_no_mlx_claim",
    }
    if status != "ok":
        report.update(
            {
                "error_type": "AggregatedShapeMismatch",
                "error_message": f"expected shape prefix {target_prefix}, got {list(aggregated.shape[:2])}",
                "invalidates_full_tribe_compatibility": True,
            }
        )
    _write_json(output_dir / "aggregation_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--required-hidden-states-dtype", choices=["float32", "float16", "any"], default="float32")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = aggregate_reference(args)
    if args.json or True:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
