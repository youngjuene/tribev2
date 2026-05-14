"""Repo-local MLX/precomputed feature extractor shims.

The released TRIBE inference stack expects neuralset-style extractors whose
``prepare`` method is called once and whose ``__call__`` method returns a torch
Tensor shaped like ``(layers, feature_dim, time)`` for each segment.  MLX arrays
are converted back to torch at this boundary because ``SegmentData`` and the
released TRIBE brain model are PyTorch-based.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class PrecomputedFeatureExtractor:
    """Neuralset-compatible shim backed by cached numpy feature tensors.

    Parameters
    ----------
    feature_path:
        Path to a ``.npy`` file containing ``(layers, feature_dim, time)``.
    event_types:
        Event type string used by ``EventTypesHelper`` filtering.
    frequency:
        Temporal feature frequency in Hz. Used for deterministic segment
        slicing; the caller controls exact segment duration.
    allow_missing:
        If ``True``, return zeros when the feature file is absent. Defaults to
        ``False`` to avoid silent false-positive MLX claims.
    """

    feature_path: str | Path
    event_types: str = "Word"
    frequency: float = 2.0
    layers: list[float] | None = None
    layer_aggregation: str | None = "group_mean"
    allow_missing: bool = False
    aggregation: str = "sum"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.feature_path = Path(self.feature_path)
        self._prepared = False
        self._feature: torch.Tensor | None = None
        from neuralset.events.etypes import EventTypesHelper
        self._event_types_helper = EventTypesHelper(self.event_types)

    def prepare(self, events: Any) -> None:  # neuralset passes a standardized DataFrame
        self._prepared = True
        if self.feature_path.exists():
            arr = np.load(self.feature_path)
            self._feature = torch.from_numpy(np.asarray(arr, dtype=np.float32))
        elif not self.allow_missing:
            raise FileNotFoundError(f"Precomputed feature file not found: {self.feature_path}")

    def __call__(self, events: Any, start: float, duration: float, trigger: Any = None) -> torch.Tensor:
        if not self._prepared:
            self.prepare(events)
        if self._feature is None:
            length = max(1, int(round(duration * self.frequency)))
            return torch.zeros((1, 1, length), dtype=torch.float32)
        feature = self._feature
        if feature.ndim != 3:
            raise ValueError(
                f"Expected precomputed feature shape (layers, feature_dim, time), got {tuple(feature.shape)}"
            )
        start_idx = max(0, int(round(start * self.frequency)))
        length = max(1, int(round(duration * self.frequency)))
        end_idx = min(feature.shape[-1], start_idx + length)
        if start_idx >= feature.shape[-1]:
            out = torch.zeros((*feature.shape[:2], length), dtype=feature.dtype)
        else:
            out = feature[..., start_idx:end_idx]
            if out.shape[-1] < length:
                pad = torch.zeros((*feature.shape[:2], length - out.shape[-1]), dtype=feature.dtype)
                out = torch.cat([out, pad], dim=-1)
        return out.to(torch.float32)


@dataclass
class MLXProofFeatureExtractor(PrecomputedFeatureExtractor):
    """Generate deterministic proof features with MLX, then expose them to TRIBE.

    This is a backend/contract proof, not a semantically equivalent replacement
    for Llama/V-JEPA2/Wav2Vec-BERT.  It is useful for verifying that an MLX
    feature artifact can cross the neuralset/TRIBE boundary safely.
    """

    feature_path: str | Path = "cache_mac_m2/mlx_features/mlx_proof_text.npy"
    event_types: str = "Word"
    layers_count: int = 2
    feature_dim: int = 3072
    time_steps: int = 80
    seed_text: str = "tribev2-mlx-proof"
    metadata_path: str | Path | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.metadata_path is None:
            self.metadata_path = self.feature_path.with_suffix(".json")
        else:
            self.metadata_path = Path(self.metadata_path)

    def prepare(self, events: Any) -> None:
        self.feature_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.feature_path.exists():
            self._write_mlx_features(events)
        super().prepare(events)

    def _event_digest(self, events: Any) -> str:
        try:
            payload = events.to_json(orient="records", default_handler=str)
        except Exception:
            payload = repr(events)
        return hashlib.sha256((self.seed_text + payload).encode("utf-8")).hexdigest()

    def _write_mlx_features(self, events: Any) -> None:
        import mlx.core as mx

        digest = self._event_digest(events)
        # Deterministic low-amplitude feature field generated by MLX ops on the
        # default Apple device.  Keep values small to avoid saturating the random
        # released projector weights in smoke tests.
        offset = int(digest[:8], 16) % 1000 / 1000.0
        total = self.layers_count * self.feature_dim * self.time_steps
        base = mx.arange(total, dtype=mx.float32).reshape(
            (self.layers_count, self.feature_dim, self.time_steps)
        )
        arr = mx.sin(base * 0.0001 + offset) * 0.01
        mx.eval(arr)
        np_arr = np.array(arr, dtype=np.float32)
        np.save(self.feature_path, np_arr)
        meta = {
            "backend": "mlx",
            "device": str(mx.default_device()),
            "shape": list(np_arr.shape),
            "dtype": str(np_arr.dtype),
            "event_digest": digest,
            "semantic_equivalence": "backend_proof_only",
        }
        Path(self.metadata_path).write_text(json.dumps(meta, indent=2))
        self.metadata = meta


@dataclass
class MLXLmTextFeatureExtractor(PrecomputedFeatureExtractor):
    """Generate checkpoint-shaped text features with an MLX-LM Llama model.

    This adapter closes the previous "proof tensor only" gap for text by using
    real MLX-LM hidden states.  It still does **not** claim exact neuralset
    parity: neuralset's released HuggingFaceText extractor aggregates selected
    intermediate layers, while MLX-LM exposes the final normalized residual for
    the supported public API used here.  The adapter therefore records explicit
    non-parity metadata and remains opt-in.
    """

    feature_path: str | Path = "cache_mac_m2/mlx_features/mlx_lm_text.npy"
    event_types: str = "Word"
    model_name: str = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    feature_dim: int = 3072
    max_tokens: int = 128
    metadata_path: str | Path | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.metadata_path is None:
            self.metadata_path = self.feature_path.with_suffix(".json")
        else:
            self.metadata_path = Path(self.metadata_path)

    def prepare(self, events: Any) -> None:
        self.feature_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.feature_path.exists():
            self._write_mlx_lm_features(events)
        super().prepare(events)

    def _write_mlx_lm_features(self, events: Any) -> None:
        import mlx.core as mx
        from mlx_lm import load

        model, tokenizer = load(self.model_name)
        word_events = self._word_events(events)
        if not word_events:
            raise ValueError("MLXLmTextFeatureExtractor requires at least one Word event")

        max_end = max(start + duration for _, start, duration in word_events)
        time_steps = max(1, int(np.ceil(max_end * self.frequency)))
        features = np.zeros((2, self.feature_dim, time_steps), dtype=np.float32)
        counts = np.zeros(time_steps, dtype=np.float32)

        for text, start, duration in word_events:
            mean_hidden, last_hidden = self._encode_text_with_mlx_lm(
                text, model=model, tokenizer=tokenizer, mx=mx
            )
            start_idx = max(0, int(np.floor(start * self.frequency)))
            end_idx = max(start_idx + 1, int(np.ceil((start + duration) * self.frequency)))
            end_idx = min(time_steps, end_idx)
            features[0, :, start_idx:end_idx] += mean_hidden[:, None]
            features[1, :, start_idx:end_idx] += last_hidden[:, None]
            counts[start_idx:end_idx] += 1

        active = counts > 0
        features[:, :, active] /= counts[active][None, None, :]
        np.save(self.feature_path, features)
        meta = {
            "backend": "mlx-lm",
            "device": str(mx.default_device()),
            "model_name": self.model_name,
            "shape": list(features.shape),
            "dtype": str(features.dtype),
            "event_count": len(word_events),
            "semantic_equivalence": "mlx_lm_final_state_not_neuralset_layer_parity",
            "aggregation": {
                "layer0": "mean_token_final_hidden_state",
                "layer1": "last_token_final_hidden_state",
                "time": f"{self.frequency}Hz event-window average",
            },
        }
        Path(self.metadata_path).write_text(json.dumps(meta, indent=2))
        self.metadata = meta

    def _encode_text_with_mlx_lm(self, text: str, *, model: Any, tokenizer: Any, mx: Any) -> tuple[np.ndarray, np.ndarray]:
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        if not token_ids:
            token_ids = tokenizer.encode(" ", add_special_tokens=True)
        token_ids = token_ids[: self.max_tokens]
        inputs = mx.array([token_ids])
        hidden_model = getattr(model, "model", model)
        hidden = hidden_model(inputs)
        mx.eval(hidden)
        arr = np.array(hidden[0], dtype=np.float32)
        if arr.ndim != 2 or arr.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected MLX-LM hidden states (*, {self.feature_dim}), got {arr.shape}"
            )
        return arr.mean(axis=0), arr[-1]

    @staticmethod
    def _word_events(events: Any) -> list[tuple[str, float, float]]:
        rows: list[tuple[str, float, float]] = []
        if not hasattr(events, "iterrows"):
            return rows
        for _, row in events.iterrows():
            if str(row.get("type", "")) != "Word":
                continue
            text = (
                row.get("text")
                or row.get("word")
                or row.get("sentence")
                or row.get("context")
                or ""
            )
            text = str(text).strip() or " "
            start = float(row.get("start", 0.0) or 0.0)
            duration = float(row.get("duration", 1.0) or 1.0)
            rows.append((text, start, max(duration, 1.0 / 1000)))
        return rows
