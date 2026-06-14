# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Pure NumPy/Pandas transforms for TRIBE prediction matrices."""

from __future__ import annotations

import math
import typing as tp

import numpy as np
import pandas as pd

DEFAULT_FSAVERAGE5_VERTICES = 20484


def validate_predictions(
    preds: tp.Any,
    *,
    expected_vertices: int | None = DEFAULT_FSAVERAGE5_VERTICES,
    require_finite: bool = True,
) -> np.ndarray:
    """Return a checked 2-D prediction matrix.

    TRIBE inference emits arrays shaped ``(n_segments, n_vertices)``.  The
    released fsaverage5 checkpoint has 20,484 vertices.
    """

    arr = np.asarray(preds)
    if arr.ndim != 2:
        raise ValueError(f"predictions must be 2-D, got shape {arr.shape}")
    if expected_vertices is not None and arr.shape[1] != expected_vertices:
        raise ValueError(
            "prediction vertex count mismatch: "
            f"expected {expected_vertices}, got {arr.shape[1]}"
        )
    if require_finite and not np.isfinite(arr).all():
        raise ValueError("predictions contain NaN or infinite values")
    return arr


def _safe_get(obj: tp.Any, name: str, default: tp.Any = None) -> tp.Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _event_types_from_events(events: tp.Any) -> str | None:
    if events is None:
        return None
    if hasattr(events, "__len__") and len(events) == 0:
        return ""
    if hasattr(events, "__getitem__") and "type" in getattr(events, "columns", []):
        return ",".join(sorted(map(str, events["type"].dropna().unique())))
    types = []
    try:
        for event in events:
            event_type = _safe_get(event, "type", None)
            if event_type is None:
                event_type = event.__class__.__name__
            types.append(str(event_type))
    except TypeError:
        return None
    return ",".join(sorted(set(types)))


def _event_count(segment: tp.Any) -> int | None:
    ns_events = _safe_get(segment, "ns_events", None)
    if ns_events is not None:
        try:
            return len(ns_events)
        except TypeError:
            pass
    events = _safe_get(segment, "events", None)
    if events is not None:
        try:
            return len(events)
        except TypeError:
            pass
    return None


def _segment_timing(segment: tp.Any) -> dict[str, tp.Any]:
    start = _safe_get(segment, "start", None)
    offset = _safe_get(segment, "offset", None)
    duration = _safe_get(segment, "duration", None)
    absolute_start = None
    stop = None
    if start is not None and offset is not None:
        absolute_start = float(start) + float(offset)
    elif start is not None:
        absolute_start = float(start)
    elif offset is not None:
        absolute_start = float(offset)
    if absolute_start is not None and duration is not None:
        stop = absolute_start + float(duration)
    return {
        "start": start,
        "offset": offset,
        "duration": duration,
        "absolute_start": absolute_start,
        "stop": stop,
    }


def make_segment_table(
    segments: tp.Any | None,
    *,
    n_rows: int | None = None,
) -> pd.DataFrame:
    """Serialize TRIBE segment objects into a row-aligned metadata table."""

    if segments is None:
        if n_rows is None:
            return pd.DataFrame(columns=["row"])
        return pd.DataFrame({"row": np.arange(n_rows, dtype=int)})

    if isinstance(segments, pd.DataFrame):
        table = segments.copy()
        if "row" not in table.columns:
            table.insert(0, "row", np.arange(len(table), dtype=int))
        if n_rows is not None and len(table) != n_rows:
            raise ValueError(f"segment row count mismatch: {len(table)} != {n_rows}")
        return table

    rows = []
    for row, segment in enumerate(list(segments)):
        events = _safe_get(segment, "events", None)
        ns_events = _safe_get(segment, "ns_events", None)
        event_types = _event_types_from_events(events)
        if event_types is None:
            event_types = _event_types_from_events(ns_events)
        timing = _segment_timing(segment)
        rows.append(
            {
                "row": row,
                **timing,
                "event_count": _event_count(segment),
                "event_types": event_types,
                "timeline": _safe_get(segment, "timeline", None),
                "subject": _safe_get(segment, "subject", None),
                "segment_repr": repr(segment),
            }
        )
    table = pd.DataFrame(rows)
    if n_rows is not None and len(table) != n_rows:
        raise ValueError(f"segment row count mismatch: {len(table)} != {n_rows}")
    return table


def _valid_label(label: tp.Any) -> bool:
    if label is None:
        return False
    if isinstance(label, float) and math.isnan(label):
        return False
    if isinstance(label, str):
        return bool(label)
    if isinstance(label, (int, np.integer, float, np.floating)):
        return label > 0
    return True


def roi_timeseries(
    preds: tp.Any,
    labels_or_indices: tp.Mapping[str, tp.Sequence[int]] | tp.Sequence[tp.Any],
    *,
    expected_vertices: int | None = DEFAULT_FSAVERAGE5_VERTICES,
) -> pd.DataFrame:
    """Average vertex predictions into ROI columns."""

    arr = validate_predictions(preds, expected_vertices=expected_vertices)
    out: dict[str, np.ndarray] = {}
    if isinstance(labels_or_indices, dict):
        for label, indices in labels_or_indices.items():
            idx = np.asarray(indices, dtype=int)
            if idx.size == 0:
                continue
            out[str(label)] = arr[:, idx].mean(axis=1)
    else:
        labels = np.asarray(labels_or_indices, dtype=object)
        if labels.shape[0] != arr.shape[1]:
            raise ValueError(
                "ROI label length mismatch: "
                f"expected {arr.shape[1]}, got {labels.shape[0]}"
            )
        for label in sorted({x for x in labels.tolist() if _valid_label(x)}, key=str):
            mask = labels == label
            if mask.any():
                out[str(label)] = arr[:, mask].mean(axis=1)
    table = pd.DataFrame(out)
    table.insert(0, "row", np.arange(arr.shape[0], dtype=int))
    return table


def _condition_rows(
    segment_table: pd.DataFrame,
    condition_col: str,
) -> tuple[np.ndarray, list[str]]:
    if condition_col not in segment_table.columns:
        raise ValueError(f"condition column not found: {condition_col}")
    values = segment_table[condition_col]
    mask = values.notna() & (values.astype(str) != "")
    conditions = sorted(values[mask].astype(str).unique().tolist())
    return mask.to_numpy(), conditions


def label_segments_from_events(
    segment_table: pd.DataFrame,
    events: pd.DataFrame,
    *,
    condition_col: str = "type",
    output_col: str | None = None,
    segment_start_col: str = "absolute_start",
    segment_stop_col: str = "stop",
    event_start_col: str = "start",
    event_duration_col: str = "duration",
) -> pd.DataFrame:
    """Attach segment-level labels from overlapping event intervals.

    The first overlapping event label is used.  This keeps the mapping
    deterministic and conservative when a segment contains several events.
    """

    if output_col is None:
        output_col = condition_col
    for col in (segment_start_col, segment_stop_col):
        if col not in segment_table.columns:
            raise ValueError(f"segment timing column not found: {col}")
    for col in (event_start_col, condition_col):
        if col not in events.columns:
            raise ValueError(f"event column not found: {col}")

    out = segment_table.copy()
    event_starts = events[event_start_col].to_numpy(dtype=float)
    if event_duration_col in events.columns:
        event_stops = event_starts + events[event_duration_col].fillna(0).to_numpy(
            dtype=float
        )
    elif "stop" in events.columns:
        event_stops = events["stop"].to_numpy(dtype=float)
    else:
        event_stops = event_starts
    labels = events[condition_col].astype(str).to_numpy()

    assigned: list[str | None] = []
    for segment in out.itertuples(index=False):
        seg_start = float(getattr(segment, segment_start_col))
        seg_stop = float(getattr(segment, segment_stop_col))
        overlap = (event_starts < seg_stop) & (event_stops > seg_start)
        if overlap.any():
            assigned.append(labels[np.flatnonzero(overlap)[0]])
        else:
            assigned.append(None)
    out[output_col] = assigned
    return out


def pattern_matrix(
    preds: tp.Any,
    segment_table: pd.DataFrame,
    *,
    condition_col: str = "condition",
    expected_vertices: int | None = DEFAULT_FSAVERAGE5_VERTICES,
) -> tuple[np.ndarray, list[str]]:
    """Average predictions by a segment-level condition column."""

    arr = validate_predictions(preds, expected_vertices=expected_vertices)
    if len(segment_table) != arr.shape[0]:
        raise ValueError(
            f"segment table length {len(segment_table)} != predictions rows {arr.shape[0]}"
        )
    mask, conditions = _condition_rows(segment_table, condition_col)
    if not conditions:
        raise ValueError(f"no non-empty conditions found in column {condition_col!r}")
    patterns = []
    cond_values = segment_table[condition_col].astype(str).to_numpy()
    for condition in conditions:
        row_mask = mask & (cond_values == condition)
        patterns.append(arr[row_mask].mean(axis=0))
    return np.vstack(patterns), conditions


def contrast_vectors(
    patterns: tp.Any,
    condition_names: tp.Sequence[str],
    contrast_specs: tp.Mapping[str, tp.Mapping[str, float]] | tp.Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    """Build named linear contrasts from condition patterns.

    String specs use the compact form ``A-B``.
    """

    matrix = np.asarray(patterns)
    if matrix.ndim != 2:
        raise ValueError(f"patterns must be 2-D, got shape {matrix.shape}")
    if matrix.shape[0] != len(condition_names):
        raise ValueError("condition name count must match pattern rows")
    condition_to_row = {str(name): i for i, name in enumerate(condition_names)}

    parsed: dict[str, dict[str, float]] = {}
    if isinstance(contrast_specs, dict):
        parsed = {
            str(name): {str(k): float(v) for k, v in weights.items()}
            for name, weights in contrast_specs.items()
        }
    else:
        for spec in contrast_specs:
            if "-" not in spec:
                raise ValueError(f"contrast string must look like A-B, got {spec!r}")
            left, right = [part.strip() for part in spec.split("-", 1)]
            parsed[spec] = {left: 1.0, right: -1.0}

    rows = []
    names = []
    for name, weights in parsed.items():
        vector = np.zeros(matrix.shape[1], dtype=matrix.dtype)
        for condition, weight in weights.items():
            if condition not in condition_to_row:
                raise ValueError(f"unknown condition in contrast {name!r}: {condition}")
            vector = vector + weight * matrix[condition_to_row[condition]]
        rows.append(vector)
        names.append(name)
    return np.vstack(rows), names


def event_locked_average(
    values: tp.Any,
    segment_table: pd.DataFrame,
    onsets: tp.Sequence[float],
    window: tp.Sequence[int],
    *,
    time_col: str = "absolute_start",
) -> tuple[np.ndarray, np.ndarray]:
    """Average rows around onset times using nearest segment indices."""

    arr = np.asarray(values)
    if arr.ndim != 2:
        raise ValueError(f"values must be 2-D, got shape {arr.shape}")
    if len(segment_table) != arr.shape[0]:
        raise ValueError("segment table length must match values rows")
    if time_col not in segment_table.columns:
        raise ValueError(f"time column not found: {time_col}")
    times = segment_table[time_col].to_numpy(dtype=float)
    rel = np.asarray(window, dtype=int)
    epochs = []
    for onset in onsets:
        idx = int(np.argmin(np.abs(times - float(onset))))
        take = idx + rel
        if take.min() >= 0 and take.max() < len(arr):
            epochs.append(arr[take])
    if not epochs:
        raise ValueError("no complete event-locked windows were available")
    return np.stack(epochs).mean(axis=0), rel


def standardize_matrix(
    matrix: tp.Any,
    *,
    axis: int = 0,
    method: tp.Literal["zscore", "center", "none"] = "zscore",
    eps: float = 1e-8,
) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    if method == "none":
        return arr
    mean = arr.mean(axis=axis, keepdims=True)
    centered = arr - mean
    if method == "center":
        return centered
    if method != "zscore":
        raise ValueError(f"unknown standardization method: {method}")
    scale = arr.std(axis=axis, keepdims=True)
    return centered / np.maximum(scale, eps)


def rdm(
    patterns: tp.Any,
    *,
    metric: tp.Literal["correlation", "euclidean"] = "correlation",
) -> np.ndarray:
    """Compute a representational dissimilarity matrix."""

    matrix = np.asarray(patterns, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"patterns must be 2-D, got shape {matrix.shape}")
    if metric == "correlation":
        z = standardize_matrix(matrix, axis=1)
        sim = np.corrcoef(z)
        out = 1.0 - sim
    elif metric == "euclidean":
        diffs = matrix[:, None, :] - matrix[None, :, :]
        out = np.sqrt(np.sum(diffs * diffs, axis=-1))
    else:
        raise ValueError(f"unknown RDM metric: {metric}")
    np.fill_diagonal(out, 0.0)
    return out


def vectorize_rdm(matrix: tp.Any, *, upper: bool = True) -> np.ndarray:
    arr = np.asarray(matrix)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"RDM must be square, got shape {arr.shape}")
    if upper:
        idx = np.triu_indices(arr.shape[0], k=1)
    else:
        idx = np.tril_indices(arr.shape[0], k=-1)
    return arr[idx]
