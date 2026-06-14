# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""File IO and manifest helpers for vectorized analysis outputs."""

from __future__ import annotations

import hashlib
import json
import typing as tp
from pathlib import Path

import numpy as np
import pandas as pd

from .vectorize import validate_predictions


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: tp.Mapping[str, tp.Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def load_predictions(
    path: Path,
    *,
    expected_vertices: int | None = 20484,
) -> np.ndarray:
    preds = np.load(path)
    return validate_predictions(preds, expected_vertices=expected_vertices)


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def load_events(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    return load_table(path)


def load_segments(path: Path | None, *, n_rows: int | None = None) -> pd.DataFrame | None:
    if path is None:
        return None
    table = load_table(path)
    if "row" not in table.columns:
        table.insert(0, "row", np.arange(len(table), dtype=int))
    if n_rows is not None and len(table) != n_rows:
        raise ValueError(f"segment row count mismatch: {len(table)} != {n_rows}")
    return table


def write_table(path: Path, table: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        table.to_parquet(path, index=False)
    elif suffix in {".tsv", ".tab"}:
        table.to_csv(path, index=False, sep="\t")
    else:
        table.to_csv(path, index=False)
    return path


def write_array(path: Path, array: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(array))
    return path


def _artifact_record(path: Path, *, root: Path | None = None) -> dict[str, tp.Any]:
    rec: dict[str, tp.Any] = {
        "path": str(path.relative_to(root) if root is not None else path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }
    if path.suffix.lower() == ".npy":
        arr = np.load(path, mmap_mode="r")
        rec["shape"] = list(arr.shape)
        rec["dtype"] = str(arr.dtype)
    return rec


def build_manifest(
    *,
    output_root: Path,
    artifacts: tp.Iterable[Path],
    metadata: tp.Mapping[str, tp.Any] | None = None,
) -> dict[str, tp.Any]:
    output_root = output_root.resolve()
    records = [
        _artifact_record(path.resolve(), root=output_root)
        for path in artifacts
        if path.exists()
    ]
    return {
        "schema": "tribev2.vectorized-analysis.v1",
        "metadata": dict(metadata or {}),
        "artifacts": records,
    }
