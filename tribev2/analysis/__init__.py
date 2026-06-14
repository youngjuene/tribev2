# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Vectorized analysis helpers for TRIBE prediction artifacts."""

from .io import (
    build_manifest,
    file_sha256,
    load_events,
    load_predictions,
    load_segments,
    write_json,
    write_table,
)
from .vectorize import (
    contrast_vectors,
    event_locked_average,
    label_segments_from_events,
    make_segment_table,
    pattern_matrix,
    rdm,
    roi_timeseries,
    standardize_matrix,
    validate_predictions,
    vectorize_rdm,
)

__all__ = [
    "build_manifest",
    "contrast_vectors",
    "event_locked_average",
    "file_sha256",
    "label_segments_from_events",
    "load_events",
    "load_predictions",
    "load_segments",
    "make_segment_table",
    "pattern_matrix",
    "rdm",
    "roi_timeseries",
    "standardize_matrix",
    "validate_predictions",
    "vectorize_rdm",
    "write_json",
    "write_table",
]
