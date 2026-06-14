from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tribev2.analysis.surface import split_hemispheres
from tribev2.analysis.vectorize import (
    contrast_vectors,
    label_segments_from_events,
    make_segment_table,
    pattern_matrix,
    rdm,
    roi_timeseries,
    validate_predictions,
    vectorize_rdm,
)


def test_validate_predictions_rejects_wrong_shape_and_nonfinite():
    with pytest.raises(ValueError, match="2-D"):
        validate_predictions(np.zeros(4), expected_vertices=2)
    with pytest.raises(ValueError, match="vertex count"):
        validate_predictions(np.zeros((3, 5)), expected_vertices=4)
    bad = np.zeros((3, 4))
    bad[1, 2] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate_predictions(bad, expected_vertices=4)


def test_make_segment_table_serializes_object_like_segments():
    segment = SimpleNamespace(
        start=10.0,
        offset=2.0,
        duration=1.5,
        timeline="default",
        subject="average",
        ns_events=[SimpleNamespace(type="Word"), SimpleNamespace(type="Audio")],
    )
    table = make_segment_table([segment], n_rows=1)
    assert table.loc[0, "row"] == 0
    assert table.loc[0, "absolute_start"] == 12.0
    assert table.loc[0, "stop"] == 13.5
    assert table.loc[0, "event_count"] == 2


def test_roi_timeseries_accepts_label_vector():
    preds = np.array(
        [
            [1.0, 3.0, 10.0, 20.0],
            [5.0, 7.0, 30.0, 40.0],
        ]
    )
    labels = np.array([1, 1, 2, 2])
    table = roi_timeseries(preds, labels, expected_vertices=4)
    assert table.columns.tolist() == ["row", "1", "2"]
    np.testing.assert_allclose(table["1"], [2.0, 6.0])
    np.testing.assert_allclose(table["2"], [15.0, 35.0])


def test_patterns_contrasts_and_rdm_are_vectorized():
    preds = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [3.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 2.0],
            [0.0, 0.0, 4.0, 4.0],
        ]
    )
    segments = pd.DataFrame({"condition": ["A", "A", "B", "B"]})
    patterns, names = pattern_matrix(
        preds,
        segments,
        condition_col="condition",
        expected_vertices=4,
    )
    assert names == ["A", "B"]
    np.testing.assert_allclose(patterns, [[2.0, 2.0, 0.0, 0.0], [0.0, 0.0, 3.0, 3.0]])

    contrasts, contrast_names = contrast_vectors(patterns, names, ["A-B"])
    assert contrast_names == ["A-B"]
    np.testing.assert_allclose(contrasts[0], [2.0, 2.0, -3.0, -3.0])

    matrix = rdm(patterns)
    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == 0
    assert matrix[1, 1] == 0
    assert vectorize_rdm(matrix).shape == (1,)


def test_label_segments_from_events_uses_first_overlapping_event():
    segments = pd.DataFrame(
        {
            "absolute_start": [0.0, 1.0, 2.0],
            "stop": [1.0, 2.0, 3.0],
        }
    )
    events = pd.DataFrame(
        {
            "start": [0.5, 2.0],
            "duration": [0.75, 0.5],
            "type": ["Word", "Video"],
        }
    )
    labeled = label_segments_from_events(segments, events, condition_col="type")
    assert labeled["type"].tolist() == ["Word", "Word", "Video"]


def test_split_hemispheres_rejects_wrong_vertex_count():
    left, right = split_hemispheres(np.arange(1284), mesh="fsaverage3")
    assert left.shape == (642,)
    assert right.shape == (642,)
    with pytest.raises(ValueError, match="length mismatch"):
        split_hemispheres(np.arange(8), mesh="fsaverage3")
