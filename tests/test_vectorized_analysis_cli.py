import json
import subprocess
import sys

import numpy as np
import pandas as pd


def test_vectorized_analysis_cli_builds_saved_prediction_artifacts(tmp_path):
    preds = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [3.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 2.0],
            [0.0, 0.0, 4.0, 4.0],
        ],
        dtype=np.float32,
    )
    predictions = tmp_path / "preds.npy"
    segments = tmp_path / "segments.csv"
    labels = tmp_path / "labels.npy"
    output_root = tmp_path / "out"
    np.save(predictions, preds)
    pd.DataFrame(
        {
            "row": [0, 1, 2, 3],
            "absolute_start": [0.0, 1.0, 2.0, 3.0],
            "duration": [1.0, 1.0, 1.0, 1.0],
            "condition": ["A", "A", "B", "B"],
        }
    ).to_csv(segments, index=False)
    np.save(labels, np.array([1, 1, 2, 2]))

    out = subprocess.check_output(
        [
            sys.executable,
            "tools/run_vectorized_analysis.py",
            "--predictions",
            str(predictions),
            "--segments",
            str(segments),
            "--roi-labels",
            str(labels),
            "--condition-column",
            "condition",
            "--contrast",
            "A-B",
            "--write-rdm",
            "--expected-vertices",
            "4",
            "--output-root",
            str(output_root),
        ],
        text=True,
    )
    report = json.loads(out)
    assert report["status"] == "completed"
    assert report["predictions_shape"] == [4, 4]
    assert report["statistical_scope"] == "descriptive_model_prediction"
    assert (output_root / "predictions.npy").exists()
    assert (output_root / "roi_timeseries.csv").exists()
    assert (output_root / "event_patterns.npy").exists()
    assert (output_root / "contrast_vectors.npy").exists()
    assert (output_root / "rdm_vector.npy").exists()
    manifest = json.loads((output_root / "source_data_manifest.json").read_text())
    assert manifest["schema"] == "tribev2.vectorized-analysis.v1"


def test_vectorized_analysis_cli_dry_runs_freesurfer_without_surface_write(tmp_path):
    predictions = tmp_path / "preds.npy"
    output_root = tmp_path / "out"
    np.save(predictions, np.zeros((2, 4), dtype=np.float32))

    out = subprocess.check_output(
        [
            sys.executable,
            "tools/run_vectorized_analysis.py",
            "--predictions",
            str(predictions),
            "--expected-vertices",
            "4",
            "--output-root",
            str(output_root),
            "--dry-run-freesurfer",
            "--subjects-dir",
            str(tmp_path / "subjects"),
        ],
        text=True,
    )
    report = json.loads(out)
    assert report["status"] == "completed"
    assert "freesurfer_commands" in report
    assert report["freesurfer_surf2vol_method"] == "projection_merge"
    assert report["freesurfer_commands"]["surf2vol_lh"][0] == "mri_surf2vol"
    assert report["freesurfer_commands"]["surf2vol_rh_merge"][0] == "mri_surf2vol"
    assert not (output_root / "surface_maps").exists()


def test_vectorized_analysis_cli_maps_conditions_from_events(tmp_path):
    predictions = tmp_path / "preds.npy"
    segments = tmp_path / "segments.csv"
    events = tmp_path / "events.csv"
    output_root = tmp_path / "out"
    np.save(
        predictions,
        np.array(
            [
                [1.0, 1.0, 0.0, 0.0],
                [3.0, 3.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 2.0],
                [0.0, 0.0, 4.0, 4.0],
            ],
            dtype=np.float32,
        ),
    )
    pd.DataFrame(
        {
            "row": [0, 1, 2, 3],
            "absolute_start": [0.0, 1.0, 2.0, 3.0],
            "stop": [1.0, 2.0, 3.0, 4.0],
        }
    ).to_csv(segments, index=False)
    pd.DataFrame(
        {
            "start": [0.0, 2.0],
            "duration": [2.0, 2.0],
            "type": ["A", "B"],
        }
    ).to_csv(events, index=False)

    out = subprocess.check_output(
        [
            sys.executable,
            "tools/run_vectorized_analysis.py",
            "--predictions",
            str(predictions),
            "--segments",
            str(segments),
            "--events",
            str(events),
            "--condition-column",
            "type",
            "--expected-vertices",
            "4",
            "--output-root",
            str(output_root),
        ],
        text=True,
    )
    report = json.loads(out)
    assert report["status"] == "completed"
    assert (output_root / "event_patterns.npy").exists()
    rewritten_segments = pd.read_csv(output_root / "segments.csv")
    assert rewritten_segments["type"].tolist() == ["A", "A", "B", "B"]


def test_vectorized_analysis_cli_writes_surface_gifti_without_plotting_extra(tmp_path):
    predictions = tmp_path / "preds.npy"
    output_root = tmp_path / "out"
    np.save(predictions, np.zeros((1, 1284), dtype=np.float32))

    subprocess.check_output(
        [
            sys.executable,
            "tools/run_vectorized_analysis.py",
            "--predictions",
            str(predictions),
            "--expected-vertices",
            "1284",
            "--mesh",
            "fsaverage3",
            "--write-surface",
            "--output-root",
            str(output_root),
        ],
        text=True,
    )
    left = output_root / "surface_maps" / "prediction_row-00000_hemi-L.func.gii"
    right = output_root / "surface_maps" / "prediction_row-00000_hemi-R.func.gii"
    assert left.exists()
    assert right.exists()
    assert "<GIFTI" in left.read_text()
