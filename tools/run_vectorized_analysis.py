#!/usr/bin/env python3
"""Create vectorized TRIBE analysis artifacts from predictions."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tribev2.analysis.freesurfer import (
    preflight,
    run_command,
    surf2vol_command,
    surf2vol_projection_command,
    surfcluster_command,
)
from tribev2.analysis.io import (
    build_manifest,
    load_events,
    load_predictions,
    load_segments,
    write_array,
    write_json,
    write_table,
)
from tribev2.analysis.surface import write_gifti_pair
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


def _parse_rows(value: str | None) -> list[int] | None:
    if value is None or value == "":
        return None
    rows: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        rows.append(int(part))
    return rows


def _freesurfer_subject(args: argparse.Namespace) -> str:
    if args.freesurfer_subject:
        return args.freesurfer_subject
    if str(args.mesh).startswith("fsaverage"):
        return str(args.mesh)
    return "fsaverage"


def _load_or_predict(
    args: argparse.Namespace,
) -> tuple[np.ndarray, Any, Any, dict[str, Any]]:
    if args.predictions is not None:
        preds = load_predictions(
            args.predictions, expected_vertices=args.expected_vertices
        )
        segments = load_segments(args.segments, n_rows=preds.shape[0])
        events = load_events(args.events)
        return (
            preds,
            segments,
            events,
            {"mode": "saved_predictions", "predictions": str(args.predictions)},
        )

    source_args = [args.video, args.audio, args.text]
    if sum(path is not None for path in source_args) != 1:
        raise ValueError(
            "provide --predictions or exactly one of --video, --audio, --text"
        )

    from tribev2 import TribeModel

    model = TribeModel.from_pretrained(
        args.checkpoint,
        cache_folder=args.cache_folder,
        device=args.device,
    )
    events = model.get_events_dataframe(
        video_path=str(args.video) if args.video else None,
        audio_path=str(args.audio) if args.audio else None,
        text_path=str(args.text) if args.text else None,
    )
    preds, segments = model.predict(events=events, verbose=not args.quiet)
    preds = validate_predictions(preds, expected_vertices=args.expected_vertices)
    return preds, segments, events, {
        "mode": "direct_inference",
        "checkpoint": args.checkpoint,
        "cache_folder": str(args.cache_folder),
        "device": args.device,
    }


def _write_base_artifacts(
    *,
    preds: np.ndarray,
    segments: Any,
    events: Any,
    output_root: Path,
    provenance: dict[str, Any],
    mesh: str,
    hemodynamic_offset_sec: float,
) -> tuple[list[Path], Any]:
    artifacts: list[Path] = []
    predictions_path = write_array(
        output_root / "predictions.npy", preds.astype(np.float32)
    )
    artifacts.append(predictions_path)

    segment_table = make_segment_table(segments, n_rows=preds.shape[0])
    artifacts.append(write_table(output_root / "segments.csv", segment_table))

    if events is not None:
        artifacts.append(write_table(output_root / "events.csv", events))

    metadata = {
        "mesh": mesh,
        "expected_vertices": preds.shape[1],
        "n_segments": preds.shape[0],
        "hemodynamic_offset_sec": hemodynamic_offset_sec,
        **provenance,
    }
    artifacts.append(write_json(output_root / "metadata.json", metadata))
    return artifacts, segment_table


def _write_optional_vector_artifacts(
    *,
    preds: np.ndarray,
    segment_table,
    events,
    args: argparse.Namespace,
    output_root: Path,
) -> tuple[list[Path], dict[str, Any]]:
    artifacts: list[Path] = []
    report: dict[str, Any] = {}

    if args.roi_labels is not None:
        labels = np.load(args.roi_labels, allow_pickle=True)
        roi_table = roi_timeseries(
            preds,
            labels,
            expected_vertices=args.expected_vertices,
        )
        path = write_table(output_root / "roi_timeseries.csv", roi_table)
        artifacts.append(path)
        report["roi_timeseries"] = {"path": str(path), "shape": list(roi_table.shape)}

    if args.condition_column:
        if (
            events is not None
            and args.condition_column not in segment_table.columns
        ):
            if events is not None and args.condition_column in events.columns:
                try:
                    segment_table = label_segments_from_events(
                        segment_table,
                        events,
                        condition_col=args.condition_column,
                    )
                    write_table(output_root / "segments.csv", segment_table)
                except ValueError as exc:
                    report.setdefault("warnings", []).append(str(exc))
        try:
            patterns, conditions = pattern_matrix(
                preds,
                segment_table,
                condition_col=args.condition_column,
                expected_vertices=args.expected_vertices,
            )
        except ValueError as exc:
            report.setdefault("warnings", []).append(str(exc))
        else:
            pattern_path = write_array(
                output_root / "event_patterns.npy", patterns.astype(np.float32)
            )
            condition_path = write_json(
                output_root / "conditions.json", {"conditions": conditions}
            )
            artifacts.extend([pattern_path, condition_path])
            report["event_patterns"] = {
                "path": str(pattern_path),
                "shape": list(patterns.shape),
            }

            if args.contrast:
                contrasts, contrast_names = contrast_vectors(
                    patterns, conditions, args.contrast
                )
                contrast_path = write_array(
                    output_root / "contrast_vectors.npy",
                    contrasts.astype(np.float32),
                )
                contrast_name_path = write_json(
                    output_root / "contrasts.json",
                    {"contrasts": contrast_names},
                )
                artifacts.extend([contrast_path, contrast_name_path])
                report["contrast_vectors"] = {
                    "path": str(contrast_path),
                    "shape": list(contrasts.shape),
                }

            if args.write_rdm:
                matrix = rdm(patterns, metric=args.rdm_metric)
                rdm_path = write_array(output_root / "rdm.npy", matrix.astype(np.float32))
                rdm_vec_path = write_array(
                    output_root / "rdm_vector.npy",
                    vectorize_rdm(matrix).astype(np.float32),
                )
                artifacts.extend([rdm_path, rdm_vec_path])
                report["rdm"] = {"path": str(rdm_path), "shape": list(matrix.shape)}

    return artifacts, report


def _write_surface_and_freesurfer(
    *,
    preds: np.ndarray,
    args: argparse.Namespace,
    output_root: Path,
) -> tuple[list[Path], dict[str, Any]]:
    artifacts: list[Path] = []
    report: dict[str, Any] = {}
    rows = _parse_rows(args.surface_rows) or [0]

    if args.write_surface or args.run_freesurfer:
        surface_dir = output_root / "surface_maps"
        surface_outputs: list[Path] = []
        for row in rows:
            if row < 0 or row >= preds.shape[0]:
                raise ValueError(f"surface row {row} is outside prediction rows")
            left, right = write_gifti_pair(
                preds[row],
                surface_dir / f"prediction_row-{row:05d}",
                mesh=args.mesh,
            )
            surface_outputs.extend([left, right])
        artifacts.extend(surface_outputs)
        report["surface_maps"] = [str(path) for path in surface_outputs]

    if args.dry_run_freesurfer or args.run_freesurfer:
        subjects_dir = args.subjects_dir or Path(preflight().subjects_dir or "")
        fs_subject = _freesurfer_subject(args)
        left_surface = subjects_dir / fs_subject / "surf" / "lh.white"
        right_surface = subjects_dir / fs_subject / "surf" / "rh.white"
        if not report.get("surface_maps"):
            surface_dir = output_root / "surface_maps"
            row = rows[0]
            report["surface_maps"] = [
                str(surface_dir / f"prediction_row-{row:05d}_hemi-L.func.gii"),
                str(surface_dir / f"prediction_row-{row:05d}_hemi-R.func.gii"),
            ]
        left_overlay = Path(report["surface_maps"][0])
        right_overlay = Path(report["surface_maps"][1])
        volume_output = (
            output_root / "freesurfer" / "volume" / "prediction_row-00000.nii.gz"
        )
        left_volume_output = (
            output_root / "freesurfer" / "volume" / "prediction_row-00000_lh.nii.gz"
        )
        cluster_left = output_root / "freesurfer" / "clusters" / "lh.summary"
        cluster_right = output_root / "freesurfer" / "clusters" / "rh.summary"
        fs_status = preflight(subjects_dir=args.subjects_dir)
        report["freesurfer_preflight"] = {
            "ok": fs_status.ok,
            "freesurfer_home": fs_status.freesurfer_home,
            "subjects_dir": fs_status.subjects_dir,
            "commands": fs_status.commands,
            "missing": fs_status.missing,
        }
        ribbon = subjects_dir / fs_subject / "mri" / "ribbon.mgz"
        template = args.freesurfer_template or (
            subjects_dir / fs_subject / "mri" / "orig.mgz"
        )
        if ribbon.exists():
            surf2vol_commands = {
                "surf2vol": surf2vol_command(
                    left_surface=left_surface,
                    left_overlay=left_overlay,
                    right_surface=right_surface,
                    right_overlay=right_overlay,
                    output_volume=volume_output,
                    template=args.freesurfer_template,
                    subject=fs_subject,
                    subjects_dir=args.subjects_dir,
                )
            }
            report["freesurfer_surf2vol_method"] = "ribbon"
        else:
            surf2vol_commands = {
                "surf2vol_lh": surf2vol_projection_command(
                    in_file=left_overlay,
                    hemi="lh",
                    output_volume=left_volume_output,
                    subject=fs_subject,
                    template=template,
                    subjects_dir=args.subjects_dir,
                ),
                "surf2vol_rh_merge": surf2vol_projection_command(
                    in_file=right_overlay,
                    hemi="rh",
                    output_volume=volume_output,
                    subject=fs_subject,
                    merge=left_volume_output,
                    subjects_dir=args.subjects_dir,
                ),
            }
            report["freesurfer_surf2vol_method"] = "projection_merge"
        report["freesurfer_commands"] = {
            **surf2vol_commands,
            "surfcluster_lh": surfcluster_command(
                in_file=left_overlay,
                hemi="lh",
                summary_file=cluster_left,
                subject=fs_subject,
                thmin=args.cluster_thmin,
                thmax=args.cluster_thmax,
                sign=args.cluster_sign,
                minarea=args.cluster_minarea,
                subjects_dir=args.subjects_dir,
            ),
            "surfcluster_rh": surfcluster_command(
                in_file=right_overlay,
                hemi="rh",
                summary_file=cluster_right,
                subject=fs_subject,
                thmin=args.cluster_thmin,
                thmax=args.cluster_thmax,
                sign=args.cluster_sign,
                minarea=args.cluster_minarea,
                subjects_dir=args.subjects_dir,
            ),
        }
        if args.run_freesurfer:
            if not fs_status.ok:
                missing = ", ".join(fs_status.missing)
                raise RuntimeError(f"FreeSurfer preflight failed: {missing}")
            volume_output.parent.mkdir(parents=True, exist_ok=True)
            cluster_left.parent.mkdir(parents=True, exist_ok=True)
            report["freesurfer_runs"] = {
                name: run_command(cmd)
                for name, cmd in report["freesurfer_commands"].items()
            }

    return artifacts, report


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output_root: Path = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    preds, segments, events, provenance = _load_or_predict(args)
    artifacts, segment_table = _write_base_artifacts(
        preds=preds,
        segments=segments,
        events=events,
        output_root=output_root,
        provenance=provenance,
        mesh=args.mesh,
        hemodynamic_offset_sec=args.hemodynamic_offset_sec,
    )
    optional_artifacts, optional_report = _write_optional_vector_artifacts(
        preds=preds,
        segment_table=segment_table,
        events=events,
        args=args,
        output_root=output_root,
    )
    artifacts.extend(optional_artifacts)
    surface_artifacts, surface_report = _write_surface_and_freesurfer(
        preds=preds,
        args=args,
        output_root=output_root,
    )
    artifacts.extend(surface_artifacts)
    manifest = build_manifest(
        output_root=output_root,
        artifacts=artifacts,
        metadata={
            "mesh": args.mesh,
            "statistical_scope": "descriptive_model_prediction",
            "hemodynamic_offset_sec": args.hemodynamic_offset_sec,
        },
    )
    manifest_path = write_json(output_root / "source_data_manifest.json", manifest)
    report = {
        "status": "completed",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "output_root": str(output_root),
        "predictions_shape": list(preds.shape),
        "statistical_scope": "descriptive_model_prediction",
        "manifest": str(manifest_path),
        **optional_report,
        **surface_report,
    }
    report_path = write_json(output_root / "report.json", report)
    report["report"] = str(report_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--segments", type=Path, default=None)
    parser.add_argument("--events", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-vertices", type=int, default=20484)
    parser.add_argument("--mesh", default="fsaverage5")
    parser.add_argument("--hemodynamic-offset-sec", type=float, default=5.0)

    parser.add_argument("--roi-labels", type=Path, default=None)
    parser.add_argument("--condition-column", default=None)
    parser.add_argument("--contrast", action="append", default=[])
    parser.add_argument("--write-rdm", action="store_true")
    parser.add_argument(
        "--rdm-metric", choices=["correlation", "euclidean"], default="correlation"
    )

    parser.add_argument("--write-surface", action="store_true")
    parser.add_argument(
        "--surface-rows",
        default=None,
        help="Comma-separated prediction rows; default 0.",
    )

    parser.add_argument("--dry-run-freesurfer", action="store_true")
    parser.add_argument("--run-freesurfer", action="store_true")
    parser.add_argument("--subjects-dir", type=Path, default=None)
    parser.add_argument("--freesurfer-subject", default=None)
    parser.add_argument("--freesurfer-template", type=Path, default=None)
    parser.add_argument("--cluster-thmin", type=float, default=0.0)
    parser.add_argument("--cluster-thmax", type=float, default=None)
    parser.add_argument("--cluster-sign", choices=["pos", "neg", "abs"], default="pos")
    parser.add_argument("--cluster-minarea", type=float, default=0.0)

    parser.add_argument("--checkpoint", default="facebook/tribev2")
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--audio", type=Path, default=None)
    parser.add_argument("--text", type=Path, default=None)
    parser.add_argument("--cache-folder", type=Path, default=Path("cache"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        report = run(args)
    except Exception as exc:
        payload = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
        print(json.dumps(payload, indent=2), file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
