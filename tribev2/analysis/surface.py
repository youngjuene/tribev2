# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Surface export helpers for TRIBE cortical vectors."""

from __future__ import annotations

from pathlib import Path

import numpy as np

FSAVERAGE_VERTICES_PER_HEMI = {
    "fsaverage3": 642,
    "fsaverage4": 2562,
    "fsaverage5": 10242,
    "fsaverage6": 40962,
    "fsaverage7": 163842,
}


def _write_ascii_gifti(path: Path, values: np.ndarray, *, intent: str) -> None:
    data = " ".join(f"{float(value):.9g}" for value in values)
    text = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE GIFTI SYSTEM "http://www.nitrc.org/frs/download.php/1594/gifti.dtd">
<GIFTI Version="1.0" NumberOfDataArrays="1">
  <MetaData/>
  <LabelTable/>
  <DataArray Intent="{intent}" DataType="NIFTI_TYPE_FLOAT32" ArrayIndexingOrder="RowMajorOrder" Dimensionality="1" Dim0="{values.shape[0]}" Encoding="ASCII" Endian="LittleEndian" ExternalFileName="" ExternalFileOffset="">
    <MetaData/>
    <Data>{data}</Data>
  </DataArray>
</GIFTI>
"""
    path.write_text(text)


def split_hemispheres(
    vector: np.ndarray, *, mesh: str = "fsaverage5"
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(vector)
    if arr.ndim != 1:
        raise ValueError(f"surface vector must be 1-D, got shape {arr.shape}")
    if mesh not in FSAVERAGE_VERTICES_PER_HEMI:
        raise ValueError(f"unknown fsaverage mesh: {mesh}")
    n_hemi = FSAVERAGE_VERTICES_PER_HEMI[mesh]
    expected = 2 * n_hemi
    if arr.shape[0] != expected:
        raise ValueError(
            f"surface vector length mismatch: expected {expected}, got {arr.shape[0]}"
        )
    return arr[:n_hemi], arr[n_hemi:]


def write_gifti_pair(
    vector: np.ndarray,
    out_prefix: Path,
    *,
    mesh: str = "fsaverage5",
    intent: str = "NIFTI_INTENT_SHAPE",
) -> tuple[Path, Path]:
    """Write left/right GIfTI functional files for a cortical vector."""

    left, right = split_hemispheres(vector, mesh=mesh)
    paths = (
        out_prefix.with_name(out_prefix.name + "_hemi-L.func.gii"),
        out_prefix.with_name(out_prefix.name + "_hemi-R.func.gii"),
    )
    try:
        import nibabel as nib
    except ModuleNotFoundError:
        for values, path in zip((left, right), paths):
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_ascii_gifti(path, np.asarray(values, dtype=np.float32), intent=intent)
    else:
        for values, path in zip((left, right), paths):
            path.parent.mkdir(parents=True, exist_ok=True)
            darray = nib.gifti.GiftiDataArray(
                np.asarray(values, dtype=np.float32), intent=intent
            )
            nib.save(nib.gifti.GiftiImage(darrays=[darray]), str(path))
    return paths


def write_surface_series(
    matrix: np.ndarray,
    out_dir: Path,
    *,
    rows: list[int] | None = None,
    mesh: str = "fsaverage5",
    prefix: str = "row",
) -> list[Path]:
    arr = np.asarray(matrix)
    if arr.ndim != 2:
        raise ValueError(f"surface series must be 2-D, got shape {arr.shape}")
    selected = rows if rows is not None else list(range(arr.shape[0]))
    outputs: list[Path] = []
    for row in selected:
        left, right = write_gifti_pair(
            arr[row], out_dir / f"{prefix}-{row:05d}", mesh=mesh
        )
        outputs.extend([left, right])
    return outputs


def plot_surface_panel(
    vector_or_matrix: np.ndarray,
    out_png: Path,
    *,
    mesh: str = "fsaverage5",
    **kwargs,
) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from tribev2.plotting import PlotBrain

    plotter = PlotBrain(mesh=mesh)
    arr = np.asarray(vector_or_matrix)
    if arr.ndim == 1:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4), subplot_kw={"projection": "3d"})
        plotter.plot_surf(arr, axes=ax, **kwargs)
    elif arr.ndim == 2:
        fig = plotter.plot_timesteps(arr, **kwargs)
    else:
        raise ValueError(
            f"surface plot input must be 1-D or 2-D, got shape {arr.shape}"
        )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_png


def plot_brain_movie(
    matrix: np.ndarray,
    out_mp4: Path,
    *,
    mesh: str = "fsaverage5",
    segments=None,
    **kwargs,
) -> Path:
    from tribev2.plotting import PlotBrain

    plotter = PlotBrain(mesh=mesh)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    plotter.plot_timesteps_mp4(np.asarray(matrix), out_mp4, segments=segments, **kwargs)
    return out_mp4
