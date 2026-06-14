# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""FreeSurfer preflight and command builders for vectorized outputs."""

from __future__ import annotations

import os
import shutil
import subprocess
import typing as tp
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FreeSurferPreflight:
    ok: bool
    freesurfer_home: str | None
    subjects_dir: str | None
    commands: dict[str, str | None]
    missing: list[str]


def preflight(
    *,
    subjects_dir: Path | None = None,
    required_commands: tp.Sequence[str] = ("mri_surf2vol", "mri_surfcluster"),
) -> FreeSurferPreflight:
    fs_home = os.environ.get("FREESURFER_HOME")
    sd = str(subjects_dir) if subjects_dir is not None else os.environ.get("SUBJECTS_DIR")
    commands = {cmd: shutil.which(cmd) for cmd in required_commands}
    missing = []
    if not fs_home:
        missing.append("FREESURFER_HOME")
    if not sd:
        missing.append("SUBJECTS_DIR")
    missing.extend([cmd for cmd, path in commands.items() if path is None])
    return FreeSurferPreflight(
        ok=not missing,
        freesurfer_home=fs_home,
        subjects_dir=sd,
        commands=commands,
        missing=missing,
    )


def surf2vol_command(
    *,
    left_surface: Path,
    left_overlay: Path,
    right_surface: Path,
    right_overlay: Path,
    output_volume: Path,
    template: Path | None = None,
    subject: str = "fsaverage",
    subjects_dir: Path | None = None,
) -> list[str]:
    cmd = [
        "mri_surf2vol",
        "--so",
        str(left_surface),
        str(left_overlay),
        "--so",
        str(right_surface),
        str(right_overlay),
        "--subject",
        subject,
        "--o",
        str(output_volume),
    ]
    if template is not None:
        cmd.extend(["--template", str(template)])
    if subjects_dir is not None:
        cmd.extend(["--sd", str(subjects_dir)])
    return cmd


def surf2vol_projection_command(
    *,
    in_file: Path,
    hemi: tp.Literal["lh", "rh"],
    output_volume: Path,
    subject: str = "fsaverage5",
    surface: str = "white",
    template: Path | None = None,
    merge: Path | None = None,
    subjects_dir: Path | None = None,
) -> list[str]:
    cmd = [
        "mri_surf2vol",
        "--surfval",
        str(in_file),
        "--hemi",
        hemi,
        "--surf",
        surface,
        "--identity",
        subject,
    ]
    if merge is not None:
        cmd.extend(["--merge", str(merge)])
    elif template is not None:
        cmd.extend(["--template", str(template)])
    cmd.extend(["--o", str(output_volume)])
    if subjects_dir is not None:
        cmd.extend(["--sd", str(subjects_dir)])
    return cmd


def surfcluster_command(
    *,
    in_file: Path,
    hemi: tp.Literal["lh", "rh"],
    summary_file: Path,
    subject: str = "fsaverage",
    surface: str = "white",
    thmin: float = 0.0,
    thmax: float | None = None,
    sign: tp.Literal["pos", "neg", "abs"] = "pos",
    minarea: float = 0.0,
    subjects_dir: Path | None = None,
    annot: str | None = None,
    no_adjust: bool = True,
) -> list[str]:
    cmd = [
        "mri_surfcluster",
        "--in",
        str(in_file),
        "--subject",
        subject,
        "--hemi",
        hemi,
        "--surf",
        surface,
        "--thmin",
        str(thmin),
        "--sign",
        sign,
        "--minarea",
        str(minarea),
        "--sum",
        str(summary_file),
    ]
    if thmax is not None:
        cmd.extend(["--thmax", str(thmax)])
    if annot is not None:
        cmd.extend(["--annot", annot])
    if no_adjust:
        cmd.append("--no-adjust")
    if subjects_dir is not None:
        cmd.extend(["--sd", str(subjects_dir)])
    return cmd


def surf2surf_command(
    *,
    srcsubject: str,
    srcsurfval: Path,
    trgsubject: str,
    trgsurfval: Path,
    hemi: tp.Literal["lh", "rh"],
    subjects_dir: Path | None = None,
) -> list[str]:
    cmd = [
        "mri_surf2surf",
        "--srcsubject",
        srcsubject,
        "--srcsurfval",
        str(srcsurfval),
        "--trgsubject",
        trgsubject,
        "--trgsurfval",
        str(trgsurfval),
        "--hemi",
        hemi,
    ]
    if subjects_dir is not None:
        cmd.extend(["--sd", str(subjects_dir)])
    return cmd


def run_command(cmd: list[str], *, dry_run: bool = False) -> dict[str, tp.Any]:
    if dry_run:
        return {"status": "dry_run", "command": cmd}
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    return {
        "status": "completed" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "command": cmd,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
