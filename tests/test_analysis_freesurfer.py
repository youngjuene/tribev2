from pathlib import Path

from tribev2.analysis.freesurfer import (
    surf2surf_command,
    surf2vol_command,
    surf2vol_projection_command,
    surfcluster_command,
)


def test_surf2vol_command_contains_surface_overlay_pairs():
    cmd = surf2vol_command(
        left_surface=Path("subjects/fsaverage/surf/lh.white"),
        left_overlay=Path("lh.func.gii"),
        right_surface=Path("subjects/fsaverage/surf/rh.white"),
        right_overlay=Path("rh.func.gii"),
        output_volume=Path("out.nii.gz"),
        template=Path("template.nii.gz"),
        subjects_dir=Path("subjects"),
    )
    assert cmd[:3] == ["mri_surf2vol", "--so", "subjects/fsaverage/surf/lh.white"]
    assert "--template" in cmd
    assert "template.nii.gz" in cmd
    assert "--sd" in cmd


def test_surf2vol_projection_command_can_merge_hemisphere_volumes():
    cmd = surf2vol_projection_command(
        in_file=Path("rh.func.gii"),
        hemi="rh",
        output_volume=Path("merged.nii.gz"),
        subject="fsaverage5",
        merge=Path("lh.nii.gz"),
        subjects_dir=Path("subjects"),
    )
    assert cmd[:4] == ["mri_surf2vol", "--surfval", "rh.func.gii", "--hemi"]
    assert cmd[cmd.index("--identity") + 1] == "fsaverage5"
    assert cmd[cmd.index("--merge") + 1] == "lh.nii.gz"
    assert cmd[cmd.index("--o") + 1] == "merged.nii.gz"
    assert cmd[cmd.index("--sd") + 1] == "subjects"


def test_surfcluster_command_is_descriptive_thresholdable_surface_cluster():
    cmd = surfcluster_command(
        in_file=Path("lh.func.gii"),
        hemi="lh",
        summary_file=Path("lh.summary"),
        thmin=2.0,
        sign="abs",
        minarea=25.0,
    )
    assert cmd[0] == "mri_surfcluster"
    assert cmd[cmd.index("--hemi") + 1] == "lh"
    assert cmd[cmd.index("--thmin") + 1] == "2.0"
    assert cmd[cmd.index("--sign") + 1] == "abs"
    assert "--thmax" not in cmd
    assert "--no-adjust" in cmd


def test_surfcluster_command_includes_thmax_only_when_set():
    cmd = surfcluster_command(
        in_file=Path("lh.func.gii"),
        hemi="lh",
        summary_file=Path("lh.summary"),
        thmax=4.0,
    )
    assert cmd[cmd.index("--thmax") + 1] == "4.0"


def test_surf2surf_command_records_subjects_and_hemi():
    cmd = surf2surf_command(
        srcsubject="fsaverage",
        srcsurfval=Path("lh.func.gii"),
        trgsubject="bert",
        trgsurfval=Path("lh.bert.func.gii"),
        hemi="lh",
    )
    assert cmd == [
        "mri_surf2surf",
        "--srcsubject",
        "fsaverage",
        "--srcsurfval",
        "lh.func.gii",
        "--trgsubject",
        "bert",
        "--trgsurfval",
        "lh.bert.func.gii",
        "--hemi",
        "lh",
    ]
