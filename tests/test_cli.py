from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import trimesh


def test_help_lists_phase_commands() -> None:
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "custom_shoe_tree.cli", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "measure" in result.stdout
    assert "template" in result.stdout
    assert "warp" in result.stdout
    assert "refine" in result.stdout
    assert "finalize" in result.stdout
    assert "split" in result.stdout
    assert "pipeline" in result.stdout


def test_measure_command_writes_phase1_artifacts(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    sample_scan = project_root / "sample-foot-scans" / "0014-B.obj"
    output_dir = tmp_path / "phase1" / "0014-B"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "custom_shoe_tree.cli",
            "measure",
            str(sample_scan),
            "-o",
            str(output_dir),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (output_dir / "mesh_aligned.obj").exists()
    assert (output_dir / "annotated.obj").exists()
    assert (output_dir / "annotated.png").exists()
    assert (output_dir / "measurements.json").exists()
    assert (output_dir / "audit.json").exists()


def test_template_command_writes_phase2_artifacts(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "phase2"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "custom_shoe_tree.cli",
            "template",
            "-o",
            str(output_dir),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (output_dir / "base_shoe_tree_decimated.obj").exists()
    assert (output_dir / "base_shoe_tree_measurements.json").exists()
    assert (output_dir / "template_landmarks.json").exists()
    assert (output_dir / "template_landmarks.png").exists()


def test_warp_command_writes_phase3_artifacts(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    sample_scan = project_root / "sample-foot-scans" / "0014-B.obj"
    output_dir = tmp_path / "phase3" / "0014-B"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "custom_shoe_tree.cli",
            "warp",
            str(sample_scan),
            "-o",
            str(output_dir),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (output_dir / "shoe_tree_warp.obj").exists()
    assert (output_dir / "render.png").exists()
    assert (output_dir / "warp_report.json").exists()


def test_refine_command_writes_phase4_artifacts(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    sample_scan = project_root / "sample-foot-scans" / "0014-B.obj"
    output_dir = tmp_path / "phase4" / "0014-B"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "custom_shoe_tree.cli",
            "refine",
            str(sample_scan),
            "-o",
            str(output_dir),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (output_dir / "shoe_tree_refined.obj").exists()
    assert (output_dir / "render_pass1.png").exists()
    assert (output_dir / "comparison.png").exists()
    assert (output_dir / "refine_report.json").exists()


def test_split_command_writes_phase6_artifacts(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    mesh_path = tmp_path / "shoe_tree_box.stl"
    trimesh.creation.box(extents=(50.0, 120.0, 30.0)).export(mesh_path)
    output_dir = tmp_path / "phase6_split" / "box"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "custom_shoe_tree.cli",
            "split",
            str(mesh_path),
            "-o",
            str(output_dir),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (output_dir / "shoe_tree_box_heel_tabs.stl").exists()
    assert (output_dir / "shoe_tree_box_toe_sockets.stl").exists()
    assert (output_dir / "split_report.json").exists()
