from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from last_generator.finalize import run_best


def test_finalize_run_best_writes_report_and_exports(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    sample_scan = project_root.parent / "Sample Foot Scans" / "0014-B.obj"
    output_dir = tmp_path / "phase5" / "0014-B"

    artifacts = run_best(sample_scan, output_dir, allowance_mm=3.0)

    assert artifacts.obj_path.exists()
    assert artifacts.stl_path.exists()
    assert artifacts.report_path.exists()
    assert artifacts.render_path.exists()

    payload = json.loads(artifacts.report_path.read_text(encoding="utf-8"))
    assert payload["phase"] == "phase5"
    assert payload["scan_id"] == "0014-B"
    assert payload["source"] in {"phase4", "phase3-fallback"}
    assert isinstance(payload["volume_ml"], float)
    assert payload["volume_ml"] == payload["volume_ml"]
    assert payload["face_count"] > 0
    assert payload["vertex_count"] > 0


def test_finalize_cli_writes_phase5_artifacts(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    sample_scan = project_root.parent / "Sample Foot Scans" / "0014-B.obj"
    phase3_output_dir = tmp_path / "phase3" / "0014-B"
    warp_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "last_generator.cli",
            "warp",
            str(sample_scan),
            "-o",
            str(phase3_output_dir),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert warp_result.returncode == 0, warp_result.stderr

    output_dir = tmp_path / "phase5" / "0014-B"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "last_generator.cli",
            "finalize",
            str(phase3_output_dir / "last_warp.obj"),
            "-o",
            str(output_dir),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (output_dir / "last_0014-B.obj").exists()
    assert (output_dir / "last_0014-B.stl").exists()
    assert (output_dir / "pipeline_report.json").exists()
    assert (output_dir / "review.png").exists()
