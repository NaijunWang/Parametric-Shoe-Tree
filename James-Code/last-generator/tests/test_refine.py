from __future__ import annotations

import json
from pathlib import Path

import trimesh

from last_generator.refine import run


def test_refine_run_writes_report_and_selected_mesh(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    sample_scan = project_root.parent / "Sample Foot Scans" / "0014-B.obj"
    output_dir = tmp_path / "phase4" / "0014-B"

    artifacts = run(sample_scan, output_dir, allowance_mm=3.0)

    assert artifacts.mesh_path.exists()
    assert artifacts.comparison_path.exists()
    assert artifacts.report_path.exists()
    assert len(artifacts.pass_render_paths) == 3

    payload = json.loads(artifacts.report_path.read_text(encoding="utf-8"))
    assert payload["phase"] == "phase4"
    assert payload["scan_id"] == "0014-B"
    assert "selected_source" in payload
    assert "pass_records" in payload
    assert "final_drift_vs_phase3" in payload

    mesh = trimesh.load(artifacts.mesh_path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    assert mesh.bounds[0][2] >= -1e-6
    assert int(mesh.extents.argmax()) == 1
