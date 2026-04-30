from __future__ import annotations

import json
from pathlib import Path

import trimesh

from last_generator.warp import run


def test_warp_run_writes_artifacts_and_improves_ball_width(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    sample_scan = project_root.parent / "Sample Foot Scans" / "0014-B.obj"
    output_dir = tmp_path / "phase3" / "0014-B"

    artifacts = run(sample_scan, output_dir, allowance_mm=3.0)

    assert artifacts.mesh_path.exists()
    assert artifacts.render_path.exists()
    assert artifacts.report_path.exists()

    payload = json.loads(artifacts.report_path.read_text(encoding="utf-8"))
    assert payload["section_count"] == 60
    assert payload["allowance_mm"] == 3.0
    assert payload["scan_id"] == "0014-B"

    mesh = trimesh.load(artifacts.mesh_path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    assert mesh.bounds[0][2] >= -1e-6
    assert int(mesh.extents.argmax()) == 1
    assert abs(payload["warped_measurements"]["length_mm"] - payload["target_measurements"]["length_mm"]) < 1e-6

    template_toe = payload["review_summary"]["template"]["toe_width_mm"]
    target_toe = payload["review_summary"]["targets"]["toe_width_mm"]
    achieved_toe = payload["review_summary"]["achieved"]["toe_width_mm"]
    assert abs(achieved_toe - target_toe) < abs(template_toe - target_toe)
