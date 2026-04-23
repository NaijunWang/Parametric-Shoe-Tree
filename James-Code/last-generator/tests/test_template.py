from __future__ import annotations

import json
from pathlib import Path

from last_generator.template import run


def test_template_run_writes_expected_outputs(tmp_path: Path) -> None:
    artifacts = run(tmp_path / "phase2")
    assert artifacts.decimated_mesh_path.exists()
    assert artifacts.measurements_path.exists()
    assert artifacts.output_landmarks_path.exists()
    assert artifacts.render_path.exists()
    payload = json.loads(artifacts.output_landmarks_path.read_text(encoding="utf-8"))
    assert set(payload["landmarks"]) == {
        "heel_back",
        "ball_inside",
        "ball_outside",
        "toe_tip",
        "arch_peak",
        "instep_top",
    }
    source_payload = json.loads(
        (Path(__file__).resolve().parents[1] / "src" / "last_generator" / "template_landmarks.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["mesh_vertex_count"] == source_payload["mesh_vertex_count"]
    assert payload["mesh_face_count"] == source_payload["mesh_face_count"]
    assert payload["mesh_vertex_count"] < 29640
    assert payload["mesh_face_count"] < 58986
    assert artifacts.measurements.length_mm > 260.0
    assert artifacts.measurements.max_width_mm < artifacts.measurements.length_mm
