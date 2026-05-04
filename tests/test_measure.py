from __future__ import annotations

from pathlib import Path

from custom_shoe_tree.align import align_to_canonical
from custom_shoe_tree.io import load_scan
from custom_shoe_tree.measure import measure_mesh


def test_measure_mesh_extracts_expected_fields() -> None:
    project_root = Path(__file__).resolve().parents[1]
    scan_path = project_root / "sample-foot-scans" / "0027-A.obj"
    context = measure_mesh(align_to_canonical(load_scan(scan_path)))
    measurements = context.measurements
    assert measurements.length_mm > 250.0
    assert measurements.ball_width_mm > measurements.heel_width_mm
    assert measurements.ball_perimeter_mm > 200.0
    assert measurements.arch_type in {"flat", "small", "medium", "tall"}
    assert measurements.toe_box_type in {"standard", "square", "angled"}
    assert len(context.adaptive_ball_profile) == 21
