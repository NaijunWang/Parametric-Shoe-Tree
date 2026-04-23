from __future__ import annotations

from pathlib import Path

from last_generator.io import (
    load_scan,
    mesh_audit_from_metadata,
    resolve_output_dir,
    scan_id_from_path,
)


def test_scan_id_from_path_uses_stem() -> None:
    assert scan_id_from_path("Sample Foot Scans/0030-A.obj") == "0030-A"


def test_resolve_output_dir_defaults_to_phase_and_scan(tmp_path: Path) -> None:
    scan_path = tmp_path / "0027-A.obj"
    scan_path.write_text("# stub\n", encoding="utf-8")
    destination = resolve_output_dir(tmp_path / "phase0" / "0027-A", phase_name="phase0", scan_path=scan_path)
    assert destination.name == "0027-A"
    assert destination.exists()


def test_load_scan_repairs_small_open_hole() -> None:
    project_root = Path(__file__).resolve().parents[1]
    scan_path = project_root.parent / "Sample Foot Scans" / "0030-A.obj"
    mesh = load_scan(scan_path)
    audit = mesh_audit_from_metadata(mesh)
    assert audit.units == "m"
    assert audit.repair_applied is True
    assert audit.pre_repair_boundary_edge_count == 3
    assert audit.watertight is True
    assert mesh.is_watertight is True
