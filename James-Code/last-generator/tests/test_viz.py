from __future__ import annotations

from pathlib import Path

from last_generator.measure import run


def test_run_writes_nonempty_png_and_obj(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    scan_path = project_root.parent / "Sample Foot Scans" / "0014-B.obj"
    artifacts = run(scan_path, tmp_path / "phase1" / "0014-B")
    assert artifacts.annotated_png_path.stat().st_size > 10_000
    assert artifacts.annotated_obj_path.stat().st_size > 10_000
