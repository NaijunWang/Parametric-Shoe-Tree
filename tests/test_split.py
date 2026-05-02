from __future__ import annotations

import json
from pathlib import Path

import trimesh

from custom_shoe_tree.split import run, split_mesh_y


def _box_mesh() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(50.0, 120.0, 30.0))


def test_split_mesh_y_returns_heel_and_toe_halves() -> None:
    mesh = _box_mesh()
    heel_half, toe_half, split_y = split_mesh_y(mesh, fraction=0.5)

    assert split_y == 0.0
    assert heel_half.bounds[1][1] <= split_y + 1e-6
    assert toe_half.bounds[0][1] >= split_y - 1e-6
    assert heel_half.is_watertight
    assert toe_half.is_watertight


def test_run_writes_split_stls_and_report(tmp_path: Path) -> None:
    mesh_path = tmp_path / "shoe_tree_box.stl"
    _box_mesh().export(mesh_path)
    output_dir = tmp_path / "phase6_split" / "box"

    artifacts = run(mesh_path, output_dir)

    assert artifacts.heel_tabs_stl_path == output_dir / "shoe_tree_box_heel_tabs.stl"
    assert artifacts.toe_sockets_stl_path == output_dir / "shoe_tree_box_toe_sockets.stl"
    assert artifacts.heel_tabs_stl_path.exists()
    assert artifacts.toe_sockets_stl_path.exists()
    assert artifacts.report_path.exists()
    assert artifacts.clips_applied is True

    payload = json.loads(artifacts.report_path.read_text(encoding="utf-8"))
    assert payload["phase"] == "phase6_split"
    assert payload["scan_id"] == "box"
    assert payload["clips_applied"] is True
    assert payload["heel_tabs"]["stl_path"].endswith("shoe_tree_box_heel_tabs.stl")
    assert payload["toe_sockets"]["stl_path"].endswith("shoe_tree_box_toe_sockets.stl")
