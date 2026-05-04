from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import Polygon
import trimesh

from custom_shoe_tree.io import (
    coerce_trimesh,
    ensure_directory,
    ensure_input_path,
    resolve_output_dir,
    save_binary_stl,
    write_json,
)

LOGGER = logging.getLogger(__name__)

TAB_EMBED_MM = 2.0
DEFAULT_CLIP_PARAMS = {
    "tab_width": 8.0,
    "tab_height": 3.0,
    "tab_length": 10.0,
    "chamfer": 1.5,
    "socket_tolerance": 0.3,
}


@dataclass(slots=True)
class Phase6Artifacts:
    output_dir: Path
    heel_tabs_stl_path: Path
    toe_sockets_stl_path: Path
    report_path: Path
    split_y_mm: float
    clips_applied: bool
    heel_watertight: bool
    toe_watertight: bool


@dataclass(slots=True)
class MeshStats:
    vertex_count: int
    face_count: int
    watertight: bool
    volume: bool
    extents_mm: list[float]


@dataclass(slots=True)
class ClipPlacement:
    split_face_x_min_mm: float
    split_face_x_max_mm: float
    split_face_z_min_mm: float
    split_face_z_max_mm: float
    x_positions_mm: list[float]
    z_position_mm: float
    used_bounds_fallback: bool


def _round(value: float) -> float:
    return round(float(value), 6)


def _stats(mesh: trimesh.Trimesh) -> MeshStats:
    extents = mesh.extents if mesh.extents is not None else np.zeros(3)
    return MeshStats(
        vertex_count=int(len(mesh.vertices)),
        face_count=int(len(mesh.faces)),
        watertight=bool(mesh.is_watertight),
        volume=bool(mesh.is_volume),
        extents_mm=[_round(value) for value in extents],
    )


def _rebuild_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    rebuilt = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), process=False)
    rebuilt.metadata.update(mesh.metadata)
    rebuilt.remove_unreferenced_vertices()
    rebuilt.merge_vertices()
    return rebuilt


def _infer_scan_id(mesh_path: Path) -> str:
    if mesh_path.stem in {"shoe_tree_warp", "shoe_tree_refined"} and mesh_path.parent.name:
        return mesh_path.parent.name
    if mesh_path.stem.startswith("shoe_tree_"):
        return mesh_path.stem.removeprefix("shoe_tree_")
    return mesh_path.stem


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    input_path = ensure_input_path(path)
    loaded = trimesh.load(input_path, force="mesh", process=False)
    return _rebuild_mesh(coerce_trimesh(loaded))


def _edge_use_counts(mesh: trimesh.Trimesh) -> np.ndarray:
    return np.bincount(mesh.edges_unique_inverse, minlength=len(mesh.edges_unique))


def _boundary_components(mesh: trimesh.Trimesh) -> list[list[int]]:
    boundary_edges = mesh.edges_unique[_edge_use_counts(mesh) == 1]
    adjacency: dict[int, set[int]] = {}
    for a, b in boundary_edges:
        adjacency.setdefault(int(a), set()).add(int(b))
        adjacency.setdefault(int(b), set()).add(int(a))

    components: list[list[int]] = []
    seen: set[int] = set()
    for vertex in adjacency:
        if vertex in seen:
            continue
        stack = [vertex]
        component: list[int] = []
        seen.add(vertex)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def _order_boundary_loop(component: list[int], mesh: trimesh.Trimesh) -> list[int] | None:
    component_set = set(component)
    boundary_edges = mesh.edges_unique[_edge_use_counts(mesh) == 1]
    adjacency: dict[int, list[int]] = {vertex: [] for vertex in component}
    for a_raw, b_raw in boundary_edges:
        a = int(a_raw)
        b = int(b_raw)
        if a in component_set and b in component_set:
            adjacency[a].append(b)
            adjacency[b].append(a)

    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        return None

    start = component[0]
    loop = [start]
    previous: int | None = None
    current = start
    for _ in range(len(component) + 1):
        neighbors = adjacency[current]
        next_vertex = neighbors[0] if neighbors[0] != previous else neighbors[1]
        if next_vertex == start:
            return loop
        if next_vertex in loop:
            return None
        loop.append(next_vertex)
        previous, current = current, next_vertex
    return None


def _cap_boundary_loops(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, dict[str, int]]:
    components = _boundary_components(mesh)
    if not components:
        return mesh, {"boundary_components": 0, "caps_added": 0, "faces_added": 0}

    vertices = mesh.vertices.tolist()
    faces = mesh.faces.tolist()
    caps_added = 0
    faces_added = 0
    for component in components:
        loop = _order_boundary_loop(component, mesh)
        if loop is None or len(loop) < 3:
            continue
        center = mesh.vertices[loop].mean(axis=0)
        center_index = len(vertices)
        vertices.append(center.tolist())
        for index, vertex_index in enumerate(loop):
            next_index = loop[(index + 1) % len(loop)]
            faces.append([int(vertex_index), int(next_index), center_index])
            faces_added += 1
        caps_added += 1

    if caps_added == 0:
        return mesh, {
            "boundary_components": len(components),
            "caps_added": 0,
            "faces_added": 0,
        }

    capped = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
    trimesh.repair.fix_normals(capped, multibody=False)
    capped.remove_unreferenced_vertices()
    capped.merge_vertices()
    return _rebuild_mesh(capped), {
        "boundary_components": len(components),
        "caps_added": caps_added,
        "faces_added": faces_added,
    }


def clean_mesh(mesh: trimesh.Trimesh, *, use_pymeshfix: bool = True) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    cleaned = mesh.copy()
    before = _stats(cleaned)

    trimesh.repair.fix_normals(cleaned, multibody=False)
    trimesh.repair.fix_inversion(cleaned)
    faces_before_fill = int(len(cleaned.faces))
    trimesh.repair.fill_holes(cleaned)
    cleaned.remove_unreferenced_vertices()
    cleaned.merge_vertices()
    cleaned = _rebuild_mesh(cleaned)

    cap_report = {"boundary_components": 0, "caps_added": 0, "faces_added": 0}
    if not cleaned.is_watertight:
        cleaned, cap_report = _cap_boundary_loops(cleaned)

    pymeshfix_status = "not-needed"
    if use_pymeshfix and not cleaned.is_watertight:
        try:
            import pymeshfix  # type: ignore[import-not-found]

            meshfix = pymeshfix.MeshFix(cleaned.vertices, cleaned.faces)
            meshfix.repair()
            repaired = meshfix.mesh
            faces = repaired.faces.reshape(-1, 4)[:, 1:]
            cleaned = trimesh.Trimesh(vertices=np.asarray(repaired.points), faces=faces, process=True)
            cleaned = _rebuild_mesh(cleaned)
            pymeshfix_status = "applied"
        except ImportError:
            pymeshfix_status = "unavailable"
            LOGGER.info("pymeshfix is not installed; keeping trimesh-cleaned split input")
        except Exception as exc:  # pragma: no cover - depends on optional native package behavior
            pymeshfix_status = f"failed: {exc}"
            LOGGER.warning("pymeshfix repair failed; keeping trimesh-cleaned split input: %s", exc)

    after = _stats(cleaned)
    return cleaned, {
        "before": asdict(before),
        "after": asdict(after),
        "trimesh_fill_faces_added": int(len(cleaned.faces) - faces_before_fill),
        "boundary_loop_caps": cap_report,
        "pymeshfix": pymeshfix_status,
    }


def split_mesh_y(mesh: trimesh.Trimesh, *, fraction: float = 0.5) -> tuple[trimesh.Trimesh, trimesh.Trimesh, float]:
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"split fraction must be between 0 and 1, got {fraction}")

    y_min = float(mesh.bounds[0][1])
    y_max = float(mesh.bounds[1][1])
    split_y = y_min + fraction * (y_max - y_min)
    origin = [0.0, split_y, 0.0]

    heel_half = trimesh.intersections.slice_mesh_plane(mesh, [0, -1, 0], origin, cap=True)
    toe_half = trimesh.intersections.slice_mesh_plane(mesh, [0, 1, 0], origin, cap=True)
    if heel_half is None or toe_half is None:
        raise RuntimeError("failed to split mesh along the Y axis")

    return _rebuild_mesh(heel_half), _rebuild_mesh(toe_half), split_y


def _clip_tab(split_y: float, x_pos: float, z_pos: float, params: dict[str, float]) -> trimesh.Trimesh:
    width = params["tab_width"]
    height = params["tab_height"]
    length = params["tab_length"]
    chamfer = params["chamfer"]

    profile = Polygon(
        [
            (-width / 2, 0),
            (width / 2, 0),
            (width / 2, height),
            (width / 2 - chamfer, height + chamfer),
            (-width / 2 + chamfer, height + chamfer),
            (-width / 2, height),
        ]
    )
    tab = trimesh.creation.extrude_polygon(profile, height=length + TAB_EMBED_MM)
    tab.apply_transform(trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0]))
    tab.apply_transform(trimesh.transformations.rotation_matrix(math.pi, [1, 0, 0]))
    tab.apply_translation([x_pos, split_y + length, z_pos])
    return _rebuild_mesh(tab)


def _clip_socket(split_y: float, x_pos: float, z_pos: float, params: dict[str, float]) -> trimesh.Trimesh:
    width = params["tab_width"]
    height = params["tab_height"] + params["chamfer"]
    length = params["tab_length"]
    tolerance = params["socket_tolerance"]

    profile = Polygon(
        [
            (-width / 2 - tolerance, 0),
            (width / 2 + tolerance, 0),
            (width / 2 + tolerance, height + tolerance),
            (-width / 2 - tolerance, height + tolerance),
        ]
    )
    socket = trimesh.creation.extrude_polygon(profile, height=length + tolerance)
    socket.apply_transform(trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0]))
    socket.apply_transform(trimesh.transformations.rotation_matrix(math.pi, [1, 0, 0]))
    socket.apply_translation([x_pos, split_y + length + tolerance, z_pos])
    return _rebuild_mesh(socket)


def _coerce_boolean_result(result: trimesh.Trimesh | trimesh.Scene | None, operation: str) -> trimesh.Trimesh:
    if result is None:
        raise RuntimeError(f"boolean {operation} failed")
    mesh = _rebuild_mesh(coerce_trimesh(result))
    if len(mesh.faces) == 0 or mesh.extents is None:
        raise RuntimeError(f"boolean {operation} produced an empty mesh")
    return mesh


def _split_face_placement(heel_half: trimesh.Trimesh, split_y: float) -> ClipPlacement:
    tolerance = max(0.01, float(heel_half.extents[1]) * 1e-6)
    on_plane = np.abs(heel_half.vertices[:, 1] - split_y) < tolerance
    face_vertices = heel_half.vertices[on_plane]
    used_bounds_fallback = len(face_vertices) < 3

    if used_bounds_fallback:
        x_min, _, z_min = heel_half.bounds[0]
        x_max, _, z_max = heel_half.bounds[1]
    else:
        x_min = float(face_vertices[:, 0].min())
        x_max = float(face_vertices[:, 0].max())
        z_min = float(face_vertices[:, 2].min())
        z_max = float(face_vertices[:, 2].max())

    x_center = (float(x_min) + float(x_max)) / 2
    z_center = (float(z_min) + float(z_max)) / 2
    x_offset = (float(x_max) - float(x_min)) * 0.25
    x_positions = [x_center + x_offset, x_center - x_offset]

    return ClipPlacement(
        split_face_x_min_mm=_round(x_min),
        split_face_x_max_mm=_round(x_max),
        split_face_z_min_mm=_round(z_min),
        split_face_z_max_mm=_round(z_max),
        x_positions_mm=[_round(value) for value in x_positions],
        z_position_mm=_round(z_center),
        used_bounds_fallback=used_bounds_fallback,
    )


def add_snap_clips(
    heel_half: trimesh.Trimesh,
    toe_half: trimesh.Trimesh,
    *,
    split_y: float,
    clip_params: dict[str, float] | None = None,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh, ClipPlacement]:
    params = {**DEFAULT_CLIP_PARAMS, **(clip_params or {})}
    placement = _split_face_placement(heel_half, split_y)

    clipped_heel = heel_half
    socketed_toe = toe_half
    for x_pos in placement.x_positions_mm:
        tab = _clip_tab(split_y, x_pos, placement.z_position_mm, params)
        socket = _clip_socket(split_y, x_pos, placement.z_position_mm, params)
        clipped_heel = _coerce_boolean_result(
            trimesh.boolean.union([clipped_heel, tab], engine="manifold", check_volume=False),
            "union",
        )
        socketed_toe = _coerce_boolean_result(
            trimesh.boolean.difference([socketed_toe, socket], engine="manifold", check_volume=False),
            "difference",
        )

    trimesh.repair.fix_normals(clipped_heel, multibody=False)
    trimesh.repair.fix_normals(socketed_toe, multibody=False)
    return _rebuild_mesh(clipped_heel), _rebuild_mesh(socketed_toe), placement


def run(
    mesh_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    split_fraction: float = 0.5,
    add_clips: bool = True,
    use_pymeshfix: bool = True,
) -> Phase6Artifacts:
    input_path = ensure_input_path(mesh_path)
    scan_id = _infer_scan_id(input_path)
    destination = (
        ensure_directory(output_dir)
        if output_dir is not None
        else resolve_output_dir(None, phase_name="phase6_split", scan_path=Path(f"{scan_id}.obj"))
    )

    mesh = load_mesh(input_path)
    cleaned_mesh, clean_report = clean_mesh(mesh, use_pymeshfix=use_pymeshfix)
    heel_half, toe_half, split_y = split_mesh_y(cleaned_mesh, fraction=split_fraction)
    base_heel_half = heel_half.copy()
    base_toe_half = toe_half.copy()
    placement: ClipPlacement | None = None
    clip_error: str | None = None
    clips_applied = False

    if add_clips:
        try:
            heel_half, toe_half, placement = add_snap_clips(heel_half, toe_half, split_y=split_y)
            clips_applied = True
        except Exception as exc:
            clip_error = str(exc)
            LOGGER.warning("snap-fit clip booleans failed; exporting clean split halves without clips: %s", exc)
            heel_half = base_heel_half
            toe_half = base_toe_half

    heel_path = save_binary_stl(heel_half, destination / f"shoe_tree_{scan_id}_heel_tabs.stl")
    toe_path = save_binary_stl(toe_half, destination / f"shoe_tree_{scan_id}_toe_sockets.stl")
    report_payload = {
        "phase": "phase6_split",
        "scan_id": scan_id,
        "input_mesh_path": str(input_path),
        "split_axis": "y",
        "split_fraction": _round(split_fraction),
        "split_y_mm": _round(split_y),
        "clips_requested": bool(add_clips),
        "clips_applied": clips_applied,
        "clip_error": clip_error,
        "clip_params": {**DEFAULT_CLIP_PARAMS, "tab_embed": TAB_EMBED_MM},
        "clip_placement": asdict(placement) if placement is not None else None,
        "cleaning": clean_report,
        "heel_tabs": {
            "stl_path": str(heel_path),
            "stats": asdict(_stats(heel_half)),
        },
        "toe_sockets": {
            "stl_path": str(toe_path),
            "stats": asdict(_stats(toe_half)),
        },
    }
    report_path = write_json(destination / "split_report.json", report_payload)

    LOGGER.info("wrote phase 6 split artifacts to %s", destination)
    return Phase6Artifacts(
        output_dir=destination,
        heel_tabs_stl_path=heel_path,
        toe_sockets_stl_path=toe_path,
        report_path=report_path,
        split_y_mm=split_y,
        clips_applied=clips_applied,
        heel_watertight=bool(heel_half.is_watertight),
        toe_watertight=bool(toe_half.is_watertight),
    )
