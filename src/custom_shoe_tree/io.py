from __future__ import annotations

import csv
import json
import logging
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import trimesh

LOGGER = logging.getLogger(__name__)

OPEN3D_HEADER_MARKER = "Created by Open3D"
UNIT_SCALE_TO_MM = {"m": 1000.0, "cm": 10.0, "mm": 1.0}
SMALL_HOLE_EDGE_THRESHOLD = 5
TEMPLATE_FILENAME = "cc_base_last.obj"


class MeshDefectWarning(UserWarning):
    """Warning raised when the input mesh needs repair or review."""


@dataclass(slots=True)
class MeshAudit:
    source_path: str
    units: Literal["m", "cm", "mm"]
    unit_scale_to_mm: float
    unit_sniff_reason: str
    watertight: bool
    boundary_edge_count: int
    nonmanifold_edge_count: int
    component_count: int
    vertex_count: int
    face_count: int
    bbox_min_mm: list[float]
    bbox_max_mm: list[float]
    extents_mm: list[float]
    source_header: list[str] = field(default_factory=list)
    repair_applied: bool = False
    repair_faces_added: int = 0
    repair_reason: str | None = None
    pre_repair_boundary_edge_count: int | None = None
    pre_repair_watertight: bool | None = None
    boundary_vertex_y_range_mm: list[float] | None = None
    boundary_vertex_z_range_mm: list[float] | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MeshAudit":
        return cls(**payload)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    return project_root()


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_input_path(path: str | Path) -> Path:
    input_path = Path(path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"input path does not exist: {input_path}")
    return input_path


def scan_id_from_path(path: str | Path) -> str:
    return Path(path).stem


def resolve_output_dir(
    output_dir: str | Path | None,
    *,
    phase_name: str,
    scan_path: str | Path | None = None,
) -> Path:
    if output_dir is not None:
        return ensure_directory(output_dir)
    base_dir = project_root() / "out" / phase_name
    if scan_path is None:
        return ensure_directory(base_dir)
    return ensure_directory(base_dir / scan_id_from_path(scan_path))


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


def write_csv(path: str | Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def save_mesh(mesh: trimesh.Trimesh, path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(destination)
    return destination


def save_binary_stl(mesh: trimesh.Trimesh, path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    exported = mesh.export(file_type="stl")
    if not isinstance(exported, bytes):
        raise TypeError(f"expected STL export to return bytes, got {type(exported)!r}")
    destination.write_bytes(exported)
    return destination


def save_scene_as_obj(scene: trimesh.Scene, path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    exported = scene.export(file_type="obj")
    if not isinstance(exported, str):
        raise TypeError(f"expected OBJ export to return str, got {type(exported)!r}")
    obj_text = "\n".join(
        f"g {line[2:]}" if line.startswith("o ") else line
        for line in exported.splitlines()
    )
    destination.write_text(obj_text + "\n", encoding="utf-8")
    return destination


def read_obj_header(path: str | Path, max_lines: int = 20) -> list[str]:
    header_lines: list[str] = []
    with ensure_input_path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for _ in range(max_lines):
            line = handle.readline()
            if not line:
                break
            stripped = line.rstrip("\n")
            if stripped.startswith("#"):
                header_lines.append(stripped)
                continue
            if stripped.startswith("v ") or stripped.startswith("o ") or stripped.startswith("g "):
                break
    return header_lines


def coerce_trimesh(loaded: trimesh.Trimesh | trimesh.Scene) -> trimesh.Trimesh:
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    if isinstance(loaded, trimesh.Scene):
        meshes = [geometry for geometry in loaded.geometry.values() if isinstance(geometry, trimesh.Trimesh)]
        if not meshes:
            raise ValueError("loaded scene does not contain any mesh geometry")
        return trimesh.util.concatenate(meshes)
    raise TypeError(f"unsupported trimesh payload: {type(loaded)!r}")


def sniff_units(
    mesh: trimesh.Trimesh,
    header_lines: list[str] | None = None,
) -> tuple[Literal["m", "cm", "mm"], str]:
    header_lines = header_lines or []
    if any(OPEN3D_HEADER_MARKER in line for line in header_lines):
        return "m", "Open3D header"
    max_extent = float(np.max(mesh.extents))
    if max_extent < 1.0:
        return "m", "max extent < 1"
    if max_extent < 50.0:
        return "cm", "max extent < 50"
    return "mm", "max extent >= 50"


def _edge_use_counts(mesh: trimesh.Trimesh) -> np.ndarray:
    return np.bincount(mesh.edges_unique_inverse, minlength=len(mesh.edges_unique))


def _boundary_vertex_ranges(
    mesh: trimesh.Trimesh,
    edge_use_counts: np.ndarray,
) -> tuple[list[float] | None, list[float] | None]:
    boundary_mask = edge_use_counts == 1
    if not np.any(boundary_mask):
        return None, None
    boundary_edges = mesh.edges_unique[boundary_mask]
    boundary_vertices = mesh.vertices[np.unique(boundary_edges.reshape(-1))]
    y_range = [
        round(float(boundary_vertices[:, 1].min()), 6),
        round(float(boundary_vertices[:, 1].max()), 6),
    ]
    z_range = [
        round(float(boundary_vertices[:, 2].min()), 6),
        round(float(boundary_vertices[:, 2].max()), 6),
    ]
    return y_range, z_range


def audit_mesh(
    mesh: trimesh.Trimesh,
    *,
    source_path: str | Path = "",
    units: Literal["m", "cm", "mm"] = "mm",
    unit_scale_to_mm: float = 1.0,
    unit_sniff_reason: str = "unspecified",
    source_header: list[str] | None = None,
) -> MeshAudit:
    edge_use_counts = _edge_use_counts(mesh)
    boundary_count = int(np.sum(edge_use_counts == 1))
    nonmanifold_count = int(np.sum(edge_use_counts > 2))
    y_range, z_range = _boundary_vertex_ranges(mesh, edge_use_counts)
    return MeshAudit(
        source_path=str(source_path),
        units=units,
        unit_scale_to_mm=unit_scale_to_mm,
        unit_sniff_reason=unit_sniff_reason,
        watertight=bool(mesh.is_watertight),
        boundary_edge_count=boundary_count,
        nonmanifold_edge_count=nonmanifold_count,
        component_count=len(mesh.split(only_watertight=False)),
        vertex_count=int(len(mesh.vertices)),
        face_count=int(len(mesh.faces)),
        bbox_min_mm=[round(float(value), 6) for value in mesh.bounds[0]],
        bbox_max_mm=[round(float(value), 6) for value in mesh.bounds[1]],
        extents_mm=[round(float(value), 6) for value in mesh.extents],
        source_header=list(source_header or []),
        boundary_vertex_y_range_mm=y_range,
        boundary_vertex_z_range_mm=z_range,
    )


def _should_repair_small_hole(mesh: trimesh.Trimesh, audit: MeshAudit) -> bool:
    if audit.boundary_edge_count == 0 or audit.boundary_edge_count >= SMALL_HOLE_EDGE_THRESHOLD:
        return False
    if audit.boundary_vertex_z_range_mm is None:
        return False
    max_z = float(mesh.bounds[1][2])
    boundary_min_z = audit.boundary_vertex_z_range_mm[0]
    return boundary_min_z >= max_z - max(5.0, 0.05 * float(mesh.extents[2]))


def _load_mesh(
    path: str | Path,
    *,
    metadata_id: str,
    repair_small_holes: bool,
    warn_on_open_mesh: bool,
) -> trimesh.Trimesh:
    input_path = ensure_input_path(path)
    header_lines = read_obj_header(input_path)
    loaded = trimesh.load(input_path, process=False)
    mesh = coerce_trimesh(loaded)
    units, reason = sniff_units(mesh, header_lines)
    scale = UNIT_SCALE_TO_MM[units]
    LOGGER.info("loading %s", input_path.name)
    LOGGER.info("sniffed units=%s using %s", units, reason)
    if scale != 1.0:
        mesh.apply_scale(scale)
        LOGGER.info("scaled mesh by %.3f to convert %s to mm", scale, units)

    audit = audit_mesh(
        mesh,
        source_path=input_path,
        units=units,
        unit_scale_to_mm=scale,
        unit_sniff_reason=reason,
        source_header=header_lines,
    )
    if audit.boundary_edge_count and warn_on_open_mesh:
        message = (
            f"{input_path.name} has {audit.boundary_edge_count} boundary edges "
            f"and {audit.nonmanifold_edge_count} non-manifold edges"
        )
        warnings.warn(message, MeshDefectWarning)
        LOGGER.warning(message)

    if repair_small_holes and _should_repair_small_hole(mesh, audit):
        LOGGER.info("repairing small top-side hole on %s", input_path.name)
        pre_repair_boundary_edges = audit.boundary_edge_count
        pre_repair_watertight = audit.watertight
        face_count_before = len(mesh.faces)
        trimesh.repair.fill_holes(mesh)
        repaired_audit = audit_mesh(
            mesh,
            source_path=input_path,
            units=units,
            unit_scale_to_mm=scale,
            unit_sniff_reason=reason,
            source_header=header_lines,
        )
        repaired_audit.repair_applied = True
        repaired_audit.repair_faces_added = int(len(mesh.faces) - face_count_before)
        repaired_audit.repair_reason = "filled small boundary loop near scan top"
        repaired_audit.pre_repair_boundary_edge_count = pre_repair_boundary_edges
        repaired_audit.pre_repair_watertight = pre_repair_watertight
        repaired_audit.notes.append("repair applied during load_scan")
        audit = repaired_audit
        LOGGER.info(
            "hole repair added %s face(s); watertight=%s",
            audit.repair_faces_added,
            audit.watertight,
        )

    mesh.metadata["source_path"] = str(input_path)
    mesh.metadata["scan_id"] = metadata_id
    mesh.metadata["mesh_audit"] = audit.to_dict()
    return mesh


def load_scan(path: str | Path) -> trimesh.Trimesh:
    input_path = ensure_input_path(path)
    return _load_mesh(
        input_path,
        metadata_id=scan_id_from_path(input_path),
        repair_small_holes=True,
        warn_on_open_mesh=True,
    )


def load_reference_mesh(path: str | Path, *, metadata_id: str) -> trimesh.Trimesh:
    return _load_mesh(
        path,
        metadata_id=metadata_id,
        repair_small_holes=False,
        warn_on_open_mesh=False,
    )


def template_source_path() -> Path:
    return repo_root() / "template" / TEMPLATE_FILENAME


def load_template_source(path: str | Path | None = None) -> trimesh.Trimesh:
    source_path = template_source_path() if path is None else ensure_input_path(path)
    return load_reference_mesh(source_path, metadata_id="base_shoe_tree")


def mesh_audit_from_metadata(mesh: trimesh.Trimesh) -> MeshAudit:
    payload = mesh.metadata.get("mesh_audit")
    if payload is None:
        raise KeyError("mesh metadata does not contain mesh_audit")
    return MeshAudit.from_dict(payload)
