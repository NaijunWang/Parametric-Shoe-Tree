from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from math import pi
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from last_generator.align import align_to_canonical
from last_generator.io import (
    MeshAudit,
    ensure_input_path,
    load_scan,
    mesh_audit_from_metadata,
    resolve_output_dir,
    save_mesh,
    save_scene_as_obj,
    scan_id_from_path,
    write_json,
)
from last_generator.viz import annotate_scan, render_annotated_png

LOGGER = logging.getLogger(__name__)

SECTION_Y_PCTS = {
    "heel": 10.0,
    "ball": 55.0,
    "toe_height": 85.0,
    "toe_width": 90.0,
}
ADAPTIVE_BALL_RANGE = np.arange(60.0, 81.0, 1.0)


class MeasurementError(RuntimeError):
    """Raised when the measurement extraction cannot find a required feature."""


@dataclass(slots=True)
class SectionSlice:
    name: str
    y_pct: float
    y_mm: float
    width_mm: float
    height_mm: float
    perimeter_mm: float
    area_mm2: float
    bounds_xz_mm: list[float]
    discrete_loops: list[np.ndarray]


@dataclass(slots=True)
class FootMeasurements:
    scan_id: str
    length_mm: float
    max_width_mm: float
    max_height_mm: float
    heel_width_mm: float
    ball_width_mm: float
    ball_perimeter_mm: float
    ball_height_mm: float
    arch_height_mm: float
    arch_length_mm: float
    toe_box_width_mm: float
    toe_box_height_mm: float
    adaptive_ball_y_pct: float
    arch_ratio: float
    arch_type: str
    toe_section_circularity: float
    toe_section_aspect_ratio: float
    toe_angle_deg: float
    toe_box_type: str
    control_sections_mm: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary_row(self) -> dict[str, Any]:
        return {
            "scan_id": self.scan_id,
            "length_mm": round(self.length_mm, 3),
            "max_width_mm": round(self.max_width_mm, 3),
            "max_height_mm": round(self.max_height_mm, 3),
            "heel_width_mm": round(self.heel_width_mm, 3),
            "ball_width_mm": round(self.ball_width_mm, 3),
            "ball_perimeter_mm": round(self.ball_perimeter_mm, 3),
            "ball_height_mm": round(self.ball_height_mm, 3),
            "arch_height_mm": round(self.arch_height_mm, 3),
            "toe_box_width_mm": round(self.toe_box_width_mm, 3),
            "toe_box_height_mm": round(self.toe_box_height_mm, 3),
            "adaptive_ball_y_pct": round(self.adaptive_ball_y_pct, 3),
            "arch_ratio": round(self.arch_ratio, 6),
            "arch_type": self.arch_type,
            "toe_section_circularity": round(self.toe_section_circularity, 6),
            "toe_section_aspect_ratio": round(self.toe_section_aspect_ratio, 6),
            "toe_angle_deg": round(self.toe_angle_deg, 6),
            "toe_box_type": self.toe_box_type,
        }


@dataclass(slots=True)
class MeasurementContext:
    scan_id: str
    measurements: FootMeasurements
    sections: dict[str, SectionSlice]
    landmarks: dict[str, np.ndarray]
    adaptive_ball_profile: list[dict[str, float]]


@dataclass(slots=True)
class Phase1Artifacts:
    output_dir: Path
    mesh_aligned_path: Path
    annotated_obj_path: Path
    annotated_png_path: Path
    measurements_path: Path
    audit_path: Path
    measurements: FootMeasurements
    audit: MeshAudit


def _round(value: float) -> float:
    return round(float(value), 6)


def _section_at_y(mesh: trimesh.Trimesh, *, y_pct: float, y_mm: float, name: str) -> SectionSlice:
    section = mesh.section(plane_origin=[0.0, y_mm, 0.0], plane_normal=[0.0, 1.0, 0.0])
    if section is None:
        raise MeasurementError(f"{name}: no cross-section found at y={y_mm:.3f} mm ({y_pct:.1f}%)")
    planar, _ = section.to_2D()
    polygons = [polygon for polygon in planar.polygons_full if not polygon.is_empty]
    if not polygons:
        raise MeasurementError(f"{name}: section at y={y_mm:.3f} mm has no closed polygon")
    polygon = max(polygons, key=lambda item: item.area)
    min_x, min_z, max_x, max_z = polygon.bounds
    loops = [np.asarray(loop) for loop in section.discrete if len(loop) >= 2]
    if not loops:
        raise MeasurementError(f"{name}: section at y={y_mm:.3f} mm has no discrete loops")
    return SectionSlice(
        name=name,
        y_pct=y_pct,
        y_mm=_round(y_mm),
        width_mm=_round(max_x - min_x),
        height_mm=_round(max_z - min_z),
        perimeter_mm=_round(section.length),
        area_mm2=_round(polygon.area),
        bounds_xz_mm=[_round(min_x), _round(min_z), _round(max_x), _round(max_z)],
        discrete_loops=loops,
    )


def _section_at_pct(mesh: trimesh.Trimesh, *, y_pct: float, name: str) -> SectionSlice:
    length_mm = float(mesh.extents[1])
    return _section_at_y(mesh, y_pct=y_pct, y_mm=length_mm * (y_pct / 100.0), name=name)


def _slab_peak_vertex(mesh: trimesh.Trimesh, start_pct: float, end_pct: float) -> np.ndarray:
    length_mm = float(mesh.extents[1])
    start_y = length_mm * (start_pct / 100.0)
    end_y = length_mm * (end_pct / 100.0)
    mask = (mesh.vertices[:, 1] >= start_y) & (mesh.vertices[:, 1] <= end_y)
    slab_vertices = mesh.vertices[mask]
    if len(slab_vertices) == 0:
        raise MeasurementError(
            f"no vertices found in slab y=[{start_pct:.1f}%, {end_pct:.1f}%] for arch measurement"
        )
    return slab_vertices[np.argmax(slab_vertices[:, 2])]


def _adaptive_ball_profile(mesh: trimesh.Trimesh) -> list[dict[str, float]]:
    samples: list[dict[str, float]] = []
    for y_pct in ADAPTIVE_BALL_RANGE:
        section = _section_at_pct(mesh, y_pct=float(y_pct), name=f"adaptive_ball_{int(y_pct)}")
        samples.append(
            {
                "y_pct": _round(y_pct),
                "y_mm": section.y_mm,
                "perimeter_mm": section.perimeter_mm,
            }
        )
    return samples


def _arch_type(arch_ratio: float) -> str:
    if arch_ratio < 0.45:
        return "flat"
    if arch_ratio < 0.55:
        return "small"
    if arch_ratio < 0.65:
        return "medium"
    return "tall"


def _toe_angle_deg(mesh: trimesh.Trimesh) -> float:
    length_mm = float(mesh.extents[1])
    front_vertices = mesh.vertices[mesh.vertices[:, 1] >= length_mm * 0.85]
    if len(front_vertices) == 0:
        raise MeasurementError("no vertices found in front slab for toe angle classification")
    split_x = float(np.median(front_vertices[:, 0]))
    left = front_vertices[front_vertices[:, 0] <= split_x]
    right = front_vertices[front_vertices[:, 0] > split_x]
    if len(left) == 0 or len(right) == 0:
        raise MeasurementError(
            "toe angle classification could not split the forefoot into two sides"
        )
    tip_delta = abs(float(left[:, 1].max() - right[:, 1].max()))
    front_width = max(float(front_vertices[:, 0].max() - front_vertices[:, 0].min()), 1e-6)
    return _round(np.degrees(np.arctan2(tip_delta, front_width)))


def _toe_box_type(circularity: float, aspect_ratio: float, toe_angle_deg: float) -> str:
    if toe_angle_deg >= 6.0:
        return "angled"
    if aspect_ratio >= 1.45 and circularity <= 0.78:
        return "square"
    return "standard"


def _landmarks(mesh: trimesh.Trimesh, sections: dict[str, SectionSlice], arch_peak: np.ndarray) -> dict[str, np.ndarray]:
    max_width = float(mesh.extents[0])
    heel = np.array([0.0, 0.0, 0.0])
    toe_tip = mesh.vertices[np.argmax(mesh.vertices[:, 1])]
    ball_section = sections["ball"]
    heel_section = sections["heel"]
    toe_width_section = sections["toe_width"]
    toe_height_section = sections["toe_height"]
    return {
        "heel_origin": heel,
        "toe_tip": toe_tip,
        "arch_peak": arch_peak,
        "ball_center": np.array([0.0, ball_section.y_mm, ball_section.bounds_xz_mm[3]]),
        "heel_center": np.array([0.0, heel_section.y_mm, heel_section.bounds_xz_mm[3]]),
        "toe_width_center": np.array([0.0, toe_width_section.y_mm, toe_width_section.bounds_xz_mm[3]]),
        "toe_height_center": np.array([max_width * 0.05, toe_height_section.y_mm, toe_height_section.bounds_xz_mm[3]]),
    }


def measure_mesh(mesh: trimesh.Trimesh) -> MeasurementContext:
    scan_id = mesh.metadata.get("scan_id", "scan")
    length_mm = float(mesh.extents[1])
    max_width_mm = float(mesh.extents[0])
    max_height_mm = float(mesh.extents[2])

    sections = {
        "heel": _section_at_pct(mesh, y_pct=SECTION_Y_PCTS["heel"], name="heel"),
        "ball": _section_at_pct(mesh, y_pct=SECTION_Y_PCTS["ball"], name="ball"),
        "toe_height": _section_at_pct(mesh, y_pct=SECTION_Y_PCTS["toe_height"], name="toe_height"),
        "toe_width": _section_at_pct(mesh, y_pct=SECTION_Y_PCTS["toe_width"], name="toe_width"),
    }

    arch_peak = _slab_peak_vertex(mesh, 38.0, 42.0)
    arch_height_mm = float(arch_peak[2])
    arch_ratio = arch_height_mm / max_height_mm

    adaptive_ball_profile = _adaptive_ball_profile(mesh)
    adaptive_perimeters = np.array([item["perimeter_mm"] for item in adaptive_ball_profile], dtype=float)
    adaptive_index = int(np.argmax(adaptive_perimeters))
    adaptive_ball_y_pct = float(adaptive_ball_profile[adaptive_index]["y_pct"])
    if adaptive_index in {0, len(adaptive_ball_profile) - 1}:
        LOGGER.warning(
            "adaptive ball perimeter peak for %s is at the search boundary (%s%%)",
            scan_id,
            adaptive_ball_y_pct,
        )

    toe_width_section = sections["toe_width"]
    toe_width_height_ratio = toe_width_section.width_mm / max(toe_width_section.height_mm, 1e-6)
    toe_circularity = (4.0 * pi * toe_width_section.area_mm2) / max(toe_width_section.perimeter_mm ** 2, 1e-6)
    toe_angle_deg = _toe_angle_deg(mesh)

    measurements = FootMeasurements(
        scan_id=scan_id,
        length_mm=_round(length_mm),
        max_width_mm=_round(max_width_mm),
        max_height_mm=_round(max_height_mm),
        heel_width_mm=sections["heel"].width_mm,
        ball_width_mm=sections["ball"].width_mm,
        ball_perimeter_mm=sections["ball"].perimeter_mm,
        ball_height_mm=sections["ball"].height_mm,
        arch_height_mm=_round(arch_height_mm),
        arch_length_mm=_round(arch_height_mm),
        toe_box_width_mm=sections["toe_width"].width_mm,
        toe_box_height_mm=sections["toe_height"].height_mm,
        adaptive_ball_y_pct=_round(adaptive_ball_y_pct),
        arch_ratio=_round(arch_ratio),
        arch_type=_arch_type(arch_ratio),
        toe_section_circularity=_round(toe_circularity),
        toe_section_aspect_ratio=_round(toe_width_height_ratio),
        toe_angle_deg=toe_angle_deg,
        toe_box_type=_toe_box_type(toe_circularity, toe_width_height_ratio, toe_angle_deg),
        control_sections_mm={
            "heel": sections["heel"].y_mm,
            "ball": sections["ball"].y_mm,
            "toe_height": sections["toe_height"].y_mm,
            "toe_width": sections["toe_width"].y_mm,
            "arch": _round(length_mm * 0.40),
        },
    )
    return MeasurementContext(
        scan_id=scan_id,
        measurements=measurements,
        sections=sections,
        landmarks=_landmarks(mesh, sections, arch_peak),
        adaptive_ball_profile=adaptive_ball_profile,
    )


def log_measurement_summary(measurements: FootMeasurements) -> None:
    LOGGER.info("measurement summary for %s", measurements.scan_id)
    summary_pairs: list[tuple[str, str]] = [
        ("length_mm", f"{measurements.length_mm:.2f}"),
        ("max_width_mm", f"{measurements.max_width_mm:.2f}"),
        ("max_height_mm", f"{measurements.max_height_mm:.2f}"),
        ("heel_width_mm", f"{measurements.heel_width_mm:.2f}"),
        ("ball_width_mm", f"{measurements.ball_width_mm:.2f}"),
        ("ball_perimeter_mm", f"{measurements.ball_perimeter_mm:.2f}"),
        ("ball_height_mm", f"{measurements.ball_height_mm:.2f}"),
        ("arch_height_mm", f"{measurements.arch_height_mm:.2f}"),
        ("toe_box_width_mm", f"{measurements.toe_box_width_mm:.2f}"),
        ("toe_box_height_mm", f"{measurements.toe_box_height_mm:.2f}"),
        ("adaptive_ball_y_pct", f"{measurements.adaptive_ball_y_pct:.2f}"),
        ("arch_type", measurements.arch_type),
        ("toe_box_type", measurements.toe_box_type),
    ]
    for label, value in summary_pairs:
        LOGGER.info("  %-22s %s", label, value)


def _measurements_payload(
    source_path: Path,
    context: MeasurementContext,
) -> dict[str, Any]:
    return {
        **context.measurements.to_dict(),
        "source_path": str(source_path),
        "sections": {
            name: {
                "y_pct": slice_.y_pct,
                "y_mm": slice_.y_mm,
                "width_mm": slice_.width_mm,
                "height_mm": slice_.height_mm,
                "perimeter_mm": slice_.perimeter_mm,
                "area_mm2": slice_.area_mm2,
                "bounds_xz_mm": slice_.bounds_xz_mm,
            }
            for name, slice_ in context.sections.items()
        },
        "landmarks_mm": {
            name: [_round(value) for value in point]
            for name, point in context.landmarks.items()
        },
        "adaptive_ball_profile": context.adaptive_ball_profile,
    }


def run(scan_path: str | Path, output_dir: str | Path | None = None) -> Phase1Artifacts:
    input_path = ensure_input_path(scan_path)
    destination = resolve_output_dir(output_dir, phase_name="phase1", scan_path=input_path)
    mesh = load_scan(input_path)
    audit = mesh_audit_from_metadata(mesh)
    aligned_mesh = align_to_canonical(mesh)
    aligned_mesh.metadata.update(mesh.metadata)
    context = measure_mesh(aligned_mesh)
    scene = annotate_scan(aligned_mesh, context)

    mesh_aligned_path = save_mesh(aligned_mesh, destination / "mesh_aligned.obj")
    annotated_obj_path = save_scene_as_obj(scene, destination / "annotated.obj")
    annotated_png_path = render_annotated_png(aligned_mesh, context, destination / "annotated.png")
    measurements_path = write_json(
        destination / "measurements.json",
        _measurements_payload(input_path, context),
    )
    audit_path = write_json(destination / "audit.json", audit.to_dict())

    log_measurement_summary(context.measurements)
    LOGGER.info("wrote phase 1 artifacts to %s", destination)

    return Phase1Artifacts(
        output_dir=destination,
        mesh_aligned_path=mesh_aligned_path,
        annotated_obj_path=annotated_obj_path,
        annotated_png_path=annotated_png_path,
        measurements_path=measurements_path,
        audit_path=audit_path,
        measurements=context.measurements,
        audit=audit,
    )
