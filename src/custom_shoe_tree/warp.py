from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
import trimesh

from custom_shoe_tree.align import assert_alignment, align_to_canonical
from custom_shoe_tree.io import ensure_input_path, load_scan, resolve_output_dir, save_mesh, write_json
from custom_shoe_tree.measure import FootMeasurements, MeasurementContext, SectionSlice, measure_mesh
from custom_shoe_tree.template import load_decimated_template, measure_template_mesh
from custom_shoe_tree.viz import render_warp_review_png

LOGGER = logging.getLogger(__name__)

SECTION_COUNT = 60
SOLE_THRESHOLD_MM = 3.0
SOLE_BLEND_MM = 6.0
DEFAULT_SECTION_HALF_WIDTH_MM = 2.5
GAUSSIAN_SIGMA_SECTIONS = 1.2
MIN_SECTION_POINT_COUNT = 24


@dataclass(slots=True)
class SectionProfile:
    y_norm: np.ndarray
    y_mm: np.ndarray
    width_mm: np.ndarray
    height_mm: np.ndarray
    top_z_mm: np.ndarray


@dataclass(slots=True)
class Phase3Artifacts:
    output_dir: Path
    mesh_path: Path
    render_path: Path
    report_path: Path
    warped_measurements: FootMeasurements


def _round(value: float) -> float:
    return round(float(value), 6)


def _smoothstep(values: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
    if edge1 <= edge0:
        return np.ones_like(values, dtype=float)
    t = np.clip((values - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _section_window(mesh: trimesh.Trimesh, y_mm: float, *, minimum_points: int) -> np.ndarray:
    half_width_mm = DEFAULT_SECTION_HALF_WIDTH_MM
    max_half_width_mm = max(float(mesh.extents[1]) * 0.04, half_width_mm * 4.0)
    while half_width_mm <= max_half_width_mm:
        mask = np.abs(mesh.vertices[:, 1] - y_mm) <= half_width_mm
        slab = mesh.vertices[mask]
        if len(slab) >= minimum_points:
            return slab
        half_width_mm *= 1.5
    raise RuntimeError(
        f"failed to sample a section near y={y_mm:.3f} mm with at least {minimum_points} vertices"
    )


def _sample_profile(mesh: trimesh.Trimesh, *, section_count: int = SECTION_COUNT) -> SectionProfile:
    length_mm = float(mesh.extents[1])
    y_norm = np.linspace(0.0, 1.0, section_count)
    y_mm = y_norm * length_mm
    widths = np.zeros(section_count, dtype=float)
    heights = np.zeros(section_count, dtype=float)
    top_z = np.zeros(section_count, dtype=float)

    for index, y_value in enumerate(y_mm):
        slab = _section_window(mesh, float(y_value), minimum_points=MIN_SECTION_POINT_COUNT)
        widths[index] = float(slab[:, 0].max() - slab[:, 0].min())
        heights[index] = float(slab[:, 2].max() - slab[:, 2].min())
        top_z[index] = float(slab[:, 2].max())

    return SectionProfile(
        y_norm=y_norm,
        y_mm=y_mm,
        width_mm=widths,
        height_mm=heights,
        top_z_mm=top_z,
    )


def _interp(values: np.ndarray, x: np.ndarray, x_query: float | np.ndarray) -> np.ndarray:
    return np.interp(x_query, x, values)


def _toe_tip_width_target(
    template_profile: SectionProfile,
    scan_context: MeasurementContext,
) -> float:
    toe_width = _section_world_bounds(scan_context.sections["toe_width"])["width_mm"]
    toe_ratio = toe_width / max(
        float(_interp(template_profile.width_mm, template_profile.y_norm, 0.90)),
        1e-6,
    )
    tip_factor = {
        "square": 0.92,
        "standard": 0.78,
        "angled": 0.62,
    }[scan_context.measurements.toe_box_type]
    template_tip_width = float(template_profile.width_mm[-1])
    return max(template_tip_width * toe_ratio * tip_factor, 4.0)


def _width_control_points(
    template_profile: SectionProfile,
    scan_context: MeasurementContext,
) -> tuple[np.ndarray, np.ndarray]:
    heel_width = scan_context.measurements.heel_width_mm
    ball_width = _section_world_bounds(scan_context.sections["ball"])["width_mm"]
    toe_height_width = _section_world_bounds(scan_context.sections["toe_height"])["width_mm"]
    toe_width = _section_world_bounds(scan_context.sections["toe_width"])["width_mm"]
    heel_ratio = scan_context.measurements.heel_width_mm / max(
        float(_interp(template_profile.width_mm, template_profile.y_norm, 0.10)),
        1e-6,
    )
    ball_ratio = ball_width / max(
        float(_interp(template_profile.width_mm, template_profile.y_norm, 0.55)),
        1e-6,
    )
    toe_ratio = toe_width / max(
        float(_interp(template_profile.width_mm, template_profile.y_norm, 0.90)),
        1e-6,
    )
    mid_forefoot_target = float(_interp(template_profile.width_mm, template_profile.y_norm, 0.72)) * (
        0.65 * ball_ratio + 0.35 * toe_ratio
    )
    control_y = np.array([0.00, 0.10, 0.55, 0.72, 0.85, 0.90, 1.00], dtype=float)
    control_widths = np.array(
        [
            float(template_profile.width_mm[0]) * heel_ratio,
            heel_width,
            ball_width,
            mid_forefoot_target,
            toe_height_width,
            toe_width,
            _toe_tip_width_target(template_profile, scan_context),
        ],
        dtype=float,
    )
    return control_y, control_widths


def _height_control_points(
    template_profile: SectionProfile,
    scan_context: MeasurementContext,
) -> tuple[np.ndarray, np.ndarray]:
    ball_top = _section_world_bounds(scan_context.sections["ball"])["top_z_mm"]
    toe_height_top = _section_world_bounds(scan_context.sections["toe_height"])["top_z_mm"]
    toe_width_top = _section_world_bounds(scan_context.sections["toe_width"])["top_z_mm"]
    ball_ratio = ball_top / max(
        float(_interp(template_profile.top_z_mm, template_profile.y_norm, 0.55)),
        1e-6,
    )
    arch_ratio = scan_context.measurements.arch_height_mm / max(
        float(_interp(template_profile.top_z_mm, template_profile.y_norm, 0.40)),
        1e-6,
    )
    midfoot_target = float(_interp(template_profile.top_z_mm, template_profile.y_norm, 0.70)) * (
        0.5 * arch_ratio
        + 0.5 * ball_ratio
    )
    control_y = np.array([0.00, 0.10, 0.40, 0.55, 0.70, 0.85, 0.90, 1.00], dtype=float)
    control_heights = np.array(
        [
            float(template_profile.top_z_mm[0]),
            float(_interp(template_profile.top_z_mm, template_profile.y_norm, 0.10)),
            scan_context.measurements.arch_height_mm,
            ball_top,
            midfoot_target,
            toe_height_top,
            toe_width_top,
            float(template_profile.top_z_mm[-1]),
        ],
        dtype=float,
    )
    return control_y, control_heights


def _section_world_bounds(section: SectionSlice) -> dict[str, float]:
    loops = [loop for loop in section.discrete_loops if len(loop) >= 2]
    if not loops:
        raise RuntimeError(f"section {section.name} does not contain any world-space loops")
    points = np.vstack(loops)
    return {
        "width_mm": float(points[:, 0].max() - points[:, 0].min()),
        "top_z_mm": float(points[:, 2].max()),
        "height_mm": float(points[:, 2].max() - points[:, 2].min()),
    }


def _build_target_curve(
    y_norm: np.ndarray,
    control_y: np.ndarray,
    control_values: np.ndarray,
) -> np.ndarray:
    curve = np.interp(y_norm, control_y, control_values)
    smoothed = gaussian_filter1d(curve, sigma=GAUSSIAN_SIGMA_SECTIONS, mode="nearest")
    smoothed[0] = control_values[0]
    smoothed[-1] = control_values[-1]
    return smoothed


def trim_collar_fins(
    mesh: trimesh.Trimesh,
    *,
    collar_y_pct: float = 0.22,
    aspect_ratio_threshold: float = 8.0,
) -> trimesh.Trimesh:
    """Remove degenerate spiked faces from the ankle-collar region.

    The warp's asymmetric X/Z scaling near the collar rim folds thin
    triangles into high-aspect-ratio fins. We inspect faces that touch
    the heel collar region, drop the worst aspect-ratio outliers, and
    discard any tiny disconnected collar fragments that remain. The open
    collar boundary that remains is intentional.
    """
    length_mm = float(mesh.extents[1])
    collar_y_limit = length_mm * collar_y_pct

    verts = mesh.vertices
    faces = mesh.faces

    # Allow faces that straddle the collar band so the cleanup reaches the rim.
    in_collar = np.any(verts[faces, 1] < collar_y_limit, axis=1)
    collar_indices = np.where(in_collar)[0]
    if collar_indices.size == 0:
        LOGGER.debug("trim_collar_fins: no faces in collar zone (y < %.1f mm)", collar_y_limit)
        return mesh

    f = faces[collar_indices]
    v0, v1, v2 = verts[f[:, 0]], verts[f[:, 1]], verts[f[:, 2]]
    a = v1 - v0
    b = v2 - v1
    c = v0 - v2
    len_a = np.linalg.norm(a, axis=1)
    len_b = np.linalg.norm(b, axis=1)
    len_c = np.linalg.norm(c, axis=1)
    cross = np.cross(a, -c)
    area = np.linalg.norm(cross, axis=1) / 2.0
    s = (len_a + len_b + len_c) / 2.0
    circumradius = (len_a * len_b * len_c) / (4.0 * np.maximum(area, 1e-10))
    inradius = area / np.maximum(s, 1e-10)
    aspect = circumradius / np.maximum(inradius, 1e-10)

    fin_global = collar_indices[aspect > aspect_ratio_threshold]
    working = mesh
    if fin_global.size == 0:
        LOGGER.debug("trim_collar_fins: no degenerate faces found in collar region")
    else:
        keep = np.ones(len(faces), dtype=bool)
        keep[fin_global] = False
        LOGGER.info(
            "trim_collar_fins: removed %d degenerate face(s) from collar (aspect > %.1f)",
            fin_global.size,
            aspect_ratio_threshold,
        )
        working = trimesh.Trimesh(vertices=verts.copy(), faces=faces[keep], process=False)
        working.remove_unreferenced_vertices()
        working.merge_vertices()
        working._cache.clear()
        working.metadata.update(mesh.metadata)

    return _drop_small_collar_components(working, collar_y_pct=collar_y_pct)


def _drop_small_collar_components(
    mesh: trimesh.Trimesh,
    *,
    collar_y_pct: float,
    min_face_count: int = 50,
    min_face_ratio: float = 0.005,
) -> trimesh.Trimesh:
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh

    collar_y_limit = float(mesh.extents[1]) * collar_y_pct
    face_threshold = max(min_face_count, int(len(mesh.faces) * min_face_ratio))
    kept_parts: list[trimesh.Trimesh] = []
    dropped_parts = 0
    for part in parts:
        if len(part.faces) < face_threshold and float(part.vertices[:, 1].max()) < collar_y_limit:
            dropped_parts += 1
            continue
        kept_parts.append(part)

    if dropped_parts == 0 or not kept_parts:
        return mesh

    combined = kept_parts[0].copy() if len(kept_parts) == 1 else trimesh.util.concatenate(kept_parts)
    clean = trimesh.Trimesh(vertices=combined.vertices.copy(), faces=combined.faces.copy(), process=False)
    clean.remove_unreferenced_vertices()
    clean.merge_vertices()
    clean._cache.clear()
    clean.metadata.update(mesh.metadata)
    LOGGER.info(
        "trim_collar_fins: dropped %d small collar component(s) below %d faces",
        dropped_parts,
        face_threshold,
    )
    return clean


def _normalize_translation(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    normalized = mesh.copy()
    vertices = normalized.vertices.copy()
    min_v = vertices.min(axis=0)
    max_v = vertices.max(axis=0)
    vertices[:, 0] -= (min_v[0] + max_v[0]) / 2.0
    vertices[:, 1] -= min_v[1]
    vertices[:, 2] -= min_v[2]
    normalized.vertices = vertices
    normalized._cache.clear()
    normalized.metadata.update(mesh.metadata)
    return normalized


def _apply_allowance(
    mesh: trimesh.Trimesh,
    *,
    allowance_mm: float,
    sole_threshold_mm: float,
    blend_mm: float,
) -> trimesh.Trimesh:
    if allowance_mm <= 0.0:
        return mesh.copy()

    offset_mesh = mesh.copy()
    normals = np.asarray(offset_mesh.vertex_normals).copy()
    normals[:, 1] = 0.0
    magnitudes = np.linalg.norm(normals, axis=1)
    safe_normals = normals.copy()
    valid = magnitudes > 1e-6
    safe_normals[valid] /= magnitudes[valid][:, None]
    blend = _smoothstep(offset_mesh.vertices[:, 2], sole_threshold_mm, sole_threshold_mm + blend_mm)
    offset = safe_normals * (allowance_mm * blend[:, None])
    offset_mesh.vertices = offset_mesh.vertices + offset
    offset_mesh.vertices[:, 2] = np.maximum(offset_mesh.vertices[:, 2], 0.0)
    offset_mesh._cache.clear()
    offset_mesh.metadata.update(mesh.metadata)
    return offset_mesh


def _warp_vertices(
    template_mesh: trimesh.Trimesh,
    *,
    width_scale_curve: np.ndarray,
    height_scale_curve: np.ndarray,
    y_norm_samples: np.ndarray,
    length_scale: float,
    sole_threshold_mm: float,
    blend_mm: float,
) -> trimesh.Trimesh:
    warped = template_mesh.copy()
    vertices = warped.vertices.copy()
    y_norm = vertices[:, 1] / max(float(template_mesh.extents[1]), 1e-6)
    x_scale = np.interp(y_norm, y_norm_samples, width_scale_curve)
    z_scale = np.interp(y_norm, y_norm_samples, height_scale_curve)
    upper_blend = _smoothstep(vertices[:, 2], sole_threshold_mm, sole_threshold_mm + blend_mm)

    vertices[:, 1] *= length_scale
    vertices[:, 0] *= 1.0 + upper_blend * (x_scale - 1.0)
    vertices[:, 2] *= 1.0 + upper_blend * (z_scale - 1.0)

    warped.vertices = vertices
    warped._cache.clear()
    warped.metadata.update(template_mesh.metadata)
    return warped


def _profile_payload(
    template_profile: SectionProfile,
    target_width_mm: np.ndarray,
    target_height_mm: np.ndarray,
    achieved_profile: SectionProfile,
) -> list[dict[str, float]]:
    samples: list[dict[str, float]] = []
    for index, y_norm in enumerate(template_profile.y_norm):
        template_width = float(template_profile.width_mm[index])
        template_height = float(template_profile.top_z_mm[index])
        target_width = float(target_width_mm[index])
        target_height = float(target_height_mm[index])
        achieved_width = float(achieved_profile.width_mm[index])
        achieved_height = float(achieved_profile.top_z_mm[index])
        samples.append(
            {
                "y_pct": _round(y_norm * 100.0),
                "template_width_mm": _round(template_width),
                "template_top_z_mm": _round(template_height),
                "target_width_mm": _round(target_width),
                "target_top_z_mm": _round(target_height),
                "achieved_width_mm": _round(achieved_width),
                "achieved_top_z_mm": _round(achieved_height),
                "width_scale": _round(target_width / max(template_width, 1e-6)),
                "height_scale": _round(target_height / max(template_height, 1e-6)),
            }
        )
    return samples


def _scale_summary(width_scale_curve: np.ndarray, height_scale_curve: np.ndarray) -> dict[str, float]:
    return {
        "width_scale_min": _round(np.min(width_scale_curve)),
        "width_scale_max": _round(np.max(width_scale_curve)),
        "height_scale_min": _round(np.min(height_scale_curve)),
        "height_scale_max": _round(np.max(height_scale_curve)),
    }


def _review_summary(
    *,
    template_measurements: FootMeasurements,
    width_control_values: np.ndarray,
    height_control_values: np.ndarray,
    achieved_profile: SectionProfile,
) -> dict[str, dict[str, float]]:
    return {
        "targets": {
            "heel_width_mm": _round(width_control_values[1]),
            "ball_width_mm": _round(width_control_values[2]),
            "toe_width_mm": _round(width_control_values[5]),
            "arch_top_z_mm": _round(height_control_values[2]),
            "ball_top_z_mm": _round(height_control_values[3]),
            "toe_top_z_mm": _round(height_control_values[6]),
        },
        "template": {
            "heel_width_mm": _round(template_measurements.heel_width_mm),
            "ball_width_mm": _round(template_measurements.ball_width_mm),
            "toe_width_mm": _round(template_measurements.toe_box_width_mm),
        },
        "achieved": {
            "heel_width_mm": _round(float(_interp(achieved_profile.width_mm, achieved_profile.y_norm, 0.10))),
            "ball_width_mm": _round(float(_interp(achieved_profile.width_mm, achieved_profile.y_norm, 0.55))),
            "toe_width_mm": _round(float(_interp(achieved_profile.width_mm, achieved_profile.y_norm, 0.90))),
            "arch_top_z_mm": _round(float(_interp(achieved_profile.top_z_mm, achieved_profile.y_norm, 0.40))),
            "ball_top_z_mm": _round(float(_interp(achieved_profile.top_z_mm, achieved_profile.y_norm, 0.55))),
            "toe_top_z_mm": _round(float(_interp(achieved_profile.top_z_mm, achieved_profile.y_norm, 0.90))),
        },
    }


def _warp_report(
    *,
    scan_path: Path,
    template_measurements: FootMeasurements,
    target_measurements: FootMeasurements,
    warped_measurements: FootMeasurements,
    template_profile: SectionProfile,
    target_width_mm: np.ndarray,
    target_height_mm: np.ndarray,
    achieved_profile: SectionProfile,
    width_control_y: np.ndarray,
    width_control_values: np.ndarray,
    height_control_y: np.ndarray,
    height_control_values: np.ndarray,
    width_scale_curve: np.ndarray,
    height_scale_curve: np.ndarray,
    allowance_mm: float,
    sole_threshold_mm: float,
    length_scale: float,
) -> dict[str, Any]:
    return {
        "phase": "phase3",
        "scan_id": target_measurements.scan_id,
        "source_scan_path": str(scan_path),
        "allowance_mm": _round(allowance_mm),
        "sole_threshold_mm": _round(sole_threshold_mm),
        "section_count": int(len(template_profile.y_norm)),
        "length_scale": _round(length_scale),
        "template_measurements": template_measurements.to_dict(),
        "target_measurements": target_measurements.to_dict(),
        "warped_measurements": warped_measurements.to_dict(),
        "width_control_points": [
            {"y_pct": _round(y_value * 100.0), "target_width_mm": _round(width_value)}
            for y_value, width_value in zip(width_control_y, width_control_values)
        ],
        "height_control_points": [
            {"y_pct": _round(y_value * 100.0), "target_height_mm": _round(height_value)}
            for y_value, height_value in zip(height_control_y, height_control_values)
        ],
        "review_summary": _review_summary(
            template_measurements=template_measurements,
            width_control_values=width_control_values,
            height_control_values=height_control_values,
            achieved_profile=achieved_profile,
        ),
        "scale_summary": _scale_summary(width_scale_curve, height_scale_curve),
        "section_samples": _profile_payload(
            template_profile,
            target_width_mm,
            target_height_mm,
            achieved_profile,
        ),
    }


def build_warp(
    template_mesh: trimesh.Trimesh,
    template_measurements: FootMeasurements,
    scan_context: MeasurementContext,
    *,
    allowance_mm: float,
    sole_threshold_mm: float = SOLE_THRESHOLD_MM,
) -> tuple[trimesh.Trimesh, FootMeasurements, dict[str, Any]]:
    template_profile = _sample_profile(template_mesh)

    width_control_y, width_control_values = _width_control_points(template_profile, scan_context)
    height_control_y, height_control_values = _height_control_points(template_profile, scan_context)

    target_width_mm = _build_target_curve(
        template_profile.y_norm,
        width_control_y,
        width_control_values,
    )
    target_height_mm = _build_target_curve(
        template_profile.y_norm,
        height_control_y,
        height_control_values,
    )

    width_scale_curve = target_width_mm / np.maximum(template_profile.width_mm, 1e-6)
    height_scale_curve = target_height_mm / np.maximum(template_profile.top_z_mm, 1e-6)
    length_scale = scan_context.measurements.length_mm / max(template_measurements.length_mm, 1e-6)

    warped = _warp_vertices(
        template_mesh,
        width_scale_curve=width_scale_curve,
        height_scale_curve=height_scale_curve,
        y_norm_samples=template_profile.y_norm,
        length_scale=length_scale,
        sole_threshold_mm=sole_threshold_mm,
        blend_mm=SOLE_BLEND_MM,
    )
    warped = _apply_allowance(
        warped,
        allowance_mm=allowance_mm,
        sole_threshold_mm=sole_threshold_mm,
        blend_mm=SOLE_BLEND_MM,
    )
    warped = _normalize_translation(warped)
    warped = trim_collar_fins(warped)
    assert_alignment(warped)

    warped_measurements = measure_template_mesh(warped)
    achieved_profile = _sample_profile(warped, section_count=SECTION_COUNT)
    report = _warp_report(
        scan_path=Path(scan_context.measurements.scan_id),
        template_measurements=template_measurements,
        target_measurements=scan_context.measurements,
        warped_measurements=warped_measurements,
        template_profile=template_profile,
        target_width_mm=target_width_mm,
        target_height_mm=target_height_mm,
        achieved_profile=achieved_profile,
        width_control_y=width_control_y,
        width_control_values=width_control_values,
        height_control_y=height_control_y,
        height_control_values=height_control_values,
        width_scale_curve=width_scale_curve,
        height_scale_curve=height_scale_curve,
        allowance_mm=allowance_mm,
        sole_threshold_mm=sole_threshold_mm,
        length_scale=length_scale,
    )
    return warped, warped_measurements, report


def run(
    scan_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    allowance_mm: float = 3.0,
) -> Phase3Artifacts:
    input_path = ensure_input_path(scan_path)
    destination = resolve_output_dir(output_dir, phase_name="phase3", scan_path=input_path)

    scan_mesh = load_scan(input_path)
    aligned_scan = align_to_canonical(scan_mesh)
    aligned_scan.metadata.update(scan_mesh.metadata)
    scan_context = measure_mesh(aligned_scan)

    template_cache_dir = resolve_output_dir(None, phase_name="phase2")
    template_mesh = load_decimated_template(template_cache_dir)
    template_measurements = measure_template_mesh(template_mesh)

    warped_mesh, warped_measurements, report = build_warp(
        template_mesh,
        template_measurements,
        scan_context,
        allowance_mm=allowance_mm,
        sole_threshold_mm=SOLE_THRESHOLD_MM,
    )
    report["source_scan_path"] = str(input_path)

    mesh_path = save_mesh(warped_mesh, destination / "shoe_tree_warp.obj")
    render_path = render_warp_review_png(aligned_scan, warped_mesh, report, destination / "render.png")
    report_path = write_json(destination / "warp_report.json", report)

    LOGGER.info(
        "phase 3 warp completed for %s: length %.2f mm, ball width %.2f mm, toe width %.2f mm",
        scan_context.measurements.scan_id,
        warped_measurements.length_mm,
        warped_measurements.ball_width_mm,
        warped_measurements.toe_box_width_mm,
    )
    LOGGER.info("wrote phase 3 artifacts to %s", destination)

    return Phase3Artifacts(
        output_dir=destination,
        mesh_path=mesh_path,
        render_path=render_path,
        report_path=report_path,
        warped_measurements=warped_measurements,
    )
