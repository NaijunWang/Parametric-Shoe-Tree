from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from custom_shoe_tree.align import align_to_canonical, assert_alignment
from custom_shoe_tree.io import (
    ensure_input_path,
    load_reference_mesh,
    load_scan,
    resolve_output_dir,
    save_mesh,
    write_json,
)
from custom_shoe_tree.measure import FootMeasurements, MeasurementContext, measure_mesh
from custom_shoe_tree.template import measure_template_mesh, run as run_template
from custom_shoe_tree.viz import render_overlay_review_png
from custom_shoe_tree.warp import SOLE_BLEND_MM, SOLE_THRESHOLD_MM, run as run_warp, trim_collar_fins

LOGGER = logging.getLogger(__name__)

PASS_CONFIGS = (
    {"pass_index": 1, "stiffness": 100.0, "landmark_weight": 3.0, "normal_weight": 0.1, "max_iter": 40},
    {"pass_index": 2, "stiffness": 20.0, "landmark_weight": 5.0, "normal_weight": 0.2, "max_iter": 40},
    {"pass_index": 3, "stiffness": 5.0, "landmark_weight": 5.0, "normal_weight": 0.3, "max_iter": 30},
)
TOLERANCES_MM = {
    "length_mm": 1.0,
    "ball_width_mm": 2.0,
    "ball_perimeter_mm": 3.0,
    "toe_box_width_mm": 2.0,
    "heel_width_mm": 2.0,
}
LANDMARK_ORDER = (
    "heel_back",
    "ball_inside",
    "ball_outside",
    "toe_tip",
    "arch_peak",
    "instep_top",
)
ADAPTIVE_BALL_HALF_WIDTH_MM = 3.0
SOURCE_BALL_Y_PCT = 55.0
LANDMARK_REMAP_MAX_DISTANCE_MM = 12.0
EARLY_ABORT_DIMENSION_SCALE = 1.35
EARLY_ABORT_DRIFT_MULTIPLIER = 5.0


@dataclass(slots=True)
class Phase4Artifacts:
    output_dir: Path
    mesh_path: Path
    pass_render_paths: list[Path]
    comparison_path: Path
    report_path: Path
    fallback: bool
    refined_measurements: FootMeasurements


def _round(value: float) -> float:
    return round(float(value), 6)


def _smoothstep(values: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
    if edge1 <= edge0:
        return np.ones_like(values, dtype=float)
    t = np.clip((values - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _normalize_translation_refine(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
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


def _phase3_dir(scan_path: Path) -> Path:
    return resolve_output_dir(None, phase_name="phase3", scan_path=scan_path)


def _phase2_landmarks_path() -> Path:
    return resolve_output_dir(None, phase_name="phase2") / "template_landmarks.json"


def _phase3_mesh_path(scan_path: Path) -> Path:
    return _phase3_dir(scan_path) / "shoe_tree_warp.obj"


def _phase3_report_path(scan_path: Path) -> Path:
    return _phase3_dir(scan_path) / "warp_report.json"


def _load_phase3_mesh(scan_path: Path) -> trimesh.Trimesh:
    phase3_mesh_path = _phase3_mesh_path(scan_path)
    return load_reference_mesh(phase3_mesh_path, metadata_id=f"{scan_path.stem}_phase3")


def _load_phase3_report(scan_path: Path) -> dict[str, Any]:
    return json.loads(_phase3_report_path(scan_path).read_text(encoding="utf-8"))


def _inflate_scan_target(
    aligned_scan: trimesh.Trimesh,
    *,
    allowance_mm: float,
    sole_threshold_mm: float,
) -> trimesh.PointCloud:
    vertices = aligned_scan.vertices.copy()
    normals = np.asarray(aligned_scan.vertex_normals).copy()
    blend = _smoothstep(vertices[:, 2], sole_threshold_mm, sole_threshold_mm + SOLE_BLEND_MM)
    vertices += normals * (allowance_mm * blend[:, None])
    vertices[:, 2] = np.maximum(vertices[:, 2], 0.0)
    return trimesh.PointCloud(vertices)


def _slab_vertices(
    mesh: trimesh.Trimesh,
    *,
    center_y_mm: float,
    half_width_mm: float,
    label: str,
    minimum_vertices: int = 1,
) -> np.ndarray:
    search_half_width = half_width_mm
    max_half_width = max(half_width_mm * 3.0, 9.0)
    while search_half_width <= max_half_width:
        mask = np.abs(mesh.vertices[:, 1] - center_y_mm) <= search_half_width
        slab = mesh.vertices[mask]
        if len(slab) >= minimum_vertices:
            return slab
        search_half_width += half_width_mm
    raise RuntimeError(
        f"{label}: no vertices found within +/-{max_half_width:.1f} mm of y={center_y_mm:.3f} mm"
    )


def _max_z_point_in_range(
    mesh: trimesh.Trimesh,
    *,
    start_pct: float,
    end_pct: float,
    label: str,
) -> np.ndarray:
    length_mm = float(mesh.extents[1])
    start_y = length_mm * (start_pct / 100.0)
    end_y = length_mm * (end_pct / 100.0)
    mask = (mesh.vertices[:, 1] >= start_y) & (mesh.vertices[:, 1] <= end_y)
    slab = mesh.vertices[mask]
    if len(slab) == 0:
        raise RuntimeError(f"{label}: no vertices found in y={start_pct:.1f}-{end_pct:.1f}% range")
    return slab[int(np.argmax(slab[:, 2]))]


def _scan_landmark_positions(
    aligned_scan: trimesh.Trimesh,
    scan_context: MeasurementContext,
) -> tuple[list[np.ndarray], dict[str, list[float]]]:
    adaptive_ball_y_mm = scan_context.measurements.length_mm * (
        scan_context.measurements.adaptive_ball_y_pct / 100.0
    )
    ball_slab = _slab_vertices(
        aligned_scan,
        center_y_mm=adaptive_ball_y_mm,
        half_width_mm=ADAPTIVE_BALL_HALF_WIDTH_MM,
        label="adaptive_ball_landmarks",
        minimum_vertices=4,
    )
    ball_inside = ball_slab[int(np.argmin(ball_slab[:, 0]))]
    ball_outside = ball_slab[int(np.argmax(ball_slab[:, 0]))]
    instep_top = _max_z_point_in_range(aligned_scan, start_pct=50.0, end_pct=62.0, label="instep_top")

    named_positions = {
        # Phase 2's heel_back landmark sits on the posterior heel collar, not the plantar
        # heel origin. Targeting the heel-center top keeps the constraint anatomically aligned.
        "heel_back": np.asarray(scan_context.landmarks["heel_center"], dtype=float),
        "ball_inside": ball_inside,
        "ball_outside": ball_outside,
        "toe_tip": np.asarray(scan_context.landmarks["toe_tip"], dtype=float),
        "arch_peak": np.asarray(scan_context.landmarks["arch_peak"], dtype=float),
        "instep_top": instep_top,
    }
    ordered = [named_positions[name] for name in LANDMARK_ORDER]
    payload = {name: [_round(value) for value in point] for name, point in named_positions.items()}
    return ordered, payload


def _source_landmark_indices_from_heuristics(source_mesh: trimesh.Trimesh) -> list[int]:
    length_mm = float(source_mesh.extents[1])
    vertices = source_mesh.vertices
    heel_mask = vertices[:, 1] <= length_mm * 0.05
    heel_vertices = vertices[heel_mask]
    heel_indices = np.where(heel_mask)[0]
    if len(heel_vertices) == 0:
        raise RuntimeError("source heel_back: no vertices found in heel slab")
    heel_origin_scores = np.linalg.norm(heel_vertices, axis=1)
    heel_back = int(heel_indices[int(np.argmin(heel_origin_scores))])

    ball_y_mm = length_mm * (SOURCE_BALL_Y_PCT / 100.0)
    search_half_width = ADAPTIVE_BALL_HALF_WIDTH_MM
    max_half_width = max(ADAPTIVE_BALL_HALF_WIDTH_MM * 3.0, 9.0)
    while search_half_width <= max_half_width:
        ball_mask = np.abs(vertices[:, 1] - ball_y_mm) <= search_half_width
        if np.count_nonzero(ball_mask) >= 4:
            break
        search_half_width += ADAPTIVE_BALL_HALF_WIDTH_MM
    if not np.any(ball_mask):
        raise RuntimeError("source ball landmarks: no vertices found in source slab")
    ball_indices = np.where(ball_mask)[0]
    ball_vertices = vertices[ball_mask]
    ball_inside = int(ball_indices[int(np.argmin(ball_vertices[:, 0]))])
    ball_outside = int(ball_indices[int(np.argmax(ball_vertices[:, 0]))])

    toe_tip = int(np.argmax(vertices[:, 1]))
    arch_mask = (vertices[:, 1] >= length_mm * 0.38) & (vertices[:, 1] <= length_mm * 0.42)
    instep_mask = (vertices[:, 1] >= length_mm * 0.50) & (vertices[:, 1] <= length_mm * 0.62)
    if not np.any(arch_mask):
        raise RuntimeError("source arch_peak: no vertices found in source arch range")
    if not np.any(instep_mask):
        raise RuntimeError("source instep_top: no vertices found in source instep range")
    arch_indices = np.where(arch_mask)[0]
    instep_indices = np.where(instep_mask)[0]
    arch_peak = int(arch_indices[int(np.argmax(vertices[arch_mask][:, 2]))])
    instep_top = int(instep_indices[int(np.argmax(vertices[instep_mask][:, 2]))])

    return [heel_back, ball_inside, ball_outside, toe_tip, arch_peak, instep_top]


def _ensure_phase4_prerequisites(scan_path: Path, allowance_mm: float) -> None:
    phase2_dir = resolve_output_dir(None, phase_name="phase2")
    if not _phase2_landmarks_path().exists():
        LOGGER.info("phase 4 requires phase 2 landmarks; generating template artifacts")
        run_template(phase2_dir)

    phase3_mesh_path = _phase3_mesh_path(scan_path)
    phase3_report_path = _phase3_report_path(scan_path)
    if not phase3_mesh_path.exists() or not phase3_report_path.exists():
        LOGGER.info("phase 4 requires phase 3 artifacts; generating warp output for %s", scan_path.name)
        run_warp(scan_path, _phase3_dir(scan_path), allowance_mm=allowance_mm)


def _remap_phase2_landmarks_to_source_mesh(
    source_mesh: trimesh.Trimesh,
    payload: dict[str, Any],
) -> tuple[list[int], dict[str, float]]:
    indices: list[int] = []
    distances: dict[str, float] = {}
    used_indices: set[int] = set()
    for name in LANDMARK_ORDER:
        point = np.asarray(payload["landmarks"][name]["point_mm"], dtype=float)
        deltas = np.linalg.norm(source_mesh.vertices - point, axis=1)
        if used_indices:
            deltas[list(used_indices)] = np.inf
        index = int(np.argmin(deltas))
        distance = float(deltas[index])
        if not np.isfinite(distance):
            raise RuntimeError(f"phase 2 landmark remap exhausted unique candidates for {name}")
        if distance > LANDMARK_REMAP_MAX_DISTANCE_MM:
            raise RuntimeError(
                f"phase 2 landmark remap for {name} exceeded {LANDMARK_REMAP_MAX_DISTANCE_MM:.1f} mm "
                f"(nearest={distance:.3f} mm)"
            )
        used_indices.add(index)
        indices.append(index)
        distances[name] = _round(distance)
    return indices, distances


def _load_source_landmark_indices(
    source_mesh: trimesh.Trimesh,
) -> tuple[list[int], str, dict[str, Any], dict[str, float]]:
    payload = json.loads(_phase2_landmarks_path().read_text(encoding="utf-8"))
    landmark_indices = [int(payload["landmarks"][name]["index"]) for name in LANDMARK_ORDER]
    if (
        int(payload["mesh_vertex_count"]) == len(source_mesh.vertices)
        and min(landmark_indices) >= 0
        and max(landmark_indices) < len(source_mesh.vertices)
    ):
        return landmark_indices, "template_indices", payload, {name: 0.0 for name in LANDMARK_ORDER}

    try:
        remapped_indices, remap_distances = _remap_phase2_landmarks_to_source_mesh(source_mesh, payload)
        LOGGER.info(
            "phase 4 source landmark selection remapped phase 2 landmarks onto the trimmed phase 3 mesh "
            "because source vertex count %s does not match phase 2 landmark mesh %s",
            len(source_mesh.vertices),
            payload["mesh_vertex_count"],
        )
        return remapped_indices, "nearest_phase2_remap", payload, remap_distances
    except Exception as exc:
        LOGGER.info(
            "phase 4 source landmark remap failed (%s); falling back to heuristics because source vertex count %s "
            "does not match phase 2 landmark mesh %s",
            exc,
            len(source_mesh.vertices),
            payload["mesh_vertex_count"],
        )
    return (
        _source_landmark_indices_from_heuristics(source_mesh),
        "heuristic_due_to_topology_change",
        payload,
        {},
    )


def _measurement_drift(
    current: FootMeasurements,
    baseline: FootMeasurements,
) -> dict[str, dict[str, float | bool]]:
    drift: dict[str, dict[str, float | bool]] = {}
    for field_name, tolerance in TOLERANCES_MM.items():
        delta = float(getattr(current, field_name) - getattr(baseline, field_name))
        drift[field_name] = {
            "baseline_mm": _round(getattr(baseline, field_name)),
            "current_mm": _round(getattr(current, field_name)),
            "delta_mm": _round(delta),
            "tolerance_mm": _round(tolerance),
            "within_tolerance": abs(delta) <= tolerance,
        }
    return drift


def _all_tolerances_pass(drift: dict[str, dict[str, float | bool]]) -> bool:
    return all(bool(item["within_tolerance"]) for item in drift.values())


def _divergence_reason(
    mesh: trimesh.Trimesh,
    *,
    baseline_measurements: FootMeasurements,
    drift: dict[str, dict[str, float | bool]],
) -> str | None:
    extents = np.asarray(mesh.extents, dtype=float)
    if int(np.argmax(extents)) != 1:
        return f"longest axis is not +Y (extents={extents.tolist()})"
    if extents[1] > baseline_measurements.length_mm * EARLY_ABORT_DIMENSION_SCALE:
        return (
            f"length extent {extents[1]:.3f} mm exceeded early-abort scale "
            f"{EARLY_ABORT_DIMENSION_SCALE:.2f}x baseline"
        )
    if extents[0] > baseline_measurements.max_width_mm * EARLY_ABORT_DIMENSION_SCALE:
        return (
            f"width extent {extents[0]:.3f} mm exceeded early-abort scale "
            f"{EARLY_ABORT_DIMENSION_SCALE:.2f}x baseline"
        )
    if extents[2] > baseline_measurements.max_height_mm * EARLY_ABORT_DIMENSION_SCALE:
        return (
            f"height extent {extents[2]:.3f} mm exceeded early-abort scale "
            f"{EARLY_ABORT_DIMENSION_SCALE:.2f}x baseline"
        )
    for field_name, item in drift.items():
        tolerance = float(item["tolerance_mm"])
        threshold = max(10.0, tolerance * EARLY_ABORT_DRIFT_MULTIPLIER)
        if abs(float(item["delta_mm"])) > threshold:
            return (
                f"{field_name} drift {float(item['delta_mm']):.3f} mm exceeded early-abort guard "
                f"({threshold:.3f} mm)"
            )
    return None


def _pass_footer_lines(
    *,
    scan_id: str,
    config: dict[str, float | int],
    measurements: FootMeasurements,
    baseline: FootMeasurements,
    drift: dict[str, dict[str, float | bool]],
    distance_threshold_mm: float,
    landmark_mode: str,
) -> list[str]:
    return [
        (
            f"Scan: {scan_id} | pass {config['pass_index']} | ws={config['stiffness']:.1f} "
            f"wl={config['landmark_weight']:.1f} wn={config['normal_weight']:.1f} "
            f"max_iter={int(config['max_iter'])}"
        ),
        (
            "Phase 3 length / current length: "
            f"{baseline.length_mm:.1f} / {measurements.length_mm:.1f} mm"
        ),
        (
            "Phase 3 ball / heel / toe width: "
            f"{baseline.ball_width_mm:.1f} / {baseline.heel_width_mm:.1f} / {baseline.toe_box_width_mm:.1f} mm"
        ),
        (
            "Current ball / heel / toe width: "
            f"{measurements.ball_width_mm:.1f} / {measurements.heel_width_mm:.1f} / {measurements.toe_box_width_mm:.1f} mm"
        ),
        (
            "Drift (L / BallW / BallP / ToeW / HeelW): "
            f"{drift['length_mm']['delta_mm']:.2f} / "
            f"{drift['ball_width_mm']['delta_mm']:.2f} / "
            f"{drift['ball_perimeter_mm']['delta_mm']:.2f} / "
            f"{drift['toe_box_width_mm']['delta_mm']:.2f} / "
            f"{drift['heel_width_mm']['delta_mm']:.2f} mm"
        ),
        (
            f"Landmark mode: {landmark_mode} | target distance threshold={distance_threshold_mm:.1f} mm"
        ),
        "Grey = aligned foot scan, orange = Phase 4 refined shoe tree.",
    ]


def _comparison_footer_lines(
    *,
    scan_id: str,
    phase3_measurements: FootMeasurements,
    refined_measurements: FootMeasurements,
    fallback: bool,
    landmark_mode: str,
) -> list[str]:
    source_label = "phase3-fallback" if fallback else "phase4"
    return [
        f"Scan: {scan_id} | selected source={source_label} | landmark mode={landmark_mode}",
        (
            "Phase 3 length / ball / heel / toe width: "
            f"{phase3_measurements.length_mm:.1f} / {phase3_measurements.ball_width_mm:.1f} / "
            f"{phase3_measurements.heel_width_mm:.1f} / {phase3_measurements.toe_box_width_mm:.1f} mm"
        ),
        (
            "Selected length / ball / heel / toe width: "
            f"{refined_measurements.length_mm:.1f} / {refined_measurements.ball_width_mm:.1f} / "
            f"{refined_measurements.heel_width_mm:.1f} / {refined_measurements.toe_box_width_mm:.1f} mm"
        ),
        (
            "Phase 3 arch / ball perimeter: "
            f"{phase3_measurements.arch_height_mm:.1f} / {phase3_measurements.ball_perimeter_mm:.1f} mm"
        ),
        (
            "Selected arch / ball perimeter: "
            f"{refined_measurements.arch_height_mm:.1f} / {refined_measurements.ball_perimeter_mm:.1f} mm"
        ),
        "Grey = Phase 3 warp, orange = Phase 4 selected mesh.",
    ]


def _render_pass_png(
    aligned_scan: trimesh.Trimesh,
    working_mesh: trimesh.Trimesh,
    *,
    footer_lines: list[str],
    path: Path,
) -> Path:
    return render_overlay_review_png(
        aligned_scan,
        working_mesh,
        footer_lines=footer_lines,
        label_a="aligned foot scan",
        label_b="phase 4 refined shoe tree",
        path=path,
    )


def _render_comparison_png(
    phase3_mesh: trimesh.Trimesh,
    selected_mesh: trimesh.Trimesh,
    *,
    footer_lines: list[str],
    path: Path,
) -> Path:
    return render_overlay_review_png(
        phase3_mesh,
        selected_mesh,
        footer_lines=footer_lines,
        label_a="phase 3 warp",
        label_b="phase 4 selected mesh",
        path=path,
    )


def run(
    scan_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    allowance_mm: float = 3.0,
) -> Phase4Artifacts:
    input_path = ensure_input_path(scan_path)
    destination = resolve_output_dir(output_dir, phase_name="phase4", scan_path=input_path)
    _ensure_phase4_prerequisites(input_path, allowance_mm)

    aligned_scan = align_to_canonical(load_scan(input_path))
    scan_context = measure_mesh(aligned_scan)
    target_point_cloud = _inflate_scan_target(
        aligned_scan,
        allowance_mm=allowance_mm,
        sole_threshold_mm=SOLE_THRESHOLD_MM,
    )

    phase3_mesh = _load_phase3_mesh(input_path)
    phase3_report = _load_phase3_report(input_path)
    phase3_measurements = FootMeasurements(**phase3_report["warped_measurements"])
    source_landmark_indices, landmark_mode, phase2_landmarks, source_landmark_distances = _load_source_landmark_indices(
        phase3_mesh
    )
    target_landmark_positions, target_landmark_payload = _scan_landmark_positions(aligned_scan, scan_context)
    target_landmark_positions_array = np.asarray(target_landmark_positions, dtype=float)

    working_mesh = phase3_mesh.copy()
    sole_mask = working_mesh.vertices[:, 2] < SOLE_THRESHOLD_MM
    sole_positions = working_mesh.vertices[sole_mask].copy()

    pass_render_paths: list[Path] = []
    pass_records: list[dict[str, Any]] = []
    refinement_error: str | None = None
    fallback = False

    try:
        for config in PASS_CONFIGS:
            distance_threshold_mm = 20.0
            normalized_threshold = distance_threshold_mm / max(float(working_mesh.scale), 1e-6)
            refined_vertices = trimesh.registration.nricp_amberg(
                working_mesh.copy(),
                trimesh.PointCloud(target_point_cloud.vertices.copy()),
                source_landmarks=np.asarray(source_landmark_indices, dtype=int),
                target_positions=target_landmark_positions_array,
                steps=[
                    [
                        config["stiffness"],
                        config["landmark_weight"],
                        config["normal_weight"],
                        config["max_iter"],
                    ]
                ],
                distance_threshold=normalized_threshold,
                return_records=False,
                use_faces=False,
                neighbors_count=8,
            )
            refined_vertices[sole_mask] = sole_positions
            working_mesh = trimesh.Trimesh(
                vertices=np.asarray(refined_vertices),
                faces=working_mesh.faces.copy(),
                process=False,
            )
            working_mesh.metadata.update(phase3_mesh.metadata)
            working_mesh = _normalize_translation_refine(working_mesh)
            sole_positions = working_mesh.vertices[sole_mask].copy()
            pass_measurements = measure_template_mesh(working_mesh)
            drift = _measurement_drift(pass_measurements, phase3_measurements)
            pass_record = {
                "pass_index": int(config["pass_index"]),
                "parameters": {
                    "stiffness": _round(config["stiffness"]),
                    "landmark_weight": _round(config["landmark_weight"]),
                    "normal_weight": _round(config["normal_weight"]),
                    "max_iter": int(config["max_iter"]),
                    "distance_threshold_mm": _round(distance_threshold_mm),
                },
                "measurements": pass_measurements.to_dict(),
                "drift_vs_phase3": drift,
                "within_tolerance": _all_tolerances_pass(drift),
            }
            divergence_reason = _divergence_reason(
                working_mesh,
                baseline_measurements=phase3_measurements,
                drift=drift,
            )
            pass_record["diverged"] = divergence_reason is not None
            if divergence_reason is not None:
                pass_record["divergence_reason"] = divergence_reason
            pass_records.append(pass_record)
            render_path = _render_pass_png(
                aligned_scan,
                working_mesh,
                footer_lines=_pass_footer_lines(
                    scan_id=scan_context.measurements.scan_id,
                    config=config,
                    measurements=pass_measurements,
                    baseline=phase3_measurements,
                    drift=drift,
                    distance_threshold_mm=distance_threshold_mm,
                    landmark_mode=landmark_mode,
                ),
                path=destination / f"render_pass{config['pass_index']}.png",
            )
            pass_render_paths.append(render_path)
            if divergence_reason is not None:
                raise RuntimeError(divergence_reason)
    except Exception as exc:
        refinement_error = str(exc)
        fallback = True
        LOGGER.warning(
            "NRICP refinement failed for %s (%s) — falling back to Phase 3 warp",
            scan_context.measurements.scan_id,
            exc,
        )
        working_mesh = phase3_mesh.copy()

    if len(pass_render_paths) < len(PASS_CONFIGS):
        for config in PASS_CONFIGS[len(pass_render_paths) :]:
            fallback_drift = _measurement_drift(phase3_measurements, phase3_measurements)
            pass_records.append(
                {
                    "pass_index": int(config["pass_index"]),
                    "parameters": {
                        "stiffness": _round(config["stiffness"]),
                        "landmark_weight": _round(config["landmark_weight"]),
                        "normal_weight": _round(config["normal_weight"]),
                        "max_iter": int(config["max_iter"]),
                        "distance_threshold_mm": _round(20.0),
                    },
                    "measurements": phase3_measurements.to_dict(),
                    "drift_vs_phase3": fallback_drift,
                    "within_tolerance": True,
                    "skipped": True,
                    "skip_reason": refinement_error or "refinement aborted before this pass",
                }
            )
            render_path = _render_pass_png(
                aligned_scan,
                phase3_mesh,
                footer_lines=[
                    f"Scan: {scan_context.measurements.scan_id} | pass {config['pass_index']} skipped",
                    f"Reason: {refinement_error or 'refinement aborted before this pass'}",
                    "Grey = aligned foot scan, orange = Phase 3 fallback mesh.",
                ],
                path=destination / f"render_pass{config['pass_index']}.png",
            )
            pass_render_paths.append(render_path)

    try:
        refined_mesh = trim_collar_fins(working_mesh)
        refined_mesh = _normalize_translation_refine(refined_mesh)
        assert_alignment(refined_mesh)
        refined_measurements = measure_template_mesh(refined_mesh)
        final_drift = _measurement_drift(refined_measurements, phase3_measurements)
        tolerance_pass = _all_tolerances_pass(final_drift)
        if not fallback and not tolerance_pass:
            fallback = True
            LOGGER.warning(
                "NRICP drift exceeded tolerance for %s — falling back to Phase 3 warp",
                scan_context.measurements.scan_id,
            )
            refined_mesh = phase3_mesh.copy()
            refined_measurements = phase3_measurements
    except Exception as exc:
        if refinement_error is None:
            refinement_error = str(exc)
        fallback = True
        LOGGER.warning(
            "NRICP post-processing failed for %s (%s) — falling back to Phase 3 warp",
            scan_context.measurements.scan_id,
            exc,
        )
        refined_mesh = phase3_mesh.copy()
        refined_measurements = phase3_measurements
        final_drift = _measurement_drift(refined_measurements, phase3_measurements)

    selected_source = "phase3-fallback" if fallback else "phase4"
    if fallback:
        final_drift = _measurement_drift(refined_measurements, phase3_measurements)

    comparison_path = _render_comparison_png(
        phase3_mesh,
        refined_mesh,
        footer_lines=_comparison_footer_lines(
            scan_id=scan_context.measurements.scan_id,
            phase3_measurements=phase3_measurements,
            refined_measurements=refined_measurements,
            fallback=fallback,
            landmark_mode=landmark_mode,
        ),
        path=destination / "comparison.png",
    )
    mesh_path = save_mesh(refined_mesh, destination / "shoe_tree_refined.obj")

    report_payload = {
        "phase": "phase4",
        "scan_id": scan_context.measurements.scan_id,
        "source_scan_path": str(input_path),
        "allowance_mm": _round(allowance_mm),
        "sole_threshold_mm": _round(SOLE_THRESHOLD_MM),
        "selected_source": selected_source,
        "fallback": fallback,
        "refinement_error": refinement_error,
        "source_landmark_mode": landmark_mode,
        "source_landmark_distances_mm": source_landmark_distances,
        "source_landmark_indices": {
            name: int(index) for name, index in zip(LANDMARK_ORDER, source_landmark_indices)
        },
        "target_landmark_positions_mm": target_landmark_payload,
        "phase2_landmarks": phase2_landmarks["landmarks"],
        "phase3_measurements": phase3_measurements.to_dict(),
        "selected_measurements": refined_measurements.to_dict(),
        "final_drift_vs_phase3": final_drift,
        "final_within_tolerance": _all_tolerances_pass(final_drift),
        "pass_records": pass_records,
        "artifacts": {
            "shoe_tree_refined_obj": str(mesh_path),
            "comparison_png": str(comparison_path),
            "pass_renders": [str(path) for path in pass_render_paths],
        },
    }
    report_path = write_json(destination / "refine_report.json", report_payload)

    LOGGER.info(
        "phase 4 refine completed for %s: source=%s fallback=%s",
        scan_context.measurements.scan_id,
        selected_source,
        fallback,
    )
    LOGGER.info("wrote phase 4 artifacts to %s", destination)

    return Phase4Artifacts(
        output_dir=destination,
        mesh_path=mesh_path,
        pass_render_paths=pass_render_paths,
        comparison_path=comparison_path,
        report_path=report_path,
        fallback=fallback,
        refined_measurements=refined_measurements,
    )
