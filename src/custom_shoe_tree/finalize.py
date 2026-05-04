from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import trimesh

from custom_shoe_tree.align import align_to_canonical, assert_alignment
from custom_shoe_tree.io import (
    audit_mesh,
    ensure_directory,
    ensure_input_path,
    load_reference_mesh,
    load_scan,
    repo_root,
    resolve_output_dir,
    save_binary_stl,
    save_mesh,
    write_json,
)
from custom_shoe_tree.measure import FootMeasurements
from custom_shoe_tree.refine import run as run_refine
from custom_shoe_tree.template import measure_template_mesh
from custom_shoe_tree.viz import render_overlay_review_png

LOGGER = logging.getLogger(__name__)

DECIMATION_FACE_THRESHOLD = 200_000
DECIMATION_TARGET_FACE_COUNT = 150_000


@dataclass(slots=True)
class Phase5Artifacts:
    output_dir: Path
    obj_path: Path
    stl_path: Path
    report_path: Path
    render_path: Path
    source: str
    measurements: FootMeasurements
    volume_ml: float


def _round(value: float) -> float:
    return round(float(value), 6)


def _phase3_dir(scan_path: Path) -> Path:
    return resolve_output_dir(None, phase_name="phase3", scan_path=scan_path)


def _phase4_dir(scan_path: Path) -> Path:
    return resolve_output_dir(None, phase_name="phase4", scan_path=scan_path)


def _phase3_mesh_path(scan_path: Path) -> Path:
    return _phase3_dir(scan_path) / "shoe_tree_warp.obj"


def _phase4_mesh_path(scan_path: Path) -> Path:
    return _phase4_dir(scan_path) / "shoe_tree_refined.obj"


def _phase4_report_path(scan_path: Path) -> Path:
    return _phase4_dir(scan_path) / "refine_report.json"


def _ensure_phase5_prerequisites(scan_path: Path, allowance_mm: float) -> None:
    if not _phase4_mesh_path(scan_path).exists() or not _phase4_report_path(scan_path).exists():
        LOGGER.info("phase 5 requires phase 4 artifacts; generating refine output for %s", scan_path.name)
        run_refine(scan_path, _phase4_dir(scan_path), allowance_mm=allowance_mm)


def _select_best_input(scan_path: Path) -> tuple[Path, str]:
    phase4_mesh_path = _phase4_mesh_path(scan_path)
    phase4_report_path = _phase4_report_path(scan_path)
    if phase4_mesh_path.exists() and phase4_report_path.exists():
        payload = json.loads(phase4_report_path.read_text(encoding="utf-8"))
        if not bool(payload.get("fallback", True)):
            return phase4_mesh_path, "phase4"
    phase3_mesh_path = _phase3_mesh_path(scan_path)
    if not phase3_mesh_path.exists():
        raise FileNotFoundError(f"phase 3 mesh not found for finalize fallback: {phase3_mesh_path}")
    return phase3_mesh_path, "phase3-fallback"


def _infer_scan_id(mesh_path: Path) -> str:
    if mesh_path.stem in {"shoe_tree_warp", "shoe_tree_refined"} and mesh_path.parent.name:
        return mesh_path.parent.name
    if mesh_path.stem.startswith("shoe_tree_"):
        return mesh_path.stem.removeprefix("shoe_tree_")
    if mesh_path.parent.name and mesh_path.parent.name not in {"phase3", "phase4", "phase5"}:
        return mesh_path.parent.name
    return mesh_path.stem


def _infer_source_label(mesh_path: Path) -> str:
    if mesh_path.name == "shoe_tree_warp.obj":
        return "phase3-fallback"
    if mesh_path.name == "shoe_tree_refined.obj":
        report_path = mesh_path.with_name("refine_report.json")
        if report_path.exists():
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            return "phase4" if not bool(payload.get("fallback", True)) else "phase3-fallback"
        return "phase4"
    return "direct"


def _candidate_scan_path(scan_id: str) -> Path | None:
    candidate = repo_root() / "sample-foot-scans" / f"{scan_id}.obj"
    return candidate if candidate.exists() else None


def _rebuild_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    rebuilt = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), process=False)
    rebuilt.metadata.update(mesh.metadata)
    return rebuilt


def _repair_mesh(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    repaired = mesh.copy()
    trimesh.repair.fix_normals(repaired, multibody=False)
    trimesh.repair.fix_inversion(repaired)

    pre_hole_mesh = repaired.copy()
    watertight_before = bool(pre_hole_mesh.is_watertight)
    face_count_before = len(pre_hole_mesh.faces)
    trimesh.repair.fill_holes(repaired)
    hole_faces_added = int(len(repaired.faces) - face_count_before)
    collar_fill_reverted = False
    if not watertight_before and repaired.is_watertight:
        collar_fill_reverted = True
        LOGGER.warning(
            "fill_holes appeared to cap the collar opening; reverting topology changes and keeping the open collar"
        )
        repaired = pre_hole_mesh

    repaired.remove_unreferenced_vertices()
    repaired.merge_vertices()
    repaired = _rebuild_mesh(repaired)
    return repaired, {
        "hole_faces_added": hole_faces_added,
        "collar_fill_reverted": collar_fill_reverted,
        "watertight_before_fill": watertight_before,
        "watertight_after_fill": bool(repaired.is_watertight),
    }


def _largest_component(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, int]:
    parts = mesh.split(only_watertight=False)
    if not parts:
        raise RuntimeError("finalize produced no connected components")
    ordered = sorted(parts, key=lambda item: len(item.faces), reverse=True)
    return _rebuild_mesh(ordered[0]), len(ordered) - 1


def _maybe_decimate(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, bool]:
    if len(mesh.faces) <= DECIMATION_FACE_THRESHOLD:
        return mesh, False
    target_faces = min(len(mesh.faces), DECIMATION_TARGET_FACE_COUNT)
    LOGGER.info("decimating finalized mesh from %s faces to %s faces", len(mesh.faces), target_faces)
    decimated = mesh.simplify_quadric_decimation(face_count=target_faces)
    decimated = _rebuild_mesh(decimated)
    trimesh.repair.fix_normals(decimated, multibody=False)
    return decimated, True


def _render_review(
    mesh: trimesh.Trimesh,
    *,
    scan_id: str,
    source: str,
    measurements: FootMeasurements,
    volume_ml: float,
    destination: Path,
    scan_path: Path | None,
) -> Path:
    comparison_mesh = align_to_canonical(load_scan(scan_path)) if scan_path is not None else mesh
    footer_lines = [
        f"Scan: {scan_id} | source={source}",
        (
            "Final length / ball / heel / toe width: "
            f"{measurements.length_mm:.1f} / {measurements.ball_width_mm:.1f} / "
            f"{measurements.heel_width_mm:.1f} / {measurements.toe_box_width_mm:.1f} mm"
        ),
        (
            "Max width / max height / volume: "
            f"{measurements.max_width_mm:.1f} / {measurements.max_height_mm:.1f} / {volume_ml:.1f} mL"
        ),
        "Grey = aligned foot scan, orange = Phase 5 finalized shoe tree.",
    ]
    return render_overlay_review_png(
        comparison_mesh,
        mesh,
        footer_lines=footer_lines,
        label_a="aligned foot scan" if scan_path is not None else "phase 5 finalized shoe tree",
        label_b="phase 5 finalized shoe tree",
        path=destination,
    )


def run(
    mesh_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    source_label: str | None = None,
    scan_path: str | Path | None = None,
) -> Phase5Artifacts:
    input_path = ensure_input_path(mesh_path)
    scan_id = _infer_scan_id(input_path)
    scan_reference = ensure_input_path(scan_path) if scan_path is not None else _candidate_scan_path(scan_id)
    destination = (
        ensure_directory(output_dir)
        if output_dir is not None
        else resolve_output_dir(None, phase_name="phase5", scan_path=Path(f"{scan_id}.obj"))
    )
    mesh = load_reference_mesh(input_path, metadata_id=f"{scan_id}_final")
    source = source_label or _infer_source_label(input_path)

    repaired_mesh, repair_payload = _repair_mesh(mesh)
    finalized_mesh, dropped_components = _largest_component(repaired_mesh)
    if dropped_components:
        LOGGER.info("phase 5 dropped %s disconnected fragment(s) before export", dropped_components)
    finalized_mesh, decimated = _maybe_decimate(finalized_mesh)
    finalized_mesh, dropped_after_decimation = _largest_component(finalized_mesh)
    dropped_components += dropped_after_decimation
    finalized_mesh = _rebuild_mesh(finalized_mesh)
    assert_alignment(finalized_mesh)

    measurements = measure_template_mesh(finalized_mesh)
    audit = audit_mesh(
        finalized_mesh,
        source_path=input_path,
        units="mm",
        unit_scale_to_mm=1.0,
        unit_sniff_reason="phase output",
    )
    watertight = bool(finalized_mesh.is_watertight)
    is_volume = bool(finalized_mesh.is_volume)
    boundary_edge_count = int(audit.boundary_edge_count)
    if not watertight:
        LOGGER.info(
            "phase 5 topology for %s remains open at the collar: watertight=%s boundary_edges=%s",
            scan_id,
            watertight,
            boundary_edge_count,
        )

    obj_path = save_mesh(finalized_mesh, destination / f"shoe_tree_{scan_id}.obj")
    stl_path = save_binary_stl(finalized_mesh, destination / f"shoe_tree_{scan_id}.stl")
    volume_ml = _round(finalized_mesh.volume / 1000.0)
    if volume_ml < 700.0 or volume_ml > 1200.0:
        LOGGER.warning(
            "phase 5 volume for %s is outside the expected range: %.1f mL",
            scan_id,
            volume_ml,
        )
    render_path = _render_review(
        finalized_mesh,
        scan_id=scan_id,
        source=source,
        measurements=measurements,
        volume_ml=volume_ml,
        destination=destination / "review.png",
        scan_path=scan_reference,
    )

    report_payload = {
        "phase": "phase5",
        "scan_id": scan_id,
        "source_scan_path": str(scan_reference) if scan_reference is not None else "",
        "input_mesh_path": str(input_path),
        "source": source,
        "length_mm": _round(measurements.length_mm),
        "max_width_mm": _round(measurements.max_width_mm),
        "max_height_mm": _round(measurements.max_height_mm),
        "ball_width_mm": _round(measurements.ball_width_mm),
        "heel_width_mm": _round(measurements.heel_width_mm),
        "toe_box_width_mm": _round(measurements.toe_box_width_mm),
        "volume_ml": volume_ml,
        "surface_area_mm2": _round(finalized_mesh.area),
        "face_count": int(len(finalized_mesh.faces)),
        "vertex_count": int(len(finalized_mesh.vertices)),
        "component_count": int(len(finalized_mesh.split(only_watertight=False))),
        "dropped_component_count": int(dropped_components),
        "is_watertight": watertight,
        "boundary_edge_count": boundary_edge_count,
        "is_volume": is_volume,
        "repair": repair_payload,
        "decimated": decimated,
        "obj_path": str(obj_path),
        "stl_path": str(stl_path),
        "review_png_path": str(render_path),
    }
    report_path = write_json(destination / "pipeline_report.json", report_payload)

    LOGGER.info(
        "%s | %.1f | %.1f | %.1f | %s | %s",
        scan_id,
        measurements.length_mm,
        measurements.ball_width_mm,
        volume_ml,
        source,
        watertight,
    )
    LOGGER.info("wrote phase 5 artifacts to %s", destination)

    return Phase5Artifacts(
        output_dir=destination,
        obj_path=obj_path,
        stl_path=stl_path,
        report_path=report_path,
        render_path=render_path,
        source=source,
        measurements=measurements,
        volume_ml=volume_ml,
    )


def run_best(
    scan_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    allowance_mm: float = 3.0,
) -> Phase5Artifacts:
    input_path = ensure_input_path(scan_path)
    _ensure_phase5_prerequisites(input_path, allowance_mm)
    mesh_path, source = _select_best_input(input_path)
    destination = resolve_output_dir(output_dir, phase_name="phase5", scan_path=input_path)
    return run(mesh_path, destination, source_label=source, scan_path=input_path)
