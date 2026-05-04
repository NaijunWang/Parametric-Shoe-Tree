from __future__ import annotations

import asyncio
import dataclasses
import logging
import sys
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

# Add the custom_shoe_tree source to sys.path so we can import it without a separate install step.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PIPELINE_SRC = _REPO_ROOT / "src"
if str(_PIPELINE_SRC) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_SRC))

from custom_shoe_tree import measure, finalize as finalize_mod, split as split_mod
from custom_shoe_tree.io import load_reference_mesh, resolve_output_dir, save_mesh, write_json
from custom_shoe_tree.measure import measure_mesh
from custom_shoe_tree.template import load_decimated_template, measure_template_mesh, run as run_template
from custom_shoe_tree.warp import build_warp, SOLE_THRESHOLD_MM
from custom_shoe_tree.refine import run as run_refine


def _predict_shoe_size(length_mm: float) -> dict[str, Any]:
    eu = round(length_mm / 6.67)
    return {
        "eu": eu,
        "us_mens": max(eu - 31, 1),
        "us_womens": max(eu - 30, 1),
        "uk": max(eu - 33, 1),
    }


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}

    def create_job(self, job_id: str, scan_path: str) -> None:
        self._jobs[job_id] = {
            "job_id": job_id,
            "scan_path": scan_path,
            "status": "measuring",
            "progress": 0,
            "message": "Starting measurement…",
            "measurements": None,
            "shoe_size": None,
            "aligned_path": None,
            "scan_id": None,
            "stl_path": None,
            "obj_path": None,
            "split_for_print": False,
            "split_stl_paths": None,
            "split_report_path": None,
            "error": None,
        }

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        return self._jobs.get(job_id)

    def update_job(self, job_id: str, **kwargs: Any) -> None:
        if job_id in self._jobs:
            self._jobs[job_id].update(kwargs)


def _measurements_to_dict(m: Any) -> dict[str, Any]:
    return {
        "scan_id": m.scan_id,
        "length_mm": round(m.length_mm, 1),
        "max_width_mm": round(m.max_width_mm, 1),
        "max_height_mm": round(m.max_height_mm, 1),
        "heel_width_mm": round(m.heel_width_mm, 1),
        "ball_width_mm": round(m.ball_width_mm, 1),
        "ball_perimeter_mm": round(m.ball_perimeter_mm, 1),
        "ball_height_mm": round(m.ball_height_mm, 1),
        "arch_height_mm": round(m.arch_height_mm, 1),
        "arch_length_mm": round(getattr(m, "arch_length_mm", m.arch_height_mm), 1),
        "toe_box_width_mm": round(m.toe_box_width_mm, 1),
        "toe_box_height_mm": round(m.toe_box_height_mm, 1),
        "adaptive_ball_y_pct": round(m.adaptive_ball_y_pct, 1),
        "arch_ratio": round(m.arch_ratio, 3),
        "arch_type": m.arch_type,
        "toe_section_circularity": round(m.toe_section_circularity, 3),
        "toe_section_aspect_ratio": round(m.toe_section_aspect_ratio, 3),
        "toe_angle_deg": round(m.toe_angle_deg, 1),
        "toe_box_type": m.toe_box_type,
    }


async def run_measure_job(job_id: str, scan_path: Path, store: JobStore) -> None:
    loop = asyncio.get_event_loop()
    try:
        store.update_job(job_id, progress=10, message="Aligning and measuring foot scan…")

        phase1 = await loop.run_in_executor(None, lambda: measure.run(scan_path))

        measurements_dict = _measurements_to_dict(phase1.measurements)
        shoe_size = _predict_shoe_size(phase1.measurements.length_mm)

        store.update_job(
            job_id,
            status="measured",
            progress=100,
            message="Measurement complete — review and confirm below.",
            measurements=measurements_dict,
            shoe_size=shoe_size,
            aligned_path=str(phase1.mesh_aligned_path),
            scan_id=phase1.measurements.scan_id,
        )
    except Exception as exc:
        LOGGER.exception("Measure job %s failed", job_id)
        store.update_job(job_id, status="error", message=str(exc), error=str(exc))


async def run_generate_job(
    job_id: str,
    user_measurements: dict[str, Any],
    allowance_mm: float,
    split_for_print: bool,
    store: JobStore,
) -> None:
    loop = asyncio.get_event_loop()
    try:
        job = store.get_job(job_id)
        scan_path = Path(job["scan_path"])

        # ── Phase 2: ensure template decimation is cached ───────────────────
        store.update_job(job_id, progress=5, message="Loading template…")

        def _load_template() -> tuple[Any, Any]:
            cache_dir = resolve_output_dir(None, phase_name="phase2")
            if not (cache_dir / "base_shoe_tree_decimated.obj").exists():
                run_template(cache_dir)
            tmesh = load_decimated_template(cache_dir)
            tmeasurements = measure_template_mesh(tmesh)
            return tmesh, tmeasurements

        template_mesh, template_measurements = await loop.run_in_executor(None, _load_template)

        # ── Phase 3: warp with (optionally patched) measurements ─────────────
        store.update_job(job_id, progress=20, message="Warping template to foot measurements…")

        def _warp() -> None:
            aligned_path = Path(job["aligned_path"])
            aligned_scan = load_reference_mesh(aligned_path, metadata_id=scan_path.stem)
            scan_context = measure_mesh(aligned_scan)

            # Apply user-edited measurements (only the numeric fields we expose)
            editable = {
                "length_mm", "heel_width_mm", "ball_width_mm", "ball_perimeter_mm",
                "ball_height_mm", "arch_height_mm", "toe_box_width_mm", "toe_box_height_mm",
            }
            patches = {k: float(v) for k, v in user_measurements.items() if k in editable and v is not None}
            if patches:
                patched_m = dataclasses.replace(scan_context.measurements, **patches)
                scan_context = dataclasses.replace(scan_context, measurements=patched_m)

            warped_mesh, _warped_m, report = build_warp(
                template_mesh,
                template_measurements,
                scan_context,
                allowance_mm=allowance_mm,
                sole_threshold_mm=SOLE_THRESHOLD_MM,
            )

            phase3_dir = resolve_output_dir(None, phase_name="phase3", scan_path=scan_path)
            phase3_dir.mkdir(parents=True, exist_ok=True)
            save_mesh(warped_mesh, phase3_dir / "shoe_tree_warp.obj")
            write_json(phase3_dir / "warp_report.json", report)

        await loop.run_in_executor(None, _warp)

        # ── Phase 4: non-rigid ICP refinement ────────────────────────────────
        store.update_job(job_id, progress=45, message="Refining shape (NRICP pass 1 of 3)…")

        def _refine() -> None:
            phase4_dir = resolve_output_dir(None, phase_name="phase4", scan_path=scan_path)
            run_refine(scan_path, phase4_dir, allowance_mm=allowance_mm)

        await loop.run_in_executor(None, _refine)

        # ── Phase 5: finalise + export STL ───────────────────────────────────
        store.update_job(job_id, progress=80, message="Finalising and exporting STL…")

        def _finalize() -> Any:
            phase5_dir = resolve_output_dir(None, phase_name="phase5", scan_path=scan_path)
            return finalize_mod.run_best(scan_path, phase5_dir, allowance_mm=allowance_mm)

        artifacts = await loop.run_in_executor(None, _finalize)

        split_stl_paths: dict[str, str] | None = None
        split_report_path: str | None = None
        if split_for_print:
            store.update_job(job_id, progress=92, message="Splitting STL for print bed…")

            def _split() -> Any:
                phase6_dir = resolve_output_dir(None, phase_name="phase6_split", scan_path=scan_path)
                return split_mod.run(artifacts.stl_path, phase6_dir)

            split_artifacts = await loop.run_in_executor(None, _split)
            split_stl_paths = {
                "heel-tabs": str(split_artifacts.heel_tabs_stl_path),
                "toe-sockets": str(split_artifacts.toe_sockets_stl_path),
            }
            split_report_path = str(split_artifacts.report_path)

        store.update_job(
            job_id,
            status="done",
            progress=100,
            message=(
                "Your split custom shoe tree files are ready to download."
                if split_for_print
                else "Your custom shoe tree is ready to download."
            ),
            stl_path=str(artifacts.stl_path),
            obj_path=str(artifacts.obj_path),
            split_for_print=split_for_print,
            split_stl_paths=split_stl_paths,
            split_report_path=split_report_path,
        )

    except Exception as exc:
        LOGGER.exception("Generate job %s failed", job_id)
        store.update_job(job_id, status="error", message=str(exc), error=str(exc))
