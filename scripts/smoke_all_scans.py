from __future__ import annotations

import json
import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps
from last_generator.finalize import Phase5Artifacts, run_best as run_finalize_best
from last_generator.io import write_csv
from last_generator.measure import Phase1Artifacts, run as run_measure
from last_generator.refine import Phase4Artifacts, run as run_refine
from last_generator.template import TemplateArtifacts, run as run_template
from last_generator.warp import Phase3Artifacts, run as run_warp

LOGGER = logging.getLogger("smoke_all_scans")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    return project_root().parent


def sample_scan_paths() -> list[Path]:
    scans_dir = repo_root() / "Sample Foot Scans"
    return sorted(scans_dir.glob("*.obj"))


def run_measurement(scan_path: Path) -> tuple[Phase1Artifacts, dict[str, object]]:
    destination = project_root() / "out" / "phase1" / scan_path.stem
    LOGGER.info("running phase 1 measure for %s", scan_path.name)
    artifacts = run_measure(scan_path, destination)
    entry = {
        "scan_id": scan_path.stem,
        "returncode": 0,
        "output_dir": str(destination),
        "measurements_path": str(artifacts.measurements_path),
        "audit_path": str(artifacts.audit_path),
        "annotated_png_path": str(artifacts.annotated_png_path),
        "repair_applied": artifacts.audit.repair_applied,
    }
    return artifacts, entry


def write_summary_csv(artifacts: list[Phase1Artifacts]) -> Path:
    destination = project_root() / "out" / "phase1" / "measurement_summary.csv"
    rows = [artifact.measurements.summary_row() for artifact in artifacts]
    return write_csv(destination, list(rows[0].keys()), rows)


def run_template_phase() -> tuple[TemplateArtifacts, dict[str, object]]:
    destination = project_root() / "out" / "phase2"
    LOGGER.info("running phase 2 template preparation")
    artifacts = run_template(destination)
    entry = {
        "output_dir": str(destination),
        "decimated_mesh_path": str(artifacts.decimated_mesh_path),
        "measurements_path": str(artifacts.measurements_path),
        "landmarks_path": str(artifacts.output_landmarks_path),
        "render_path": str(artifacts.render_path),
    }
    return artifacts, entry


def run_warp_phase(scan_path: Path) -> tuple[Phase3Artifacts, dict[str, object]]:
    destination = project_root() / "out" / "phase3" / scan_path.stem
    LOGGER.info("running phase 3 warp for %s", scan_path.name)
    artifacts = run_warp(scan_path, destination)
    entry = {
        "scan_id": scan_path.stem,
        "output_dir": str(destination),
        "mesh_path": str(artifacts.mesh_path),
        "render_path": str(artifacts.render_path),
        "report_path": str(artifacts.report_path),
        "ball_width_mm": artifacts.warped_measurements.ball_width_mm,
        "toe_box_width_mm": artifacts.warped_measurements.toe_box_width_mm,
    }
    return artifacts, entry


def run_refine_phase(scan_path: Path) -> tuple[Phase4Artifacts, dict[str, object]]:
    destination = project_root() / "out" / "phase4" / scan_path.stem
    LOGGER.info("running phase 4 refine for %s", scan_path.name)
    artifacts = run_refine(scan_path, destination)
    entry = {
        "scan_id": scan_path.stem,
        "output_dir": str(destination),
        "mesh_path": str(artifacts.mesh_path),
        "comparison_path": str(artifacts.comparison_path),
        "report_path": str(artifacts.report_path),
        "fallback": artifacts.fallback,
        "ball_width_mm": artifacts.refined_measurements.ball_width_mm,
        "toe_box_width_mm": artifacts.refined_measurements.toe_box_width_mm,
    }
    return artifacts, entry


def run_finalize_phase(scan_path: Path) -> tuple[Phase5Artifacts, dict[str, object]]:
    destination = project_root() / "out" / "phase5" / scan_path.stem
    LOGGER.info("running phase 5 finalize for %s", scan_path.name)
    artifacts = run_finalize_best(scan_path, destination)
    report_payload = json.loads(artifacts.report_path.read_text(encoding="utf-8"))
    entry = {
        "scan_id": scan_path.stem,
        "output_dir": str(destination),
        "obj_path": str(artifacts.obj_path),
        "stl_path": str(artifacts.stl_path),
        "review_path": str(artifacts.render_path),
        "report_path": str(artifacts.report_path),
        "source": artifacts.source,
        "ball_width_mm": artifacts.measurements.ball_width_mm,
        "heel_width_mm": artifacts.measurements.heel_width_mm,
        "volume_ml": artifacts.volume_ml,
        "watertight": bool(report_payload["is_watertight"]),
    }
    return artifacts, entry


def write_contact_sheet(
    entries: list[dict[str, object]],
    *,
    render_key: str,
    destination: Path,
) -> Path:
    cards: list[Image.Image] = []
    for entry in entries:
        render_path = Path(str(entry[render_key]))
        image = Image.open(render_path).convert("RGB")
        image = ImageOps.contain(image, (900, 600))
        card = Image.new("RGB", (920, 650), (245, 243, 239))
        card.paste(image, ((920 - image.width) // 2, 30))
        draw = ImageDraw.Draw(card)
        draw.text((20, 8), str(entry["scan_id"]), fill=(30, 30, 30))
        cards.append(card)

    sheet = Image.new("RGB", (1840, 1950), (238, 236, 231))
    positions = [(0, 0), (920, 0), (0, 650), (920, 650), (0, 1300)]
    for card, position in zip(cards, positions):
        sheet.paste(card, position)
    sheet.save(destination)
    return destination


def write_report(
    measurement_entries: list[dict[str, object]],
    template_entry: dict[str, object],
    warp_entries: list[dict[str, object]],
    warp_contact_sheet_path: Path,
    refine_entries: list[dict[str, object]],
    refine_contact_sheet_path: Path,
    finalize_entries: list[dict[str, object]],
    finalize_contact_sheet_path: Path,
) -> Path:
    destination = project_root() / "out" / "phase5" / "smoke_report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": "phase5",
        "scan_count": len(measurement_entries),
        "status": "ok",
        "measurement_entries": measurement_entries,
        "template": template_entry,
        "warp_entries": warp_entries,
        "warp_contact_sheet_path": str(warp_contact_sheet_path),
        "refine_entries": refine_entries,
        "refine_contact_sheet_path": str(refine_contact_sheet_path),
        "finalize_entries": finalize_entries,
        "finalize_contact_sheet_path": str(finalize_contact_sheet_path),
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


def main() -> int:
    configure_logging()
    scans = sample_scan_paths()
    if len(scans) != 5:
        raise RuntimeError(f"expected 5 sample scans, found {len(scans)}")
    results = [run_measurement(scan_path) for scan_path in scans]
    artifacts = [artifact for artifact, _ in results]
    measurement_entries = [entry for _, entry in results]
    _, template_entry = run_template_phase()
    warp_results = [run_warp_phase(scan_path) for scan_path in scans]
    warp_entries = [entry for _, entry in warp_results]
    warp_contact_sheet_path = write_contact_sheet(
        warp_entries,
        render_key="render_path",
        destination=project_root() / "out" / "phase3" / "phase3_contact_sheet.png",
    )
    refine_results = [run_refine_phase(scan_path) for scan_path in scans]
    refine_entries = [entry for _, entry in refine_results]
    refine_contact_sheet_path = write_contact_sheet(
        refine_entries,
        render_key="comparison_path",
        destination=project_root() / "out" / "phase4" / "phase4_contact_sheet.png",
    )
    finalize_results = [run_finalize_phase(scan_path) for scan_path in scans]
    finalize_entries = [entry for _, entry in finalize_results]
    finalize_contact_sheet_path = write_contact_sheet(
        finalize_entries,
        render_key="review_path",
        destination=project_root() / "out" / "phase5" / "phase5_contact_sheet.png",
    )
    summary_csv_path = write_summary_csv(artifacts)
    report_path = write_report(
        measurement_entries,
        template_entry,
        warp_entries,
        warp_contact_sheet_path,
        refine_entries,
        refine_contact_sheet_path,
        finalize_entries,
        finalize_contact_sheet_path,
    )
    LOGGER.info(
        "phase 5 smoke run completed for %s scans plus template prep, warps, refinements, and final exports",
        len(measurement_entries),
    )
    LOGGER.info("wrote phase 3 contact sheet to %s", warp_contact_sheet_path)
    LOGGER.info("wrote phase 4 contact sheet to %s", refine_contact_sheet_path)
    LOGGER.info("wrote phase 5 contact sheet to %s", finalize_contact_sheet_path)
    LOGGER.info("wrote measurement summary to %s", summary_csv_path)
    LOGGER.info("wrote smoke report to %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
