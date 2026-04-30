from __future__ import annotations

import argparse
import logging
from pathlib import Path

from custom_shoe_tree import finalize, measure, refine, template, warp
from custom_shoe_tree.io import ensure_directory, ensure_input_path, scan_id_from_path

LOGGER = logging.getLogger(__name__)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="custom-shoe-tree",
        description="FABRIC-581 foot-scan to custom shoe tree pipeline",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging level for CLI output.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    measure_parser = subparsers.add_parser(
        "measure",
        help="Phase 1: align a foot scan and extract measurements.",
    )
    measure_parser.add_argument("scan", help="Path to a single foot scan OBJ.")
    measure_parser.add_argument("-o", "--output-dir", help="Directory for phase artifacts.")
    measure_parser.set_defaults(handler=handle_measure)

    template_parser = subparsers.add_parser(
        "template",
        help="Phase 2: prepare and landmark the template shoe tree.",
    )
    template_parser.add_argument("-o", "--output-dir", help="Directory for phase artifacts.")
    template_parser.set_defaults(handler=handle_template)

    warp_parser = subparsers.add_parser(
        "warp",
        help="Phase 3: warp the template to match scan measurements.",
    )
    warp_parser.add_argument("scan", help="Path to a single foot scan OBJ.")
    warp_parser.add_argument("-o", "--output-dir", help="Directory for phase artifacts.")
    warp_parser.add_argument(
        "--allowance-mm",
        type=float,
        default=3.0,
        help="Upper-surface shoe-tree allowance in millimeters.",
    )
    warp_parser.set_defaults(handler=handle_warp)

    refine_parser = subparsers.add_parser(
        "refine",
        help="Phase 4: refine the warped shoe tree with non-rigid registration.",
    )
    refine_parser.add_argument("scan", help="Path to a single foot scan OBJ.")
    refine_parser.add_argument("-o", "--output-dir", help="Directory for phase artifacts.")
    refine_parser.add_argument(
        "--allowance-mm",
        type=float,
        default=3.0,
        help="Upper-surface shoe-tree allowance in millimeters.",
    )
    refine_parser.set_defaults(handler=handle_refine)

    finalize_parser = subparsers.add_parser(
        "finalize",
        help="Phase 5: repair and export the fabrication-ready shoe tree.",
    )
    finalize_parser.add_argument("mesh", help="Path to the warped/refined shoe tree mesh OBJ.")
    finalize_parser.add_argument("-o", "--output-dir", help="Directory for phase artifacts.")
    finalize_parser.set_defaults(handler=handle_finalize)

    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run the full custom shoe tree pipeline for one scan.",
    )
    pipeline_parser.add_argument("scan", help="Path to a single foot scan OBJ.")
    pipeline_parser.add_argument(
        "-o",
        "--output-dir",
        help="Optional root directory for pipeline artifacts.",
    )
    pipeline_parser.add_argument(
        "--allowance-mm",
        type=float,
        default=3.0,
        help="Upper-surface shoe-tree allowance in millimeters.",
    )
    pipeline_parser.set_defaults(handler=handle_pipeline)

    return parser


def handle_measure(args: argparse.Namespace) -> int:
    measure.run(args.scan, args.output_dir)
    return 0


def handle_template(args: argparse.Namespace) -> int:
    template.run(args.output_dir)
    return 0


def handle_warp(args: argparse.Namespace) -> int:
    warp.run(args.scan, args.output_dir, allowance_mm=args.allowance_mm)
    return 0


def handle_refine(args: argparse.Namespace) -> int:
    refine.run(args.scan, args.output_dir, allowance_mm=args.allowance_mm)
    return 0


def handle_finalize(args: argparse.Namespace) -> int:
    finalize.run(args.mesh, args.output_dir)
    return 0


def _pipeline_phase_output(root: Path | None, phase_name: str, scan_id: str) -> Path | None:
    if root is None:
        return None
    if phase_name == "phase2":
        return root / phase_name
    return root / phase_name / scan_id


def handle_pipeline(args: argparse.Namespace) -> int:
    scan_path = ensure_input_path(args.scan)
    scan_id = scan_id_from_path(scan_path)
    pipeline_root = ensure_directory(args.output_dir) if args.output_dir else None
    measure.run(scan_path, _pipeline_phase_output(pipeline_root, "phase1", scan_id))
    template.run(_pipeline_phase_output(pipeline_root, "phase2", scan_id))
    warp.run(
        scan_path,
        _pipeline_phase_output(pipeline_root, "phase3", scan_id),
        allowance_mm=args.allowance_mm,
    )
    refine.run(
        scan_path,
        _pipeline_phase_output(pipeline_root, "phase4", scan_id),
        allowance_mm=args.allowance_mm,
    )
    artifacts = finalize.run_best(
        scan_path,
        _pipeline_phase_output(pipeline_root, "phase5", scan_id),
        allowance_mm=args.allowance_mm,
    )
    LOGGER.info("pipeline completed for %s", scan_id)
    LOGGER.info("final pipeline report: %s", artifacts.report_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
