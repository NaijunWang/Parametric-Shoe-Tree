from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import trimesh
import trimesh.transformations as tf
from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from custom_shoe_tree.measure import MeasurementContext

SECTION_COLORS = {
    "heel": (240, 113, 103, 255),
    "ball": (39, 125, 161, 255),
    "toe_height": (67, 170, 139, 255),
    "toe_width": (249, 199, 79, 255),
}
LANDMARK_COLORS = {
    "heel_origin": (87, 117, 144, 255),
    "toe_tip": (144, 190, 109, 255),
    "arch_peak": (188, 71, 73, 255),
    "ball_center": (39, 125, 161, 255),
    "heel_center": (240, 113, 103, 255),
    "toe_width_center": (249, 199, 79, 255),
    "toe_height_center": (67, 170, 139, 255),
}


def _vertex_colored(mesh: trimesh.Trimesh, color: tuple[int, int, int, int]) -> trimesh.Trimesh:
    colored = mesh.copy()
    colored.visual.vertex_colors = np.tile(np.array(color, dtype=np.uint8), (len(colored.vertices), 1))
    return colored


def _tube_from_polyline(
    points: np.ndarray,
    radius: float,
    color: tuple[int, int, int, int],
) -> trimesh.Trimesh:
    segments: list[trimesh.Trimesh] = []
    for start, end in zip(points[:-1], points[1:]):
        if np.linalg.norm(end - start) < 1e-6:
            continue
        cylinder = trimesh.creation.cylinder(radius=radius, segment=[start, end], sections=18)
        cylinder.visual.vertex_colors = np.tile(np.array(color, dtype=np.uint8), (len(cylinder.vertices), 1))
        segments.append(cylinder)
    if not segments:
        raise ValueError("polyline did not contain any drawable segments")
    return trimesh.util.concatenate(segments)


def _ring_meshes(context: MeasurementContext, ring_radius: float) -> list[tuple[str, trimesh.Trimesh]]:
    geometries: list[tuple[str, trimesh.Trimesh]] = []
    for name, section in context.sections.items():
        color = SECTION_COLORS[name]
        for loop_index, loop in enumerate(section.discrete_loops):
            if np.linalg.norm(loop[0] - loop[-1]) > 1e-4:
                loop = np.vstack([loop, loop[0]])
            tube = _tube_from_polyline(loop, ring_radius, color)
            geometries.append((f"ring_{name}_{loop_index}", tube))
    return geometries


def _sphere(center: np.ndarray, radius: float, color: tuple[int, int, int, int]) -> trimesh.Trimesh:
    marker = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    marker.apply_translation(center)
    marker.visual.vertex_colors = np.tile(np.array(color, dtype=np.uint8), (len(marker.vertices), 1))
    return marker


def _direction_rotation(direction: np.ndarray) -> np.ndarray:
    direction = direction / np.linalg.norm(direction)
    source = np.array([0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(source, direction), -1.0, 1.0))
    if np.isclose(dot, 1.0):
        return np.eye(4)
    if np.isclose(dot, -1.0):
        return tf.rotation_matrix(math.pi, [1.0, 0.0, 0.0])
    axis = np.cross(source, direction)
    angle = math.acos(dot)
    return tf.rotation_matrix(angle, axis)


def _dimension_bar(
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    color: tuple[int, int, int, int],
) -> trimesh.Trimesh:
    direction = end - start
    length = float(np.linalg.norm(direction))
    shaft = trimesh.creation.cylinder(radius=radius, height=length, sections=18)
    shaft.apply_transform(_direction_rotation(direction))
    shaft.apply_translation((start + end) / 2.0)
    shaft.visual.vertex_colors = np.tile(np.array(color, dtype=np.uint8), (len(shaft.vertices), 1))

    sphere_radius = radius * 1.8
    start_sphere = _sphere(start, sphere_radius, color)
    end_sphere = _sphere(end, sphere_radius, color)
    return trimesh.util.concatenate([shaft, start_sphere, end_sphere])


def annotate_scan(mesh: trimesh.Trimesh, context: MeasurementContext) -> trimesh.Scene:
    scene = trimesh.Scene()
    scan_mesh = _vertex_colored(mesh, (215, 215, 215, 255))
    scene.add_geometry(scan_mesh, geom_name="scan_mesh")

    ring_radius = max(0.8, float(mesh.extents[1]) * 0.0022)
    marker_radius = ring_radius * 2.2
    for name, geometry in _ring_meshes(context, ring_radius):
        scene.add_geometry(geometry, geom_name=name)

    for name, point in context.landmarks.items():
        color = LANDMARK_COLORS.get(name, (0, 0, 0, 255))
        scene.add_geometry(_sphere(point, marker_radius, color), geom_name=f"landmark_{name}")

    bounds = mesh.bounds
    offset_x = bounds[1][0] + max(8.0, float(mesh.extents[0]) * 0.08)
    length_bar = _dimension_bar(
        np.array([offset_x, 0.0, 0.0]),
        np.array([offset_x, context.measurements.length_mm, 0.0]),
        ring_radius * 0.55,
        (87, 117, 144, 255),
    )
    width_bar = _dimension_bar(
        np.array([bounds[0][0], context.measurements.control_sections_mm["ball"], ring_radius]),
        np.array([bounds[1][0], context.measurements.control_sections_mm["ball"], ring_radius]),
        ring_radius * 0.45,
        (39, 125, 161, 255),
    )
    height_bar = _dimension_bar(
        np.array([offset_x * 0.85, context.measurements.length_mm * 0.5, 0.0]),
        np.array([offset_x * 0.85, context.measurements.length_mm * 0.5, context.measurements.max_height_mm]),
        ring_radius * 0.45,
        (67, 170, 139, 255),
    )
    scene.add_geometry(length_bar, geom_name="dimension_length")
    scene.add_geometry(width_bar, geom_name="dimension_width")
    scene.add_geometry(height_bar, geom_name="dimension_height")
    return scene


def _project_points(
    points: np.ndarray,
    *,
    axes: tuple[int, int],
    box: tuple[int, int, int, int],
    bounds: tuple[float, float, float, float],
) -> list[tuple[float, float]]:
    left, top, right, bottom = box
    min_a, min_b, max_a, max_b = bounds
    width = max(right - left, 1)
    height = max(bottom - top, 1)
    span_a = max(max_a - min_a, 1e-6)
    span_b = max(max_b - min_b, 1e-6)
    scale = min((width * 0.86) / span_a, (height * 0.86) / span_b)
    center_x = left + width / 2.0
    center_y = top + height / 2.0
    center_a = (min_a + max_a) / 2.0
    center_b = (min_b + max_b) / 2.0
    coords = points[:, axes]
    projected: list[tuple[float, float]] = []
    for value_a, value_b in coords:
        px = center_x + (float(value_a) - center_a) * scale
        py = center_y - (float(value_b) - center_b) * scale
        projected.append((px, py))
    return projected


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    color: tuple[int, int, int],
    width: int = 3,
) -> None:
    draw.line([start, end], fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    arrow_length = 10
    left = (
        end[0] - arrow_length * math.cos(angle - math.pi / 6.0),
        end[1] - arrow_length * math.sin(angle - math.pi / 6.0),
    )
    right = (
        end[0] - arrow_length * math.cos(angle + math.pi / 6.0),
        end[1] - arrow_length * math.sin(angle + math.pi / 6.0),
    )
    draw.line([end, left], fill=color, width=width)
    draw.line([end, right], fill=color, width=width)


def _axis_bounds(mesh: trimesh.Trimesh, axes: tuple[int, int]) -> tuple[float, float, float, float]:
    coords = mesh.vertices[:, axes]
    min_a = float(coords[:, 0].min())
    min_b = float(coords[:, 1].min())
    max_a = float(coords[:, 0].max())
    max_b = float(coords[:, 1].max())
    pad_a = max((max_a - min_a) * 0.08, 4.0)
    pad_b = max((max_b - min_b) * 0.08, 4.0)
    return (min_a - pad_a, min_b - pad_b, max_a + pad_a, max_b + pad_b)


def _draw_measurement_footer(
    image: Image.Image,
    context: MeasurementContext,
    box: tuple[int, int, int, int],
) -> None:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=18, fill=(250, 247, 242), outline=(220, 216, 210), width=2)
    text_x = left + 18
    text_y = top + 16
    lines = [
        f"Scan: {context.measurements.scan_id}",
        (
            "Length / Width / Height: "
            f"{context.measurements.length_mm:.1f} / "
            f"{context.measurements.max_width_mm:.1f} / "
            f"{context.measurements.max_height_mm:.1f} mm"
        ),
        (
            "Heel / Ball width / Ball perimeter: "
            f"{context.measurements.heel_width_mm:.1f} / "
            f"{context.measurements.ball_width_mm:.1f} / "
            f"{context.measurements.ball_perimeter_mm:.1f} mm"
        ),
        (
            "Arch height / Toe width / Toe height: "
            f"{context.measurements.arch_height_mm:.1f} / "
            f"{context.measurements.toe_box_width_mm:.1f} / "
            f"{context.measurements.toe_box_height_mm:.1f} mm"
        ),
        (
            "Adaptive ball peak: "
            f"{context.measurements.adaptive_ball_y_pct:.1f}% of length"
        ),
        (
            "Arch type / Toe box type: "
            f"{context.measurements.arch_type} / {context.measurements.toe_box_type}"
        ),
        (
            "Raw toe metrics: "
            f"circularity={context.measurements.toe_section_circularity:.3f}, "
            f"aspect={context.measurements.toe_section_aspect_ratio:.3f}, "
            f"angle={context.measurements.toe_angle_deg:.2f} deg"
        ),
    ]
    for line in lines:
        draw.text((text_x, text_y), line, fill=(40, 40, 40), font=font)
        text_y += 24

    legend_x = right - 280
    legend_y = top + 16
    legend_items = [
        ("heel ring", SECTION_COLORS["heel"][:3]),
        ("ball ring", SECTION_COLORS["ball"][:3]),
        ("toe height ring", SECTION_COLORS["toe_height"][:3]),
        ("toe width ring", SECTION_COLORS["toe_width"][:3]),
        ("arch peak", LANDMARK_COLORS["arch_peak"][:3]),
        ("toe tip", LANDMARK_COLORS["toe_tip"][:3]),
    ]
    for label, color in legend_items:
        draw.rounded_rectangle(
            (legend_x, legend_y, legend_x + 18, legend_y + 18),
            radius=4,
            fill=color,
        )
        draw.text((legend_x + 26, legend_y + 2), label, fill=(40, 40, 40), font=font)
        legend_y += 24


def render_annotated_png(
    mesh: trimesh.Trimesh,
    context: MeasurementContext,
    path: str | Path,
) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    width, height = 1800, 1200
    image = Image.new("RGB", (width, height), (245, 243, 239))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    panel_boxes = {
        "top": (36, 36, 582, 700),
        "side": (624, 36, 1170, 700),
        "front": (1212, 36, 1758, 700),
    }
    footer_box = (36, 742, 1758, 1146)
    views = {
        "top": {"axes": (0, 1), "title": "Top View (X / Y)"},
        "side": {"axes": (1, 2), "title": "Side View (Y / Z)"},
        "front": {"axes": (0, 2), "title": "Front View (X / Z)"},
    }

    for name, box in panel_boxes.items():
        draw.rounded_rectangle(box, radius=24, fill=(252, 251, 248), outline=(221, 218, 212), width=2)
        draw.text((box[0] + 18, box[1] + 14), views[name]["title"], fill=(38, 38, 38), font=font)
        inner_box = (box[0] + 18, box[1] + 42, box[2] - 18, box[3] - 18)
        bounds = _axis_bounds(mesh, views[name]["axes"])

        projected_vertices = _project_points(mesh.vertices, axes=views[name]["axes"], box=inner_box, bounds=bounds)
        draw.point(projected_vertices, fill=(188, 188, 188))

        for section_name, section in context.sections.items():
            color = SECTION_COLORS[section_name][:3]
            for loop in section.discrete_loops:
                projected_loop = _project_points(loop, axes=views[name]["axes"], box=inner_box, bounds=bounds)
                if len(projected_loop) >= 2:
                    draw.line(projected_loop, fill=color, width=3)

        for landmark_name, point in context.landmarks.items():
            color = LANDMARK_COLORS.get(landmark_name, (0, 0, 0, 255))[:3]
            projected_point = _project_points(
                np.asarray([point]),
                axes=views[name]["axes"],
                box=inner_box,
                bounds=bounds,
            )[0]
            radius = 5
            draw.ellipse(
                (
                    projected_point[0] - radius,
                    projected_point[1] - radius,
                    projected_point[0] + radius,
                    projected_point[1] + radius,
                ),
                fill=color,
                outline=(255, 255, 255),
            )
            draw.text(
                (projected_point[0] + 8, projected_point[1] - 8),
                landmark_name.replace("_", " "),
                fill=(38, 38, 38),
                font=font,
            )

        if name == "top":
            x_offset = float(mesh.bounds[1][0] + max(8.0, mesh.extents[0] * 0.08))
            length_points = np.asarray(
                [
                    [x_offset, 0.0, 0.0],
                    [x_offset, context.measurements.length_mm, 0.0],
                ]
            )
            width_points = np.asarray(
                [
                    [mesh.bounds[0][0], context.measurements.control_sections_mm["ball"], 0.0],
                    [mesh.bounds[1][0], context.measurements.control_sections_mm["ball"], 0.0],
                ]
            )
            projected_length = _project_points(length_points, axes=views[name]["axes"], box=inner_box, bounds=bounds)
            projected_width = _project_points(width_points, axes=views[name]["axes"], box=inner_box, bounds=bounds)
            _draw_arrow(draw, projected_length[0], projected_length[1], (87, 117, 144))
            _draw_arrow(draw, projected_width[0], projected_width[1], (39, 125, 161))
            draw.text((projected_length[1][0] + 8, projected_length[1][1] - 18), "L", fill=(87, 117, 144), font=font)
            draw.text((projected_width[1][0] + 8, projected_width[1][1] - 18), "W", fill=(39, 125, 161), font=font)

        if name == "side":
            x_offset = float(mesh.bounds[1][0] + max(8.0, mesh.extents[0] * 0.08))
            height_points = np.asarray(
                [
                    [x_offset, context.measurements.length_mm * 0.5, 0.0],
                    [x_offset, context.measurements.length_mm * 0.5, context.measurements.max_height_mm],
                ]
            )
            projected_height = _project_points(height_points, axes=views[name]["axes"], box=inner_box, bounds=bounds)
            _draw_arrow(draw, projected_height[0], projected_height[1], (67, 170, 139))
            draw.text(
                (projected_height[1][0] + 8, projected_height[1][1] - 18),
                "H",
                fill=(67, 170, 139),
                font=font,
            )

    _draw_measurement_footer(image, context, footer_box)
    image.save(destination)
    return destination


def _combined_axis_bounds(
    meshes: list[trimesh.Trimesh],
    axes: tuple[int, int],
) -> tuple[float, float, float, float]:
    coords = np.vstack([mesh.vertices[:, axes] for mesh in meshes])
    min_a = float(coords[:, 0].min())
    min_b = float(coords[:, 1].min())
    max_a = float(coords[:, 0].max())
    max_b = float(coords[:, 1].max())
    pad_a = max((max_a - min_a) * 0.08, 4.0)
    pad_b = max((max_b - min_b) * 0.08, 4.0)
    return (min_a - pad_a, min_b - pad_b, max_a + pad_a, max_b + pad_b)


def render_overlay_review_png(
    mesh_a: trimesh.Trimesh,
    mesh_b: trimesh.Trimesh,
    *,
    footer_lines: list[str],
    label_a: str,
    label_b: str,
    color_a: tuple[int, int, int] = (190, 190, 190),
    color_b: tuple[int, int, int] = (214, 111, 44),
    path: str | Path,
) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    width, height = 1800, 1200
    image = Image.new("RGB", (width, height), (245, 243, 239))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    panel_boxes = {
        "top": (36, 36, 582, 700),
        "side": (624, 36, 1170, 700),
        "front": (1212, 36, 1758, 700),
    }
    footer_box = (36, 742, 1758, 1146)
    views = {
        "top": {"axes": (0, 1), "title": "Top View (X / Y)"},
        "side": {"axes": (1, 2), "title": "Side View (Y / Z)"},
        "front": {"axes": (0, 2), "title": "Front View (X / Z)"},
    }
    bounds_by_view = {
        name: _combined_axis_bounds([mesh_a, mesh_b], config["axes"])
        for name, config in views.items()
    }

    for name, box in panel_boxes.items():
        draw.rounded_rectangle(box, radius=24, fill=(252, 251, 248), outline=(221, 218, 212), width=2)
        draw.text((box[0] + 18, box[1] + 14), views[name]["title"], fill=(38, 38, 38), font=font)
        inner_box = (box[0] + 18, box[1] + 42, box[2] - 18, box[3] - 18)
        bounds = bounds_by_view[name]
        projected_a = _project_points(mesh_a.vertices, axes=views[name]["axes"], box=inner_box, bounds=bounds)
        projected_b = _project_points(mesh_b.vertices, axes=views[name]["axes"], box=inner_box, bounds=bounds)
        draw.point(projected_a, fill=color_a)
        draw.point(projected_b, fill=color_b)

    draw.rounded_rectangle(footer_box, radius=18, fill=(250, 247, 242), outline=(220, 216, 210), width=2)
    text_y = footer_box[1] + 16
    for line in footer_lines:
        draw.text((footer_box[0] + 18, text_y), line, fill=(40, 40, 40), font=font)
        text_y += 24

    legend_x = footer_box[2] - 300
    legend_y = footer_box[1] + 16
    legend_items = [
        (label_a, color_a),
        (label_b, color_b),
    ]
    for label, color in legend_items:
        draw.rounded_rectangle((legend_x, legend_y, legend_x + 18, legend_y + 18), radius=4, fill=color)
        draw.text((legend_x + 26, legend_y + 2), label, fill=(40, 40, 40), font=font)
        legend_y += 24

    image.save(destination)
    return destination


def render_warp_review_png(
    scan_mesh: trimesh.Trimesh,
    warped_mesh: trimesh.Trimesh,
    report: dict[str, object],
    path: str | Path,
) -> Path:
    target = report["review_summary"]["targets"]
    achieved = report["review_summary"]["achieved"]
    scale_summary = report["scale_summary"]
    footer_lines = [
        (
            f"Scan: {report['scan_id']} | allowance={report['allowance_mm']:.1f} mm | "
            f"sole threshold={report['sole_threshold_mm']:.1f} mm"
        ),
        (
            "Target length / warped length: "
            f"{report['target_measurements']['length_mm']:.1f} / {report['warped_measurements']['length_mm']:.1f} mm"
        ),
        (
            "Control heel / ball / toe width: "
            f"{target['heel_width_mm']:.1f} / {target['ball_width_mm']:.1f} / {target['toe_width_mm']:.1f} mm"
        ),
        (
            "Achieved heel / ball / toe width: "
            f"{achieved['heel_width_mm']:.1f} / {achieved['ball_width_mm']:.1f} / {achieved['toe_width_mm']:.1f} mm"
        ),
        (
            "Control arch / ball / toe top Z: "
            f"{target['arch_top_z_mm']:.1f} / {target['ball_top_z_mm']:.1f} / {target['toe_top_z_mm']:.1f} mm"
        ),
        (
            "Achieved arch / ball / toe top Z: "
            f"{achieved['arch_top_z_mm']:.1f} / {achieved['ball_top_z_mm']:.1f} / {achieved['toe_top_z_mm']:.1f} mm"
        ),
        (
            "Width scale range / Height scale range: "
            f"{scale_summary['width_scale_min']:.2f}-{scale_summary['width_scale_max']:.2f} / "
            f"{scale_summary['height_scale_min']:.2f}-{scale_summary['height_scale_max']:.2f}"
        ),
        "Grey = aligned foot scan, orange = Phase 3 warped shoe tree.",
    ]
    return render_overlay_review_png(
        scan_mesh,
        warped_mesh,
        footer_lines=footer_lines,
        label_a="aligned foot scan",
        label_b="phase 3 warped shoe tree",
        path=path,
    )
