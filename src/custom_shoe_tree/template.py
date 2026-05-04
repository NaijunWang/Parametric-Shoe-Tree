from __future__ import annotations

from collections import defaultdict, deque
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import MultiPoint
import trimesh

from custom_shoe_tree.align import align_to_canonical, assert_alignment
from custom_shoe_tree.io import (
    load_template_source,
    project_root,
    resolve_output_dir,
    save_mesh,
    write_json,
)
from custom_shoe_tree.measure import FootMeasurements

LOGGER = logging.getLogger(__name__)

TARGET_FACE_COUNT = 60_000
BALL_Y_PCT = 55.0
ARCH_Y_RANGE = (38.0, 42.0)
INSTEP_Y_RANGE = (50.0, 62.0)
TOE_Y_RANGE = (85.0, 98.0)
SECTION_SLAB_HALF_WIDTH_MM = 1.25
ANKLE_FLAP_COLLAR_Y_PCT = 0.22
ANKLE_FLAP_MIN_COMPONENT_VERTICES = 24
ANKLE_FLAP_MAX_COMPONENTS = 3
ANKLE_FLAP_SIDE_IMBALANCE_RATIO = 2.0
ANKLE_FLAP_MAX_REMOVED_FACE_RATIO = 0.02
ANKLE_FLAP_MAX_ITERATIONS = 2


@dataclass(slots=True)
class TemplateLandmark:
    index: int
    point_mm: list[float]


@dataclass(slots=True)
class TemplateArtifacts:
    output_dir: Path
    decimated_mesh_path: Path
    measurements_path: Path
    output_landmarks_path: Path
    source_landmarks_path: Path
    render_path: Path
    measurements: FootMeasurements
    landmarks: dict[str, TemplateLandmark]


def _round(value: float) -> float:
    return round(float(value), 6)


def source_landmarks_path() -> Path:
    return project_root() / "src" / "custom_shoe_tree" / "template_landmarks.json"


def _cache_dir(output_dir: str | Path | None) -> Path:
    return resolve_output_dir(output_dir, phase_name="phase2")


def _cache_mesh_path(cache_dir: Path) -> Path:
    return cache_dir / "base_shoe_tree_decimated.obj"


def _largest_component(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, int]:
    parts = mesh.split(only_watertight=False)
    if not parts:
        raise RuntimeError("template decimation produced no connected components")
    ordered = sorted(parts, key=lambda item: len(item.faces), reverse=True)
    return ordered[0].copy(), len(ordered) - 1


def _clean_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    cleaned = mesh.copy()
    cleaned.update_faces(cleaned.nondegenerate_faces())
    cleaned.remove_unreferenced_vertices()
    cleaned.merge_vertices()
    cleaned = align_to_canonical(cleaned)
    cleaned = _trim_ankle_flap(cleaned)
    assert_alignment(cleaned)
    return cleaned


def _boundary_vertex_components(mesh: trimesh.Trimesh) -> list[np.ndarray]:
    edges = mesh.edges_sorted
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    if boundary_edges.size == 0:
        return []

    adjacency: dict[int, set[int]] = defaultdict(set)
    for start, end in boundary_edges:
        adjacency[int(start)].add(int(end))
        adjacency[int(end)].add(int(start))

    components: list[np.ndarray] = []
    seen: set[int] = set()
    for start in adjacency:
        if start in seen:
            continue
        queue = deque([start])
        seen.add(start)
        component: list[int] = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                queue.append(neighbor)
        components.append(np.asarray(component, dtype=int))

    components.sort(key=len, reverse=True)
    return components


def _trim_ankle_flap_pass(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, int]:
    length_mm = float(mesh.extents[1])
    collar_y_limit = length_mm * ANKLE_FLAP_COLLAR_Y_PCT
    components = _boundary_vertex_components(mesh)
    if not components:
        return mesh, 0

    side_components: dict[str, list[np.ndarray]] = {"positive": [], "negative": []}
    for component in components:
        if len(component) < ANKLE_FLAP_MIN_COMPONENT_VERTICES:
            continue
        points = mesh.vertices[component]
        centroid = points.mean(axis=0)
        if centroid[1] >= collar_y_limit or abs(float(centroid[0])) <= 5.0:
            continue
        side_components["positive" if centroid[0] > 0.0 else "negative"].append(component)

    positive_total = sum(len(component) for component in side_components["positive"])
    negative_total = sum(len(component) for component in side_components["negative"])
    if max(positive_total, negative_total) < ANKLE_FLAP_MIN_COMPONENT_VERTICES * 2:
        return mesh, 0

    if positive_total and negative_total:
        imbalance = max(positive_total, negative_total) / max(min(positive_total, negative_total), 1)
        if imbalance < ANKLE_FLAP_SIDE_IMBALANCE_RATIO:
            return mesh, 0

    side_name = "positive" if positive_total >= negative_total else "negative"
    selected = sorted(side_components[side_name], key=len, reverse=True)[:ANKLE_FLAP_MAX_COMPONENTS]
    if len(selected) < 2:
        return mesh, 0

    remove_faces = np.zeros(len(mesh.faces), dtype=bool)
    for component in selected:
        remove_faces |= np.isin(mesh.faces, component).any(axis=1)

    centroids = mesh.triangles_center
    side_sign = 1.0 if side_name == "positive" else -1.0
    remove_faces &= (side_sign * centroids[:, 0] > 0.0) & (centroids[:, 1] < collar_y_limit)
    removed_face_count = int(remove_faces.sum())
    if removed_face_count == 0:
        return mesh, 0
    if removed_face_count > int(len(mesh.faces) * ANKLE_FLAP_MAX_REMOVED_FACE_RATIO):
        LOGGER.warning(
            "trim_ankle_flap skipped an unexpectedly large removal (%s / %s faces)",
            removed_face_count,
            len(mesh.faces),
        )
        return mesh, 0

    removed_vertices = np.unique(mesh.faces[remove_faces].reshape(-1))
    removed_points = mesh.vertices[removed_vertices]
    cleaned = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces[~remove_faces], process=False)
    cleaned.update_faces(cleaned.nondegenerate_faces())
    cleaned.remove_unreferenced_vertices()
    cleaned.merge_vertices()
    cleaned._cache.clear()
    cleaned.metadata.update(mesh.metadata)
    LOGGER.info(
        "trim_ankle_flap removed %s face(s) from %s %s-side collar patch(es); bbox=%s -> %s",
        removed_face_count,
        len(selected),
        side_name,
        np.round(removed_points.min(axis=0), 3).tolist(),
        np.round(removed_points.max(axis=0), 3).tolist(),
    )
    return cleaned, removed_face_count


def _trim_ankle_flap(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    trimmed = mesh
    total_removed = 0
    for _ in range(ANKLE_FLAP_MAX_ITERATIONS):
        trimmed, removed_face_count = _trim_ankle_flap_pass(trimmed)
        if removed_face_count == 0:
            break
        total_removed += removed_face_count
    if total_removed:
        LOGGER.info("trim_ankle_flap removed %s total face(s)", total_removed)
    return trimmed


def _prepare_decimated_template(cache_dir: Path) -> Path:
    cache_path = _cache_mesh_path(cache_dir)
    if cache_path.exists():
        LOGGER.info("using cached decimated template at %s", cache_path)
        return cache_path

    template_mesh = align_to_canonical(load_template_source())
    LOGGER.info(
        "decimating template from %s faces to ~%s faces",
        len(template_mesh.faces),
        TARGET_FACE_COUNT,
    )
    decimated = template_mesh.simplify_quadric_decimation(face_count=TARGET_FACE_COUNT)
    dominant_shell, dropped_components = _largest_component(decimated)
    cleaned = _clean_component(dominant_shell)
    LOGGER.info(
        "decimated template has %s faces / %s vertices after dropping %s small fragment(s)",
        len(cleaned.faces),
        len(cleaned.vertices),
        dropped_components,
    )
    return save_mesh(cleaned, cache_path)


def load_decimated_template(cache_dir: Path, cache_path: Path | None = None) -> trimesh.Trimesh:
    cache_path = _prepare_decimated_template(cache_dir) if cache_path is None else cache_path
    decimated = trimesh.load(cache_path, process=False)
    if not isinstance(decimated, trimesh.Trimesh):
        decimated = trimesh.util.concatenate(list(decimated.geometry.values()))
    decimated.metadata["scan_id"] = "base_shoe_tree_decimated"
    decimated.metadata["source_path"] = str(cache_path)
    assert_alignment(decimated)
    return decimated


def _point_payload(mesh: trimesh.Trimesh, index: int) -> TemplateLandmark:
    return TemplateLandmark(
        index=int(index),
        point_mm=[_round(value) for value in mesh.vertices[int(index)]],
    )


def _section_slab(mesh: trimesh.Trimesh, y_pct: float, *, half_width_mm: float = SECTION_SLAB_HALF_WIDTH_MM) -> np.ndarray:
    length_mm = float(mesh.extents[1])
    y_mm = length_mm * (y_pct / 100.0)
    mask = np.abs(mesh.vertices[:, 1] - y_mm) <= half_width_mm
    slab = mesh.vertices[mask]
    if len(slab) < 3:
        raise RuntimeError(f"template slab at {y_pct:.1f}% did not contain enough vertices")
    return slab


def _hull_stats(points_xz: np.ndarray) -> tuple[float, float, float, list[float]]:
    hull = MultiPoint([tuple(point) for point in points_xz]).convex_hull
    min_x = float(points_xz[:, 0].min())
    max_x = float(points_xz[:, 0].max())
    min_z = float(points_xz[:, 1].min())
    max_z = float(points_xz[:, 1].max())
    return (
        _round(max_x - min_x),
        _round(max_z - min_z),
        _round(hull.length),
        [_round(min_x), _round(min_z), _round(max_x), _round(max_z)],
    )


def _toe_angle_deg(mesh: trimesh.Trimesh) -> float:
    length_mm = float(mesh.extents[1])
    front_vertices = mesh.vertices[mesh.vertices[:, 1] >= length_mm * (TOE_Y_RANGE[0] / 100.0)]
    split_x = float(np.median(front_vertices[:, 0]))
    left = front_vertices[front_vertices[:, 0] <= split_x]
    right = front_vertices[front_vertices[:, 0] > split_x]
    tip_delta = abs(float(left[:, 1].max() - right[:, 1].max()))
    front_width = max(float(front_vertices[:, 0].max() - front_vertices[:, 0].min()), 1e-6)
    return _round(np.degrees(np.arctan2(tip_delta, front_width)))


def _arch_type(arch_ratio: float) -> str:
    if arch_ratio < 0.45:
        return "flat"
    if arch_ratio < 0.55:
        return "small"
    if arch_ratio < 0.65:
        return "medium"
    return "tall"


def _toe_box_type(circularity: float, aspect_ratio: float, toe_angle_deg: float) -> str:
    if toe_angle_deg >= 6.0:
        return "angled"
    if aspect_ratio >= 1.45 and circularity <= 0.78:
        return "square"
    return "standard"


def measure_template_mesh(mesh: trimesh.Trimesh) -> FootMeasurements:
    length_mm = float(mesh.extents[1])
    max_width_mm = float(mesh.extents[0])
    max_height_mm = float(mesh.extents[2])

    heel_slab = _section_slab(mesh, 10.0)
    ball_slab = _section_slab(mesh, 55.0)
    toe_height_slab = _section_slab(mesh, 85.0)
    toe_width_slab = _section_slab(mesh, 90.0)

    heel_width, _, _, _ = _hull_stats(heel_slab[:, [0, 2]])
    ball_width, ball_height, ball_perimeter, _ = _hull_stats(ball_slab[:, [0, 2]])
    toe_box_width, _, toe_width_perimeter, _ = _hull_stats(toe_width_slab[:, [0, 2]])
    _, toe_box_height, _, _ = _hull_stats(toe_height_slab[:, [0, 2]])

    arch_mask = (mesh.vertices[:, 1] >= length_mm * 0.38) & (mesh.vertices[:, 1] <= length_mm * 0.42)
    arch_peak = mesh.vertices[arch_mask][np.argmax(mesh.vertices[arch_mask][:, 2])]
    arch_height_mm = float(arch_peak[2])
    arch_ratio = arch_height_mm / max_height_mm

    toe_width_area = MultiPoint([tuple(point) for point in toe_width_slab[:, [0, 2]]]).convex_hull.area
    toe_circularity = (4.0 * np.pi * toe_width_area) / max(toe_width_perimeter**2, 1e-6)
    toe_aspect_ratio = toe_box_width / max(_hull_stats(toe_width_slab[:, [0, 2]])[1], 1e-6)
    toe_angle_deg = _toe_angle_deg(mesh)

    return FootMeasurements(
        scan_id="base_shoe_tree",
        length_mm=_round(length_mm),
        max_width_mm=_round(max_width_mm),
        max_height_mm=_round(max_height_mm),
        heel_width_mm=heel_width,
        ball_width_mm=ball_width,
        ball_perimeter_mm=ball_perimeter,
        ball_height_mm=ball_height,
        arch_height_mm=_round(arch_height_mm),
        arch_length_mm=_round(arch_height_mm),
        toe_box_width_mm=toe_box_width,
        toe_box_height_mm=toe_box_height,
        adaptive_ball_y_pct=55.0,
        arch_ratio=_round(arch_ratio),
        arch_type=_arch_type(arch_ratio),
        toe_section_circularity=_round(toe_circularity),
        toe_section_aspect_ratio=_round(toe_aspect_ratio),
        toe_angle_deg=toe_angle_deg,
        toe_box_type=_toe_box_type(toe_circularity, toe_aspect_ratio, toe_angle_deg),
        control_sections_mm={
            "heel": _round(length_mm * 0.10),
            "ball": _round(length_mm * 0.55),
            "toe_height": _round(length_mm * 0.85),
            "toe_width": _round(length_mm * 0.90),
            "arch": _round(length_mm * 0.40),
        },
    )


def _select_ball_landmarks(mesh: trimesh.Trimesh) -> tuple[int, int]:
    vertices = mesh.vertices
    ball_y = float(mesh.extents[1]) * (BALL_Y_PCT / 100.0)
    slab_half_width = max(2.5, float(mesh.extents[1]) * 0.012)
    mask = np.abs(vertices[:, 1] - ball_y) <= slab_half_width
    if not np.any(mask):
        raise RuntimeError("failed to find a vertex slab near the template ball section")
    slab_indices = np.where(mask)[0]
    slab_vertices = vertices[mask]
    outside_local = int(np.argmax(slab_vertices[:, 0] - 0.15 * np.abs(slab_vertices[:, 1] - ball_y)))
    inside_local = int(np.argmin(slab_vertices[:, 0] + 0.15 * np.abs(slab_vertices[:, 1] - ball_y)))
    return int(slab_indices[inside_local]), int(slab_indices[outside_local])


def _select_landmark_in_y_range(
    mesh: trimesh.Trimesh,
    start_pct: float,
    end_pct: float,
    *,
    coordinate: int = 2,
) -> int:
    vertices = mesh.vertices
    length_mm = float(mesh.extents[1])
    start_y = length_mm * (start_pct / 100.0)
    end_y = length_mm * (end_pct / 100.0)
    mask = (vertices[:, 1] >= start_y) & (vertices[:, 1] <= end_y)
    if not np.any(mask):
        raise RuntimeError(f"failed to find vertices in template slab {start_pct:.1f}-{end_pct:.1f}%")
    indices = np.where(mask)[0]
    slab_vertices = vertices[mask]
    local_index = int(np.argmax(slab_vertices[:, coordinate]))
    return int(indices[local_index])


def compute_template_landmarks(mesh: trimesh.Trimesh) -> dict[str, TemplateLandmark]:
    heel_back = int(np.argmin(mesh.vertices[:, 1]))
    toe_tip = int(np.argmax(mesh.vertices[:, 1]))
    ball_inside, ball_outside = _select_ball_landmarks(mesh)
    arch_peak = _select_landmark_in_y_range(mesh, *ARCH_Y_RANGE)
    instep_top = _select_landmark_in_y_range(mesh, *INSTEP_Y_RANGE)
    return {
        "heel_back": _point_payload(mesh, heel_back),
        "ball_inside": _point_payload(mesh, ball_inside),
        "ball_outside": _point_payload(mesh, ball_outside),
        "toe_tip": _point_payload(mesh, toe_tip),
        "arch_peak": _point_payload(mesh, arch_peak),
        "instep_top": _point_payload(mesh, instep_top),
    }


def _load_source_landmarks(mesh: trimesh.Trimesh) -> dict[str, TemplateLandmark] | None:
    path = source_landmarks_path()
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    landmarks = {
        name: TemplateLandmark(
            index=int(item["index"]),
            point_mm=[_round(value) for value in item["point_mm"]],
        )
        for name, item in payload["landmarks"].items()
    }
    max_index = len(mesh.vertices) - 1
    for name, landmark in landmarks.items():
        if landmark.index < 0 or landmark.index > max_index:
            raise ValueError(f"template landmark {name} index {landmark.index} is out of range")
    return landmarks


def _landmarks_payload(
    mesh: trimesh.Trimesh,
    landmarks: dict[str, TemplateLandmark],
) -> dict[str, Any]:
    return {
        "mesh_vertex_count": int(len(mesh.vertices)),
        "mesh_face_count": int(len(mesh.faces)),
        "landmarks": {
            name: {
                "index": landmark.index,
                "point_mm": landmark.point_mm,
            }
            for name, landmark in landmarks.items()
        },
    }


def _measurements_payload(measurements: FootMeasurements) -> dict[str, Any]:
    return {
        **measurements.to_dict(),
        "source": "template/cc_base_last.obj",
    }


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
    scale = min((width * 0.84) / span_a, (height * 0.84) / span_b)
    center_x = left + width / 2.0
    center_y = top + height / 2.0
    center_a = (min_a + max_a) / 2.0
    center_b = (min_b + max_b) / 2.0
    projected: list[tuple[float, float]] = []
    for value_a, value_b in points[:, axes]:
        px = center_x + (float(value_a) - center_a) * scale
        py = center_y - (float(value_b) - center_b) * scale
        projected.append((px, py))
    return projected


def _axis_bounds(mesh: trimesh.Trimesh, axes: tuple[int, int]) -> tuple[float, float, float, float]:
    coords = mesh.vertices[:, axes]
    min_a = float(coords[:, 0].min())
    min_b = float(coords[:, 1].min())
    max_a = float(coords[:, 0].max())
    max_b = float(coords[:, 1].max())
    pad_a = max((max_a - min_a) * 0.08, 4.0)
    pad_b = max((max_b - min_b) * 0.08, 4.0)
    return (min_a - pad_a, min_b - pad_b, max_a + pad_a, max_b + pad_b)


def _render_landmarks_png(
    mesh: trimesh.Trimesh,
    measurements: FootMeasurements,
    landmarks: dict[str, TemplateLandmark],
    path: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1800, 1100
    image = Image.new("RGB", (width, height), (245, 243, 239))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    colors = {
        "heel_back": (240, 113, 103),
        "ball_inside": (67, 170, 139),
        "ball_outside": (39, 125, 161),
        "toe_tip": (144, 190, 109),
        "arch_peak": (188, 71, 73),
        "instep_top": (249, 199, 79),
    }
    panel_boxes = {
        "top": (36, 36, 582, 676),
        "side": (624, 36, 1170, 676),
        "front": (1212, 36, 1758, 676),
    }
    footer_box = (36, 718, 1758, 1046)
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
        draw.point(projected_vertices, fill=(176, 176, 176))

        for landmark_name, landmark in landmarks.items():
            point = np.asarray([landmark.point_mm], dtype=float)
            projected = _project_points(point, axes=views[name]["axes"], box=inner_box, bounds=bounds)[0]
            color = colors[landmark_name]
            radius = 6
            draw.ellipse(
                (
                    projected[0] - radius,
                    projected[1] - radius,
                    projected[0] + radius,
                    projected[1] + radius,
                ),
                fill=color,
                outline=(255, 255, 255),
            )
            draw.text((projected[0] + 8, projected[1] - 8), landmark_name, fill=(38, 38, 38), font=font)

    draw.rounded_rectangle(footer_box, radius=18, fill=(250, 247, 242), outline=(220, 216, 210), width=2)
    footer_lines = [
        "Template: cc_base_last.obj",
        (
            "Length / Width / Height: "
            f"{measurements.length_mm:.1f} / {measurements.max_width_mm:.1f} / {measurements.max_height_mm:.1f} mm"
        ),
        (
            "Heel / Ball width / Ball perimeter: "
            f"{measurements.heel_width_mm:.1f} / {measurements.ball_width_mm:.1f} / {measurements.ball_perimeter_mm:.1f} mm"
        ),
        (
            "Toe width / Toe height / Arch height: "
            f"{measurements.toe_box_width_mm:.1f} / {measurements.toe_box_height_mm:.1f} / {measurements.arch_height_mm:.1f} mm"
        ),
        "Landmarks are authored on the cached decimated template mesh for later NRICP constraints.",
    ]
    text_y = footer_box[1] + 18
    for line in footer_lines:
        draw.text((footer_box[0] + 18, text_y), line, fill=(40, 40, 40), font=font)
        text_y += 26

    legend_x = footer_box[2] - 300
    legend_y = footer_box[1] + 18
    for landmark_name, color in colors.items():
        draw.rounded_rectangle((legend_x, legend_y, legend_x + 18, legend_y + 18), radius=4, fill=color)
        draw.text((legend_x + 26, legend_y + 2), landmark_name, fill=(40, 40, 40), font=font)
        legend_y += 24

    image.save(path)
    return path


def run(output_dir: str | Path | None = None) -> TemplateArtifacts:
    destination = _cache_dir(output_dir)
    decimated_mesh_path = _prepare_decimated_template(destination)
    decimated_mesh = load_decimated_template(destination, decimated_mesh_path)

    template_source = align_to_canonical(load_template_source())
    source_measurements = measure_template_mesh(template_source)
    measurements_path = write_json(
        destination / "base_shoe_tree_measurements.json",
        _measurements_payload(source_measurements),
    )

    landmarks = _load_source_landmarks(decimated_mesh)
    if landmarks is None:
        LOGGER.info("source template landmark file not found, using heuristic landmark authoring")
        landmarks = compute_template_landmarks(decimated_mesh)
    else:
        LOGGER.info("loaded template landmark indices from %s", source_landmarks_path())

    output_landmarks_path = write_json(
        destination / "template_landmarks.json",
        _landmarks_payload(decimated_mesh, landmarks),
    )
    render_path = _render_landmarks_png(
        decimated_mesh,
        source_measurements,
        landmarks,
        destination / "template_landmarks.png",
    )

    LOGGER.info("wrote phase 2 artifacts to %s", destination)

    return TemplateArtifacts(
        output_dir=destination,
        decimated_mesh_path=decimated_mesh_path,
        measurements_path=measurements_path,
        output_landmarks_path=output_landmarks_path,
        source_landmarks_path=source_landmarks_path(),
        render_path=render_path,
        measurements=source_measurements,
        landmarks=landmarks,
    )
