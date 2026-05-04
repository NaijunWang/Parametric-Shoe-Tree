from __future__ import annotations

import logging

import numpy as np
import trimesh
import trimesh.transformations as tf

LOGGER = logging.getLogger(__name__)


class AlignmentError(RuntimeError):
    """Raised when a scan cannot be aligned to the canonical frame."""


def get_area_from_path3d(path3d):
    if path3d is None:
        return 0
    try:
        path2d, _ = path3d.to_2D()
        return path2d.area
    except Exception:
        return 0


def align_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    aligned = mesh.copy()

    # Normalize scale to millimeters based on bounding box size.
    max_ext = max(aligned.extents)
    scale = 1000.0 if max_ext < 1.0 else (10.0 if max_ext < 50.0 else 1.0)
    aligned.apply_scale(scale)

    to_origin, _ = trimesh.bounds.oriented_bounds(aligned)
    aligned.apply_transform(to_origin)

    # Longest side is expected to be the foot length axis.
    extents = aligned.extents
    long_axis = np.argmax(extents)
    if long_axis == 0:
        aligned.apply_transform(tf.rotation_matrix(np.radians(90), [0, 0, 1]))
    elif long_axis == 2:
        aligned.apply_transform(tf.rotation_matrix(np.radians(90), [1, 0, 0]))

    def get_section_area_at_axis(axis_idx: int) -> float:
        try:
            origin = [0, 0, 0]
            origin[axis_idx] = aligned.centroid[axis_idx]
            normal = [0, 0, 0]
            normal[axis_idx] = 1
            section = aligned.section(plane_origin=origin, plane_normal=normal)
            return get_area_from_path3d(section)
        except Exception:
            return 0

    area_x = get_section_area_at_axis(0)
    area_z = get_section_area_at_axis(2)
    if area_x > area_z:
        aligned.apply_transform(tf.rotation_matrix(np.radians(90), [0, 1, 0]))

    bounds = aligned.bounds
    z_min_section = aligned.section(
        plane_origin=[0, 0, bounds[0][2] + 5],
        plane_normal=[0, 0, 1],
    )
    z_max_section = aligned.section(
        plane_origin=[0, 0, bounds[1][2] - 5],
        plane_normal=[0, 0, 1],
    )
    area_bottom = get_area_from_path3d(z_min_section)
    area_top = get_area_from_path3d(z_max_section)

    if area_top > area_bottom:
        aligned.apply_transform(tf.rotation_matrix(np.radians(180), [1, 0, 0]))

    # Orient toes toward +Y by keeping the thinner end in front.
    current_bounds = aligned.bounds
    y_len = current_bounds[1][1] - current_bounds[0][1]
    front_points = aligned.vertices[
        aligned.vertices[:, 1] > (current_bounds[1][1] - y_len * 0.15)
    ]
    back_points = aligned.vertices[
        aligned.vertices[:, 1] < (current_bounds[0][1] + y_len * 0.15)
    ]
    front_thickness = np.ptp(front_points[:, 2]) if len(front_points) > 0 else 0
    back_thickness = np.ptp(back_points[:, 2]) if len(back_points) > 0 else 0

    if front_thickness > back_thickness:
        aligned.apply_transform(tf.rotation_matrix(np.radians(180), [0, 0, 1]))

    vertices = aligned.vertices.copy()
    min_v, max_v = vertices.min(axis=0), vertices.max(axis=0)
    vertices[:, 0] -= (min_v[0] + max_v[0]) / 2
    vertices[:, 1] -= min_v[1]
    vertices[:, 2] -= min_v[2]
    aligned.vertices = vertices
    aligned._cache.clear()

    return aligned


def _rotation_to_match_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if np.isclose(dot, 1.0):
        return np.eye(4)
    if np.isclose(dot, -1.0):
        axis = np.cross(source, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-8:
            axis = np.cross(source, np.array([0.0, 1.0, 0.0]))
        return tf.rotation_matrix(np.pi, axis)
    axis = np.cross(source, target)
    angle = np.arccos(dot)
    return tf.rotation_matrix(angle, axis)


def _fit_sole_normal(mesh: trimesh.Trimesh) -> np.ndarray:
    sole_limit = min(2.0, float(np.percentile(mesh.vertices[:, 2], 8)))
    sole_vertices = mesh.vertices[mesh.vertices[:, 2] <= sole_limit]
    if len(sole_vertices) < 3:
        raise AlignmentError("not enough sole vertices to fit a support plane")
    centered = sole_vertices - sole_vertices.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0:
        normal = -normal
    return normal


def _normalize_origin(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
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


def assert_alignment(mesh: trimesh.Trimesh) -> None:
    bounds = mesh.bounds
    extents = mesh.extents
    x_center = float((bounds[0][0] + bounds[1][0]) / 2.0)
    if not np.isclose(bounds[0][1], 0.0, atol=1e-5):
        raise AlignmentError(f"heel plane is not at y=0 (y_min={bounds[0][1]:.6f})")
    if not np.isclose(bounds[0][2], 0.0, atol=1e-5):
        raise AlignmentError(f"sole plane is not at z=0 (z_min={bounds[0][2]:.6f})")
    if abs(x_center) > 1e-4:
        raise AlignmentError(f"mesh is not centered on x=0 (x_center={x_center:.6f})")
    if int(np.argmax(extents)) != 1:
        raise AlignmentError(f"longest axis is not +Y (extents={extents.tolist()})")


def align_to_canonical(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    aligned = align_mesh(mesh)

    sole_normal = _fit_sole_normal(aligned)
    support_rotation = _rotation_to_match_vectors(
        sole_normal, np.array([0.0, 0.0, 1.0])
    )
    aligned.apply_transform(support_rotation)

    normalized = _normalize_origin(aligned)
    assert_alignment(normalized)
    LOGGER.info("alignment verified for %s", normalized.metadata.get("scan_id", "mesh"))
    return normalized
