from __future__ import annotations

import logging
from functools import lru_cache
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import trimesh
import trimesh.transformations as tf

from last_generator.io import repo_root

LOGGER = logging.getLogger(__name__)


class AlignmentError(RuntimeError):
    """Raised when a scan cannot be aligned to the canonical frame."""


def _partner_module_path() -> Path:
    return (
        repo_root()
        / "Context"
        / "Partner-Work"
        / "Parametric-Shoe-Tree"
        / "src"
        / "extract_feature.py"
    )


@lru_cache(maxsize=1)
def _partner_module():
    module_path = _partner_module_path()
    spec = spec_from_file_location("partner_extract_feature", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load partner module from {module_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_area_from_path3d(path3d):
    return _partner_module().get_area_from_path3d(path3d)


def align_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    aligned = mesh.copy()
    return _partner_module().align_mesh(aligned)


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
    support_rotation = _rotation_to_match_vectors(sole_normal, np.array([0.0, 0.0, 1.0]))
    aligned.apply_transform(support_rotation)

    normalized = _normalize_origin(aligned)
    assert_alignment(normalized)
    LOGGER.info("alignment verified for %s", normalized.metadata.get("scan_id", "mesh"))
    return normalized
