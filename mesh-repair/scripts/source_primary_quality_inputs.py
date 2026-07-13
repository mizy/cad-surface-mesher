from __future__ import annotations

from typing import Any

import numpy as np

from source_primary_quality_geometry import normalize_open_loop


def validate_quality_inputs(
    points: np.ndarray,
    faces: np.ndarray,
    interior: np.ndarray,
    patch_faces: np.ndarray,
    loops: tuple[np.ndarray, ...],
    cfg: Any,
) -> str | None:
    points = np.asarray(points)
    faces = np.asarray(faces)
    interior = np.asarray(interior)
    patch_faces = np.asarray(patch_faces)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not np.issubdtype(points.dtype, np.floating)
    ):
        return "current_points must be a floating (N, 3) array"
    if not np.all(np.isfinite(points)):
        return "current_points must contain only finite coordinates"
    if (
        faces.ndim != 2
        or faces.shape[1] != 3
        or not np.issubdtype(faces.dtype, np.integer)
    ):
        return "current_faces must be an integer (M, 3) array"
    if faces.size and np.any((faces < 0) | (faces >= points.shape[0])):
        return "current_faces contains an out-of-range point ID"
    if (
        interior.ndim != 2
        or interior.shape[1] != 3
        or not np.all(np.isfinite(interior))
    ):
        return "interior_points must be a finite (P, 3) array"
    if patch_faces.ndim != 2 or patch_faces.shape[1] != 3 or patch_faces.shape[0] == 0:
        return "patch_faces must be a non-empty integer (K, 3) array"
    if not np.issubdtype(patch_faces.dtype, np.integer):
        return "patch_faces must use integer point IDs"
    total_points = points.shape[0] + interior.shape[0]
    if np.any((patch_faces < 0) | (patch_faces >= total_points)):
        return "patch_faces contains an out-of-range point ID"
    normalized = tuple(normalize_open_loop(loop) for loop in loops)
    if not normalized or any(
        loop.size < 3 or np.unique(loop).size != loop.size for loop in normalized
    ):
        return "every boundary loop must contain at least three unique source point IDs"
    flat = np.concatenate(normalized)
    if (
        np.any((flat < 0) | (flat >= points.shape[0]))
        or np.unique(flat).size != flat.size
    ):
        return "boundary loops must contain distinct current source point IDs"
    return _validate_limits(cfg)


def _validate_limits(cfg: Any) -> str | None:
    numeric_limits = np.asarray(
        [
            cfg.minimum_triangle_area,
            cfg.minimum_area_ratio,
            cfg.maximum_aspect_ratio,
            cfg.minimum_orientation_dot,
            cfg.minimum_source_normal_dot,
            cfg.minimum_boundary_normal_dot,
            cfg.maximum_normal_transition_degrees,
            cfg.maximum_curvature_jump,
            cfg.coordinate_tolerance_ratio,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(numeric_limits)):
        return "patch quality limits must be finite"
    if cfg.minimum_triangle_area < 0.0 or cfg.minimum_area_ratio < 0.0:
        return "triangle area limits must be non-negative"
    if cfg.coordinate_tolerance_ratio < 0.0 or cfg.maximum_curvature_jump < 0.0:
        return "coordinate tolerance and curvature limits must be non-negative"
    dots = (
        cfg.minimum_orientation_dot,
        cfg.minimum_source_normal_dot,
        cfg.minimum_boundary_normal_dot,
    )
    if any(value < -1.0 or value > 1.0 for value in dots):
        return "normal dot limits must be in [-1, 1]"
    if not 0.0 <= cfg.maximum_normal_transition_degrees <= 180.0:
        return "maximum normal transition must be in [0, 180] degrees"
    if cfg.maximum_aspect_ratio <= 0.0 or cfg.maximum_intersection_candidate_pairs < 1:
        return "aspect and intersection limits must be positive"
    return None
