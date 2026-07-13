from __future__ import annotations

from typing import Any

import numpy as np

from source_primary_patch_inputs import validate_finite_config
from source_primary_patch_parameterization import polygon_self_intersections, polygon_signed_area


_HARD_CONFIG_LIMITS = {
    "max_samples": 100_000,
    "max_hits_per_sample": 4_096,
    "max_query_candidate_faces": 8_192,
    "max_footprint_candidate_faces": 250_000,
    "max_spatial_index_entries": 2_000_000,
}


def validate_proxy_sampling_inputs(
    proxy_points: np.ndarray,
    proxy_faces: np.ndarray,
    sample_uv: np.ndarray,
    footprint_uv: np.ndarray,
    center: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    normal: np.ndarray,
    scale: float,
    triangle_index: np.ndarray | None,
    component_id: np.ndarray | None,
    config: Any,
) -> tuple[str, str] | None:
    try:
        points = np.asarray(proxy_points)
        faces = np.asarray(proxy_faces)
        samples = np.asarray(sample_uv)
        footprint = np.asarray(footprint_uv)
    except (TypeError, ValueError) as exc:
        return "proxy_input_invalid", f"proxy arrays cannot be read: {exc}"
    if points.ndim != 2 or points.shape[1:] != (3,) or points.shape[0] < 3:
        return "proxy_points_invalid", "proxy_points must have shape (N, 3)"
    if (
        not np.issubdtype(points.dtype, np.number)
        or np.iscomplexobj(points)
        or not np.all(np.isfinite(points))
    ):
        return "proxy_points_invalid", "proxy_points must be finite real values"
    if faces.ndim != 2 or faces.shape[1:] != (3,) or not np.issubdtype(faces.dtype, np.integer):
        return "proxy_faces_invalid", "proxy_faces must be an integer (M, 3) array"
    if not faces.size or np.min(faces) < 0 or np.max(faces) >= points.shape[0]:
        return "proxy_faces_invalid", "proxy_faces contain no triangles or invalid point IDs"
    if np.any(
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 2] == faces[:, 0])
    ):
        return "proxy_faces_invalid", "proxy_faces contain repeated point IDs"
    if (
        samples.ndim != 2
        or samples.shape[1:] != (2,)
        or not samples.shape[0]
        or not np.issubdtype(samples.dtype, np.number)
        or np.iscomplexobj(samples)
    ):
        return "proxy_sample_uv_invalid", "sample_uv must be a non-empty real (K, 2) array"
    if samples.shape[0] > config.max_samples or not np.all(np.isfinite(samples)):
        return "proxy_sample_uv_invalid", "sample_uv exceeds limits or contains non-finite values"
    if (
        footprint.ndim != 2
        or footprint.shape[1:] != (2,)
        or footprint.shape[0] < 3
        or not np.issubdtype(footprint.dtype, np.number)
        or np.iscomplexobj(footprint)
    ):
        return "proxy_footprint_invalid", "footprint_uv must be a real polygon"
    if not np.all(np.isfinite(footprint)):
        return "proxy_footprint_invalid", "footprint_uv must contain finite values"
    footprint_float = footprint.astype(np.float64, copy=False)
    footprint_scale = footprint_diagonal_from_uv(footprint_float)
    signed_area = polygon_signed_area(footprint_float)
    if footprint_scale is None or not np.isfinite(signed_area) or abs(signed_area) <= 0.0:
        return "proxy_footprint_invalid", "footprint_uv must have finite non-zero area"
    if polygon_self_intersections(
        footprint_float, footprint_scale * np.finfo(np.float64).eps * 1024.0
    ):
        return "proxy_footprint_invalid", "footprint_uv must be a simple polygon"
    for name, value in (
        ("center", center),
        ("u_axis", u_axis),
        ("v_axis", v_axis),
        ("normal", normal),
    ):
        try:
            array = np.asarray(value)
        except (TypeError, ValueError) as exc:
            return "proxy_frame_invalid", f"{name} cannot be read: {exc}"
        if (
            array.shape != (3,)
            or not np.issubdtype(array.dtype, np.number)
            or np.iscomplexobj(array)
            or not np.all(np.isfinite(array))
        ):
            return "proxy_frame_invalid", f"{name} must be one finite real 3-vector"
    u, v, n = unit_vector(u_axis), unit_vector(v_axis), unit_vector(normal)
    if (
        u is None
        or v is None
        or n is None
        or max(abs(float(np.dot(u, v))), abs(float(np.dot(u, n))), abs(float(np.dot(v, n))))
        > 1e-8
        or float(np.dot(np.cross(u, v), n)) < 1.0 - 1e-8
    ):
        return "proxy_frame_invalid", "frame axes must be right-handed and orthonormal"
    try:
        supplied_scale = float(scale)
    except (OverflowError, TypeError, ValueError):
        return "proxy_footprint_scale_invalid", "footprint_diagonal must be finite"
    if not np.isfinite(supplied_scale) or supplied_scale <= 0.0:
        return "proxy_footprint_scale_invalid", "footprint_diagonal must be positive"
    scale_tolerance = max(
        footprint_scale * np.finfo(np.float64).eps * 64.0,
        np.nextafter(0.0, 1.0) * 64.0,
    )
    if abs(supplied_scale - footprint_scale) > scale_tolerance:
        return (
            "proxy_footprint_scale_mismatch",
            "footprint_diagonal must equal the diagonal derived from footprint_uv",
        )
    for name, values in (
        ("proxy_triangle_index", triangle_index),
        ("proxy_component_id", component_id),
    ):
        if values is None:
            continue
        try:
            array = np.asarray(values)
        except (TypeError, ValueError) as exc:
            return f"{name}_invalid", f"{name} cannot be read: {exc}"
        if array.shape != (faces.shape[0],) or not np.issubdtype(array.dtype, np.integer):
            return f"{name}_invalid", f"{name} must contain one integer per proxy face"
        if np.min(array) < 0 or np.max(array) > np.iinfo(np.int64).max:
            return f"{name}_invalid", f"{name} must contain non-negative int64 values"
        if name == "proxy_triangle_index" and (
            np.max(array) >= faces.shape[0] or np.unique(array).size != array.size
        ):
            return f"{name}_invalid", f"{name} must uniquely index the proxy face array"
    return None


def validate_proxy_sampling_config(config: Any, expected_type: type) -> str | None:
    scalar_error = validate_finite_config(
        config,
        expected_type,
        integer_fields=frozenset(
            {
                "max_samples",
                "max_hits_per_sample",
                "max_query_candidate_faces",
                "max_footprint_candidate_faces",
                "max_spatial_index_entries",
            }
        ),
    )
    if scalar_error is not None:
        return scalar_error
    if not -1.0 <= config.min_signed_depth_ratio < config.max_signed_depth_ratio <= 1.0:
        return "signed proxy depth range must be ordered within [-1, 1]"
    if not 0.0 < config.depth_cluster_tolerance_ratio <= 0.10:
        return "depth_cluster_tolerance_ratio must be within (0, 0.1]"
    if not 0.0 < config.barycentric_tolerance <= 1.0e-4:
        return "barycentric_tolerance must be within (0, 1e-4]"
    if not 0.0 <= config.minimum_normal_abs_dot <= 1.0:
        return "minimum_normal_abs_dot must be within [0, 1]"
    if not 0.0 < config.minimum_sample_coverage <= 1.0:
        return "minimum_sample_coverage must be within (0, 1]"
    if min(getattr(config, name) for name in _HARD_CONFIG_LIMITS) < 1:
        return "sample, hit, face, and index limits must be positive"
    for name, hard_limit in _HARD_CONFIG_LIMITS.items():
        if getattr(config, name) > hard_limit:
            return f"{name} exceeds hard safety limit {hard_limit}"
    if config.max_hits_per_sample > config.max_query_candidate_faces:
        return "max_hits_per_sample cannot exceed max_query_candidate_faces"
    return None


def footprint_diagonal_from_uv(footprint_uv: np.ndarray) -> float | None:
    """Return the finite bbox diagonal used for every proxy depth ratio."""

    try:
        footprint = np.asarray(footprint_uv, dtype=np.float64)
    except (OverflowError, TypeError, ValueError):
        return None
    if (
        footprint.ndim != 2
        or footprint.shape[1:] != (2,)
        or footprint.shape[0] < 3
        or not np.all(np.isfinite(footprint))
    ):
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        span = footprint.max(axis=0) - footprint.min(axis=0)
        diagonal = float(np.hypot(span[0], span[1]))
    return diagonal if np.isfinite(diagonal) and diagonal > 0.0 else None


def barycentric_2d(
    point: np.ndarray,
    triangle: np.ndarray,
    tolerance: float,
    area_tolerance: float,
) -> np.ndarray | None:
    a, b, c = triangle
    denominator = cross_2d(b - a, c - a)
    if abs(denominator) <= area_tolerance:
        return None
    second = cross_2d(point - a, c - a) / denominator
    third = cross_2d(b - a, point - a) / denominator
    result = np.asarray([1.0 - second - third, second, third], dtype=np.float64)
    return result if np.all(result >= -tolerance) else None


def cluster_depth_hits(
    hits: list[dict[str, Any]], tolerance: float
) -> list[list[dict[str, Any]]]:
    ordered = sorted(hits, key=lambda row: float(row["depth"]))
    clusters: list[list[dict[str, Any]]] = []
    for row in ordered:
        if not clusters or float(row["depth"]) - float(clusters[-1][0]["depth"]) > tolerance:
            clusters.append([row])
        else:
            clusters[-1].append(row)
    return clusters


def unit_vector(value: np.ndarray) -> np.ndarray | None:
    array = np.asarray(value, dtype=np.float64)
    length = float(np.linalg.norm(array))
    return None if not np.isfinite(length) or length <= 1e-30 else array / length


def cross_2d(left: np.ndarray, right: np.ndarray) -> float:
    return float(left[0] * right[1] - left[1] * right[0])
