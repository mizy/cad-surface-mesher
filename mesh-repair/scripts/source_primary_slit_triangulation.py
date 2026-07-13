from __future__ import annotations

from numbers import Integral, Real
from typing import Any

import numpy as np

from source_primary_patch_footprint import segments_conflict
from source_primary_patch_parameterization import points_in_polygon, polygon_signed_area


# @entry
def triangulate_source_fixed_slit_bridge(
    boundary_uv: np.ndarray,
    boundary_vertex_ids: np.ndarray,
    *,
    source_point_count: int,
    tolerance: float,
    maximum_appended_points: int,
    spacing_width_ratio: float,
) -> dict[str, Any]:
    """Triangulate a narrow footprint with source-fixed edges and interior points."""

    validated = _validate_inputs(
        boundary_uv,
        boundary_vertex_ids,
        source_point_count,
        tolerance,
        maximum_appended_points,
        spacing_width_ratio,
    )
    if isinstance(validated, dict):
        return validated
    uv, vertex_ids, tolerance, maximum_appended_points, spacing_width_ratio = validated
    start = int(np.argmin(vertex_ids))
    uv = np.roll(uv, -start, axis=0)
    vertex_ids = np.roll(vertex_ids, -start)
    patch_uv = uv[::-1].copy()
    patch_ids = vertex_ids[::-1].copy()
    if polygon_signed_area(patch_uv) <= tolerance * tolerance:
        return _failure("slit_bridge_parameterization_orientation_invalid")

    centerline = _build_centerline_points(
        patch_uv,
        tolerance=tolerance,
        maximum_points=maximum_appended_points,
        spacing_width_ratio=spacing_width_ratio,
    )
    if isinstance(centerline, dict):
        return centerline
    try:
        from scipy.spatial import Delaunay, QhullError
    except ImportError as exc:
        return _failure("slit_bridge_delaunay_unavailable", message=str(exc))
    local_points = np.vstack([patch_uv, centerline])
    try:
        raw_faces = Delaunay(local_points).simplices.astype(np.int64, copy=False)
    except QhullError as exc:
        return _failure("slit_bridge_delaunay_failed", message=str(exc))
    local_faces = _retain_footprint_faces(local_points, raw_faces, patch_uv, tolerance)
    if local_faces.shape[0] == 0:
        return _failure("slit_bridge_delaunay_no_footprint_faces")
    topology_error = _validate_local_topology(local_faces, patch_uv.shape[0])
    if topology_error is not None:
        return _failure(topology_error)
    if _interior_edges_cross_boundary(local_faces, local_points, patch_uv.shape[0], tolerance):
        return _failure("slit_bridge_delaunay_crosses_footprint")
    local_faces = _sort_oriented_faces(local_faces)

    used_interior = np.unique(local_faces[local_faces >= patch_uv.shape[0]])
    if used_interior.size == 0:
        return _failure("slit_bridge_delaunay_has_no_interior_points")
    compact = np.full(local_points.shape[0], -1, dtype=np.int64)
    compact[: patch_uv.shape[0]] = patch_ids
    compact[used_interior] = source_point_count + np.arange(
        used_interior.size, dtype=np.int64
    )
    faces = compact[local_faces]
    if np.any(faces < 0):
        return _failure("slit_bridge_delaunay_references_unused_point")
    appended_uv = local_points[used_interior]
    projected_quality = _projected_triangle_quality(local_points, local_faces)
    return {
        "success": True,
        "failure_reason_codes": [],
        "appended_uv": appended_uv,
        "faces": faces,
        "diagnostics": {
            "method": "source_fixed_centerline_constrained_delaunay",
            "boundary_vertices": int(vertex_ids.size),
            "generated_centerline_points": int(centerline.shape[0]),
            "appended_points": int(appended_uv.shape[0]),
            "generated_faces": int(faces.shape[0]),
            **projected_quality,
            "spacing_width_ratio": float(spacing_width_ratio),
            "boundary_constraints_preserved": True,
            "footprint_policy": "strict_polygon_interior_no_dilation",
        },
    }


def _build_centerline_points(
    polygon: np.ndarray,
    *,
    tolerance: float,
    maximum_points: int,
    spacing_width_ratio: float,
) -> np.ndarray | dict[str, Any]:
    center = polygon.mean(axis=0)
    try:
        _, _, axes = np.linalg.svd(polygon - center, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        return _failure("slit_bridge_centerline_svd_failed", message=str(exc))
    for axis in axes:
        pivot = int(np.argmax(np.abs(axis)))
        if axis[pivot] < 0.0:
            axis *= -1.0
    coordinates = (polygon - center) @ axes.T
    spans = np.ptp(coordinates, axis=0)
    major_axis = int(np.argmax(spans))
    minor_axis = 1 - major_axis
    width = float(spans[minor_axis])
    length = float(spans[major_axis])
    spacing = width * spacing_width_ratio
    if not np.isfinite(spacing) or spacing <= tolerance or length <= spacing:
        return _failure("slit_bridge_centerline_spacing_degenerate")
    count = max(1, int(np.ceil(length / spacing)) - 1)
    if count > maximum_points:
        return _failure(
            "slit_bridge_appended_point_limit_exceeded",
            attempted_points=count,
            maximum_appended_points=maximum_points,
        )
    major = np.linspace(
        float(coordinates[:, major_axis].min()),
        float(coordinates[:, major_axis].max()),
        count + 2,
    )[1:-1]
    local = np.zeros((count, 2), dtype=np.float64)
    local[:, major_axis] = major
    local[:, minor_axis] = 0.5 * (
        float(coordinates[:, minor_axis].min())
        + float(coordinates[:, minor_axis].max())
    )
    candidates = center + local @ axes
    inside = points_in_polygon(
        candidates, polygon, tolerance=tolerance * 8.0, include_boundary=False
    )
    candidates = candidates[inside]
    if candidates.shape[0] < 1:
        return _failure("slit_bridge_centerline_outside_footprint")
    distance = np.linalg.norm(candidates[:, None, :] - polygon[None, :, :], axis=2)
    candidates = candidates[np.min(distance, axis=1) > tolerance * 8.0]
    if candidates.shape[0] < 1:
        return _failure("slit_bridge_centerline_has_no_boundary_clearance")
    return candidates


def _retain_footprint_faces(
    points: np.ndarray,
    faces: np.ndarray,
    polygon: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    triangles = points[faces]
    signed = _cross_rows(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    area_tolerance = tolerance * max(float(np.ptp(polygon, axis=0).max()), tolerance)
    valid_area = np.abs(signed) > area_tolerance
    centroids = triangles.mean(axis=1)
    valid_center = points_in_polygon(
        centroids, polygon, tolerance=tolerance, include_boundary=False
    )
    midpoints = 0.5 * (triangles + np.roll(triangles, -1, axis=1))
    valid_edges = points_in_polygon(
        midpoints.reshape(-1, 2),
        polygon,
        tolerance=tolerance,
        include_boundary=True,
    ).reshape(-1, 3).all(axis=1)
    mask = valid_area & valid_center & valid_edges
    retained = faces[mask].copy()
    retained_signed = signed[mask]
    retained[retained_signed < 0.0] = retained[retained_signed < 0.0][:, [0, 2, 1]]
    return retained


def _validate_local_topology(faces: np.ndarray, boundary_count: int) -> str | None:
    directed = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    canonical = np.sort(directed, axis=1)
    unique, inverse, counts = np.unique(
        canonical, axis=0, return_inverse=True, return_counts=True
    )
    boundary = {
        tuple(sorted((index, (index + 1) % boundary_count)))
        for index in range(boundary_count)
    }
    present_boundary: set[tuple[int, int]] = set()
    for unique_index, (edge, count) in enumerate(zip(unique, counts, strict=True)):
        key = (int(edge[0]), int(edge[1]))
        expected = 1 if key in boundary else 2
        if int(count) != expected:
            return "slit_bridge_delaunay_boundary_constraint_missing"
        rows = directed[inverse == unique_index]
        if expected == 1:
            present_boundary.add(key)
        elif tuple(int(value) for value in rows[0]) != tuple(
            int(value) for value in rows[1][::-1]
        ):
            return "slit_bridge_delaunay_winding_conflict"
    return None if present_boundary == boundary else "slit_bridge_delaunay_boundary_constraint_missing"


def _sort_oriented_faces(faces: np.ndarray) -> np.ndarray:
    minimum_position = np.argmin(faces, axis=1)
    offsets = (minimum_position[:, None] + np.arange(3)) % 3
    rotated = np.take_along_axis(faces, offsets, axis=1)
    order = np.lexsort((rotated[:, 2], rotated[:, 1], rotated[:, 0]))
    return rotated[order]


def _interior_edges_cross_boundary(
    faces: np.ndarray,
    points: np.ndarray,
    boundary_count: int,
    tolerance: float,
) -> bool:
    edges = np.unique(
        np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0
    )
    boundary_edges = np.column_stack(
        (np.arange(boundary_count), np.roll(np.arange(boundary_count), -1))
    )
    boundary_keys = {tuple(sorted(map(int, edge))) for edge in boundary_edges}
    for edge in edges:
        if tuple(int(value) for value in edge) in boundary_keys:
            continue
        for boundary_edge in boundary_edges:
            shared = bool(set(map(int, edge)).intersection(map(int, boundary_edge)))
            if segments_conflict(
                points[edge[0]],
                points[edge[1]],
                points[boundary_edge[0]],
                points[boundary_edge[1]],
                tolerance,
                allow_one_shared_endpoint=shared,
            ):
                return True
    return False


def _validate_inputs(
    boundary_uv: np.ndarray,
    boundary_vertex_ids: np.ndarray,
    source_point_count: int,
    tolerance: float,
    maximum_appended_points: int,
    spacing_width_ratio: float,
) -> tuple[np.ndarray, np.ndarray, float, int, float] | dict[str, Any]:
    try:
        raw_uv = np.asarray(boundary_uv)
        raw_ids = np.asarray(boundary_vertex_ids)
        tolerance_value = float(tolerance)
        spacing_value = float(spacing_width_ratio)
    except (OverflowError, TypeError, ValueError) as exc:
        return _failure("slit_bridge_parameterization_invalid", message=str(exc))
    integers = (source_point_count, maximum_appended_points)
    integers_valid = all(
        isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))
        for value in integers
    )
    int64_max = np.iinfo(np.int64).max
    if (
        raw_uv.ndim != 2
        or raw_uv.shape[1:] != (2,)
        or raw_ids.ndim != 1
        or raw_uv.shape[0] != raw_ids.size
        or raw_ids.size < 3
        or not np.issubdtype(raw_uv.dtype, np.number)
        or np.iscomplexobj(raw_uv)
        or not np.issubdtype(raw_ids.dtype, np.integer)
        or not np.all(np.isfinite(raw_uv))
        or np.unique(raw_ids).size != raw_ids.size
        or np.min(raw_ids) < 0
        or np.max(raw_ids) > int64_max
        or not integers_valid
        or not 0 < int(source_point_count) <= int64_max
        or int(source_point_count) <= int(np.max(raw_ids))
        or not 1 <= int(maximum_appended_points) <= 100_000
        or int(source_point_count) + int(maximum_appended_points) > int64_max
        or not isinstance(tolerance, Real)
        or isinstance(tolerance, (bool, np.bool_))
        or not np.isfinite(tolerance_value)
        or tolerance_value <= 0.0
        or not isinstance(spacing_width_ratio, Real)
        or isinstance(spacing_width_ratio, (bool, np.bool_))
        or not np.isfinite(spacing_value)
        or not 0.1 <= spacing_value <= 2.0
    ):
        return _failure("slit_bridge_parameterization_invalid")
    return (
        raw_uv.astype(np.float64, copy=False),
        raw_ids.astype(np.int64, copy=False),
        tolerance_value,
        int(maximum_appended_points),
        spacing_value,
    )


def _cross_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return left[:, 0] * right[:, 1] - left[:, 1] * right[:, 0]


def _projected_triangle_quality(points: np.ndarray, faces: np.ndarray) -> dict[str, float]:
    triangles = points[faces]
    edges = np.stack(
        (
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 1],
            triangles[:, 0] - triangles[:, 2],
        ),
        axis=1,
    )
    lengths = np.linalg.norm(edges, axis=2)
    double_area = np.abs(_cross_rows(edges[:, 0], -edges[:, 2]))
    altitude = double_area / np.maximum(lengths.max(axis=1), np.finfo(float).tiny)
    aspect = lengths.max(axis=1) / np.maximum(altitude, np.finfo(float).tiny)
    return {
        "minimum_projected_area": float(0.5 * double_area.min()),
        "maximum_projected_aspect_ratio": float(aspect.max()),
    }


def _failure(code: str, **diagnostics: Any) -> dict[str, Any]:
    return {
        "success": False,
        "failure_reason_codes": [code],
        "appended_uv": np.empty((0, 2), dtype=np.float64),
        "faces": np.empty((0, 3), dtype=np.int64),
        "diagnostics": diagnostics,
    }
