from __future__ import annotations

from typing import Any, Mapping, TYPE_CHECKING

import numpy as np

from source_primary_patch_parameterization import points_in_polygon, polygon_signed_area

if TYPE_CHECKING:
    from source_primary_patch_contract import BoundaryMapping


# @entry
def validate_patch_footprint(
    source_points: np.ndarray,
    new_points: np.ndarray,
    new_faces: np.ndarray,
    mappings: tuple[BoundaryMapping, ...],
    normal: Mapping[str, Any],
    source_count: int,
    method: str,
    point_provenance: Mapping[str, Any],
) -> list[str]:
    """Prove that a one-loop patch stays in its true planar parameter domain."""

    if method == "paired_loop_zipper" or len(mappings) != 1:
        return []
    loop = np.asarray(mappings[0].source_vertex_ids, dtype=np.int64)
    boundary = np.asarray(source_points, dtype=np.float64)[loop]
    origin = _perimeter_center(boundary)
    unit_normal = _parameterization_normal(normal)
    recorded_origin = _read_vector(normal.get("parameterization_origin"))
    tangent = _read_vector(normal.get("parameterization_u_axis"))
    bitangent = _read_vector(normal.get("parameterization_v_axis"))
    if (
        origin is None
        or unit_normal is None
        or recorded_origin is None
        or tangent is None
        or bitangent is None
    ):
        return ["patch footprint parameterization is degenerate"]

    def project(values: np.ndarray) -> np.ndarray:
        local = np.asarray(values, dtype=np.float64) - origin
        return np.column_stack((local @ tangent, local @ bitangent))

    polygon = project(boundary)
    scale = max(float(np.linalg.norm(np.ptp(polygon, axis=0))), np.finfo(float).tiny)
    coordinate_ulp = float(np.max(np.abs(np.spacing(boundary)), initial=0.0))
    tolerance = max(
        scale * np.finfo(np.float64).eps * 1024.0,
        coordinate_ulp * 8.0,
        np.finfo(np.float64).tiny,
    )
    if float(np.linalg.norm(recorded_origin - origin)) > tolerance * 32.0:
        return ["parameterization_origin does not match the immutable source boundary"]
    polygon_area = polygon_signed_area(polygon)
    if abs(polygon_area) <= tolerance * scale:
        return ["patch footprint parameterization is degenerate"]
    all_points = np.vstack([np.asarray(source_points, dtype=np.float64), new_points])
    all_uv = project(all_points)
    new_depth = (np.asarray(new_points, dtype=np.float64) - origin) @ unit_normal
    errors: list[str] = []
    if new_points.shape[0] and not np.all(
        points_in_polygon(
            all_uv[source_count:], polygon, tolerance=tolerance, include_boundary=False
        )
    ):
        errors.append("appended_points must lie strictly inside the mapped repair footprint")
    if new_points.shape[0]:
        errors.extend(
            _validate_point_placement_provenance(
                all_uv[source_count:], new_depth, point_provenance, tolerance, scale
            )
        )
    if method == "planar_cap" and new_points.shape[0]:
        depth_tolerance = max(tolerance * 32.0, scale * np.finfo(float).eps * 4096.0)
        if np.max(np.abs(new_depth), initial=0.0) > depth_tolerance:
            errors.append("planar_cap appended_points must lie on the fitted patch plane")
    elif method in {"curved_conformal_patch", "slit_bridge"} and new_points.shape[0]:
        boundary_depth = (boundary - origin) @ unit_normal
        margin = 0.5 * scale if method == "curved_conformal_patch" else _slit_width(polygon)
        if (
            margin is None
            or np.min(new_depth) < np.min(boundary_depth) - margin
            or np.max(new_depth) > np.max(boundary_depth) + margin
        ):
            errors.append(f"{method} appended_points exceed the local normal-depth envelope")

    face_uv = all_uv[new_faces]
    signed_double_area = _cross_2d(
        face_uv[:, 1] - face_uv[:, 0], face_uv[:, 2] - face_uv[:, 0]
    )
    expected_sign = -1.0 if polygon_area > 0.0 else 1.0
    area_tolerance = tolerance * scale
    if np.any(expected_sign * signed_double_area <= area_tolerance):
        errors.append("projected patch faces must have one non-folded footprint orientation")
    if method == "slit_bridge" and _maximum_projected_aspect(face_uv) > 25.0:
        errors.append("slit_bridge projected triangle aspect ratio exceeds the hard limit")
    projected_area = float(0.5 * np.sum(np.abs(signed_double_area)))
    area_sum_tolerance = max(
        abs(polygon_area) * 1.0e-10,
        tolerance * scale * max(int(new_faces.shape[0]), 1) * 4.0,
    )
    if abs(projected_area - abs(polygon_area)) > area_sum_tolerance:
        errors.append("projected patch area must equal the mapped repair footprint area")
    centroids = face_uv.mean(axis=1)
    if not np.all(
        points_in_polygon(centroids, polygon, tolerance=tolerance, include_boundary=False)
    ):
        errors.append("appended_faces extend outside the mapped repair footprint")

    internal_edges = _unique_internal_edges(new_faces, loop)
    if internal_edges.size:
        midpoints = 0.5 * (all_uv[internal_edges[:, 0]] + all_uv[internal_edges[:, 1]])
        if not np.all(
            points_in_polygon(midpoints, polygon, tolerance=tolerance, include_boundary=False)
        ):
            errors.append("patch interior edges leave the mapped repair footprint")
        if _any_edge_crosses_boundary(internal_edges, all_uv, loop, tolerance):
            errors.append("patch interior edges cross or overlap the mapped repair boundary")
    return errors


def _unique_internal_edges(faces: np.ndarray, loop: np.ndarray) -> np.ndarray:
    boundary_edges = {
        tuple(sorted((int(loop[index]), int(loop[(index + 1) % loop.size]))))
        for index in range(loop.size)
    }
    edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    unique = np.unique(edges, axis=0)
    return np.asarray(
        [edge for edge in unique if tuple(int(value) for value in edge) not in boundary_edges],
        dtype=np.int64,
    ).reshape(-1, 2)


def _any_edge_crosses_boundary(
    internal_edges: np.ndarray,
    all_uv: np.ndarray,
    loop: np.ndarray,
    tolerance: float,
) -> bool:
    boundary_edges = np.column_stack((loop, np.roll(loop, -1)))
    boundary_segments = all_uv[boundary_edges]
    boundary_min = boundary_segments.min(axis=1) - tolerance
    boundary_max = boundary_segments.max(axis=1) + tolerance
    for edge in internal_edges:
        segment = all_uv[edge]
        possible = np.flatnonzero(
            np.all(boundary_max >= segment.min(axis=0) - tolerance, axis=1)
            & np.all(boundary_min <= segment.max(axis=0) + tolerance, axis=1)
        )
        for boundary_index in possible:
            shared = set(int(value) for value in edge).intersection(
                int(value) for value in boundary_edges[boundary_index]
            )
            if segments_conflict(
                segment[0],
                segment[1],
                boundary_segments[boundary_index, 0],
                boundary_segments[boundary_index, 1],
                tolerance,
                allow_one_shared_endpoint=bool(shared),
            ):
                return True
    return False


def segments_conflict(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    tolerance: float,
    *,
    allow_one_shared_endpoint: bool,
) -> bool:
    """Return whether two segments intersect beyond one declared shared endpoint."""

    scale = max(float(np.linalg.norm(b - a)), float(np.linalg.norm(d - c)), tolerance)
    area_tolerance = tolerance * scale
    values = np.asarray(
        (_orient(a, b, c), _orient(a, b, d), _orient(c, d, a), _orient(c, d, b))
    )
    proper = bool(
        values[0] * values[1] < -(area_tolerance * area_tolerance)
        and values[2] * values[3] < -(area_tolerance * area_tolerance)
    )
    if proper:
        return True
    touches = (
        (abs(values[0]) <= area_tolerance and _on_segment(a, b, c, tolerance))
        or (abs(values[1]) <= area_tolerance and _on_segment(a, b, d, tolerance))
        or (abs(values[2]) <= area_tolerance and _on_segment(c, d, a, tolerance))
        or (abs(values[3]) <= area_tolerance and _on_segment(c, d, b, tolerance))
    )
    if not touches:
        return False
    if not allow_one_shared_endpoint:
        return True
    if np.all(np.abs(values) <= area_tolerance):
        axis = int(np.argmax(np.abs(b - a)))
        overlap = min(max(a[axis], b[axis]), max(c[axis], d[axis])) - max(
            min(a[axis], b[axis]), min(c[axis], d[axis])
        )
        return bool(overlap > tolerance)
    shared_points = [
        left
        for left in (a, b)
        for right in (c, d)
        if np.linalg.norm(left - right) <= tolerance
    ]
    if not shared_points:
        return True
    shared = shared_points[0]
    nonshared_internal = b if np.linalg.norm(a - shared) <= tolerance else a
    nonshared_boundary = d if np.linalg.norm(c - shared) <= tolerance else c
    return bool(
        _point_on_segment(nonshared_internal, c, d, tolerance)
        or _point_on_segment(nonshared_boundary, a, b, tolerance)
    )


def _perimeter_center(boundary: np.ndarray) -> np.ndarray | None:
    lengths = np.linalg.norm(np.roll(boundary, -1, axis=0) - boundary, axis=1)
    perimeter = float(lengths.sum())
    if not np.isfinite(perimeter) or perimeter <= 0.0:
        return None
    return np.sum(
        (boundary + np.roll(boundary, -1, axis=0)) * (0.5 * lengths[:, None]), axis=0
    ) / perimeter


def _parameterization_normal(normal: Mapping[str, Any]) -> np.ndarray | None:
    raw = normal.get("parameterization_normal", normal.get("oriented_normal", []))
    try:
        value = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        return None
    length = float(np.linalg.norm(value))
    return None if length <= 1.0e-30 else value / length


def _validate_point_placement_provenance(
    actual_uv: np.ndarray,
    actual_depth: np.ndarray,
    provenance: Mapping[str, Any],
    tolerance: float,
    scale: float,
) -> list[str]:
    try:
        claimed_uv = np.asarray(provenance.get("uv"), dtype=np.float64)
        claimed_depth = np.asarray(provenance.get("normal_offset"), dtype=np.float64)
    except (OverflowError, TypeError, ValueError):
        return ["point placement provenance cannot be reconstructed"]
    placement_tolerance = max(tolerance * 32.0, scale * np.finfo(float).eps * 4096.0)
    errors: list[str] = []
    if claimed_uv.shape != actual_uv.shape or not np.allclose(
        claimed_uv, actual_uv, rtol=1.0e-12, atol=placement_tolerance
    ):
        errors.append("point uv provenance does not match appended point geometry")
    if claimed_depth.shape != actual_depth.shape or not np.allclose(
        claimed_depth, actual_depth, rtol=1.0e-12, atol=placement_tolerance
    ):
        errors.append("point normal_offset provenance does not match appended point geometry")
    return errors


def _slit_width(polygon: np.ndarray) -> float | None:
    try:
        _, _, axes = np.linalg.svd(polygon - polygon.mean(axis=0), full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    spans = np.ptp((polygon - polygon.mean(axis=0)) @ axes.T, axis=0)
    width = float(np.min(spans))
    return width if np.isfinite(width) and width > 0.0 else None


def _maximum_projected_aspect(face_uv: np.ndarray) -> float:
    edges = np.stack(
        (
            face_uv[:, 1] - face_uv[:, 0],
            face_uv[:, 2] - face_uv[:, 1],
            face_uv[:, 0] - face_uv[:, 2],
        ),
        axis=1,
    )
    lengths = np.linalg.norm(edges, axis=2)
    double_area = np.abs(_cross_2d(edges[:, 0], -edges[:, 2]))
    altitude = double_area / np.maximum(lengths.max(axis=1), np.finfo(float).tiny)
    aspect = lengths.max(axis=1) / np.maximum(altitude, np.finfo(float).tiny)
    return float(aspect.max(initial=0.0))


def _read_vector(value: Any) -> np.ndarray | None:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (OverflowError, TypeError, ValueError):
        return None
    return vector if vector.shape == (3,) and np.all(np.isfinite(vector)) else None


def _cross_2d(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return left[:, 0] * right[:, 1] - left[:, 1] * right[:, 0]


def _orient(left: np.ndarray, middle: np.ndarray, right: np.ndarray) -> float:
    first, second = middle - left, right - left
    return float(first[0] * second[1] - first[1] * second[0])


def _on_segment(left: np.ndarray, right: np.ndarray, point: np.ndarray, tolerance: float) -> bool:
    return bool(
        np.all(point >= np.minimum(left, right) - tolerance)
        and np.all(point <= np.maximum(left, right) + tolerance)
    )


def _point_on_segment(point: np.ndarray, left: np.ndarray, right: np.ndarray, tolerance: float) -> bool:
    scale = max(float(np.linalg.norm(right - left)), tolerance)
    return bool(
        abs(_orient(left, right, point)) <= tolerance * scale
        and _on_segment(left, right, point, tolerance)
    )
