from __future__ import annotations

from typing import Any

import numpy as np

from source_primary_patch_footprint import segments_conflict
from source_primary_patch_inputs import geometric_tolerance
from source_primary_patch_parameterization import (
    points_in_polygon,
    polygon_self_intersections,
    polygon_signed_area,
)


def build_multi_loop_common_plane(
    points: np.ndarray,
    analyses: list[dict[str, Any]],
    source_normals: np.ndarray,
    maximum_deviation_ratio: float,
) -> dict[str, Any]:
    vertex_ids = np.unique(
        np.concatenate(
            [np.asarray(analysis["loop"], dtype=np.int64) for analysis in analyses]
        )
    )
    boundary = points[vertex_ids]
    origin = boundary.mean(axis=0)
    centered = boundary - origin
    try:
        _, singular, axes = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        return _failure("multi_loop_common_plane_failed", message=str(exc))
    normal = axes[-1]
    source_reference = source_normals.sum(axis=0)
    reference_length = float(np.linalg.norm(source_reference))
    if reference_length <= 1.0e-30:
        return _failure("multi_loop_boundary_normal_conflict")
    source_reference /= reference_length
    if float(np.dot(normal, source_reference)) < 0.0:
        normal = -normal
    axis = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(normal)))]
    tangent = axis - float(np.dot(axis, normal)) * normal
    tangent_length = float(np.linalg.norm(tangent))
    if tangent_length <= 1.0e-30:
        return _failure("multi_loop_common_plane_failed")
    tangent /= tangent_length
    bitangent = np.cross(normal, tangent)
    scale = max(float(np.linalg.norm(np.ptp(boundary, axis=0))), 1.0e-30)
    tolerance = geometric_tolerance(boundary)
    maximum_deviation = float(np.max(np.abs(centered @ normal), initial=0.0))
    deviation_ratio = maximum_deviation / scale
    diagnostics = {
        "method": "combined_boundary_pca_with_deterministic_world_axis_tangent",
        "origin": origin.tolist(),
        "normal": normal.tolist(),
        "u_axis": tangent.tolist(),
        "v_axis": bitangent.tolist(),
        "singular_values": singular.tolist(),
        "scale": scale,
        "geometric_tolerance": tolerance,
        "maximum_plane_deviation": maximum_deviation,
        "maximum_plane_deviation_ratio": deviation_ratio,
        "maximum_plane_deviation_ratio_threshold": maximum_deviation_ratio,
    }
    if deviation_ratio > maximum_deviation_ratio:
        return _failure("multi_loop_boundaries_non_coplanar", **diagnostics)
    return {
        "success": True,
        "origin": origin,
        "normal": normal,
        "u_axis": tangent,
        "v_axis": bitangent,
        "tolerance": tolerance,
        "diagnostics": diagnostics,
    }


def resolve_multi_loop_nesting(
    points: np.ndarray,
    analyses: list[dict[str, Any]],
    plane: dict[str, Any],
) -> dict[str, Any]:
    loop_uv = [
        _project(points[np.asarray(analysis["loop"], dtype=np.int64)], plane)
        for analysis in analyses
    ]
    tolerance = float(plane["tolerance"])
    scales = [
        max(float(np.linalg.norm(np.ptp(loop, axis=0))), tolerance)
        for loop in loop_uv
    ]
    for index, polygon in enumerate(loop_uv):
        if (
            abs(polygon_signed_area(polygon)) <= tolerance * scales[index]
            or polygon_self_intersections(polygon, tolerance)
        ):
            return _failure("multi_loop_projection_invalid", boundary_index=index)
    if _boundaries_conflict(loop_uv[0], loop_uv[1], tolerance):
        return _failure("multi_loop_boundaries_touch_or_cross")
    first_contains_second = bool(
        np.all(
            points_in_polygon(
                loop_uv[1],
                loop_uv[0],
                tolerance=tolerance,
                include_boundary=False,
            )
        )
    )
    second_contains_first = bool(
        np.all(
            points_in_polygon(
                loop_uv[0],
                loop_uv[1],
                tolerance=tolerance,
                include_boundary=False,
            )
        )
    )
    if first_contains_second == second_contains_first:
        return _failure(
            "multi_loop_relationship_ambiguous",
            first_contains_second=first_contains_second,
            second_contains_first=second_contains_first,
        )
    outer_index = 0 if first_contains_second else 1
    return {
        "success": True,
        "outer_index": outer_index,
        "loop_uv": loop_uv,
        "diagnostics": {
            "method": "strict_vertex_containment_and_nonintersecting_boundaries",
            "outer_input_index": outer_index,
            "inner_input_index": 1 - outer_index,
            "unique_ownership": True,
        },
    }


def triangulate_multi_loop_annulus(
    analyses: list[dict[str, Any]],
    outer_uv: np.ndarray,
    inner_uv: np.ndarray,
    tolerance: float,
) -> dict[str, Any]:
    try:
        from scipy.spatial import Delaunay, QhullError
    except ImportError as exc:
        return _failure("multi_loop_delaunay_unavailable", message=str(exc))
    outer_ids = np.asarray(analyses[0]["loop"], dtype=np.int64)
    inner_ids = np.asarray(analyses[1]["loop"], dtype=np.int64)
    vertex_ids = np.concatenate((outer_ids, inner_ids))
    uv = np.vstack((outer_uv, inner_uv))
    try:
        raw_faces = Delaunay(uv, qhull_options="Qbb Qc Qz Q12").simplices.astype(
            np.int64, copy=False
        )
    except QhullError as exc:
        return _failure("multi_loop_delaunay_failed", message=str(exc))
    triangles = uv[raw_faces]
    signed_double_area = _cross_rows(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    centroids = triangles.mean(axis=1)
    selected = (
        (np.abs(signed_double_area) > tolerance * tolerance)
        & points_in_polygon(
            centroids, outer_uv, tolerance=tolerance, include_boundary=False
        )
        & ~points_in_polygon(
            centroids, inner_uv, tolerance=tolerance, include_boundary=True
        )
    )
    local_faces = raw_faces[selected].copy()
    selected_area = signed_double_area[selected]
    if local_faces.shape[0] == 0:
        return _failure("multi_loop_delaunay_no_annulus_faces")
    local_faces[selected_area < 0.0] = local_faces[selected_area < 0.0][:, [0, 2, 1]]
    faces = _sort_oriented_faces(vertex_ids[local_faces])
    audit = _audit_annulus_faces(
        faces,
        vertex_ids,
        uv,
        outer_ids,
        inner_ids,
        outer_uv,
        inner_uv,
        tolerance,
    )
    if not audit["success"]:
        return {
            "success": False,
            "reason_code": audit["reason_code"],
            "diagnostics": {
                "raw_delaunay_faces": int(raw_faces.shape[0]),
                "selected_faces": int(faces.shape[0]),
                **audit["diagnostics"],
            },
        }
    return {
        "success": True,
        "faces": faces,
        "diagnostics": {
            "method": "source_fixed_planar_delaunay_outer_minus_inner",
            "raw_delaunay_faces": int(raw_faces.shape[0]),
            "selected_faces": int(faces.shape[0]),
            "appended_points": 0,
            "outer_boundary_vertices": int(outer_ids.size),
            "inner_boundary_vertices": int(inner_ids.size),
            "source_boundaries_resampled": False,
            "boundary_constraints_preserved": True,
            "inner_island_filled": False,
            **audit["diagnostics"],
        },
    }


def _audit_annulus_faces(
    faces: np.ndarray,
    vertex_ids: np.ndarray,
    uv: np.ndarray,
    outer_ids: np.ndarray,
    inner_ids: np.ndarray,
    outer_uv: np.ndarray,
    inner_uv: np.ndarray,
    tolerance: float,
) -> dict[str, Any]:
    uv_by_id = {
        int(vertex_id): uv[index] for index, vertex_id in enumerate(vertex_ids)
    }
    face_uv = np.asarray(
        [[uv_by_id[int(vertex_id)] for vertex_id in face] for face in faces],
        dtype=np.float64,
    )
    signed = _cross_rows(
        face_uv[:, 1] - face_uv[:, 0], face_uv[:, 2] - face_uv[:, 0]
    )
    centroids = face_uv.mean(axis=1)
    inside = points_in_polygon(
        centroids, outer_uv, tolerance=tolerance, include_boundary=False
    ) & ~points_in_polygon(
        centroids, inner_uv, tolerance=tolerance, include_boundary=True
    )
    outer_area = abs(polygon_signed_area(outer_uv))
    inner_area = abs(polygon_signed_area(inner_uv))
    expected_area = outer_area - inner_area
    actual_area = 0.5 * float(np.sum(signed))
    area_tolerance = max(
        expected_area * 1.0e-10, tolerance * max(outer_area, 1.0) * 8.0
    )
    if np.any(signed <= tolerance * tolerance) or not np.all(inside):
        return _failure(
            "multi_loop_patch_footprint_invalid",
            outside_centroid_count=int(np.count_nonzero(~inside)),
        )
    if expected_area <= area_tolerance or abs(actual_area - expected_area) > area_tolerance:
        return _failure(
            "multi_loop_patch_area_mismatch",
            expected_area=expected_area,
            actual_area=actual_area,
            tolerance=area_tolerance,
        )
    topology = _audit_annulus_topology(faces, outer_ids, inner_ids)
    if not topology["success"]:
        return topology
    if _internal_edges_leave_annulus(
        faces, uv_by_id, outer_ids, inner_ids, outer_uv, inner_uv, tolerance
    ):
        return _failure("multi_loop_patch_footprint_invalid")
    return {
        "success": True,
        "diagnostics": {
            "projected_outer_area": outer_area,
            "projected_inner_island_area": inner_area,
            "projected_patch_area": actual_area,
            "projected_area_tolerance": area_tolerance,
            **topology["diagnostics"],
        },
    }


def _audit_annulus_topology(
    faces: np.ndarray, outer_ids: np.ndarray, inner_ids: np.ndarray
) -> dict[str, Any]:
    occurrences: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for face_id, face in enumerate(faces):
        for index in range(3):
            left = int(face[index])
            right = int(face[(index + 1) % 3])
            occurrences.setdefault(_edge_key(left, right), []).append(
                (face_id, left, right)
            )
    expected: dict[tuple[int, int], tuple[int, int]] = {}
    for loop in (outer_ids, inner_ids):
        for index, left_value in enumerate(loop):
            left = int(left_value)
            right = int(loop[(index + 1) % loop.size])
            expected[_edge_key(left, right)] = (right, left)
    boundary_errors = []
    interior_errors = []
    for edge, direction in expected.items():
        rows = occurrences.get(edge, [])
        if len(rows) != 1 or (rows[0][1], rows[0][2]) != direction:
            boundary_errors.append(list(edge))
    for edge, rows in occurrences.items():
        if edge in expected:
            continue
        if len(rows) != 2 or (rows[0][1], rows[0][2]) != (rows[1][2], rows[1][1]):
            interior_errors.append(list(edge))
    adjacency: list[set[int]] = [set() for _ in range(faces.shape[0])]
    for rows in occurrences.values():
        if len(rows) == 2:
            left, right = rows[0][0], rows[1][0]
            adjacency[left].add(right)
            adjacency[right].add(left)
    reached = {0}
    pending = [0]
    while pending:
        face_id = pending.pop()
        for neighbor in adjacency[face_id] - reached:
            reached.add(neighbor)
            pending.append(neighbor)
    vertex_count = int(np.unique(faces).size)
    euler_characteristic = vertex_count - len(occurrences) + int(faces.shape[0])
    if (
        boundary_errors
        or interior_errors
        or len(reached) != faces.shape[0]
        or euler_characteristic != 0
    ):
        return _failure(
            "multi_loop_patch_topology_invalid",
            boundary_constraint_errors=boundary_errors[:100],
            interior_edge_errors=interior_errors[:100],
            connected_face_count=len(reached),
            face_count=int(faces.shape[0]),
            euler_characteristic=euler_characteristic,
        )
    return {
        "success": True,
        "diagnostics": {
            "edge_count": len(occurrences),
            "boundary_edge_count": len(expected),
            "euler_characteristic": euler_characteristic,
            "connected_face_count": len(reached),
        },
    }


def _internal_edges_leave_annulus(
    faces: np.ndarray,
    uv_by_id: dict[int, np.ndarray],
    outer_ids: np.ndarray,
    inner_ids: np.ndarray,
    outer_uv: np.ndarray,
    inner_uv: np.ndarray,
    tolerance: float,
) -> bool:
    boundary_edges = _loop_edges(outer_ids) | _loop_edges(inner_ids)
    all_edges = {
        _edge_key(int(face[index]), int(face[(index + 1) % 3]))
        for face in faces
        for index in range(3)
    }
    boundary_segments = [
        (left, right, uv_by_id[left], uv_by_id[right])
        for left, right in sorted(boundary_edges)
    ]
    for edge in sorted(all_edges - boundary_edges):
        segment = np.asarray([uv_by_id[edge[0]], uv_by_id[edge[1]]])
        midpoint = segment.mean(axis=0, keepdims=True)
        if not bool(
            points_in_polygon(
                midpoint, outer_uv, tolerance=tolerance, include_boundary=False
            )[0]
        ) or bool(
            points_in_polygon(
                midpoint, inner_uv, tolerance=tolerance, include_boundary=True
            )[0]
        ):
            return True
        for left, right, start, end in boundary_segments:
            shared = bool({edge[0], edge[1]}.intersection((left, right)))
            if segments_conflict(
                segment[0],
                segment[1],
                start,
                end,
                tolerance,
                allow_one_shared_endpoint=shared,
            ):
                return True
    return False


def _boundaries_conflict(left: np.ndarray, right: np.ndarray, tolerance: float) -> bool:
    for left_index in range(left.shape[0]):
        for right_index in range(right.shape[0]):
            if segments_conflict(
                left[left_index],
                left[(left_index + 1) % left.shape[0]],
                right[right_index],
                right[(right_index + 1) % right.shape[0]],
                tolerance,
                allow_one_shared_endpoint=False,
            ):
                return True
    return False


def _project(points: np.ndarray, plane: dict[str, Any]) -> np.ndarray:
    local = points - plane["origin"]
    return np.column_stack((local @ plane["u_axis"], local @ plane["v_axis"]))


def _sort_oriented_faces(faces: np.ndarray) -> np.ndarray:
    minimum_position = np.argmin(faces, axis=1)
    offsets = (minimum_position[:, None] + np.arange(3)) % 3
    rotated = np.take_along_axis(faces, offsets, axis=1)
    order = np.lexsort((rotated[:, 2], rotated[:, 1], rotated[:, 0]))
    return rotated[order]


def _loop_edges(loop: np.ndarray) -> set[tuple[int, int]]:
    return {
        _edge_key(int(loop[index]), int(loop[(index + 1) % loop.size]))
        for index in range(loop.size)
    }


def _edge_key(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)


def _cross_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return left[:, 0] * right[:, 1] - left[:, 1] * right[:, 0]


def _failure(code: str, **diagnostics: Any) -> dict[str, Any]:
    return {"success": False, "reason_code": code, "diagnostics": diagnostics}
