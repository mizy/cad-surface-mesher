from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from source_primary_boundary_curvature import (
    build_boundary_vertex_normals,
    estimate_boundary_curvature,
    select_boundary_owner_one_ring,
)
from source_primary_patch_contract import (
    BoundaryMapping, RegionId, normalize_region_id, validate_source_arrays,
)
from source_primary_patch_inputs import (
    geometric_tolerance, read_bounded_real, validate_boundary_cycle_graph,
)
from source_primary_patch_parameterization import (
    polygon_self_intersections,
    polygon_signed_area,
    triangle_quality,
    world_from_local,
)

__all__ = ["BoundaryFrame", "analyze_source_boundary", "triangle_quality", "world_from_local"]

@dataclass(frozen=True)
class BoundaryFrame:
    center: np.ndarray
    u_axis: np.ndarray
    v_axis: np.ndarray
    normal: np.ndarray
    boundary_uv: np.ndarray
    boundary_depth: np.ndarray
    footprint_diagonal: float
    geometric_tolerance: float


# @entry
def analyze_source_boundary(
    source_points: np.ndarray,
    source_faces: np.ndarray,
    source_triangle_index: np.ndarray,
    boundary_loop: np.ndarray | list[int],
    *,
    region_id: RegionId,
    boundary_role: str = "hole",
    expected_oriented_normal: np.ndarray | None = None,
    minimum_normal_alignment: float = 0.20,
    minimum_face_normal_consistency: float = -0.10,
) -> dict[str, Any]:
    """Validate, orient, and describe one immutable source boundary loop."""

    errors = validate_source_arrays(source_points, source_faces, source_triangle_index)
    if errors:
        return _failure("source_mesh_invalid", errors=errors)
    try:
        durable_region_id = normalize_region_id(region_id)
    except (TypeError, ValueError) as exc:
        return _failure("region_id_invalid", message=str(exc))
    if not isinstance(boundary_role, str) or boundary_role not in {"hole", "paired_loop"}:
        return _failure("boundary_role_invalid", boundary_role=str(boundary_role))
    parsed_minimum_normal_alignment, alignment_error = read_bounded_real(
        minimum_normal_alignment, name="minimum_normal_alignment", minimum=0.0, maximum=1.0
    )
    parsed_minimum_face_normal_consistency, consistency_error = read_bounded_real(
        minimum_face_normal_consistency,
        name="minimum_face_normal_consistency", minimum=-1.0, maximum=1.0,
    )
    if alignment_error or consistency_error:
        return _failure("boundary_normal_threshold_invalid", message=alignment_error or consistency_error)
    assert parsed_minimum_normal_alignment is not None
    assert parsed_minimum_face_normal_consistency is not None
    minimum_normal_alignment = parsed_minimum_normal_alignment
    minimum_face_normal_consistency = parsed_minimum_face_normal_consistency
    points = np.asarray(source_points, dtype=np.float64)
    faces = np.asarray(source_faces, dtype=np.int64)
    source_ids = np.asarray(source_triangle_index, dtype=np.int64)
    loop_result = _validate_and_orient_loop(points, faces, boundary_loop)
    if not loop_result["success"]:
        return loop_result
    loop = loop_result["loop"]
    edge_face_ids = loop_result["edge_face_ids"]
    loop_points = points[loop]
    edge_vectors = np.roll(loop_points, -1, axis=0) - loop_points
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    tolerance = geometric_tolerance(loop_points)
    if np.any(edge_lengths <= tolerance):
        return _failure("boundary_loop_zero_length_edge")
    perimeter = float(edge_lengths.sum())
    if perimeter <= 0.0:
        return _failure("boundary_perimeter_degenerate")
    edge_centers = 0.5 * (loop_points + np.roll(loop_points, -1, axis=0))
    center = np.sum(edge_centers * edge_lengths[:, None], axis=0) / perimeter
    centered = loop_points - center
    newell = np.sum(np.cross(centered, np.roll(centered, -1, axis=0)), axis=0)
    newell_length = float(np.linalg.norm(newell))
    if newell_length <= tolerance * tolerance:
        return _failure("boundary_newell_normal_degenerate")
    patch_from_boundary = -newell / newell_length
    try:
        _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        return _failure("boundary_pca_failed", message=str(exc))
    pca_normal = vh[-1]
    if float(np.dot(pca_normal, patch_from_boundary)) < 0.0:
        pca_normal = -pca_normal

    face_normals, face_double_areas = _face_normals(points, faces[edge_face_ids])
    weighted = np.sum(face_normals * face_double_areas[:, None], axis=0)
    weighted_length = float(np.linalg.norm(weighted))
    if weighted_length <= tolerance * tolerance and boundary_role == "hole":
        return _failure("boundary_adjacent_normal_degenerate")
    adjacent_normal = weighted / weighted_length if weighted_length > tolerance**2 else face_normals[0]
    alignment = float(np.dot(adjacent_normal, patch_from_boundary))
    face_alignment = face_normals @ adjacent_normal
    consecutive_alignment = np.sum(
        face_normals * np.roll(face_normals, -1, axis=0), axis=1
    )
    orientation_reliable = bool(
        (boundary_role == "paired_loop" or alignment >= minimum_normal_alignment)
        and float(
            (
                consecutive_alignment
                if boundary_role == "paired_loop"
                else face_alignment
            ).min(initial=1.0)
        )
        >= minimum_face_normal_consistency
    )
    if not orientation_reliable:
        return _failure(
            "boundary_normal_evidence_conflict",
            boundary_patch_normal=patch_from_boundary.tolist(),
            adjacent_area_weighted_normal=adjacent_normal.tolist(),
            normal_alignment=alignment,
            minimum_normal_alignment=float(minimum_normal_alignment),
            adjacent_face_normal_dot_min=float(face_alignment.min(initial=1.0)),
            minimum_face_normal_consistency=float(minimum_face_normal_consistency),
        )
    frame_normal = patch_from_boundary if boundary_role == "paired_loop" else _unit(
        patch_from_boundary + adjacent_normal + pca_normal
    )
    if frame_normal is None:
        return _failure("boundary_consensus_normal_degenerate")
    external_normal = None
    external_alignment = None
    if expected_oriented_normal is not None:
        try:
            raw_expected = np.asarray(expected_oriented_normal)
        except (OverflowError, TypeError, ValueError) as exc:
            return _failure("boundary_external_normal_invalid", message=str(exc))
        if (
            raw_expected.shape != (3,)
            or not np.issubdtype(raw_expected.dtype, np.number)
            or np.iscomplexobj(raw_expected)
        ):
            return _failure("boundary_external_normal_invalid")
        expected = raw_expected.astype(np.float64, copy=False)
        external_normal = _unit(expected) if expected.shape == (3,) else None
        if external_normal is None:
            return _failure("boundary_external_normal_invalid")
        reference = adjacent_normal if boundary_role == "paired_loop" else frame_normal
        external_alignment = float(np.dot(external_normal, reference))
        strong_alignment_threshold = max(0.80, minimum_normal_alignment)
        if external_alignment < strong_alignment_threshold:
            return _failure(
                "boundary_external_normal_conflicts_source_winding",
                external_alignment=external_alignment,
                minimum_normal_alignment=float(strong_alignment_threshold),
            )
    u_axis = _deterministic_tangent(vh[0], frame_normal)
    if u_axis is None:
        return _failure("boundary_local_frame_degenerate")
    v_axis = np.cross(frame_normal, u_axis)
    local = np.column_stack((centered @ u_axis, centered @ v_axis, centered @ frame_normal))
    boundary_uv = local[:, :2]
    signed_area = polygon_signed_area(boundary_uv)
    if signed_area >= -(tolerance * tolerance):
        return _failure(
            "boundary_orientation_frame_conflict",
            projected_signed_area=float(signed_area),
        )
    intersections = polygon_self_intersections(boundary_uv, tolerance)
    if intersections:
        return _failure("boundary_projection_self_intersects", segment_pairs=intersections[:100])
    uv_span = np.ptp(boundary_uv, axis=0)
    footprint_diagonal = float(np.linalg.norm(uv_span))
    if footprint_diagonal <= tolerance:
        return _failure("boundary_footprint_degenerate")
    selected_faces, owner_error = select_boundary_owner_one_ring(
        faces,
        loop,
        edge_face_ids,
        points=points,
        minimum_face_normal_consistency=minimum_face_normal_consistency,
    )
    if owner_error is not None or selected_faces is None:
        return _failure(owner_error or "boundary_edge_owner_component_invalid")
    evidence_normal = adjacent_normal if boundary_role == "paired_loop" else frame_normal
    vertex_normals = build_boundary_vertex_normals(
        points, faces, loop, selected_faces, evidence_normal
    )
    curvature = estimate_boundary_curvature(
        points,
        faces,
        selected_faces,
        center,
        u_axis,
        v_axis,
        frame_normal,
        vertex_normals,
    )
    planarity = float(np.max(np.abs(local[:, 2]), initial=0.0) / footprint_diagonal)
    frame = BoundaryFrame(
        center=_readonly(center),
        u_axis=_readonly(u_axis),
        v_axis=_readonly(v_axis),
        normal=_readonly(frame_normal),
        boundary_uv=_readonly(boundary_uv),
        boundary_depth=_readonly(local[:, 2]),
        footprint_diagonal=footprint_diagonal,
        geometric_tolerance=tolerance,
    )
    mapping = BoundaryMapping(
        region_id=durable_region_id,
        source_vertex_ids=tuple(int(value) for value in loop),
        candidate_vertex_ids=tuple(int(value) for value in loop),
        source_edge_face_ids=tuple(int(value) for value in edge_face_ids),
        source_triangle_indices=tuple(int(value) for value in source_ids[edge_face_ids]),
    )
    normal_evidence = {
        "status": "computed",
        "method": "external_checked_source_winding_newell_pca_area_weighted_consensus"
        if external_normal is not None
        else "source_winding_newell_pca_area_weighted_consensus",
        "oriented_normal": evidence_normal.tolist(),
        "parameterization_normal": frame_normal.tolist(),
        "parameterization_origin": center.tolist(),
        "parameterization_u_axis": u_axis.tolist(),
        "parameterization_v_axis": v_axis.tolist(),
        "boundary_induced_patch_normal": patch_from_boundary.tolist(),
        "pca_normal": pca_normal.tolist(),
        "adjacent_area_weighted_normal": adjacent_normal.tolist(),
        "boundary_adjacent_normal_dot": alignment,
        "adjacent_face_normal_dot_min": float(face_alignment.min(initial=1.0)),
        "adjacent_face_normal_dot_mean": float(face_alignment.mean()),
        "adjacent_consecutive_normal_dot_min": float(
            consecutive_alignment.min(initial=1.0)
        ),
        "boundary_vertex_normals": vertex_normals.tolist(),
        "orientation_reliable": True,
        "external_orientation_supplied": external_normal is not None,
        "external_orientation_alignment": external_alignment,
        "external_orientation_strongly_consistent": external_normal is not None,
        "external_orientation_strong_alignment_threshold": max(0.80, float(minimum_normal_alignment)),
        "external_oriented_normal": (
            external_normal.tolist() if external_normal is not None else None
        ),
        "chart_uses_external_orientation": False,
        "orientation": (
            "source_surface_field_for_paired_loop"
            if boundary_role == "paired_loop"
            else "opposite_source_boundary_traversal"
        ),
    }
    return {
        "success": True,
        "failure_reason_codes": [],
        "loop": _readonly(loop, np.dtype(np.int64)),
        "edge_face_ids": _readonly(edge_face_ids, np.dtype(np.int64)),
        "mapping": mapping,
        "frame": frame,
        "normal": normal_evidence,
        "curvature": curvature,
        "diagnostics": {
            "stage": "source_primary_boundary_analysis",
            "region_id": durable_region_id,
            "boundary_role": boundary_role,
            "boundary_vertex_count": int(loop.size),
            "perimeter": perimeter,
            "projected_signed_area": float(signed_area),
            "projected_area": float(abs(signed_area)),
            "footprint_diagonal": footprint_diagonal,
            "planarity_ratio": planarity,
            "max_boundary_plane_distance": float(np.max(np.abs(local[:, 2]), initial=0.0)),
            "singular_values": singular_values.tolist(),
            "geometric_tolerance": tolerance,
        },
    }
def _validate_and_orient_loop(
    points: np.ndarray, faces: np.ndarray, boundary_loop: np.ndarray | list[int]
) -> dict[str, Any]:
    try:
        raw = np.asarray(boundary_loop)
    except (OverflowError, TypeError, ValueError) as exc:
        return _failure("boundary_loop_invalid_shape", message=str(exc))
    if raw.ndim != 1 or raw.size < 3 or not np.issubdtype(raw.dtype, np.integer):
        return _failure("boundary_loop_invalid_shape")
    loop = raw.astype(np.int64, copy=True)
    if loop.size > 1 and loop[0] == loop[-1]:
        loop = loop[:-1]
    if loop.size < 3 or np.unique(loop).size != loop.size:
        return _failure("boundary_loop_repeats_vertex")
    if np.min(loop) < 0 or np.max(loop) >= points.shape[0]:
        return _failure("boundary_loop_vertex_out_of_range")
    graph_error, graph_diagnostics = validate_boundary_cycle_graph(faces, loop)
    if graph_error is not None:
        return _failure(graph_error, **graph_diagnostics)
    edge_faces = _loop_edge_face_map(faces, loop, points.shape[0])
    directions: list[int] = []
    for index, left in enumerate(loop):
        left_id = int(left)
        right = int(loop[(index + 1) % loop.size])
        edge = (left_id, right) if left_id < right else (right, left_id)
        rows = edge_faces.get(edge, [])
        if len(rows) != 1:
            return _failure(
                "boundary_loop_edge_incidence_not_one",
                edge=[left_id, right],
                incidence=len(rows),
            )
        directions.append(
            1 if _face_has_directed_edge(faces[rows[0]], left_id, right) else -1
        )
    if all(value == -1 for value in directions):
        loop = loop[::-1].copy()
    elif not all(value == 1 for value in directions):
        return _failure("boundary_loop_winding_inconsistent")
    edge_face_ids = []
    for index, left in enumerate(loop):
        left_id = int(left)
        right = int(loop[(index + 1) % loop.size])
        edge = (left_id, right) if left_id < right else (right, left_id)
        edge_face_ids.append(edge_faces[edge][0])
    return {
        "success": True,
        "loop": loop,
        "edge_face_ids": np.asarray(edge_face_ids, dtype=np.int64),
    }
def _face_normals(points: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = np.cross(
        points[faces[:, 1]] - points[faces[:, 0]],
        points[faces[:, 2]] - points[faces[:, 0]],
    )
    lengths = np.linalg.norm(raw, axis=1)
    unit = np.divide(raw, lengths[:, None], out=np.zeros_like(raw), where=lengths[:, None] > 0.0)
    return unit, lengths


def _loop_edge_face_map(
    faces: np.ndarray, loop: np.ndarray, point_count: int
) -> dict[tuple[int, int], list[int]]:
    directed = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    canonical = np.sort(directed, axis=1)
    keys = canonical[:, 0] * point_count + canonical[:, 1]
    following = np.roll(loop, -1)
    wanted_edges = np.sort(np.column_stack((loop, following)), axis=1)
    wanted_keys = wanted_edges[:, 0] * point_count + wanted_edges[:, 1]
    matched = np.flatnonzero(np.isin(keys, wanted_keys))
    face_ids = np.repeat(np.arange(faces.shape[0], dtype=np.int64), 3)
    result: dict[tuple[int, int], list[int]] = {}
    for occurrence in matched:
        edge = (
            int(canonical[occurrence, 0]),
            int(canonical[occurrence, 1]),
        )
        result.setdefault(edge, []).append(int(face_ids[occurrence]))
    return result


def _face_has_directed_edge(face: np.ndarray, left: int, right: int) -> bool:
    return any(
        int(face[index]) == left and int(face[(index + 1) % 3]) == right
        for index in range(3)
    )


def _deterministic_tangent(candidate: np.ndarray, normal: np.ndarray) -> np.ndarray | None:
    raw_tangent = np.asarray(candidate, dtype=np.float64) - float(
        np.dot(candidate, normal)
    ) * normal
    tangent = _unit(raw_tangent)
    if tangent is None:
        axis = np.eye(3)[int(np.argmin(np.abs(normal)))]
        tangent = _unit(axis - float(np.dot(axis, normal)) * normal)
    if tangent is None:
        return None
    pivot = int(np.argmax(np.abs(tangent)))
    return -tangent if tangent[pivot] < 0.0 else tangent


def _unit(value: np.ndarray) -> np.ndarray | None:
    length = float(np.linalg.norm(value))
    return None if not np.isfinite(length) or length <= 1e-30 else np.asarray(value) / length


def _readonly(
    value: np.ndarray, dtype: np.dtype[Any] = np.dtype(np.float64)
) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _failure(code: str, **diagnostics: Any) -> dict[str, Any]:
    return {
        "success": False,
        "failure_reason_codes": [code],
        "diagnostics": {"stage": "source_primary_boundary_analysis", **diagnostics},
    }
