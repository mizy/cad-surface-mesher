from __future__ import annotations

from typing import Any

import numpy as np

from mesh_metrics import edge_topology, face_component_labels, inconsistent_winding_edges


def audit_boundary_sharing(
    current_points: np.ndarray,
    interior_points: np.ndarray,
    patch_faces: np.ndarray,
    boundary_edges: set[tuple[int, int]],
    boundary_vertices: np.ndarray,
    current_edges: dict[tuple[int, int], list[tuple[int, int]]],
    patch_edges: dict[tuple[int, int], list[tuple[int, int]]],
    tolerance: float,
) -> dict[str, Any]:
    current_count = current_points.shape[0]
    used_existing = set(int(value) for value in patch_faces[patch_faces < current_count])
    declared = set(int(value) for value in boundary_vertices)
    wrong_source_refs = sorted(used_existing - declared)
    missing_edges = []
    direction_errors = []
    for edge in sorted(boundary_edges):
        source_occurrences = current_edges.get(edge, [])
        patch_occurrences = patch_edges.get(edge, [])
        if len(source_occurrences) != 1 or len(patch_occurrences) != 1:
            missing_edges.append(
                {
                    "edge": list(edge),
                    "source_incidence": len(source_occurrences),
                    "patch_incidence": len(patch_occurrences),
                }
            )
        elif source_occurrences[0][1] == patch_occurrences[0][1]:
            direction_errors.append(list(edge))
    patch_edge_errors = [
        {"edge": list(edge), "incidence": len(values)}
        for edge, values in patch_edges.items()
        if (edge in boundary_edges and len(values) != 1)
        or (edge not in boundary_edges and len(values) != 2)
    ]
    new_ids = set(range(current_count, current_count + interior_points.shape[0]))
    used_new = set(int(value) for value in patch_faces[patch_faces >= current_count])
    dangling_new = sorted(new_ids - used_new)
    duplicated_boundary_points = 0
    if interior_points.size and boundary_vertices.size:
        boundary_coordinates = np.asarray(current_points[boundary_vertices], dtype=np.float64)
        for point in np.asarray(interior_points, dtype=np.float64):
            duplicated_boundary_points += int(
                np.any(np.linalg.norm(boundary_coordinates - point, axis=1) <= tolerance)
            )
    passed = not (
        wrong_source_refs
        or missing_edges
        or direction_errors
        or patch_edge_errors
        or dangling_new
        or duplicated_boundary_points
    )
    return build_gate(
        passed,
        "boundary_index_not_shared",
        {
            "declared_boundary_edges": len(boundary_edges),
            "wrong_existing_vertex_ids": wrong_source_refs[:100],
            "edge_incidence_errors": missing_edges[:100],
            "edge_direction_errors": direction_errors[:100],
            "patch_edge_incidence_errors": patch_edge_errors[:100],
            "dangling_interior_vertex_ids": dangling_new[:100],
            "interior_points_duplicating_boundary": duplicated_boundary_points,
        },
        {
            "source_and_patch_incidence": 1,
            "combined_incidence": 2,
            "opposite_edge_direction": True,
            "interior_edge_incidence": 2,
            "duplicated_boundary_points": 0,
        },
    )


def audit_interior_footprint(
    source_points: np.ndarray,
    interior_points: np.ndarray,
    loops: tuple[np.ndarray, ...],
    tolerance: float,
) -> dict[str, Any]:
    """Require every new point to project strictly inside one hole footprint."""

    interior = np.asarray(interior_points, dtype=np.float64)
    if not interior.shape[0]:
        return build_gate(True, "patch_interior_point_outside_footprint", 0, 0)
    if len(loops) != 1:
        return build_gate(
            False,
            "patch_interior_footprint_ambiguous",
            {"interior_points": int(interior.shape[0]), "boundary_loops": len(loops)},
            {"boundary_loops": 1},
        )
    boundary = np.asarray(source_points, dtype=np.float64)[loops[0]]
    center = boundary.mean(axis=0)
    _, _, vh = np.linalg.svd(boundary - center, full_matrices=False)
    axes = vh[:2].T
    polygon = (boundary - center) @ axes
    queries = (interior - center) @ axes
    inside = np.asarray([_point_strictly_inside_polygon(point, polygon, tolerance) for point in queries])
    outside_ids = np.flatnonzero(~inside)
    return build_gate(
        bool(np.all(inside)),
        "patch_interior_point_outside_footprint",
        {
            "point_count": int(interior.shape[0]),
            "outside_point_count": int(outside_ids.size),
            "outside_point_ids": outside_ids[:100].astype(int).tolist(),
        },
        {"outside_point_count": 0, "boundary_policy": "strictly_inside"},
    )


def audit_topology_delta(
    before_faces: np.ndarray,
    after_faces: np.ndarray,
    boundary_edge_count: int,
) -> dict[str, Any]:
    before = count_topology(before_faces)
    after = count_topology(after_faces)
    expected_boundary = before["boundary_edges"] - boundary_edge_count
    passed = (
        after["boundary_edges"] == expected_boundary
        and after["non_manifold_edges"] <= before["non_manifold_edges"]
        and after["inconsistent_winding_edges"] <= before["inconsistent_winding_edges"]
        and after["components"] <= before["components"]
    )
    return build_gate(
        passed,
        "patch_topology_delta_unexpected",
        {"before": before, "after": after},
        {
            "boundary_edges": expected_boundary,
            "non_manifold_edges_max": before["non_manifold_edges"],
            "inconsistent_winding_edges_max": before["inconsistent_winding_edges"],
            "components_max": before["components"],
        },
    )


def audit_patch_internal_continuity(
    points: np.ndarray,
    patch_faces: np.ndarray,
    boundary_edges: set[tuple[int, int]],
    patch_edges: dict[tuple[int, int], list[tuple[int, int]]],
    minimum_normal_dot: float,
    maximum_normal_transition_degrees: float,
    maximum_curvature_jump: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    normals = calculate_triangle_geometry(points, patch_faces)["normals"]
    internal_edges = [
        (edge, occurrences)
        for edge, occurrences in sorted(patch_edges.items())
        if edge not in boundary_edges
    ]
    incidence_errors = []
    invalid_edges = []
    normal_dots = []
    angles = []
    edge_lengths = []
    edge_curvatures = []
    normalized_jumps = []
    for edge, occurrences in internal_edges:
        if len(occurrences) != 2:
            incidence_errors.append({"edge": list(edge), "incidence": len(occurrences)})
            continue
        first_face_id, second_face_id = occurrences[0][0], occurrences[1][0]
        dot = float(np.dot(normals[first_face_id], normals[second_face_id]))
        angle = calculate_normal_angle(normals[first_face_id], normals[second_face_id])
        edge_length = float(np.linalg.norm(points[edge[0]] - points[edge[1]]))
        if not np.isfinite(dot + angle + edge_length) or edge_length <= 1e-30:
            invalid_edges.append(
                {
                    "edge": list(edge),
                    "faces": [first_face_id, second_face_id],
                    "normal_dot": dot,
                    "dihedral_radians": angle,
                    "edge_length": edge_length,
                }
            )
            continue
        edge_curvature = angle / edge_length
        normal_dots.append(dot)
        angles.append(angle)
        edge_lengths.append(edge_length)
        edge_curvatures.append(edge_curvature)
        normalized_jumps.append(edge_curvature * edge_length)

    dots = np.asarray(normal_dots, dtype=np.float64)
    angle_values = np.asarray(angles, dtype=np.float64)
    angle_degrees = np.degrees(angle_values)
    length_values = np.asarray(edge_lengths, dtype=np.float64)
    curvature_values = np.asarray(edge_curvatures, dtype=np.float64)
    jump_values = np.asarray(normalized_jumps, dtype=np.float64)
    complete = (
        not incidence_errors
        and not invalid_edges
        and dots.size == len(internal_edges)
    )
    normal_gate = build_gate(
        complete
        and bool(np.all(dots >= minimum_normal_dot))
        and bool(np.all(angle_degrees <= maximum_normal_transition_degrees)),
        "patch_internal_normal_transition_failed",
        {
            "internal_edges": len(internal_edges),
            "checked_edges": int(dots.size),
            "incidence_errors": incidence_errors[:100],
            "invalid_edges": invalid_edges[:100],
            "minimum_dot": float(dots.min()) if dots.size else None,
            "maximum_degrees": float(angle_degrees.max())
            if angle_degrees.size
            else None,
        },
        {
            "required_edges": len(internal_edges),
            "incidence": 2,
            "minimum_dot": float(minimum_normal_dot),
            "maximum_degrees": float(maximum_normal_transition_degrees),
        },
    )
    curvature_gate = build_gate(
        complete
        and jump_values.size == len(internal_edges)
        and bool(np.all(jump_values <= maximum_curvature_jump)),
        "patch_internal_curvature_continuity_failed",
        {
            "internal_edges": len(internal_edges),
            "checked_edges": int(jump_values.size),
            "incidence_errors": incidence_errors[:100],
            "invalid_edges": invalid_edges[:100],
            "edge_curvature": summarize_values(curvature_values),
            "normalized_jump": summarize_values(jump_values),
            "normalization_length": summarize_values(length_values),
        },
        {"maximum_normalized_jump": float(maximum_curvature_jump)},
    )
    return normal_gate, curvature_gate


def describe_local_geometry(
    points: np.ndarray,
    loops: tuple[np.ndarray, ...],
    boundary_edges: set[tuple[int, int]],
    edge_map: dict[tuple[int, int], list[tuple[int, int]]],
    local_normal: np.ndarray | None,
) -> dict[str, Any]:
    reports = []
    for loop in loops:
        coordinates = np.asarray(points[loop], dtype=np.float64)
        center = coordinates.mean(axis=0)
        centered = coordinates - center
        _, singular, vh = np.linalg.svd(centered, full_matrices=False)
        pca_normal = vh[-1]
        newell = np.sum(np.cross(centered, np.roll(centered, -1, axis=0)), axis=0)
        newell_length = float(np.linalg.norm(newell))
        newell_normal = newell / newell_length if newell_length > 1e-30 else np.zeros(3)
        if local_normal is not None and float(np.dot(pca_normal, local_normal)) < 0.0:
            pca_normal = -pca_normal
        source_direction_matches = []
        for index, left_value in enumerate(loop):
            left, right = int(left_value), int(loop[(index + 1) % loop.size])
            key = (left, right) if left < right else (right, left)
            occurrences = edge_map.get(key, [])
            listed_direction = 1 if (left, right) == key else -1
            if len(occurrences) == 1:
                source_direction_matches.append(occurrences[0][1] == listed_direction)
        patch_orientation_normal = None
        if len(source_direction_matches) == loop.size and (
            all(source_direction_matches) or not any(source_direction_matches)
        ):
            orientation = -1.0 if all(source_direction_matches) else 1.0
            patch_orientation_normal = (newell_normal * orientation).tolist()
        reports.append(
            {
                "source_vertex_ids": loop.astype(int).tolist(),
                "stable_center": center.tolist(),
                "pca_normal": pca_normal.tolist(),
                "newell_normal": newell_normal.tolist(),
                "patch_orientation_normal": patch_orientation_normal,
                "planarity_ratio": float(singular[-1] / max(singular[0], 1e-30)),
                "maximum_plane_deviation": float(np.max(np.abs(centered @ pca_normal))),
            }
        )
    incident = collect_incident_source_faces(boundary_edges, edge_map)
    return {
        "area_weighted_incident_normal": local_normal.tolist() if local_normal is not None else None,
        "incident_source_face_ids": incident.astype(int).tolist(),
        "boundary_loops": reports,
        "patch_orientation_normal": (
            reports[0]["patch_orientation_normal"] if len(reports) == 1 else None
        ),
    }


def calculate_triangle_geometry(points: np.ndarray, faces: np.ndarray) -> dict[str, np.ndarray]:
    triangles = np.asarray(points, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    edges = np.stack(
        [
            np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
            np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
            np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
        ],
        axis=1,
    )
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    normals = np.divide(
        cross,
        double_area[:, None],
        out=np.zeros_like(cross),
        where=double_area[:, None] > 1e-30,
    )
    maximum_edge = edges.max(axis=1)
    altitude = np.divide(
        double_area,
        maximum_edge,
        out=np.zeros_like(double_area),
        where=maximum_edge > 1e-30,
    )
    aspect = np.divide(
        maximum_edge,
        altitude,
        out=np.full_like(maximum_edge, np.inf),
        where=altitude > 1e-30,
    )
    return {"areas": 0.5 * double_area, "normals": normals, "aspect": aspect}


def calculate_area_weighted_normal(
    normals: np.ndarray,
    areas: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray]:
    weighted = np.sum(normals * areas[:, None], axis=0)
    length = float(np.linalg.norm(weighted))
    if length <= 1e-30:
        return None, np.zeros(0, dtype=np.float64)
    normal = weighted / length
    return normal, normals @ normal


def collect_incident_source_faces(
    boundary_edges: set[tuple[int, int]],
    edge_map: dict[tuple[int, int], list[tuple[int, int]]],
) -> np.ndarray:
    return np.asarray(
        sorted({face_id for edge in boundary_edges for face_id, _ in edge_map.get(edge, [])}),
        dtype=np.int64,
    )


def build_directed_edge_map(faces: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    result: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for face_id, face in enumerate(np.asarray(faces, dtype=np.int64)):
        for index in range(3):
            left, right = int(face[index]), int(face[(index + 1) % 3])
            key = (left, right) if left < right else (right, left)
            result.setdefault(key, []).append((face_id, 1 if (left, right) == key else -1))
    return result


def count_topology(faces: np.ndarray) -> dict[str, int]:
    topology, edge_faces = edge_topology(np.asarray(faces, dtype=np.int64))
    labels = face_component_labels(np.asarray(faces).shape[0], edge_faces)
    return {
        "boundary_edges": int(topology["boundary_edges"]),
        "non_manifold_edges": int(topology["non_manifold_edges"]),
        "inconsistent_winding_edges": int(inconsistent_winding_edges(faces)),
        "components": int(labels.max()) + 1 if labels.size else 0,
    }


def collect_loop_edges(loops: tuple[np.ndarray, ...]) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for loop in loops:
        for index in range(loop.size):
            left = int(loop[index])
            right = int(loop[(index + 1) % loop.size])
            edges.add((left, right) if left < right else (right, left))
    return edges


def normalize_open_loop(loop: np.ndarray) -> np.ndarray:
    result = np.asarray(loop, dtype=np.int64).reshape(-1)
    return result[:-1] if result.size > 1 and result[0] == result[-1] else result


def calculate_coordinate_tolerance(points: np.ndarray, ratio: float) -> float:
    diagonal = float(np.linalg.norm(np.ptp(np.asarray(points, dtype=np.float64), axis=0)))
    return max(diagonal * float(ratio), 1e-15)


def calculate_normal_angle(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.arccos(np.clip(float(np.dot(left, right)), -1.0, 1.0)))


def compare_prefix_bytes(reference: np.ndarray, candidate: np.ndarray) -> bool:
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if candidate.ndim != reference.ndim or candidate.shape[1:] != reference.shape[1:]:
        return False
    if candidate.shape[0] < reference.shape[0] or candidate.dtype != reference.dtype:
        return False
    prefix = candidate[: reference.shape[0]]
    return np.ascontiguousarray(reference).tobytes() == np.ascontiguousarray(prefix).tobytes()


def summarize_values(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {
            "min": None,
            "p50": None,
            "p95": None,
            "max": None,
            "non_finite_count": int(values.size),
        }
    return {
        "min": float(finite.min()),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(finite.max()),
        "non_finite_count": int(values.size - finite.size),
    }


def build_gate(passed: bool, reason_code: str, actual: Any, threshold: Any) -> dict[str, Any]:
    return {
        "required": True,
        "status": "computed",
        "passed": bool(passed),
        "actual": actual,
        "threshold": threshold,
        "reason_code": None if passed else reason_code,
        "evidence_paths": [],
    }


def _point_strictly_inside_polygon(point: np.ndarray, polygon: np.ndarray, tolerance: float) -> bool:
    inside = False
    for index, left in enumerate(polygon):
        right = polygon[(index + 1) % polygon.shape[0]]
        edge = right - left
        squared = float(np.dot(edge, edge))
        fraction = float(np.clip(np.dot(point - left, edge) / max(squared, 1e-30), 0.0, 1.0))
        if np.linalg.norm(point - (left + fraction * edge)) <= tolerance:
            return False
        crosses = (left[1] > point[1]) != (right[1] > point[1])
        if crosses:
            x_intersection = left[0] + (point[1] - left[1]) * (right[0] - left[0]) / (right[1] - left[1])
            if x_intersection > point[0]:
                inside = not inside
    return inside
