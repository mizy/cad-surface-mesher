from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from source_primary_outside_evidence import summarize_incident_outside_directions


def compute_face_geometry(
    points: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    triangles = points[faces]
    area_vectors = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    double_areas = np.linalg.norm(area_vectors, axis=1)
    normals = np.divide(
        area_vectors,
        double_areas[:, None],
        out=np.zeros_like(area_vectors),
        where=double_areas[:, None] > 0.0,
    )
    return triangles.mean(axis=1), normals, double_areas * 0.5


def build_local_curvature_samples(
    points: np.ndarray,
    face_normals: np.ndarray,
    manifold_edges: np.ndarray,
    manifold_face_pairs: np.ndarray,
    boundary_face_ids: np.ndarray,
) -> dict[int, list[tuple[int, float, float]]]:
    boundary_faces = {int(value) for value in boundary_face_ids}
    samples: dict[int, list[tuple[int, float, float]]] = {}
    for edge_index, (edge, pair) in enumerate(zip(manifold_edges, manifold_face_pairs, strict=True)):
        left, right = int(pair[0]), int(pair[1])
        if left not in boundary_faces and right not in boundary_faces:
            continue
        length = float(np.linalg.norm(points[int(edge[1])] - points[int(edge[0])]))
        if length <= 0.0:
            continue
        dot = float(np.clip(np.dot(face_normals[left], face_normals[right]), -1.0, 1.0))
        angle = float(np.arccos(dot))
        sample = (edge_index, angle, angle / length)
        if left in boundary_faces:
            samples.setdefault(left, []).append(sample)
        if right in boundary_faces:
            samples.setdefault(right, []).append(sample)
    return samples


def describe_boundary_geometry(
    points: np.ndarray,
    edges: Sequence[tuple[int, int]],
    ordered_loops: Sequence[Sequence[int]],
    incident_face_ids: Sequence[int],
    face_normals: np.ndarray,
    face_areas: np.ndarray,
    curvature_samples: Mapping[int, Sequence[tuple[int, float, float]]],
    face_external_directions: np.ndarray | None,
    relation_kind: str,
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    edge_array = np.asarray(edges, dtype=np.int64).reshape((-1, 2))
    segment_points = points[edge_array] if edge_array.size else np.empty((0, 2, 3))
    edge_lengths = (
        np.linalg.norm(segment_points[:, 1] - segment_points[:, 0], axis=1)
        if segment_points.size
        else np.zeros(0)
    )
    line_center, covariance = _line_moments(segment_points, edge_lengths)
    pca = _pca_plane(
        covariance,
        line_center,
        thresholds["pca_min_normal_eigengap_ratio"],
    )
    loop_metrics = [_polygon_metrics(points[np.asarray(loop, dtype=np.int64)], pca) for loop in ordered_loops]
    center, center_method, footprint_area = _stable_center(
        line_center,
        loop_metrics,
        relation_kind,
        thresholds["minimum_polygon_area"],
    )
    pca["origin"] = center.tolist()
    newell = _newell_plane(
        points,
        ordered_loops,
        center,
        thresholds["minimum_polygon_area"],
        thresholds["newell_min_area_scale_ratio"],
    )
    unique_vertices = np.unique(edge_array) if edge_array.size else np.zeros(0, dtype=np.int64)
    local_points = points[unique_vertices] if unique_vertices.size else np.empty((0, 3))
    scale = _scale_metrics(local_points, edge_lengths, pca, footprint_area)
    planarity = _planarity_metrics(local_points, center, np.asarray(pca["normal"]), scale["characteristic"])
    planarity["pca_smallest_to_largest_ratio"] = pca["smallest_to_largest_ratio"]
    source_normals = _source_normal_metrics(
        incident_face_ids,
        face_normals,
        face_areas,
        thresholds["normal_consistency_min_dot"],
    )
    outside = _outside_direction(
        incident_face_ids,
        face_areas,
        face_external_directions,
        thresholds["outside_direction_min_abs_dot"],
    )
    oriented = _orient_normal(
        source_normals,
        newell,
        pca,
        outside,
        thresholds["outside_direction_min_abs_dot"],
    )
    curvature = _curvature_metrics(incident_face_ids, curvature_samples, scale["characteristic"])
    boundary_side = _boundary_side_evidence(
        newell,
        source_normals,
        thresholds["hole_boundary_min_newell_source_opposition"],
    )
    pca_newell_dot = float(
        np.dot(np.asarray(pca["normal"]), np.asarray(newell["normal"]))
    ) if newell["status"] == "computed" else None
    return {
        "stable_center": {"value": center.tolist(), "method": center_method},
        "planes": {
            "pca": pca,
            "newell": newell,
            "pca_newell_abs_dot": None if pca_newell_dot is None else abs(pca_newell_dot),
        },
        "normals": {
            "area_weighted_source": source_normals,
            "boundary_side": boundary_side,
            "external_direction": outside,
            "oriented": oriented,
        },
        "planarity": planarity,
        "scale": scale,
        "local_curvature": curvature,
    }


def _line_moments(segments: np.ndarray, lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total = float(lengths.sum())
    if total <= 0.0:
        return np.zeros(3), np.zeros((3, 3))
    left, right = segments[:, 0], segments[:, 1]
    center = np.sum(lengths[:, None] * (left + right) * 0.5, axis=0) / total
    covariance = np.zeros((3, 3))
    for a, b, length in zip(left, right, lengths, strict=True):
        a, b = a - center, b - center
        covariance += length * (
            (np.outer(a, a) + np.outer(b, b)) / 3.0
            + (np.outer(a, b) + np.outer(b, a)) / 6.0
        )
    covariance /= total
    return center, (covariance + covariance.T) * 0.5


def _pca_plane(
    covariance: np.ndarray,
    origin: np.ndarray,
    minimum_normal_eigengap_ratio: float,
) -> dict[str, Any]:
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values, vectors = np.maximum(values[order], 0.0), vectors[:, order]
    normal = _stable_axis(vectors[:, 2])
    in_plane_degenerate = bool((values[0] - values[1]) / max(values[0], 1e-30) <= 1e-10)
    if in_plane_degenerate:
        global_axes = np.eye(3)
        axis_index = int(np.argmax(1.0 - (global_axes @ normal) ** 2))
        major = global_axes[axis_index] - np.dot(global_axes[axis_index], normal) * normal
        major = _stable_axis(major)
    else:
        major = _stable_axis(vectors[:, 0])
    minor = np.cross(normal, major)
    if np.linalg.norm(minor) <= 1e-15:
        minor = _stable_axis(vectors[:, 1])
        normal = _stable_axis(np.cross(major, minor))
    else:
        minor /= np.linalg.norm(minor)
    normal_eigengap_ratio = float((values[1] - values[2]) / max(values[0], 1e-30))
    reliable_normal = bool(values[0] > 0.0 and normal_eigengap_ratio >= minimum_normal_eigengap_ratio)
    return {
        "status": "computed" if reliable_normal else "degenerate_normal",
        "origin": origin.tolist(),
        "major_axis": major.tolist(),
        "minor_axis": minor.tolist(),
        "normal": normal.tolist(),
        "eigenvalues": values.tolist(),
        "smallest_to_largest_ratio": float(values[2] / max(values[0], 1e-30)),
        "normal_eigengap_ratio": normal_eigengap_ratio,
        "minimum_normal_eigengap_ratio": minimum_normal_eigengap_ratio,
        "in_plane_axis_degenerate": in_plane_degenerate,
    }


def _polygon_metrics(loop_points: np.ndarray, pca: Mapping[str, Any]) -> dict[str, Any]:
    origin = np.asarray(pca["origin"])
    major, minor = np.asarray(pca["major_axis"]), np.asarray(pca["minor_axis"])
    local = loop_points - origin
    polygon = np.column_stack((local @ major, local @ minor))
    cross = polygon[:, 0] * np.roll(polygon[:, 1], -1) - np.roll(polygon[:, 0], -1) * polygon[:, 1]
    signed_area = float(cross.sum() * 0.5)
    if abs(signed_area) <= 1e-30:
        centroid = polygon.mean(axis=0) if polygon.size else np.zeros(2)
    else:
        centroid = np.asarray([
            np.sum((polygon[:, 0] + np.roll(polygon[:, 0], -1)) * cross),
            np.sum((polygon[:, 1] + np.roll(polygon[:, 1], -1)) * cross),
        ]) / (6.0 * signed_area)
    world = origin + centroid[0] * major + centroid[1] * minor
    return {"signed_area": signed_area, "absolute_area": abs(signed_area), "centroid": world, "polygon": polygon}


def _stable_center(
    fallback: np.ndarray,
    loops: Sequence[Mapping[str, Any]],
    relation_kind: str,
    minimum_area: float,
) -> tuple[np.ndarray, str, float]:
    valid = [loop for loop in loops if loop["absolute_area"] > minimum_area]
    if not valid:
        return fallback, "boundary_arclength_integral", 0.0
    if relation_kind in {"near_coincident_part_seam", "compatible_paired_boundary_loops"}:
        center = np.mean(np.vstack([loop["centroid"] for loop in valid]), axis=0)
        return center, "symmetric_paired_polygon_centroids", float(np.mean([loop["absolute_area"] for loop in valid]))
    if relation_kind == "nested_loops" and len(valid) > 1:
        outer = max(valid, key=lambda item: item["absolute_area"])
        weights = [loop["absolute_area"] if loop is outer else -loop["absolute_area"] for loop in valid]
        total = float(sum(weights))
        if total > minimum_area:
            center = sum(weight * loop["centroid"] for weight, loop in zip(weights, valid, strict=True)) / total
            return np.asarray(center), "outer_minus_inner_polygon_centroids", total
    total = float(sum(loop["absolute_area"] for loop in valid))
    center = sum(loop["absolute_area"] * loop["centroid"] for loop in valid) / total
    return np.asarray(center), "polygon_area_weighted_centroid", total


def _newell_plane(
    points: np.ndarray,
    loops: Sequence[Sequence[int]],
    center: np.ndarray,
    minimum_polygon_area: float,
    minimum_area_scale_ratio: float,
) -> dict[str, Any]:
    vectors = []
    for loop in loops:
        local = points[np.asarray(loop, dtype=np.int64)] - center
        vectors.append(np.sum(np.cross(local, np.roll(local, -1, axis=0)), axis=0) * 0.5)
    vector = np.sum(vectors, axis=0) if vectors else np.zeros(3)
    magnitude = float(np.linalg.norm(vector))
    loop_vertex_ids = sorted({int(value) for loop in loops for value in loop})
    loop_points = points[np.asarray(loop_vertex_ids, dtype=np.int64)] if loop_vertex_ids else np.empty((0, 3))
    scale = float(np.linalg.norm(np.ptp(loop_points, axis=0))) if loop_points.size else 0.0
    minimum_reliable_area = max(minimum_polygon_area, scale * scale * minimum_area_scale_ratio)
    computed = magnitude >= minimum_reliable_area
    return {
        "status": "computed" if computed else "degenerate_below_area_tolerance",
        "origin": center.tolist(),
        "area_vector": vector.tolist(),
        "normal": (vector / magnitude).tolist() if computed else [0.0, 0.0, 0.0],
        "oriented_projected_area": magnitude,
        "sum_absolute_loop_area": float(sum(np.linalg.norm(value) for value in vectors)),
        "minimum_reliable_area": minimum_reliable_area,
    }


def _scale_metrics(
    points: np.ndarray,
    edge_lengths: np.ndarray,
    pca: Mapping[str, Any],
    footprint_area: float,
) -> dict[str, Any]:
    if points.size:
        centered = points - np.asarray(pca["origin"])
        major_values = centered @ np.asarray(pca["major_axis"])
        minor_values = centered @ np.asarray(pca["minor_axis"])
        major_extent = float(np.ptp(major_values))
        minor_extent = float(np.ptp(minor_values))
        bbox_diagonal = float(np.linalg.norm(np.ptp(points, axis=0)))
    else:
        major_extent = minor_extent = bbox_diagonal = 0.0
    perimeter = float(edge_lengths.sum())
    characteristic = max(major_extent, bbox_diagonal, float(edge_lengths.max(initial=0.0)), 1e-30)
    return {
        "characteristic": characteristic,
        "perimeter": perimeter,
        "major_extent": major_extent,
        "minor_extent": minor_extent,
        "aspect_ratio": major_extent / max(minor_extent, 1e-30),
        "bbox_diagonal": bbox_diagonal,
        "footprint_area": footprint_area,
        "equivalent_diameter": float(np.sqrt(4.0 * max(footprint_area, 0.0) / np.pi)),
        "hydraulic_diameter": 4.0 * footprint_area / max(perimeter, 1e-30),
        "estimated_width": 2.0 * footprint_area / max(perimeter, 1e-30),
        "min_edge_length": float(edge_lengths.min()) if edge_lengths.size else 0.0,
        "median_edge_length": float(np.median(edge_lengths)) if edge_lengths.size else 0.0,
        "max_edge_length": float(edge_lengths.max(initial=0.0)),
    }


def _planarity_metrics(points: np.ndarray, center: np.ndarray, normal: np.ndarray, scale: float) -> dict[str, Any]:
    distances = np.abs((points - center) @ normal) if points.size else np.zeros(0)
    rms = float(np.sqrt(np.mean(distances * distances))) if distances.size else 0.0
    maximum = float(distances.max(initial=0.0))
    return {
        "rms_distance": rms,
        "max_distance": maximum,
        "rms_distance_scale_ratio": rms / max(scale, 1e-30),
        "max_distance_scale_ratio": maximum / max(scale, 1e-30),
    }


def _source_normal_metrics(
    face_ids: Sequence[int],
    normals: np.ndarray,
    areas: np.ndarray,
    minimum_dot: float,
) -> dict[str, Any]:
    ids = np.asarray(sorted(set(int(value) for value in face_ids)), dtype=np.int64)
    valid = ids[(areas[ids] > 0.0) & (np.linalg.norm(normals[ids], axis=1) > 0.0)] if ids.size else ids
    total_area = float(areas[valid].sum()) if valid.size else 0.0
    vector = np.sum(areas[valid, None] * normals[valid], axis=0) if valid.size else np.zeros(3)
    magnitude = float(np.linalg.norm(vector))
    unit = vector / magnitude if magnitude > 0.0 else np.zeros(3)
    dots = normals[valid] @ unit if valid.size and magnitude > 0.0 else np.zeros(0)
    opposing_area = float(areas[valid][dots < minimum_dot].sum()) if dots.size else total_area
    return {
        "status": "computed" if magnitude > 0.0 else "unavailable",
        "normal": unit.tolist(),
        "area_vector": vector.tolist(),
        "valid_face_count": int(valid.size),
        "total_area": total_area,
        "resultant_ratio": magnitude / max(total_area, 1e-30),
        "minimum_dot": float(dots.min(initial=1.0)),
        "mean_dot": float(np.average(dots, weights=areas[valid])) if dots.size else None,
        "p05_dot": float(np.quantile(dots, 0.05)) if dots.size else None,
        "opposing_area_ratio": opposing_area / max(total_area, 1e-30),
        "consistent": bool(dots.size and opposing_area == 0.0),
    }


def _outside_direction(
    face_ids: Sequence[int],
    face_areas: np.ndarray,
    directions: np.ndarray | None,
    minimum_dot: float,
) -> dict[str, Any]:
    return summarize_incident_outside_directions(
        face_ids, face_areas, directions, minimum_dot
    )


def _boundary_side_evidence(
    newell: Mapping[str, Any],
    source: Mapping[str, Any],
    minimum_opposition: float,
) -> dict[str, Any]:
    if newell["status"] != "computed" or source["status"] != "computed":
        return {"status": "unresolved", "newell_source_normal_dot": None}
    dot = float(np.dot(np.asarray(newell["normal"]), np.asarray(source["normal"])))
    if dot <= -minimum_opposition:
        status = "missing_surface_inside_loop"
    elif dot >= minimum_opposition:
        status = "open_component_perimeter_or_inner_island"
    else:
        status = "orientation_evidence_ambiguous"
    return {"status": status, "newell_source_normal_dot": dot}


def _orient_normal(
    source: Mapping[str, Any],
    newell: Mapping[str, Any],
    pca: Mapping[str, Any],
    outside: Mapping[str, Any],
    minimum_abs_dot: float,
) -> dict[str, Any]:
    if source["status"] == "computed":
        raw, method = np.asarray(source["normal"]), "area_weighted_adjacent_source_faces"
    elif newell["status"] == "computed":
        raw, method = np.asarray(newell["normal"]), "newell_fallback"
    else:
        raw, method = np.asarray(pca["normal"]), "pca_fallback"
    if outside["status"] != "computed":
        return {"status": outside["status"], "resolved": False, "normal": raw.tolist(), "raw_normal": raw.tolist(), "method": method, "pre_alignment_dot": None, "orientation_sign": 0}
    dot = float(np.dot(raw, np.asarray(outside["direction"])))
    if abs(dot) < minimum_abs_dot:
        return {"status": "external_direction_tangent_or_conflicting", "resolved": False, "normal": raw.tolist(), "raw_normal": raw.tolist(), "method": method, "pre_alignment_dot": dot, "orientation_sign": 0}
    sign = 1 if dot > 0.0 else -1
    return {"status": "aligned" if sign == 1 else "flipped_descriptor_only", "resolved": True, "normal": (raw * sign).tolist(), "raw_normal": raw.tolist(), "method": method, "pre_alignment_dot": dot, "orientation_sign": sign}


def _curvature_metrics(
    face_ids: Sequence[int],
    samples: Mapping[int, Sequence[tuple[int, float, float]]],
    scale: float,
) -> dict[str, Any]:
    unique: list[tuple[int, float, float]] = []
    seen: set[int] = set()
    for face_id in sorted(set(int(value) for value in face_ids)):
        for sample in samples.get(face_id, ()):
            if sample[0] not in seen:
                seen.add(sample[0])
                unique.append(sample)
    if not unique:
        return {"status": "unavailable", "method": "one_ring_normal_angle_per_edge_length", "sample_count": 0}
    angles = np.asarray([value[1] for value in unique])
    inverse = np.asarray([value[2] for value in unique])
    return {
        "status": "computed",
        "method": "one_ring_normal_angle_per_edge_length",
        "sample_count": len(unique),
        "angle_radians": {"mean": float(angles.mean()), "rms": float(np.sqrt(np.mean(angles * angles))), "max": float(angles.max())},
        "inverse_length": {"mean": float(inverse.mean()), "rms": float(np.sqrt(np.mean(inverse * inverse))), "max": float(inverse.max())},
        "dimensionless_rms": float(np.sqrt(np.mean(inverse * inverse)) * scale),
    }


def _stable_axis(axis: np.ndarray) -> np.ndarray:
    result = np.asarray(axis, dtype=np.float64)
    magnitude = float(np.linalg.norm(result))
    if magnitude <= 0.0:
        return np.asarray([1.0, 0.0, 0.0])
    result = result / magnitude
    pivot = int(np.argmax(np.abs(result)))
    return -result if result[pivot] < 0.0 else result
