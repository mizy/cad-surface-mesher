from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from mesh_metrics import edge_topology, face_component_labels, mesh_report, self_intersection_report

REJECTION_BITS = {
    "source_face_unreliable": 1,
    "distance_gate_failed": 2,
    "source_edge_or_vertex_gate_failed": 4,
    "source_defect_edge_gate_failed": 8,
    "normal_orientation_gate_failed": 16,
    "normal_orientation_ambiguous": 32,
    "source_component_spatial_mismatch": 64,
    "projected_vertex_collision": 128,
    "triangle_quality_rollback": 256,
    "reported_self_intersection_rollback": 512,
}

@dataclass(frozen=True)
class ProjectionThresholds:
    """Dimensioned and dimensionless gates for exact source projection."""

    max_projection_distance: float | None = None
    max_distance_bbox_ratio: float = 0.04
    min_signed_normal_dot: float = 0.35
    orientation_vote_min_margin: float = 0.10
    source_edge_barycentric_margin: float = 0.015
    defect_edge_barycentric_margin: float = 0.08
    collision_tolerance_bbox_ratio: float = 1e-9
    max_aspect_ratio: float = 40.0
    min_triangle_area_ratio: float = 0.05
    max_edge_length_ratio: float = 4.0
    max_quality_rollback_iterations: int = 6
    require_component_consistency: bool = True

def _validate_mesh(name: str, points: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(f"{name}_points must be a non-empty (N, 3) array")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] == 0:
        raise ValueError(f"{name}_faces must be a non-empty (M, 3) array")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name}_points must be finite")
    if np.any((faces < 0) | (faces >= points.shape[0])):
        raise ValueError(f"{name}_faces contains an out-of-range point index")
    return points, faces

def _unit_face_normals(points: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    triangles = points[faces]
    raw = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(raw, axis=1)
    normals = np.divide(
        raw,
        lengths[:, None],
        out=np.zeros_like(raw),
        where=lengths[:, None] > 1e-15,
    )
    return normals, lengths * 0.5

def _vertex_normals(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    face_normals, double_areas = _unit_face_normals(points, faces)
    normals = np.zeros_like(points)
    weighted = face_normals * double_areas[:, None]
    for corner in range(3):
        np.add.at(normals, faces[:, corner], weighted)
    lengths = np.linalg.norm(normals, axis=1)
    return np.divide(
        normals,
        lengths[:, None],
        out=np.zeros_like(normals),
        where=lengths[:, None] > 1e-15,
    )

def _source_topology(
    points: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    topology, edge_faces = edge_topology(faces)
    components = face_component_labels(faces.shape[0], edge_faces)
    defect_edges: set[tuple[int, int]] = set()
    directed: dict[tuple[int, int], list[int]] = {}
    for face in faces:
        for left, right in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            a, b = int(left), int(right)
            key = (a, b) if a < b else (b, a)
            directed.setdefault(key, []).append(1 if (a, b) == key else -1)
    for edge, signs in directed.items():
        if len(signs) != 2 or signs[0] == signs[1]:
            defect_edges.add(edge)
    flags = np.zeros((faces.shape[0], 3), dtype=bool)
    for face_id, (a, b, c) in enumerate(faces):
        flags[face_id] = [
            tuple(sorted((int(b), int(c)))) in defect_edges,
            tuple(sorted((int(c), int(a)))) in defect_edges,
            tuple(sorted((int(a), int(b)))) in defect_edges,
        ]
    _, areas = _unit_face_normals(points, faces)
    scale = float(np.linalg.norm(np.ptp(points, axis=0)))
    degenerate = areas <= max(scale * scale * 1e-16, 1e-30)
    stats = {**topology, "winding_or_topology_defect_edges": len(defect_edges)}
    stats["degenerate_faces"] = int(np.count_nonzero(degenerate))
    return components, flags, degenerate, stats

def _closest_source_points(
    source_points: np.ndarray,
    source_faces: np.ndarray,
    query_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    from vtkmodules.vtkCommonCore import mutable, vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData, vtkStaticCellLocator

    vtk_points = vtkPoints()
    vtk_points.SetData(numpy_to_vtk(np.ascontiguousarray(source_points), deep=True))
    packed = np.column_stack([np.full(source_faces.shape[0], 3), source_faces]).astype(np.int64).ravel()
    cells = vtkCellArray()
    cells.ImportLegacyFormat(numpy_to_vtkIdTypeArray(packed, deep=True))
    poly = vtkPolyData()
    poly.SetPoints(vtk_points)
    poly.SetPolys(cells)
    locator = vtkStaticCellLocator()
    locator.SetDataSet(poly)
    locator.BuildLocator()
    closest_points = np.empty_like(query_points)
    face_ids = np.empty(query_points.shape[0], dtype=np.int64)
    distances = np.empty(query_points.shape[0], dtype=np.float64)
    closest = [0.0, 0.0, 0.0]
    cell_id, sub_id, distance2 = mutable(0), mutable(0), mutable(0.0)
    for point_id, point in enumerate(query_points):
        locator.FindClosestPoint(point, closest, cell_id, sub_id, distance2)
        closest_points[point_id] = closest
        face_ids[point_id] = int(cell_id)
        distances[point_id] = np.sqrt(float(distance2))
    return closest_points, face_ids, distances

def _barycentric(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    v0, v1, v2 = b - a, c - a, points - a
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)
    denominator = d00 * d11 - d01 * d01
    v = np.divide(d11 * d20 - d01 * d21, denominator, out=np.zeros_like(d00), where=np.abs(denominator) > 1e-24)
    w = np.divide(d00 * d21 - d01 * d20, denominator, out=np.zeros_like(d00), where=np.abs(denominator) > 1e-24)
    return np.column_stack((1.0 - v - w, v, w))

def _orientation_gate(
    raw_dot: np.ndarray,
    components: np.ndarray,
    eligible: np.ndarray,
    thresholds: ProjectionThresholds,
) -> tuple[np.ndarray, dict[str, Any]]:
    rejected = np.zeros(raw_dot.shape[0], dtype=np.uint16)
    component_votes: dict[str, Any] = {}
    for component_id in np.unique(components[eligible]):
        ids = np.flatnonzero(eligible & (components == component_id))
        magnitudes = np.abs(raw_dot[ids])
        signed_vote = float(np.sum(raw_dot[ids]))
        vote_total = float(np.sum(magnitudes))
        margin = abs(signed_vote) / max(vote_total, 1e-15)
        if margin < thresholds.orientation_vote_min_margin:
            rejected[ids] |= REJECTION_BITS["normal_orientation_ambiguous"]
            orientation = 0
        else:
            orientation = 1 if signed_vote >= 0.0 else -1
            signed_alignment = raw_dot[ids] * orientation
            failed = ids[signed_alignment < thresholds.min_signed_normal_dot]
            rejected[failed] |= REJECTION_BITS["normal_orientation_gate_failed"]
        component_votes[str(int(component_id))] = {
            "query_vertices": int(ids.size),
            "orientation_sign": orientation,
            "vote_margin": margin,
        }
    return rejected, component_votes

def _component_consistency_rejections(
    proxy_faces: np.ndarray,
    source_components: np.ndarray,
    accepted: np.ndarray,
) -> tuple[np.ndarray, int]:
    rejected = np.zeros(accepted.shape[0], dtype=bool)
    inconsistent_faces = 0
    for face in proxy_faces:
        ids = face[accepted[face]]
        if ids.size >= 2 and np.unique(source_components[ids]).size > 1:
            rejected[ids] = True
            inconsistent_faces += 1
    return rejected, inconsistent_faces

def _collision_rejections(
    candidate_points: np.ndarray,
    proxy_points: np.ndarray,
    proxy_faces: np.ndarray,
    accepted: np.ndarray,
    priority: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, int]:
    rejected = np.zeros(accepted.shape[0], dtype=bool)
    if np.count_nonzero(accepted) == 0:
        return rejected, 0
    pairs = cKDTree(candidate_points).query_pairs(tolerance, output_type="ndarray")
    if pairs.size == 0:
        return rejected, 0
    pairs = np.asarray(pairs, dtype=np.int64)
    adjacent = {tuple(sorted((int(a), int(b)))) for face in proxy_faces for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))}
    conflicts = [
        pair
        for pair in pairs
        if tuple(sorted(map(int, pair))) not in adjacent
        and (accepted[pair[0]] or accepted[pair[1]])
        and np.linalg.norm(proxy_points[pair[0]] - proxy_points[pair[1]]) > tolerance
    ]
    ordered = sorted(conflicts, key=lambda pair: max(priority[pair[0]], priority[pair[1]]), reverse=True)
    for left_value, right_value in ordered:
        left, right = int(left_value), int(right_value)
        if rejected[left] or rejected[right]:
            continue
        if accepted[left] and not accepted[right]:
            rejected[left] = True
        elif accepted[right] and not accepted[left]:
            rejected[right] = True
        else:
            loser = right if priority[left] >= priority[right] else left
            rejected[loser] = True
    return rejected, len(conflicts)

def _triangle_issue_mask(
    points: np.ndarray,
    proxy_points: np.ndarray,
    faces: np.ndarray,
    thresholds: ProjectionThresholds,
) -> np.ndarray:
    candidate = points[faces]
    baseline = proxy_points[faces]
    candidate_edges = np.stack([candidate[:, 1] - candidate[:, 0], candidate[:, 2] - candidate[:, 1], candidate[:, 0] - candidate[:, 2]], axis=1)
    baseline_edges = np.stack([baseline[:, 1] - baseline[:, 0], baseline[:, 2] - baseline[:, 1], baseline[:, 0] - baseline[:, 2]], axis=1)
    candidate_lengths = np.linalg.norm(candidate_edges, axis=2)
    baseline_lengths = np.linalg.norm(baseline_edges, axis=2)
    candidate_raw = np.cross(candidate_edges[:, 0], -candidate_edges[:, 2])
    baseline_raw = np.cross(baseline_edges[:, 0], -baseline_edges[:, 2])
    candidate_area = np.linalg.norm(candidate_raw, axis=1) * 0.5
    baseline_area = np.linalg.norm(baseline_raw, axis=1) * 0.5
    max_edge = candidate_lengths.max(axis=1)
    altitude = np.divide(2.0 * candidate_area, max_edge, out=np.zeros_like(max_edge), where=max_edge > 1e-15)
    aspect = np.divide(max_edge, altitude, out=np.full_like(max_edge, np.inf), where=altitude > 1e-15)
    area_ratio = np.divide(candidate_area, baseline_area, out=np.ones_like(candidate_area), where=baseline_area > 1e-30)
    edge_ratio = np.divide(candidate_lengths, baseline_lengths, out=np.ones_like(candidate_lengths), where=baseline_lengths > 1e-15)
    flipped = np.einsum("ij,ij->i", candidate_raw, baseline_raw) <= 0.0
    return flipped | (aspect > thresholds.max_aspect_ratio) | (area_ratio < thresholds.min_triangle_area_ratio) | np.any(edge_ratio > thresholds.max_edge_length_ratio, axis=1)

def _rollback_triangle_issues(
    points: np.ndarray,
    proxy_points: np.ndarray,
    faces: np.ndarray,
    accepted: np.ndarray,
    thresholds: ProjectionThresholds,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, int]]]:
    result = points.copy()
    applied = accepted.copy()
    iterations: list[dict[str, int]] = []
    for iteration in range(1, thresholds.max_quality_rollback_iterations + 1):
        issue_faces = np.flatnonzero(_triangle_issue_mask(result, proxy_points, faces, thresholds))
        vertices = np.unique(faces[issue_faces].ravel()) if issue_faces.size else np.zeros(0, dtype=np.int64)
        vertices = vertices[applied[vertices]]
        if vertices.size == 0:
            break
        result[vertices] = proxy_points[vertices]
        applied[vertices] = False
        iterations.append({"iteration": iteration, "issue_faces": int(issue_faces.size), "rolled_back_vertices": int(vertices.size)})
    return result, applied, iterations

def _percentiles(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {"min": None, "p50": None, "p95": None, "p99": None, "max": None}
    return {name: float(np.percentile(values, percentile)) for name, percentile in (("min", 0), ("p50", 50), ("p95", 95), ("p99", 99), ("max", 100))}

def _volume_drift(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_volume = before["volume"].get("signed_abs")
    after_volume = after["volume"].get("signed_abs")
    if before_volume is None or after_volume is None or before_volume <= 0.0:
        return {
            "status": "unavailable",
            "role": "diagnostic_only",
            "reference_role": "closure_connectivity_template_not_geometry_truth",
            "reason": "reliable closed-volume metrics required",
        }
    difference = float(after_volume - before_volume)
    return {
        "status": "computed",
        "role": "diagnostic_only",
        "reference_role": "closure_connectivity_template_not_geometry_truth",
        "before": float(before_volume),
        "after": float(after_volume),
        "absolute": difference,
        "relative": difference / float(before_volume),
    }

def _rejection_counts(mask: np.ndarray) -> dict[str, int]:
    return {name: int(np.count_nonzero(mask & bit)) for name, bit in REJECTION_BITS.items()}

def _projection_provenance(
    faces: np.ndarray,
    applied: np.ndarray,
    face_ids: np.ndarray,
    component_ids: np.ndarray,
    distances: np.ndarray,
    alignment: np.ndarray,
    rejection_mask: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    projected_vertex_count = applied[faces].sum(axis=1).astype(np.uint8)
    face_origin = np.where(
        projected_vertex_count == 3,
        1,  # source_projected
        np.where(projected_vertex_count == 0, 3, 2),  # sdf_generated | mixed_transition
    ).astype(np.uint8)
    return {
        "point_data": {
            "source_projection_applied": applied.astype(np.uint8),
            "source_projection_fallback": (~applied).astype(np.uint8),
            "nearest_source_face_id": face_ids,
            "nearest_source_component_id": component_ids,
            "source_projection_distance": distances,
            "source_projection_raw_normal_dot": alignment,
            "source_projection_rejection_mask": rejection_mask,
        },
        "cell_data": {
            "source_projected_vertex_count": projected_vertex_count,
            "source_projected_vertex_fraction": applied[faces].mean(axis=1).astype(np.float32),
            "face_origin": face_origin,
        },
    }

def _build_report(
    proxy_points: np.ndarray, faces: np.ndarray, candidate: np.ndarray, closest: np.ndarray,
    distances: np.ndarray, applied: np.ndarray, rejection: np.ndarray,
    thresholds: dict[str, Any], gate_reports: dict[str, Any], intersections: dict[str, Any],
) -> dict[str, Any]:
    before = mesh_report(proxy_points, faces)
    after = mesh_report(candidate, faces)
    residual = _percentiles(np.linalg.norm(candidate[applied] - closest[applied], axis=1))
    projected = int(np.count_nonzero(applied))
    fallback = int(applied.size - projected)
    topology = {
        "status": "computed",
        "faces_equal": True,
        "point_count_equal": candidate.shape[0] == proxy_points.shape[0],
        "boundary_edges_before": before["topology"]["boundary_edges"],
        "boundary_edges_after": after["topology"]["boundary_edges"],
        "non_manifold_edges_before": before["topology"]["non_manifold_edges"],
        "non_manifold_edges_after": after["topology"]["non_manifold_edges"],
    }
    return {
        "method": "exact_source_surface_projection_with_proxy_fallback",
        "thresholds": thresholds,
        "projection": {
            "vertices": int(applied.size),
            "source_projected": projected,
            "explicit_proxy_fallback": fallback,
            "unclassified": 0,
            "distance_before": _percentiles(distances),
            "source_distance_after_for_projected": residual,
            "rejection_counts": _rejection_counts(rejection),
        },
        **gate_reports,
        "quality": {"proxy": before, "projected": after},
        "comparisons": {
            "projection_topology": topology,
            "projection_mapping": {
                "status": "computed",
                "source_projected": projected,
                "explicit_proxy_fallback": fallback,
                "unclassified": 0,
            },
            "source_distance": {
                "status": "computed",
                "scope": "source_projected_vertices",
                "residual": residual,
            },
            "volume_drift": _volume_drift(before, after),
            "self_intersection": intersections,
        },
    }

def _project_with_gates(
    proxy_points: np.ndarray, proxy_faces: np.ndarray, source_points: np.ndarray, source_faces: np.ndarray,
    thresholds: ProjectionThresholds, reliable_source_face_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    scale = float(np.linalg.norm(np.ptp(source_points, axis=0)))
    max_distance = thresholds.max_projection_distance
    if max_distance is None:
        max_distance = max(scale * thresholds.max_distance_bbox_ratio, 1e-15)
    source_components, defect_edges, degenerate, source_topology = _source_topology(source_points, source_faces)
    closest, face_ids, distances = _closest_source_points(source_points, source_faces, proxy_points)
    target_components = source_components[face_ids]
    barycentric = _barycentric(closest, source_points[source_faces[face_ids]])
    source_normals, _ = _unit_face_normals(source_points, source_faces)
    raw_dot = np.einsum("ij,ij->i", _vertex_normals(proxy_points, proxy_faces), source_normals[face_ids])
    rejection = np.zeros(proxy_points.shape[0], dtype=np.uint16)
    reliable = np.ones(source_faces.shape[0], dtype=bool) if reliable_source_face_mask is None else np.asarray(reliable_source_face_mask, dtype=bool)
    if reliable.shape != (source_faces.shape[0],):
        raise ValueError("reliable_source_face_mask must contain one value per source face")
    rejection[~reliable[face_ids] | degenerate[face_ids]] |= REJECTION_BITS["source_face_unreliable"]
    rejection[distances > max_distance] |= REJECTION_BITS["distance_gate_failed"]
    rejection[np.min(barycentric, axis=1) < thresholds.source_edge_barycentric_margin] |= REJECTION_BITS["source_edge_or_vertex_gate_failed"]
    near_defect = np.any((barycentric < thresholds.defect_edge_barycentric_margin) & defect_edges[face_ids], axis=1)
    rejection[near_defect] |= REJECTION_BITS["source_defect_edge_gate_failed"]
    normal_rejection, votes = _orientation_gate(raw_dot, target_components, rejection == 0, thresholds)
    rejection |= normal_rejection
    if thresholds.require_component_consistency:
        mismatch, mismatch_faces = _component_consistency_rejections(proxy_faces, target_components, rejection == 0)
        rejection[mismatch] |= REJECTION_BITS["source_component_spatial_mismatch"]
    else:
        mismatch_faces = 0
    collision_tolerance = max(scale * thresholds.collision_tolerance_bbox_ratio, 1e-15)
    priority = np.abs(raw_dot) - distances / max(max_distance, 1e-15)
    collisions = np.zeros(proxy_points.shape[0], dtype=bool)
    collision_pairs, iterations = 0, []
    for safety_round in range(1, max(1, thresholds.max_quality_rollback_iterations) + 1):
        accepted = rejection == 0
        candidate = proxy_points.copy()
        candidate[accepted] = closest[accepted]
        new_collisions, pair_count = _collision_rejections(
            candidate, proxy_points, proxy_faces, accepted, priority, collision_tolerance,
        )
        collision_pairs += pair_count
        collisions |= new_collisions
        rejection[new_collisions] |= REJECTION_BITS["projected_vertex_collision"]
        accepted = rejection == 0
        candidate = proxy_points.copy()
        candidate[accepted] = closest[accepted]
        candidate, quality_applied, quality_rows = _rollback_triangle_issues(
            candidate, proxy_points, proxy_faces, accepted, thresholds,
        )
        quality_rollback = accepted & ~quality_applied
        rejection[quality_rollback] |= REJECTION_BITS["triangle_quality_rollback"]
        iterations.extend({**row, "safety_round": safety_round} for row in quality_rows)
        if not np.any(new_collisions) and not np.any(quality_rollback):
            break
    gate_reports = {
        "component_gate": {"source_component_votes": votes, "inconsistent_proxy_faces": mismatch_faces},
        "collision_gate": {"candidate_nonadjacent_pairs": collision_pairs, "rolled_back_vertices": int(np.count_nonzero(collisions))},
        "triangle_quality_rollback": {"iterations": iterations, "remaining_issue_faces": int(np.count_nonzero(_triangle_issue_mask(candidate, proxy_points, proxy_faces, thresholds)))},
        "source_topology": source_topology,
    }
    resolved = {**asdict(thresholds), "resolved_max_projection_distance": float(max_distance), "resolved_collision_tolerance": collision_tolerance}
    return candidate, closest, face_ids, target_components, distances, raw_dot, rejection, gate_reports, resolved

# @entry
def project_closure_to_source(
    proxy_points: np.ndarray,
    proxy_faces: np.ndarray,
    source_points: np.ndarray,
    source_faces: np.ndarray,
    *,
    thresholds: ProjectionThresholds | None = None,
    reliable_source_face_mask: np.ndarray | None = None,
    compute_self_intersections: bool = False,
    self_intersection_max_candidate_pairs: int = 2_000_000,
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Project trusted vertices exactly while preserving closure connectivity."""
    proxy_points, proxy_faces_i64 = _validate_mesh("proxy", proxy_points, proxy_faces)
    source_points, source_faces = _validate_mesh("source", source_points, source_faces)
    thresholds = thresholds or ProjectionThresholds()
    candidate, closest, face_ids, target_components, distances, raw_dot, rejection, gate_reports, resolved = _project_with_gates(
        proxy_points, proxy_faces_i64, source_points, source_faces, thresholds,
        reliable_source_face_mask,
    )
    applied = rejection == 0
    intersections = self_intersection_report(candidate, proxy_faces_i64, max_candidate_pairs=self_intersection_max_candidate_pairs) if compute_self_intersections else {"status": "not_computed", "passed": False, "reason": "global self-intersection validation belongs to the caller acceptance path"}
    provenance = _projection_provenance(proxy_faces_i64, applied, face_ids, target_components, distances, raw_dot, rejection)
    report = _build_report(
        proxy_points,
        proxy_faces_i64,
        candidate,
        closest,
        distances,
        applied,
        rejection,
        resolved,
        gate_reports,
        intersections,
    )
    return candidate, np.asarray(proxy_faces).copy(), provenance, report

def rollback_reported_self_intersections(
    points: np.ndarray,
    proxy_points: np.ndarray,
    faces: np.ndarray,
    provenance: dict[str, dict[str, np.ndarray]],
    intersection_report: dict[str, Any],
    *,
    vertex_rings: int = 0,
) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Fallback projected vertices touched by reported non-adjacent intersections."""
    if vertex_rings < 0:
        raise ValueError("vertex_rings must be non-negative")
    pairs = np.asarray(intersection_report.get("reported_pairs", []), dtype=np.int64).reshape(-1, 2)
    result = np.asarray(points, dtype=np.float64).copy()
    updated = {group: {name: values.copy() for name, values in arrays.items()} for group, arrays in provenance.items()}
    applied = updated["point_data"]["source_projection_applied"].astype(bool)
    triangle_faces = np.asarray(faces, dtype=np.int64)
    core_vertices = (
        np.unique(triangle_faces[pairs].ravel())
        if pairs.size
        else np.zeros(0, dtype=np.int64)
    )
    vertices = core_vertices
    if vertices.size and vertex_rings:
        active = np.zeros(result.shape[0], dtype=bool)
        active[vertices] = True
        for _ in range(int(vertex_rings)):
            incident_faces = np.any(active[triangle_faces], axis=1)
            active[triangle_faces[incident_faces].ravel()] = True
        vertices = np.flatnonzero(active)
    vertices = vertices[applied[vertices]]
    result[vertices] = np.asarray(proxy_points, dtype=np.float64)[vertices]
    applied[vertices] = False
    updated["point_data"]["source_projection_applied"] = applied.astype(np.uint8)
    updated["point_data"]["source_projection_fallback"] = (~applied).astype(np.uint8)
    updated["point_data"]["source_projection_rejection_mask"][vertices] |= REJECTION_BITS["reported_self_intersection_rollback"]
    updated["cell_data"]["source_projected_vertex_count"] = applied[triangle_faces].sum(axis=1).astype(np.uint8)
    updated["cell_data"]["source_projected_vertex_fraction"] = applied[triangle_faces].mean(axis=1).astype(np.float32)
    return result, updated, {
        "reported_pairs": int(pairs.shape[0]),
        "intersection_core_vertices": int(core_vertices.size),
        "rollback_vertex_rings": int(vertex_rings),
        "rolled_back_vertices": int(vertices.size),
        "requires_global_recheck": True,
    }
