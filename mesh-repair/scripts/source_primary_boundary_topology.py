from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from source_primary_boundary_relations import find_loop_self_intersections


SCHEMA_VERSION = "source-primary-boundary-inventory/1.0"
DEFAULT_THRESHOLDS = {
    "absolute_tolerance": 0.0,
    "minimum_polygon_area": 0.0,
    "pca_min_normal_eigengap_ratio": 1.0e-8,
    "newell_min_area_scale_ratio": 1.0e-10,
    "normal_consistency_min_dot": 0.0,
    "normal_consistency_min_resultant": 0.50,
    "hole_boundary_min_newell_source_opposition": 0.20,
    "outside_direction_min_abs_dot": 0.05,
    "pair_min_similarity": 0.55,
    "pair_min_normal_abs_dot": 0.75,
    "near_seam_max_gap_edge_ratio": 0.50,
    "near_seam_max_gap_scale_ratio": 0.01,
    "pair_max_gap_edge_ratio": 4.0,
    "pair_max_gap_scale_ratio": 0.25,
    "pair_max_center_distance_scale_ratio": 1.25,
    "pair_min_axial_fraction": 0.35,
    "pair_score_ambiguity_margin": 0.05,
    "nested_min_normal_abs_dot": 0.90,
    "nested_max_plane_offset_scale_ratio": 0.03,
    "nested_max_center_distance_scale_ratio": 1.0,
    "nested_max_planarity_scale_ratio": 0.03,
    "nested_boundary_clearance_scale_ratio": 1.0e-6,
    "slit_max_width_major_ratio": 0.08,
    "slit_min_aspect_ratio": 8.0,
    "slit_min_major_edge_ratio": 6.0,
    "planar_max_distance_scale_ratio": 0.02,
    "planar_max_curvature_dimensionless_rms": 0.05,
}


def build_edge_topology(faces: np.ndarray) -> dict[str, np.ndarray]:
    if faces.size == 0:
        empty_edges = np.empty((0, 2), dtype=np.int64)
        return {
            "edges": empty_edges,
            "starts": np.zeros(0, dtype=np.int64),
            "counts": np.zeros(0, dtype=np.int64),
            "face_ids": np.zeros(0, dtype=np.int64),
            "directed": empty_edges.copy(),
            "boundary_edges": empty_edges.copy(),
            "boundary_faces": np.zeros(0, dtype=np.int64),
            "boundary_directed": empty_edges.copy(),
            "manifold_edges": empty_edges.copy(),
            "manifold_face_pairs": empty_edges.copy(),
            "degenerate_occurrences": np.empty((0, 3), dtype=np.int64),
        }
    directed = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape((-1, 2))
    face_ids = np.repeat(np.arange(faces.shape[0], dtype=np.int64), 3)
    valid = directed[:, 0] != directed[:, 1]
    degenerate = np.column_stack((face_ids[~valid], directed[~valid]))
    directed, face_ids = directed[valid], face_ids[valid]
    canonical = np.sort(directed, axis=1)
    order = np.lexsort((canonical[:, 1], canonical[:, 0]))
    canonical, directed, face_ids = canonical[order], directed[order], face_ids[order]
    changes = np.ones(canonical.shape[0], dtype=bool)
    changes[1:] = np.any(canonical[1:] != canonical[:-1], axis=1)
    starts = np.flatnonzero(changes)
    counts = np.diff(np.append(starts, canonical.shape[0]))
    edges = canonical[starts]
    boundary_groups = np.flatnonzero(counts == 1)
    boundary_positions = starts[boundary_groups]
    manifold_groups = np.flatnonzero(counts == 2)
    manifold_positions = starts[manifold_groups]
    manifold_pairs = np.column_stack((face_ids[manifold_positions], face_ids[manifold_positions + 1]))
    distinct = manifold_pairs[:, 0] != manifold_pairs[:, 1]
    return {
        "edges": edges,
        "starts": starts,
        "counts": counts,
        "face_ids": face_ids,
        "directed": directed,
        "boundary_edges": edges[boundary_groups],
        "boundary_faces": face_ids[boundary_positions],
        "boundary_directed": directed[boundary_positions],
        "manifold_edges": edges[manifold_groups][distinct],
        "manifold_face_pairs": manifold_pairs[distinct],
        "degenerate_occurrences": degenerate,
    }


def extract_boundary_graphs(
    points: np.ndarray,
    topology: Mapping[str, np.ndarray],
    source_triangles: np.ndarray,
    source_vertices: np.ndarray,
    components: np.ndarray,
    tolerance: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    edges, directed = topology["boundary_edges"], topology["boundary_directed"]
    face_ids = topology["boundary_faces"]
    adjacency: dict[int, list[int]] = {}
    for edge_id, edge in enumerate(edges):
        adjacency.setdefault(int(edge[0]), []).append(edge_id)
        adjacency.setdefault(int(edge[1]), []).append(edge_id)
    remaining, rows, diagnostics = set(range(edges.shape[0])), [], []
    while remaining:
        stack, edge_ids = [min(remaining)], []
        remaining.remove(stack[0])
        while stack:
            edge_id = stack.pop()
            edge_ids.append(edge_id)
            for vertex in edges[edge_id]:
                for neighbor in adjacency[int(vertex)]:
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
        edge_ids.sort()
        graph_edges = edges[edge_ids]
        graph_directed = directed[edge_ids]
        graph_faces = sorted(set(int(face_ids[value]) for value in edge_ids))
        graph_component_ids = sorted(set(int(components[value]) for value in graph_faces))
        degrees: dict[int, int] = {}
        incoming: dict[int, int] = {}
        outgoing: dict[int, int] = {}
        for edge, source_edge in zip(graph_edges, graph_directed, strict=True):
            for vertex in edge:
                degrees[int(vertex)] = degrees.get(int(vertex), 0) + 1
            outgoing[int(source_edge[0])] = outgoing.get(int(source_edge[0]), 0) + 1
            incoming[int(source_edge[1])] = incoming.get(int(source_edge[1]), 0) + 1
        reasons = []
        if len(graph_component_ids) != 1:
            reasons.append("boundary_graph_spans_multiple_face_components")
        if len(graph_edges) < 3:
            reasons.append("boundary_graph_has_fewer_than_three_edges")
        if any(value == 1 for value in degrees.values()):
            reasons.append("open_boundary_chain")
        if any(value > 2 for value in degrees.values()):
            reasons.append("branched_boundary_graph")
        if any(value != 2 for value in degrees.values()):
            reasons.append("boundary_vertex_degree_not_two")
        if any(incoming.get(vertex, 0) != 1 or outgoing.get(vertex, 0) != 1 for vertex in degrees):
            reasons.append("boundary_winding_conflict")
        if any(float(np.linalg.norm(points[int(edge[1])] - points[int(edge[0])])) <= tolerance for edge in graph_edges):
            reasons.append("zero_length_boundary_edge")
        ordered = _traverse_directed_loop(graph_directed) if not reasons else None
        if ordered is None and not reasons:
            reasons.append("boundary_loop_traversal_failed")
        intersections = find_loop_self_intersections(points, ordered, tolerance) if ordered else []
        if intersections:
            reasons.append("self_intersecting_boundary_loop")
        edge_pairs = [(int(edge[0]), int(edge[1])) for edge in graph_edges]
        source_edges = [sorted((int(source_vertices[left]), int(source_vertices[right]))) for left, right in edge_pairs]
        edge_records = [
            {
                "edge": list(edge_pairs[index]),
                "source_edge": source_edges[index],
                "directed_edge": graph_directed[index].astype(int).tolist(),
                "source_directed_edge": [
                    int(source_vertices[int(graph_directed[index, 0])]),
                    int(source_vertices[int(graph_directed[index, 1])]),
                ],
                "incident_face_id": int(face_ids[edge_ids[index]]),
                "source_triangle_id": int(source_triangles[int(face_ids[edge_ids[index]])]),
                "incidence": 1,
            }
            for index in range(len(edge_ids))
        ]
        graph_id = stable_id("boundary_graph", {
            "edges": sorted(source_edges),
            "source_triangles": sorted(int(source_triangles[value]) for value in graph_faces),
        })
        row = {
            "graph_id": graph_id,
            "boundary_edges": [list(edge) for edge in edge_pairs],
            "boundary_edge_records": edge_records,
            "source_boundary_edges": sorted(source_edges),
            "incident_face_ids": graph_faces,
            "source_triangle_ids": [int(source_triangles[value]) for value in graph_faces],
            "source_face_provenance": [
                {"face_id": value, "source_triangle_id": int(source_triangles[value])}
                for value in graph_faces
            ],
            "component_ids": graph_component_ids,
            "ordered_vertex_ids": ordered,
            "ordered_source_vertex_ids": (
                [int(source_vertices[value]) for value in ordered]
                if ordered is not None
                else None
            ),
        }
        rows.append(row)
        if reasons:
            diagnostics.append({
                **row,
                "reason_codes": sorted(set(reasons)),
                "vertex_degrees": [{"vertex_id": key, "degree": degrees[key]} for key in sorted(degrees)],
                "self_intersection_segment_pairs": intersections,
            })
    return rows, sorted(diagnostics, key=lambda item: item["graph_id"])


def topology_diagnostics(
    topology: Mapping[str, np.ndarray],
    source_triangles: np.ndarray,
    source_vertices: np.ndarray,
) -> dict[str, Any]:
    non_manifold, winding = [], []
    for group in np.flatnonzero(topology["counts"] > 2):
        start, count = int(topology["starts"][group]), int(topology["counts"][group])
        face_ids = sorted(set(int(value) for value in topology["face_ids"][start:start + count]))
        non_manifold.append({
            "edge": topology["edges"][group].astype(int).tolist(),
            "source_edge": sorted(int(source_vertices[value]) for value in topology["edges"][group]),
            "occurrence_count": count,
            "face_ids": face_ids,
            "source_triangle_ids": sorted(int(source_triangles[value]) for value in face_ids),
        })
    for group in np.flatnonzero(topology["counts"] == 2):
        start = int(topology["starts"][group])
        edge = topology["edges"][group]
        directed = topology["directed"][start:start + 2]
        signs = directed[:, 0] == edge[0]
        if bool(signs[0]) == bool(signs[1]):
            face_ids = sorted(set(int(value) for value in topology["face_ids"][start:start + 2]))
            winding.append({
                "edge": edge.astype(int).tolist(),
                "source_edge": sorted(int(source_vertices[value]) for value in edge),
                "face_ids": face_ids,
                "source_triangle_ids": sorted(int(source_triangles[value]) for value in face_ids),
            })
    degenerate = [
        {
            "face_id": int(row[0]),
            "edge": [int(row[1]), int(row[2])],
            "source_edge": sorted((int(source_vertices[int(row[1])]), int(source_vertices[int(row[2])]))),
            "source_triangle_id": int(source_triangles[int(row[0])]),
        }
        for row in topology["degenerate_occurrences"]
    ]
    return {
        "non_manifold_edges": non_manifold,
        "inconsistent_winding_edges": winding,
        "degenerate_edge_occurrences": degenerate,
    }


def face_components(face_count: int, topology: Mapping[str, np.ndarray]) -> np.ndarray:
    parent = np.arange(face_count, dtype=np.int64)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    for group in np.flatnonzero(topology["counts"] > 1):
        start, count = int(topology["starts"][group]), int(topology["counts"][group])
        face_ids = sorted(set(int(value) for value in topology["face_ids"][start:start + count]))
        for face_id in face_ids[1:]:
            left, right = find(face_ids[0]), find(face_id)
            parent[max(left, right)] = min(left, right)
    roots = [find(value) for value in range(face_count)]
    labels = {root: index for index, root in enumerate(sorted(set(roots)))}
    return np.asarray([labels[root] for root in roots], dtype=np.int64)


def canonical_cycle(values: Sequence[int]) -> list[int]:
    values = [int(value) for value in values]
    pivot = values.index(min(values))
    forward = values[pivot:] + values[:pivot]
    reversed_values = list(reversed(values))
    reverse_pivot = reversed_values.index(min(reversed_values))
    backward = reversed_values[reverse_pivot:] + reversed_values[:reverse_pivot]
    return min(forward, backward)


def validated_inputs(
    points: np.ndarray,
    faces: np.ndarray,
    source_triangles: np.ndarray | None,
    source_vertices: np.ndarray | None,
    external: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("points and triangle faces must have shapes (N, 3) and (M, 3)")
    if not np.all(np.isfinite(points)):
        raise ValueError("points must be finite")
    if faces.size and (int(faces.min()) < 0 or int(faces.max()) >= points.shape[0]):
        raise ValueError("faces contain out-of-range point ids")
    source_triangles = np.arange(faces.shape[0], dtype=np.int64) if source_triangles is None else np.asarray(source_triangles, dtype=np.int64)
    source_vertices = np.arange(points.shape[0], dtype=np.int64) if source_vertices is None else np.asarray(source_vertices, dtype=np.int64)
    if source_triangles.shape != (faces.shape[0],) or source_vertices.shape != (points.shape[0],):
        raise ValueError("source ids must match face and point counts")
    if np.unique(source_vertices).size != source_vertices.size:
        raise ValueError("source_vertex_ids must be unique for durable loop identity")
    if external is not None:
        external = np.asarray(external, dtype=np.float64)
        if external.shape not in {(3,), (faces.shape[0], 3)} or not np.all(np.isfinite(external)):
            raise ValueError("face_external_directions must have shape (3,) or (face_count, 3) and be finite")
        lengths = np.linalg.norm(external, axis=-1, keepdims=True)
        external = np.divide(external, lengths, out=np.zeros_like(external), where=lengths > 0.0)
    return points, faces, source_triangles, source_vertices, external


def validated_component_ids(values: np.ndarray, face_count: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.int64)
    if result.shape != (face_count,):
        raise ValueError("face_component_ids must match face count")
    return result


def normalized_thresholds(points: np.ndarray, values: Mapping[str, float] | None) -> dict[str, float]:
    unknown = set(values or {}) - set(DEFAULT_THRESHOLDS)
    if unknown:
        raise ValueError(f"unknown boundary inventory threshold(s): {', '.join(sorted(unknown))}")
    result = {**DEFAULT_THRESHOLDS, **(values or {})}
    if any(not np.isfinite(value) or value < 0.0 for value in result.values()):
        raise ValueError("boundary inventory thresholds must be finite and non-negative")
    diagonal = float(np.linalg.norm(np.ptp(points, axis=0))) if points.size else 0.0
    if result["absolute_tolerance"] == 0.0:
        result["absolute_tolerance"] = max(diagonal * 1.0e-12, np.finfo(np.float64).eps)
    if result["minimum_polygon_area"] == 0.0:
        result["minimum_polygon_area"] = result["absolute_tolerance"] ** 2
    return {key: float(value) for key, value in result.items()}


def stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(SCHEMA_VERSION.encode("utf-8") + b"\0" + encoded).hexdigest()[:16]
    return f"{prefix}_{digest}"


def geometry_fingerprint(points: np.ndarray, tolerance: float) -> str:
    local = points - points[0] if points.size else points
    quantized = np.rint(local / max(tolerance, 1e-30)).astype("<i8", copy=False)
    return hashlib.sha256(quantized.tobytes(order="C")).hexdigest()


def mesh_fingerprint(
    points: np.ndarray,
    faces: np.ndarray,
    source_triangles: np.ndarray,
    source_vertices: np.ndarray,
) -> str:
    digest = hashlib.sha256(SCHEMA_VERSION.encode("utf-8"))
    for values, dtype in ((points, "<f8"), (faces, "<i8"), (source_triangles, "<i8"), (source_vertices, "<i8")):
        digest.update(np.asarray(values, dtype=dtype).tobytes(order="C"))
    return digest.hexdigest()


def _traverse_directed_loop(directed: np.ndarray) -> list[int] | None:
    outgoing = {int(edge[0]): int(edge[1]) for edge in directed}
    start = min(outgoing)
    ordered, current = [], start
    for _ in range(len(directed)):
        if current in ordered or current not in outgoing:
            return None
        ordered.append(current)
        current = outgoing[current]
    return ordered if current == start and len(ordered) == len(directed) else None
