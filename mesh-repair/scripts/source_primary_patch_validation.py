from __future__ import annotations

from typing import Any, Mapping, TYPE_CHECKING

import numpy as np

from source_primary_patch_footprint import validate_patch_footprint
from source_primary_patch_inputs import validate_boundary_cycle_graph
from source_primary_patch_provenance_validation import (
    validate_curvature_evidence,
    validate_delta_provenance,
    validate_normal_evidence,
    validate_proxy_provenance,
)
from source_primary_patch_source_evidence import validate_source_normal_evidence

if TYPE_CHECKING:
    from source_primary_patch_contract import (
        BoundaryMapping,
        PatchCandidate,
        PatchDelta,
    )


# @entry
def validate_patch_candidate(
    source_points: np.ndarray,
    source_faces: np.ndarray,
    source_triangle_index: np.ndarray,
    candidate: PatchCandidate,
) -> list[str]:
    from source_primary_patch_contract import (
        PATCH_METHODS,
        build_source_provenance,
        validate_source_arrays,
    )

    errors = validate_source_arrays(source_points, source_faces, source_triangle_index)
    if errors:
        return errors
    if candidate.status not in {"candidate", "rejected"}:
        errors.append("candidate status must be candidate or rejected")
    if candidate.method not in PATCH_METHODS:
        errors.append(f"unsupported patch method: {candidate.method!r}")
    if candidate.status == "rejected":
        if candidate.delta.appended_points.size or candidate.delta.appended_faces.size:
            errors.append("a rejected candidate must expose an empty consumable delta")
        if not candidate.failure_reason_codes:
            errors.append("a rejected candidate must include a stable failure reason")
        errors.extend(validate_proxy_provenance(candidate.proxy_provenance))
        return errors
    if candidate.failure_reason_codes:
        errors.append("a consumable candidate cannot include failure reason codes")
    if candidate.method == "slit_weld":
        errors.append("slit_weld cannot consume a delta without changing source connectivity")
    expected_source = build_source_provenance(source_points, source_faces, source_triangle_index)
    if dict(candidate.source_provenance) != expected_source:
        errors.append("candidate source provenance does not match the immutable source")
    errors.extend(validate_normal_evidence(candidate.normal, candidate.method))
    errors.extend(
        validate_source_normal_evidence(
            np.asarray(source_points),
            np.asarray(source_faces),
            candidate.boundary_mapping,
            candidate.normal,
            candidate.method,
            candidate.delta,
        )
    )
    errors.extend(validate_curvature_evidence(candidate.curvature))
    errors.extend(
        validate_proxy_provenance(
            candidate.proxy_provenance,
            consumable=True,
            candidate_method=candidate.method,
            candidate_normal=candidate.normal,
        )
    )
    errors.extend(
        _validate_delta(
            np.asarray(source_points),
            np.asarray(source_faces),
            np.asarray(source_triangle_index),
            candidate.delta,
            candidate.boundary_mapping,
            candidate.method,
            candidate.normal,
        )
    )
    return errors


def _validate_delta(
    source_points: np.ndarray,
    source_faces: np.ndarray,
    source_triangle_index: np.ndarray,
    delta: PatchDelta,
    mappings: tuple[BoundaryMapping, ...],
    method: str,
    normal: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    new_points = delta.appended_points
    new_faces = delta.appended_faces
    source_count = source_points.shape[0]
    total_count = source_count + new_points.shape[0]
    if not mappings:
        errors.append("a consumable patch must map at least one source boundary")
        return errors
    if not np.all(np.isfinite(new_points)):
        errors.append("appended_points contain non-finite values")
    if method == "paired_loop_zipper" and new_points.shape[0]:
        errors.append("paired_loop_zipper must bridge mapped source loops without new points")
    if new_faces.shape[0] == 0:
        errors.append("a consumable patch must append at least one face")
        return errors
    if np.min(new_faces) < 0 or np.max(new_faces) >= total_count:
        errors.append("appended_faces contain out-of-range point IDs")
        return errors
    if np.any(
        (new_faces[:, 0] == new_faces[:, 1])
        | (new_faces[:, 1] == new_faces[:, 2])
        | (new_faces[:, 2] == new_faces[:, 0])
    ):
        errors.append("appended_faces contain repeated point IDs")
    mapped_ids: set[int] = set()
    expected_edges: dict[tuple[int, int], tuple[int, int]] = {}
    source_edge_faces = _mapped_edge_face_map(source_faces, mappings, source_count)
    mapping_errors: list[str] = []
    for mapping in mappings:
        mapping_errors.extend(
            _validate_boundary_mapping(
                source_faces,
                source_triangle_index,
                source_edge_faces,
                mapping,
                source_count,
                expected_edges,
                mapped_ids,
            )
        )
    errors.extend(mapping_errors)
    if mapping_errors:
        return errors
    referenced_source = set(
        int(value) for value in np.unique(new_faces[new_faces < source_count])
    )
    if not referenced_source.issubset(mapped_ids):
        errors.append("appended_faces reference source vertices outside mapped boundaries")
    if new_points.shape[0]:
        referenced_new = set(
            int(value) for value in np.unique(new_faces[new_faces >= source_count])
        )
        expected_new = set(range(source_count, total_count))
        if referenced_new != expected_new:
            errors.append("every appended point must be referenced by an appended face")
    errors.extend(
        validate_delta_provenance(
            delta,
            new_points.shape[0],
            new_faces.shape[0],
            method,
            mappings,
        )
    )
    errors.extend(_validate_patch_topology(new_faces, expected_edges, len(mappings), method))
    all_points = np.vstack([source_points.astype(np.float64, copy=False), new_points])
    triangles = all_points[new_faces]
    double_area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    edge_lengths = np.linalg.norm(triangles - np.roll(triangles, -1, axis=1), axis=2)
    local_scale = np.maximum(edge_lengths.max(axis=1), np.finfo(np.float64).tiny)
    coordinate_ulp = np.max(np.abs(np.spacing(triangles)), axis=(1, 2))
    area_tolerance = np.maximum(
        np.finfo(np.float64).eps * local_scale * local_scale * 64.0,
        coordinate_ulp * local_scale * 8.0,
    )
    if np.any(double_area <= area_tolerance):
        errors.append("appended_faces contain numerically degenerate triangles")
    errors.extend(
        validate_patch_footprint(
            source_points,
            new_points,
            new_faces,
            mappings,
            normal,
            source_count,
            method,
            delta.point_provenance,
        )
    )
    return errors


def _validate_boundary_mapping(
    source_faces: np.ndarray,
    source_triangle_index: np.ndarray,
    source_edge_faces: Mapping[tuple[int, int], list[int]],
    mapping: BoundaryMapping,
    source_count: int,
    expected_edges: dict[tuple[int, int], tuple[int, int]],
    mapped_ids: set[int],
) -> list[str]:
    errors: list[str] = []
    source_ids = tuple(int(value) for value in mapping.source_vertex_ids)
    if source_ids != tuple(int(value) for value in mapping.candidate_vertex_ids):
        errors.append(f"region {mapping.region_id}: boundary mapping is not identity")
    if mapping.orientation != "source_face_winding_induced":
        errors.append(f"region {mapping.region_id}: boundary orientation contract is invalid")
    if not mapping.closed:
        errors.append(f"region {mapping.region_id}: local repair boundaries must be closed")
    minimum_vertices = 3 if mapping.closed else 2
    if len(source_ids) < minimum_vertices or len(set(source_ids)) != len(source_ids):
        errors.append(f"region {mapping.region_id}: boundary vertex IDs are invalid")
        return errors
    if min(source_ids) < 0 or max(source_ids) >= source_count:
        errors.append(f"region {mapping.region_id}: boundary vertex ID is out of range")
        return errors
    graph_error, graph_diagnostics = validate_boundary_cycle_graph(
        source_faces, np.asarray(source_ids, dtype=np.int64)
    )
    if graph_error is not None:
        errors.append(
            f"region {mapping.region_id}: {graph_error}: {graph_diagnostics}"
        )
    overlap = mapped_ids.intersection(source_ids)
    if overlap:
        errors.append(f"region {mapping.region_id}: mapped boundaries share source vertices")
    edge_count = len(source_ids) if mapping.closed else len(source_ids) - 1
    if len(mapping.source_edge_face_ids) != edge_count:
        errors.append(f"region {mapping.region_id}: source edge-face mapping length mismatch")
        return errors
    if len(mapping.source_triangle_indices) != edge_count:
        errors.append(f"region {mapping.region_id}: source triangle mapping length mismatch")
        return errors
    mapped_ids.update(source_ids)
    for index in range(edge_count):
        left = source_ids[index]
        right = source_ids[(index + 1) % len(source_ids)]
        face_id = int(mapping.source_edge_face_ids[index])
        if face_id < 0 or face_id >= source_faces.shape[0]:
            errors.append(f"region {mapping.region_id}: boundary face ID is out of range")
            continue
        if int(mapping.source_triangle_indices[index]) != int(source_triangle_index[face_id]):
            errors.append(f"region {mapping.region_id}: source triangle provenance mismatch")
        if not _face_has_directed_edge(source_faces[face_id], left, right):
            errors.append(
                f"region {mapping.region_id}: boundary orientation is not source-face induced"
            )
        key = (left, right) if left < right else (right, left)
        if source_edge_faces.get(key, []) != [face_id]:
            errors.append(f"region {mapping.region_id}: source boundary edge incidence is not one")
        if key in expected_edges:
            errors.append(f"region {mapping.region_id}: duplicate mapped boundary edge")
        expected_edges[key] = (right, left)
    return errors


def _validate_patch_topology(
    faces: np.ndarray,
    expected_edges: Mapping[tuple[int, int], tuple[int, int]],
    mapping_count: int,
    method: str,
) -> list[str]:
    occurrences: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for face_id, face in enumerate(faces):
        for left, right in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            left_id, right_id = int(left), int(right)
            key = (
                (left_id, right_id)
                if left_id < right_id
                else (right_id, left_id)
            )
            occurrences.setdefault(key, []).append((face_id, left_id, right_id))
    errors: list[str] = []
    expected_mapping_count = 2 if method == "paired_loop_zipper" else 1
    if mapping_count != expected_mapping_count:
        errors.append(
            f"{method} must map exactly {expected_mapping_count} source boundary loop(s)"
        )
    for edge, expected_direction in expected_edges.items():
        rows = occurrences.get(edge, [])
        directions = [(row[1], row[2]) for row in rows]
        if directions != [expected_direction]:
            errors.append(f"mapped boundary edge {edge} is not shared once in opposite direction")
    for edge, rows in occurrences.items():
        if edge in expected_edges:
            continue
        directions = [(row[1], row[2]) for row in rows]
        if len(rows) != 2 or directions[0] != tuple(reversed(directions[1])):
            errors.append(f"patch interior edge {edge} is not an opposite directed pair")
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
    if len(reached) != faces.shape[0]:
        errors.append("patch faces must form one boundary-connected component")
    vertex_count = int(np.unique(faces).size)
    euler_characteristic = vertex_count - len(occurrences) + int(faces.shape[0])
    expected_euler = 2 - expected_mapping_count
    if euler_characteristic != expected_euler:
        errors.append(
            "patch topology is not a genus-zero disk/annulus for its mapped boundaries"
        )
    return errors


def _face_has_directed_edge(face: np.ndarray, left: int, right: int) -> bool:
    return any(
        int(face[index]) == left and int(face[(index + 1) % 3]) == right
        for index in range(3)
    )


def _mapped_edge_face_map(
    faces: np.ndarray,
    mappings: tuple[BoundaryMapping, ...],
    point_count: int,
) -> dict[tuple[int, int], list[int]]:
    directed = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    canonical = np.sort(directed, axis=1)
    keys = canonical[:, 0] * point_count + canonical[:, 1]
    wanted: list[tuple[int, int]] = []
    for mapping in mappings:
        ids = mapping.source_vertex_ids
        edge_count = len(ids) if mapping.closed else max(len(ids) - 1, 0)
        for index in range(edge_count):
            left = ids[index]
            right = ids[(index + 1) % len(ids)]
            wanted.append((left, right) if left < right else (right, left))
    wanted_keys = np.asarray(
        [left * point_count + right for left, right in wanted], dtype=np.int64
    )
    matched = np.flatnonzero(np.isin(keys, wanted_keys)) if wanted else np.empty(0, dtype=np.int64)
    face_ids = np.repeat(np.arange(faces.shape[0], dtype=np.int64), 3)
    result: dict[tuple[int, int], list[int]] = {}
    for occurrence in matched:
        edge = (
            int(canonical[occurrence, 0]),
            int(canonical[occurrence, 1]),
        )
        result.setdefault(edge, []).append(int(face_ids[occurrence]))
    return result
