from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from source_primary_outside_evidence import summarize_incident_outside_directions


def derive_expected_patch_normals(
    source_points: np.ndarray,
    source_faces: np.ndarray,
    patch_faces: np.ndarray,
    boundary_loops: Sequence[np.ndarray],
    face_external_directions: np.ndarray | None,
    *,
    minimum_outside_dot: float = 0.05,
) -> dict[str, Any]:
    points = np.asarray(source_points, dtype=np.float64)
    faces = np.asarray(source_faces, dtype=np.int64)
    patch = np.asarray(patch_faces, dtype=np.int64)
    loops = tuple(_open_loop(loop) for loop in boundary_loops)
    edge_faces = _edge_face_map(faces)
    boundary_edges = [
        _edge(int(loop[index]), int(loop[(index + 1) % loop.size]))
        for loop in loops
        for index in range(loop.size)
    ]
    owners = [edge_faces.get(edge, []) for edge in boundary_edges]
    if not boundary_edges or any(len(values) != 1 for values in owners):
        return _failure("boundary_index_not_shared")
    incident = np.asarray(sorted({values[0] for values in owners}), dtype=np.int64)
    source_normals, source_areas = _face_geometry(points, faces)
    outside = summarize_incident_outside_directions(
        incident,
        source_areas,
        face_external_directions,
        minimum_outside_dot,
    )
    if outside["status"] != "computed":
        return _failure(str(outside["status"]), outside=outside)
    outward = np.asarray(outside["direction"], dtype=np.float64)
    weighted = np.sum(
        source_areas[incident, None] * source_normals[incident], axis=0
    )
    local = _unit(weighted)
    if local is None or float(np.dot(local, outward)) < minimum_outside_dot:
        return _failure("external_direction_conflicts_source_winding", outside=outside)
    vertex_samples: dict[int, list[np.ndarray]] = {}
    for edge, values in zip(boundary_edges, owners, strict=True):
        normal = source_normals[values[0]]
        if float(np.dot(normal, outward)) < minimum_outside_dot:
            return _failure(
                "external_direction_conflicts_source_winding", outside=outside
            )
        vertex_samples.setdefault(edge[0], []).append(normal)
        vertex_samples.setdefault(edge[1], []).append(normal)
    vertex_normals = {
        vertex: _unit(np.sum(samples, axis=0))
        for vertex, samples in vertex_samples.items()
    }
    if any(value is None for value in vertex_normals.values()):
        return _failure("source_boundary_normal_unavailable", outside=outside)
    expected = []
    for face in patch:
        samples = [
            vertex_normals[int(vertex)]
            for vertex in np.unique(face)
            if int(vertex) in vertex_normals
        ]
        normal = _unit(np.sum(samples, axis=0)) if samples else local
        if normal is None or float(np.dot(normal, outward)) < minimum_outside_dot:
            return _failure("source_boundary_normal_unavailable", outside=outside)
        expected.append(normal)
    return {
        "success": True,
        "reason_code": None,
        "expected_face_normals": np.asarray(expected, dtype=np.float64),
        "outside": outside,
        "incident_source_face_ids": incident.astype(int).tolist(),
        "source_reference_normal": local.tolist(),
    }


def _face_geometry(
    points: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    triangles = points[faces]
    raw = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    lengths = np.linalg.norm(raw, axis=1)
    normals = np.divide(
        raw,
        lengths[:, None],
        out=np.zeros_like(raw),
        where=lengths[:, None] > 0.0,
    )
    return normals, 0.5 * lengths


def _edge_face_map(faces: np.ndarray) -> dict[tuple[int, int], list[int]]:
    result: dict[tuple[int, int], list[int]] = {}
    for face_id, face in enumerate(faces):
        for index in range(3):
            result.setdefault(
                _edge(int(face[index]), int(face[(index + 1) % 3])), []
            ).append(face_id)
    return result


def _open_loop(values: np.ndarray) -> np.ndarray:
    loop = np.asarray(values, dtype=np.int64).reshape(-1)
    return loop[:-1] if loop.size > 1 and loop[0] == loop[-1] else loop


def _edge(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)


def _unit(value: np.ndarray) -> np.ndarray | None:
    length = float(np.linalg.norm(value))
    return None if not np.isfinite(length) or length <= 1.0e-30 else value / length


def _failure(reason_code: str, **details: Any) -> dict[str, Any]:
    return {
        "success": False,
        "reason_code": reason_code,
        "expected_face_normals": None,
        **details,
    }
