from __future__ import annotations

from typing import Any

import numpy as np


__all__ = [
    "build_boundary_vertex_normals",
    "estimate_boundary_curvature",
    "select_boundary_owner_one_ring",
]


def select_boundary_owner_one_ring(
    faces: np.ndarray,
    loop: np.ndarray,
    edge_face_ids: np.ndarray,
    *,
    points: np.ndarray,
    minimum_face_normal_consistency: float,
) -> tuple[np.ndarray | None, str | None]:
    """Select the manifold face fan owned by the loop's boundary edges."""

    candidate_ids = np.flatnonzero(np.any(np.isin(faces, loop), axis=1))
    candidate_set = {int(face_id) for face_id in candidate_ids}
    adjacency: dict[int, set[int]] = {face_id: set() for face_id in candidate_set}
    edge_owners: dict[tuple[int, int], list[int]] = {}
    for face_id in candidate_ids:
        face = faces[int(face_id)]
        for index in range(3):
            left, right = sorted(
                (int(face[index]), int(face[(index + 1) % 3]))
            )
            edge = (left, right)
            edge_owners.setdefault(edge, []).append(int(face_id))
    for owners in edge_owners.values():
        for owner_face_id in owners:
            adjacency[owner_face_id].update(
                other for other in owners if other != owner_face_id
            )
    seeds = {int(face_id) for face_id in edge_face_ids}
    if not seeds or not seeds.issubset(candidate_set):
        return None, "boundary_edge_owner_component_invalid"
    owner_component: set[int] = set()
    pending: list[int] = [next(iter(seeds))]
    while pending:
        current_face_id = pending.pop()
        if current_face_id in owner_component:
            continue
        owner_component.add(current_face_id)
        pending.extend(adjacency[current_face_id] - owner_component)
    if not seeds.issubset(owner_component):
        return None, "boundary_edge_owner_component_ambiguous"
    owner_face_ids = candidate_ids[
        np.asarray([int(face_id) in owner_component for face_id in candidate_ids])
    ]
    if not _is_manifold_boundary_one_ring(faces, loop, owner_face_ids):
        return None, "boundary_owner_one_ring_non_manifold"
    orientation_error = _owner_face_orientation_error(
        points,
        faces,
        loop,
        owner_face_ids,
        minimum_face_normal_consistency,
    )
    if orientation_error is not None:
        return None, orientation_error
    return owner_face_ids.astype(np.int64, copy=False), None


def estimate_boundary_curvature(
    points: np.ndarray,
    faces: np.ndarray,
    selected_faces: np.ndarray,
    center: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    normal: np.ndarray,
    vertex_normals: np.ndarray,
) -> dict[str, Any]:
    """Estimate one-ring height curvature and its reliability diagnostics."""

    neighbor_ids = np.unique(faces[selected_faces])
    local_points = points[neighbor_ids] - center
    uv = np.column_stack((local_points @ u_axis, local_points @ v_axis))
    depth = local_points @ normal
    scale = max(float(np.linalg.norm(np.ptp(uv, axis=0))), 1e-30)
    x, y = uv[:, 0] / scale, uv[:, 1] / scale
    design = np.column_stack((x * x, x * y, y * y, x, y, np.ones_like(x)))
    try:
        coefficients, _, rank, singular = np.linalg.lstsq(
            design, depth / scale, rcond=None
        )
    except np.linalg.LinAlgError as exc:
        return {
            "status": "failed",
            "method": "source_one_ring_quadratic_height_and_normal_variation",
            "reliable": False,
            "reason_code": "curvature_quadratic_fit_failed",
            "message": str(exc),
        }
    predicted = design @ coefficients * scale
    residual = float(np.sqrt(np.mean((predicted - depth) ** 2))) if depth.size else np.inf
    hessian = np.asarray(
        [
            [2.0 * coefficients[0], coefficients[1]],
            [coefficients[1], 2.0 * coefficients[2]],
        ]
    ) / scale
    try:
        principal = np.linalg.eigvalsh(hessian)
    except np.linalg.LinAlgError as exc:
        return {
            "status": "failed",
            "method": "source_one_ring_quadratic_height_and_normal_variation",
            "reliable": False,
            "reason_code": "curvature_eigendecomposition_failed",
            "message": str(exc),
        }
    consecutive_dot = np.sum(
        vertex_normals * np.roll(vertex_normals, -1, axis=0), axis=1
    )
    normal_turn = np.arccos(np.clip(consecutive_dot, -1.0, 1.0))
    fit_rms_ratio = residual / scale
    condition = (
        float(singular[0] / singular[-1])
        if singular.size == 6 and singular[-1] > 0.0
        else float("inf")
    )
    reasons = []
    if rank != 6 or neighbor_ids.size < 6:
        reasons.append("curvature_quadratic_fit_underdetermined")
    if not np.isfinite(condition) or condition > 1.0e8:
        reasons.append("curvature_quadratic_fit_ill_conditioned")
    if not np.isfinite(fit_rms_ratio) or fit_rms_ratio > 0.05:
        reasons.append("curvature_quadratic_fit_residual_exceeded")
    if not np.all(np.isfinite(principal)):
        reasons.append("curvature_principal_values_non_finite")
    reliable = not reasons
    return {
        "status": "computed" if reliable else "underdetermined",
        "method": "source_one_ring_quadratic_height_and_normal_variation",
        "reliable": reliable,
        "reason_codes": reasons,
        "neighbor_vertex_count": int(neighbor_ids.size),
        "design_rank": int(rank),
        "design_singular_values": singular.tolist(),
        "principal_curvatures": principal.tolist(),
        "mean_curvature": float(0.5 * principal.sum()),
        "gaussian_curvature": float(np.prod(principal)),
        "quadratic_coefficients_normalized": coefficients.tolist(),
        "height_hessian": hessian.tolist(),
        "fit_scale": scale,
        "fit_rms": residual,
        "fit_rms_ratio": fit_rms_ratio,
        "maximum_fit_rms_ratio": 0.05,
        "design_condition_number": condition,
        "maximum_design_condition_number": 1.0e8,
        "normal_turn_radians_max": float(normal_turn.max(initial=0.0)),
        "normal_turn_radians_mean": (
            float(normal_turn.mean()) if normal_turn.size else 0.0
        ),
    }


def build_boundary_vertex_normals(
    points: np.ndarray,
    faces: np.ndarray,
    loop: np.ndarray,
    selected_faces: np.ndarray,
    fallback: np.ndarray,
) -> np.ndarray:
    """Accumulate area-weighted source normals at immutable boundary vertices."""

    raw = np.cross(
        points[faces[:, 1]] - points[faces[:, 0]],
        points[faces[:, 2]] - points[faces[:, 0]],
    )
    positions = {int(vertex_id): index for index, vertex_id in enumerate(loop)}
    accumulated = np.zeros((loop.size, 3), dtype=np.float64)
    for face_id in selected_faces:
        for vertex_id in faces[int(face_id)]:
            index = positions.get(int(vertex_id))
            if index is not None:
                accumulated[index] += raw[int(face_id)]
    result = np.empty_like(accumulated)
    for index, value in enumerate(accumulated):
        unit = _unit(value)
        result[index] = fallback if unit is None else unit
    return result


def _is_manifold_boundary_one_ring(
    faces: np.ndarray,
    loop: np.ndarray,
    selected_faces: np.ndarray,
) -> bool:
    for loop_index, vertex_id in enumerate(loop):
        incident = selected_faces[
            np.any(faces[selected_faces] == int(vertex_id), axis=1)
        ]
        links: dict[int, set[int]] = {}
        link_edges: set[tuple[int, int]] = set()
        for face_id in incident:
            others = [
                int(value)
                for value in faces[int(face_id)]
                if int(value) != int(vertex_id)
            ]
            if len(others) != 2:
                return False
            left, right = sorted(others)
            edge = (left, right)
            if edge in link_edges:
                return False
            link_edges.add(edge)
            links.setdefault(others[0], set()).add(others[1])
            links.setdefault(others[1], set()).add(others[0])
        endpoints = {node for node, neighbors in links.items() if len(neighbors) == 1}
        if (
            not links
            or any(len(neighbors) not in {1, 2} for neighbors in links.values())
            or endpoints
            != {
                int(loop[(loop_index - 1) % loop.size]),
                int(loop[(loop_index + 1) % loop.size]),
            }
        ):
            return False
        connected: set[int] = set()
        pending = [next(iter(links))]
        while pending:
            node = pending.pop()
            if node in connected:
                continue
            connected.add(node)
            pending.extend(links[node] - connected)
        if len(connected) != len(links):
            return False
    return True


def _owner_face_orientation_error(
    points: np.ndarray,
    faces: np.ndarray,
    loop: np.ndarray,
    selected_faces: np.ndarray,
    minimum_face_normal_consistency: float,
) -> str | None:
    selected = faces[selected_faces]
    raw = np.cross(
        points[selected[:, 1]] - points[selected[:, 0]],
        points[selected[:, 2]] - points[selected[:, 0]],
    )
    lengths = np.linalg.norm(raw, axis=1)
    if (
        not np.all(np.isfinite(raw))
        or not np.all(np.isfinite(lengths))
        or np.any(lengths <= 1e-30)
    ):
        return "boundary_owner_one_ring_normal_invalid"
    normals = raw / lengths[:, None]
    occurrences: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for local_face_id, face in enumerate(selected):
        for index in range(3):
            left = int(face[index])
            right = int(face[(index + 1) % 3])
            edge = (left, right) if left < right else (right, left)
            occurrences.setdefault(edge, []).append((left, right, local_face_id))
    boundary_edges = {
        tuple(sorted((int(loop[index]), int(loop[(index + 1) % loop.size]))))
        for index in range(loop.size)
    }
    loop_vertices = {int(value) for value in loop}
    for edge, rows in occurrences.items():
        internal = len(rows) > 1 or (
            edge not in boundary_edges
            and (edge[0] in loop_vertices or edge[1] in loop_vertices)
        )
        if not internal:
            continue
        if len(rows) != 2 or rows[0][:2] != tuple(reversed(rows[1][:2])):
            return "boundary_owner_one_ring_winding_inconsistent"
        normal_dot = float(np.dot(normals[rows[0][2]], normals[rows[1][2]]))
        if not np.isfinite(normal_dot) or normal_dot < minimum_face_normal_consistency:
            return "boundary_owner_one_ring_normal_conflict"
    return None


def _unit(value: np.ndarray) -> np.ndarray | None:
    length = float(np.linalg.norm(value))
    if not np.isfinite(length) or length <= 1e-30:
        return None
    return np.asarray(value) / length
