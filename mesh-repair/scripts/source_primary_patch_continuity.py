from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from source_primary_patch_contract import BoundaryMapping


def audit_boundary_normal_continuity(
    points: np.ndarray,
    source_faces: np.ndarray,
    patch_faces: np.ndarray,
    mappings: tuple[BoundaryMapping, ...],
    threshold: float,
) -> dict[str, Any]:
    """Compare each patch seam face with its exact immutable source neighbor."""

    patch_normals = _face_normals(points, patch_faces)
    occurrences: dict[tuple[int, int], list[int]] = {}
    for face_id, face in enumerate(patch_faces):
        for left, right in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            left_id, right_id = int(left), int(right)
            edge = (
                (left_id, right_id)
                if left_id < right_id
                else (right_id, left_id)
            )
            occurrences.setdefault(edge, []).append(face_id)
    dots: list[float] = []
    for mapping in mappings:
        source_face_ids = np.asarray(mapping.source_edge_face_ids, dtype=np.int64)
        source_normals = _face_normals(points, source_faces[source_face_ids])
        loop = mapping.source_vertex_ids
        for index, left in enumerate(loop):
            right = loop[(index + 1) % len(loop)]
            edge = (left, right) if left < right else (right, left)
            patch_ids = occurrences.get(edge, [])
            if len(patch_ids) != 1:
                return {"passed": False, "reason": "mapped_boundary_face_missing"}
            dots.append(float(np.dot(patch_normals[patch_ids[0]], source_normals[index])))
    values = np.asarray(dots, dtype=np.float64)
    return {
        "passed": bool(values.size and np.min(values) >= threshold),
        "minimum_normal_dot": float(values.min()) if values.size else None,
        "mean_normal_dot": float(values.mean()) if values.size else None,
        "threshold": float(threshold),
    }


def _face_normals(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = np.asarray(points, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    raw = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(raw, axis=1)
    return np.divide(
        raw, lengths[:, None], out=np.zeros_like(raw), where=lengths[:, None] > 0.0
    )
