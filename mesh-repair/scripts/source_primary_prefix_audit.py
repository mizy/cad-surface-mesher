from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from source_primary_quality_geometry import build_gate, compare_prefix_bytes


def audit_source_prefix(
    source_points: np.ndarray,
    source_faces: np.ndarray,
    source_triangle_index: np.ndarray,
    candidate_points: np.ndarray,
    candidate_faces: np.ndarray,
    candidate_source_triangle_index: np.ndarray,
) -> dict[str, Any]:
    """Require byte-identical source geometry and provenance prefixes."""

    source_points = np.asarray(source_points)
    source_faces = np.asarray(source_faces)
    source_index = np.asarray(source_triangle_index)
    final_points = np.asarray(candidate_points)
    final_faces = np.asarray(candidate_faces)
    final_index = np.asarray(candidate_source_triangle_index)
    point_equal = compare_prefix_bytes(source_points, final_points)
    face_equal = compare_prefix_bytes(source_faces, final_faces)
    index_equal = compare_prefix_bytes(source_index, final_index)
    index_rows_match_faces = final_index.shape == (final_faces.shape[0],)
    appended_indices = (
        final_index[source_faces.shape[0] :] if final_index.ndim == 1 else np.empty(0)
    )
    generated_unmapped = (
        index_rows_match_faces
        and appended_indices.shape[0]
        == max(final_faces.shape[0] - source_faces.shape[0], 0)
        and bool(np.all(appended_indices == -1))
    )
    gates = {
        "source_vertex_prefix_bitwise_equal": build_gate(
            point_equal, "source_vertex_prefix_changed", point_equal, True
        ),
        "source_face_connectivity_bitwise_equal": build_gate(
            face_equal, "source_face_connectivity_changed", face_equal, True
        ),
        "source_triangle_index_bitwise_equal": build_gate(
            index_equal and generated_unmapped,
            "source_triangle_index_changed",
            {
                "source_prefix_equal": index_equal,
                "generated_faces_are_minus_one": generated_unmapped,
                "row_count_matches_final_faces": index_rows_match_faces,
            },
            True,
        ),
    }
    failed = [row["reason_code"] for row in gates.values() if not row["passed"]]
    return {
        "passed": not failed,
        "reason_codes": failed,
        "source_point_count": int(source_points.shape[0]),
        "final_point_count": int(final_points.shape[0]) if final_points.ndim else 0,
        "source_face_count": int(source_faces.shape[0]),
        "final_face_count": int(final_faces.shape[0]) if final_faces.ndim else 0,
        "appended_point_count": max(
            int(final_points.shape[0] - source_points.shape[0]), 0
        ),
        "appended_face_count": max(
            int(final_faces.shape[0] - source_faces.shape[0]), 0
        ),
        "source_hashes": {
            "points_sha256": _array_sha256(source_points),
            "faces_sha256": _array_sha256(source_faces),
            "source_triangle_index_sha256": _array_sha256(source_index),
        },
        "final_prefix_hashes": {
            "points_sha256": _array_sha256(final_points[: source_points.shape[0]]),
            "faces_sha256": _array_sha256(final_faces[: source_faces.shape[0]]),
            "source_triangle_index_sha256": _array_sha256(
                final_index[: source_index.shape[0]]
            ),
        },
        "gates": gates,
    }


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()
