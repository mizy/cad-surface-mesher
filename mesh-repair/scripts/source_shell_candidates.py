from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def face_component_labels(faces: np.ndarray) -> np.ndarray:
    """Label face components connected by shared undirected vertex-index edges.

    Vertex-only contact and equal coordinates with different point IDs do not
    connect components. Labels are dense and ordered by each component's first
    input face.
    """
    face_array = _face_array(faces)
    face_count = face_array.shape[0]
    if face_count == 0:
        return np.empty(0, dtype=np.int64)

    face_ids = np.arange(face_count, dtype=np.int64)
    edges = np.concatenate(
        (face_array[:, (0, 1)], face_array[:, (1, 2)], face_array[:, (2, 0)])
    )
    edges.sort(axis=1)
    edge_face_ids = np.tile(face_ids, 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    ordered_edges = edges[order]
    ordered_faces = edge_face_ids[order]
    same_as_previous = np.all(ordered_edges[1:] == ordered_edges[:-1], axis=1)

    union_find = _UnionFind(face_count)
    for offset in np.flatnonzero(same_as_previous):
        union_find.union(int(ordered_faces[offset]), int(ordered_faces[offset + 1]))
    roots = np.fromiter(
        (union_find.find(face_id) for face_id in range(face_count)),
        dtype=np.int64,
        count=face_count,
    )
    return _labels_in_first_face_order(roots)


def describe_components(
    points: np.ndarray,
    faces: np.ndarray,
    source_triangle_index: np.ndarray,
    labels: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Describe shared-edge components without changing their source geometry."""
    point_array, face_array, source_index = _mesh_arrays(
        points, faces, source_triangle_index
    )
    component_labels = (
        face_component_labels(face_array)
        if labels is None
        else _label_array(labels, face_array.shape[0])
    )
    if face_array.shape[0] == 0:
        return []

    triangles = point_array[face_array].astype(np.float64, copy=False)
    areas = _triangle_areas(triangles)
    descriptions: list[dict[str, Any]] = []
    for component_id in np.unique(component_labels):
        face_ids = np.flatnonzero(component_labels == component_id)
        point_ids = np.unique(face_array[face_ids].ravel())
        component_points = point_array[point_ids]
        minimum = component_points.min(axis=0)
        maximum = component_points.max(axis=0)
        descriptions.append(
            {
                "component_id": int(component_id),
                "face_ids": face_ids.astype(int).tolist(),
                "face_count": int(face_ids.size),
                "surface_area": float(areas[face_ids].sum()),
                "bbox": {
                    "min": minimum.astype(float).tolist(),
                    "max": maximum.astype(float).tolist(),
                    "extents": (maximum - minimum).astype(float).tolist(),
                },
                "source_triangle_ids": [
                    int(value) for value in source_index[face_ids]
                ],
            }
        )
    return descriptions


def remove_low_risk_faces(
    points: np.ndarray,
    faces: np.ndarray,
    source_triangle_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Remove exactly degenerate and exactly coincident source triangles.

    Duplicate detection compares exact vertex coordinates, independent of face
    winding and point IDs. No points are moved, merged, projected, or created.
    The first input occurrence of an exact duplicate is retained.
    """
    point_array, face_array, source_index = _mesh_arrays(
        points, faces, source_triangle_index
    )
    triangles = point_array[face_array].astype(np.float64, copy=False)
    degenerate = _exact_degenerate_mask(triangles)
    duplicate = _exact_duplicate_mask(triangles, eligible=~degenerate)
    keep = ~(degenerate | duplicate)
    degenerate_ids = np.flatnonzero(degenerate)
    duplicate_ids = np.flatnonzero(duplicate)
    report = {
        "input_face_count": int(face_array.shape[0]),
        "kept_face_count": int(np.count_nonzero(keep)),
        "removed_face_count": int(np.count_nonzero(~keep)),
        "removed_degenerate_face_ids": degenerate_ids.astype(int).tolist(),
        "removed_duplicate_face_ids": duplicate_ids.astype(int).tolist(),
        "removed_degenerate_source_triangle_ids": [
            int(value) for value in source_index[degenerate_ids]
        ],
        "removed_duplicate_source_triangle_ids": [
            int(value) for value in source_index[duplicate_ids]
        ],
        "duplicate_rule": "exact_coordinate_vertex_set_ignoring_winding",
        "degenerate_rule": "exact_zero_area",
    }
    return (
        point_array.copy(),
        face_array[keep].copy(),
        source_index[keep].copy(),
        report,
    )


def merge_kept_components(
    points: np.ndarray,
    faces: np.ndarray,
    source_triangle_index: np.ndarray,
    labels: np.ndarray,
    keep_component_ids: Iterable[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pack selected components into one points/faces mesh without welding."""
    point_array, face_array, source_index = _mesh_arrays(
        points, faces, source_triangle_index
    )
    component_labels = _label_array(labels, face_array.shape[0])
    requested = np.asarray(list(keep_component_ids), dtype=np.int64).reshape(-1)
    available = np.unique(component_labels)
    unknown = np.setdiff1d(np.unique(requested), available)
    if unknown.size:
        raise ValueError(f"unknown component IDs: {unknown.astype(int).tolist()}")

    keep = np.isin(component_labels, requested)
    selected_faces = face_array[keep]
    selected_sources = source_index[keep].copy()
    if selected_faces.shape[0] == 0:
        return (
            np.empty((0, 3), dtype=point_array.dtype),
            np.empty((0, 3), dtype=np.int64),
            selected_sources,
            np.empty(0, dtype=np.int64),
        )

    source_point_index, inverse = np.unique(
        selected_faces.ravel(), return_inverse=True
    )
    compact_faces = inverse.reshape((-1, 3)).astype(np.int64, copy=False)
    return (
        point_array[source_point_index].copy(),
        compact_faces,
        selected_sources,
        source_point_index.astype(np.int64, copy=False),
    )


def _mesh_arrays(
    points: np.ndarray,
    faces: np.ndarray,
    source_triangle_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point_array = np.asarray(points)
    if point_array.ndim != 2 or point_array.shape[1] != 3:
        raise ValueError("points must have shape (point_count, 3)")
    if not np.issubdtype(point_array.dtype, np.number):
        raise ValueError("points must be numeric")
    if not np.all(np.isfinite(point_array)):
        raise ValueError("points must contain only finite coordinates")
    face_array = _face_array(faces)
    if face_array.size and int(face_array.max()) >= point_array.shape[0]:
        raise ValueError("faces contain a point index outside points")
    source_index = np.asarray(source_triangle_index)
    if source_index.ndim != 1 or source_index.shape[0] != face_array.shape[0]:
        raise ValueError("source_triangle_index must contain one value per face")
    if not np.issubdtype(source_index.dtype, np.integer):
        raise ValueError("source_triangle_index must be an integer array")
    return point_array, face_array, source_index


def _face_array(faces: np.ndarray) -> np.ndarray:
    values = np.asarray(faces)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("faces must have shape (face_count, 3)")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("faces must be an integer array")
    result = values.astype(np.int64, copy=False)
    if result.size and int(result.min()) < 0:
        raise ValueError("faces cannot contain negative point indices")
    return result


def _label_array(labels: np.ndarray, face_count: int) -> np.ndarray:
    values = np.asarray(labels)
    if values.ndim != 1 or values.shape[0] != face_count:
        raise ValueError("labels must contain one value per face")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("labels must be an integer array")
    result = values.astype(np.int64, copy=False)
    if result.size and int(result.min()) < 0:
        raise ValueError("labels cannot contain negative component IDs")
    return result


def _labels_in_first_face_order(roots: np.ndarray) -> np.ndarray:
    unique_roots, first_indices = np.unique(roots, return_index=True)
    ordered_roots = unique_roots[np.argsort(first_indices)]
    root_to_label = np.full(roots.shape[0], -1, dtype=np.int64)
    root_to_label[ordered_roots] = np.arange(ordered_roots.size, dtype=np.int64)
    return root_to_label[roots]


def _triangle_areas(triangles: np.ndarray) -> np.ndarray:
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    return np.linalg.norm(cross, axis=1) * 0.5


def _exact_degenerate_mask(triangles: np.ndarray) -> np.ndarray:
    if triangles.shape[0] == 0:
        return np.empty(0, dtype=bool)
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    return np.all(cross == 0.0, axis=1)


def _exact_duplicate_mask(
    triangles: np.ndarray,
    *,
    eligible: np.ndarray,
) -> np.ndarray:
    duplicate = np.zeros(triangles.shape[0], dtype=bool)
    eligible_ids = np.flatnonzero(eligible)
    if eligible_ids.size == 0:
        return duplicate
    candidates = triangles[eligible_ids]
    order = np.lexsort(
        (candidates[:, :, 2], candidates[:, :, 1], candidates[:, :, 0]), axis=1
    )
    canonical = np.take_along_axis(candidates, order[:, :, None], axis=1)
    _, first = np.unique(canonical.reshape((-1, 9)), axis=0, return_index=True)
    keep_local = np.zeros(eligible_ids.size, dtype=bool)
    keep_local[first] = True
    duplicate[eligible_ids[~keep_local]] = True
    return duplicate
