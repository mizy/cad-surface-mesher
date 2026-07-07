from __future__ import annotations

from typing import Any

import numpy as np


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            self.parent[root_left] = root_right
        elif self.rank[root_left] > self.rank[root_right]:
            self.parent[root_right] = root_left
        else:
            self.parent[root_right] = root_left
            self.rank[root_left] += 1


def edge_topology(faces: np.ndarray) -> tuple[dict[str, int], dict[tuple[int, int], list[int]]]:
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for face_index, (a, b, c) in enumerate(faces):
        for edge in ((a, b), (b, c), (c, a)):
            key = tuple(sorted((int(edge[0]), int(edge[1]))))
            edge_to_faces.setdefault(key, []).append(face_index)
    counts = [len(face_ids) for face_ids in edge_to_faces.values()]
    return {
        "edges": len(counts),
        "boundary_edges": sum(1 for count in counts if count == 1),
        "manifold_edges": sum(1 for count in counts if count == 2),
        "non_manifold_edges": sum(1 for count in counts if count > 2),
    }, edge_to_faces


def connected_components(face_count: int, edge_to_faces: dict[tuple[int, int], list[int]]) -> dict[str, Any]:
    if face_count == 0:
        return {"count": 0, "largest_faces": []}
    union_find = UnionFind(face_count)
    for face_ids in edge_to_faces.values():
        for face_id in face_ids[1:]:
            union_find.union(face_ids[0], face_id)
    counts: dict[int, int] = {}
    for face_index in range(face_count):
        root = union_find.find(face_index)
        counts[root] = counts.get(root, 0) + 1
    sizes = sorted(counts.values(), reverse=True)
    return {"count": len(sizes), "largest_faces": sizes[:10]}


def percentile_summary(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {"min": None, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "min": float(values.min()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def triangle_quality(points: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    if faces.size == 0:
        return {"surface_area": 0.0, "degenerate_faces": 0, "area": {}, "aspect_ratio": {}}
    triangles = points[faces]
    e0 = np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1)
    e1 = np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1)
    e2 = np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1)
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = np.linalg.norm(cross, axis=1) * 0.5
    max_edge = np.maximum.reduce([e0, e1, e2])
    min_edge = np.minimum.reduce([e0, e1, e2])
    min_alt = np.divide(2.0 * areas, max_edge, out=np.zeros_like(areas), where=max_edge > 1e-15)
    aspect = np.divide(max_edge, min_alt, out=np.full_like(max_edge, np.inf), where=min_alt > 1e-15)
    finite_aspect = aspect[np.isfinite(aspect)]
    eps_area = max(float(np.nanmedian(areas)) * 1e-12, 1e-18)
    return {
        "surface_area": float(areas.sum()),
        "degenerate_faces": int(np.count_nonzero((areas <= eps_area) | (min_edge <= 1e-15))),
        "area": percentile_summary(areas),
        "aspect_ratio": percentile_summary(finite_aspect),
    }


def bounds_dict(points: np.ndarray) -> dict[str, Any]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    extents = maxs - mins
    return {
        "min": mins.tolist(),
        "max": maxs.tolist(),
        "extents": extents.tolist(),
        "length_x": float(extents[0]),
        "width_y": float(extents[1]),
        "height_z": float(extents[2]),
    }


def signed_volume(points: np.ndarray, faces: np.ndarray) -> float:
    triangles = points[faces]
    volume = np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum()
    return float(volume / 6.0)


def mesh_report(points: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    topology, edge_to_faces = edge_topology(faces)
    quality = triangle_quality(points, faces)
    components = connected_components(faces.shape[0], edge_to_faces)
    volume_reliable = (
        topology["boundary_edges"] == 0
        and topology["non_manifold_edges"] == 0
        and components["count"] == 1
        and quality["degenerate_faces"] == 0
        and faces.shape[0] > 0
    )
    volume = abs(signed_volume(points, faces)) if volume_reliable else None
    return {
        "points": int(points.shape[0]),
        "triangles": int(faces.shape[0]),
        "bounds": bounds_dict(points),
        "surface_area": quality["surface_area"],
        "topology": {**topology, "components": components},
        "quality": {
            "degenerate_faces": quality["degenerate_faces"],
            "area": quality["area"],
            "aspect_ratio": quality["aspect_ratio"],
        },
        "volume": {"reliable": volume_reliable, "signed_abs": volume},
    }


def bbox_drift_from_reports(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    ref_min = np.asarray(reference["bounds"]["min"], dtype=np.float64)
    ref_max = np.asarray(reference["bounds"]["max"], dtype=np.float64)
    cand_min = np.asarray(candidate["bounds"]["min"], dtype=np.float64)
    cand_max = np.asarray(candidate["bounds"]["max"], dtype=np.float64)
    ref_extents = np.maximum(ref_max - ref_min, 1e-12)
    delta_min = cand_min - ref_min
    delta_max = cand_max - ref_max
    max_abs = float(max(np.max(np.abs(delta_min)), np.max(np.abs(delta_max))))
    return {
        "delta_min": delta_min.tolist(),
        "delta_max": delta_max.tolist(),
        "delta_extents": ((cand_max - cand_min) - (ref_max - ref_min)).tolist(),
        "max_abs": max_abs,
        "max_ratio": float(max_abs / np.max(ref_extents)),
    }


def silhouette_drift_from_meshes(
    reference_points: np.ndarray,
    reference_faces: np.ndarray,
    candidate_points: np.ndarray,
    candidate_faces: np.ndarray,
    views: list[Any],
    *,
    max_size: int = 720,
) -> dict[str, Any]:
    per_view = {}
    for view in views:
        reference_mask, candidate_mask = silhouette_masks(
            reference_points,
            reference_faces,
            candidate_points,
            candidate_faces,
            view,
            max_size=max_size,
        )
        per_view[view.name] = silhouette_mask_drift(reference_mask, candidate_mask)

    return {
        "method": "shared_projection_silhouette_occupancy",
        "grid_max_size": max_size,
        "per_view": per_view,
        "summary": {
            "reference_only_ratio_max": max(row["reference_only_ratio_of_union"] for row in per_view.values()),
            "candidate_only_ratio_max": max(row["candidate_only_ratio_of_union"] for row in per_view.values()),
            "changed_ratio_max": max(row["changed_ratio_of_union"] for row in per_view.values()),
            "overlap_ratio_min": min(row["overlap_ratio_of_union"] for row in per_view.values()),
        },
    }


def silhouette_masks(
    reference_points: np.ndarray,
    reference_faces: np.ndarray,
    candidate_points: np.ndarray,
    candidate_faces: np.ndarray,
    view: Any,
    *,
    max_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    reference_centroids = reference_points[reference_faces].mean(axis=1)
    candidate_centroids = candidate_points[candidate_faces].mean(axis=1)
    all_coords = np.vstack([
        reference_centroids[:, view.project_axes],
        candidate_centroids[:, view.project_axes],
    ])
    rows, cols, mins, spans = projection_grid(all_coords, max_size)
    return (
        silhouette_mask(reference_centroids, view, rows, cols, mins, spans),
        silhouette_mask(candidate_centroids, view, rows, cols, mins, spans),
    )


def projection_grid(coords: np.ndarray, max_size: int) -> tuple[int, int, np.ndarray, np.ndarray]:
    mins = coords.min(axis=0)
    spans = np.maximum(coords.max(axis=0) - mins, 1e-12)
    if spans[0] >= spans[1]:
        cols = max_size
        rows = max(64, int(round(max_size * spans[1] / spans[0])))
    else:
        rows = max_size
        cols = max(64, int(round(max_size * spans[0] / spans[1])))
    return rows, cols, mins, spans


def silhouette_mask(
    centroids: np.ndarray,
    view: Any,
    rows: int,
    cols: int,
    mins: np.ndarray,
    spans: np.ndarray,
) -> np.ndarray:
    coords = centroids[:, view.project_axes]
    uv = np.clip((coords - mins) / spans, 0.0, 1.0)
    col = np.minimum((uv[:, 0] * cols).astype(np.int64), cols - 1)
    row = np.minimum((uv[:, 1] * rows).astype(np.int64), rows - 1)
    mask = np.zeros(rows * cols, dtype=bool)
    mask[row * cols + col] = True
    return mask.reshape(rows, cols)


def silhouette_mask_drift(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    both = reference & candidate
    reference_only = reference & ~candidate
    candidate_only = candidate & ~reference
    union = int(np.count_nonzero(reference | candidate))
    reference_only_count = int(np.count_nonzero(reference_only))
    candidate_only_count = int(np.count_nonzero(candidate_only))
    overlap_count = int(np.count_nonzero(both))
    return {
        "reference_pixels": int(np.count_nonzero(reference)),
        "candidate_pixels": int(np.count_nonzero(candidate)),
        "overlap_pixels": overlap_count,
        "reference_only_pixels": reference_only_count,
        "candidate_only_pixels": candidate_only_count,
        "reference_only_ratio_of_union": float(reference_only_count / max(union, 1)),
        "candidate_only_ratio_of_union": float(candidate_only_count / max(union, 1)),
        "changed_ratio_of_union": float((reference_only_count + candidate_only_count) / max(union, 1)),
        "overlap_ratio_of_union": float(overlap_count / max(union, 1)),
    }
