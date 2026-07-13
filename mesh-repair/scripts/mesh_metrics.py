from __future__ import annotations

from typing import Any

import numpy as np

from solid_triangle_raster import SIX_VIEWS, rasterize_solid_views
from mesh_vertex_topology import count_non_manifold_vertices


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


def edge_topology(
    faces: np.ndarray,
) -> tuple[dict[str, int], dict[tuple[int, int], list[int]]]:
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for face_index, (a, b, c) in enumerate(faces):
        for edge in ((a, b), (b, c), (c, a)):
            key = (min(int(edge[0]), int(edge[1])), max(int(edge[0]), int(edge[1])))
            edge_to_faces.setdefault(key, []).append(face_index)
    counts = [len(face_ids) for face_ids in edge_to_faces.values()]
    return {
        "edges": len(counts),
        "boundary_edges": sum(1 for count in counts if count == 1),
        "manifold_edges": sum(1 for count in counts if count == 2),
        "non_manifold_edges": sum(1 for count in counts if count > 2),
        "non_manifold_vertices": count_non_manifold_vertices(faces, edge_to_faces),
    }, edge_to_faces


def connected_components(
    face_count: int, edge_to_faces: dict[tuple[int, int], list[int]]
) -> dict[str, Any]:
    if face_count == 0:
        return {"count": 0, "largest_faces": []}
    labels = face_component_labels(face_count, edge_to_faces)
    sizes = sorted(np.bincount(labels).astype(int).tolist(), reverse=True)
    return {"count": len(sizes), "largest_faces": sizes[:10]}


def face_component_labels(
    face_count: int,
    edge_to_faces: dict[tuple[int, int], list[int]],
) -> np.ndarray:
    if face_count == 0:
        return np.zeros(0, dtype=np.int64)
    union_find = UnionFind(face_count)
    for face_ids in edge_to_faces.values():
        for face_id in face_ids[1:]:
            union_find.union(face_ids[0], face_id)
    roots = np.asarray(
        [union_find.find(face_index) for face_index in range(face_count)],
        dtype=np.int64,
    )
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int64, copy=False)


def inconsistent_winding_edges(faces: np.ndarray) -> int:
    """Count manifold edges whose two incident faces traverse the edge in the same direction."""
    directions: dict[tuple[int, int], list[int]] = {}
    for a, b, c in faces:
        for left, right in ((a, b), (b, c), (c, a)):
            left_i, right_i = int(left), int(right)
            key = (left_i, right_i) if left_i < right_i else (right_i, left_i)
            directions.setdefault(key, []).append(1 if (left_i, right_i) == key else -1)
    return sum(
        1
        for values in directions.values()
        if len(values) == 2 and values[0] == values[1]
    )


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
        return {
            "surface_area": 0.0,
            "degenerate_faces": 0,
            "area": {},
            "aspect_ratio": {},
        }
    triangles = points[faces]
    e0 = np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1)
    e1 = np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1)
    e2 = np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1)
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    areas = np.linalg.norm(cross, axis=1) * 0.5
    max_edge = np.maximum.reduce([e0, e1, e2])
    min_edge = np.minimum.reduce([e0, e1, e2])
    min_alt = np.divide(
        2.0 * areas, max_edge, out=np.zeros_like(areas), where=max_edge > 1e-15
    )
    aspect = np.divide(
        max_edge, min_alt, out=np.full_like(max_edge, np.inf), where=min_alt > 1e-15
    )
    finite_aspect = aspect[np.isfinite(aspect)]
    eps_area = max(float(np.nanmedian(areas)) * 1e-12, 1e-18)
    return {
        "surface_area": float(areas.sum()),
        "degenerate_faces": int(
            np.count_nonzero((areas <= eps_area) | (min_edge <= 1e-15))
        ),
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
        "extent_x": float(extents[0]),
        "extent_y": float(extents[1]),
        "extent_z": float(extents[2]),
    }


def signed_volume(points: np.ndarray, faces: np.ndarray) -> float:
    triangles = points[faces]
    volume = np.einsum(
        "ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])
    ).sum()
    return float(volume / 6.0)


def component_volume_report(
    points: np.ndarray,
    faces: np.ndarray,
    edge_to_faces: dict[tuple[int, int], list[int]],
    *,
    topology_closed: bool,
    degenerate_faces: int,
    winding_errors: int,
) -> dict[str, Any]:
    labels = face_component_labels(faces.shape[0], edge_to_faces)
    signed_components = (
        [
            signed_volume(points, faces[labels == component_id])
            for component_id in range(int(labels.max()) + 1)
        ]
        if labels.size
        else []
    )
    scale = (
        float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
        if points.size
        else 0.0
    )
    volume_epsilon = max(scale**3 * 1e-15, 1e-30)
    nonzero_signs = [
        int(np.sign(value))
        for value in signed_components
        if abs(value) > volume_epsilon
    ]
    orientation_consistent = bool(
        winding_errors == 0
        and nonzero_signs
        and (
            all(sign > 0 for sign in nonzero_signs)
            or all(sign < 0 for sign in nonzero_signs)
        )
    )
    reliable = bool(
        topology_closed
        and degenerate_faces == 0
        and winding_errors == 0
        and signed_components
        and len(nonzero_signs) == len(signed_components)
    )
    return {
        "reliable": reliable,
        "signed_abs": float(sum(abs(value) for value in signed_components))
        if reliable
        else None,
        "signed_sum": float(sum(signed_components)) if reliable else None,
        "component_signed": [float(value) for value in signed_components],
        "component_count": len(signed_components),
        "orientation_consistent": orientation_consistent,
    }


def mesh_report(points: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    topology, edge_to_faces = edge_topology(faces)
    quality = triangle_quality(points, faces)
    components = connected_components(faces.shape[0], edge_to_faces)
    winding_errors = inconsistent_winding_edges(faces)
    topology_closed = (
        topology["boundary_edges"] == 0
        and topology["non_manifold_edges"] == 0
        and topology["non_manifold_vertices"] == 0
        and faces.shape[0] > 0
    )
    volume = component_volume_report(
        points,
        faces,
        edge_to_faces,
        topology_closed=topology_closed,
        degenerate_faces=quality["degenerate_faces"],
        winding_errors=winding_errors,
    )
    return {
        "points": int(points.shape[0]),
        "triangles": int(faces.shape[0]),
        "bounds": bounds_dict(points),
        "surface_area": quality["surface_area"],
        "topology": {
            **topology,
            "components": components,
            "inconsistent_winding_edges": winding_errors,
            "closed_manifold": topology_closed,
        },
        "quality": {
            "degenerate_faces": quality["degenerate_faces"],
            "area": quality["area"],
            "aspect_ratio": quality["aspect_ratio"],
        },
        "volume": volume,
    }


def self_intersection_report(
    points: np.ndarray,
    faces: np.ndarray,
    *,
    focus_face_ids: np.ndarray | list[int] | None = None,
    max_candidate_pairs: int = 2_000_000,
    max_reported_pairs: int = 200,
) -> dict[str, Any]:
    """Detect non-adjacent intersections through the shared robust scanner."""
    from mesh_self_intersection import bounded_triangle_self_intersections

    report = bounded_triangle_self_intersections(
        points,
        faces,
        focus_face_ids=focus_face_ids,
        max_candidate_pairs=max_candidate_pairs,
        max_reported_pairs=max_reported_pairs,
    )
    return {
        **report,
        "method": "vtk_static_cell_locator_triangle_intersection",
        "adjacency_policy": (
            "only contact confined to a shared topological vertex or edge is excluded"
        ),
    }


def bbox_drift_from_reports(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
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
    views: list[Any] | None = None,
    *,
    max_size: int = 720,
) -> dict[str, Any]:
    """Compare solid silhouettes in the fixed six-camera shared frame.

    ``views`` remains accepted so legacy callers do not need wrapper changes,
    but formal evidence always uses the canonical six directions.  In
    particular, triangle centroids are never used as silhouette pixels.
    """

    reference_shared, candidate_shared = _pad_points_to_shared_bounds(
        reference_points, candidate_points
    )
    reference = rasterize_solid_views(
        reference_shared, reference_faces, grid_size=max_size
    )
    candidate = rasterize_solid_views(
        candidate_shared, candidate_faces, grid_size=max_size
    )
    per_view = {
        reference_view.view.name: silhouette_mask_drift(
            reference_view.silhouette, candidate_view.silhouette
        )
        for reference_view, candidate_view in zip(
            reference.views, candidate.views, strict=True
        )
    }
    return {
        "method": "shared_bbox_six_direction_conservative_solid_triangle_raster",
        "grid_max_size": max_size,
        "camera_contract": [view.name for view in SIX_VIEWS],
        "legacy_view_argument_policy": (
            "accepted_for_call_compatibility; fixed_six_camera_contract_used"
        ),
        "per_view": per_view,
        "summary": {
            "reference_only_ratio_max": max(
                row["reference_only_ratio_of_union"] for row in per_view.values()
            ),
            "candidate_only_ratio_max": max(
                row["candidate_only_ratio_of_union"] for row in per_view.values()
            ),
            "changed_ratio_max": max(
                row["changed_ratio_of_union"] for row in per_view.values()
            ),
            "overlap_ratio_min": min(
                row["overlap_ratio_of_union"] for row in per_view.values()
            ),
        },
    }


def _pad_points_to_shared_bounds(
    reference_points: np.ndarray,
    candidate_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(reference_points, dtype=np.float64)
    candidate = np.asarray(candidate_points, dtype=np.float64)
    if reference.ndim != 2 or reference.shape[1:] != (3,) or not reference.size:
        raise ValueError("reference_points must have non-empty shape (N, 3)")
    if candidate.ndim != 2 or candidate.shape[1:] != (3,) or not candidate.size:
        raise ValueError("candidate_points must have non-empty shape (N, 3)")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
        raise ValueError("silhouette points must be finite")
    minimum = np.minimum(reference.min(axis=0), candidate.min(axis=0))
    maximum = np.maximum(reference.max(axis=0), candidate.max(axis=0))
    corners = np.asarray(
        [
            [x, y, z]
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ],
        dtype=np.float64,
    )
    return np.vstack((reference, corners)), np.vstack((candidate, corners))


def silhouette_mask_drift(
    reference: np.ndarray, candidate: np.ndarray
) -> dict[str, Any]:
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
        "changed_ratio_of_union": float(
            (reference_only_count + candidate_only_count) / max(union, 1)
        ),
        "overlap_ratio_of_union": float(overlap_count / max(union, 1)),
    }
