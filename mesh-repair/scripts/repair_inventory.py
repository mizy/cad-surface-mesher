from __future__ import annotations

from typing import Any

import numpy as np

from boundary_classification import classify_boundary_regions
from mesh_metrics import UnionFind, edge_topology, mesh_report


INVENTORY_REGION_SECTIONS = (
    "boundary_regions",
    "gap_regions",
    "non_manifold_regions",
    "overlap_regions",
    "semantic_opening_regions",
)


def build_inventory(
    points: np.ndarray,
    faces: np.ndarray,
    source_indices: np.ndarray,
    stage: str,
    *,
    max_items: int,
    exterior_face_score: np.ndarray | None = None,
    sealed_exterior_face_mask: np.ndarray | None = None,
    component_thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    metrics = mesh_report(points, faces)
    topology, edge_to_faces = edge_topology(faces)
    component_ids = face_component_ids(faces, edge_to_faces)
    boundary_regions = connected_edge_regions(
        edge_to_faces,
        lambda count: count == 1,
        face_component_ids=component_ids,
    )
    non_manifold_regions = connected_edge_regions(edge_to_faces, lambda count: count > 2)
    boundary_classifications, boundary_classification = classify_boundary_regions(
        points,
        faces,
        boundary_regions,
        edge_to_faces,
        component_ids,
        exterior_face_score=exterior_face_score,
        sealed_exterior_face_mask=sealed_exterior_face_mask,
        component_thresholds=component_thresholds,
    )
    for row in boundary_classifications:
        paired_index = row.get("paired_region_index")
        row["paired_region_id"] = (
            f"boundary_loop_{int(paired_index) + 1:04d}"
            if paired_index is not None
            else None
        )
    exact_overlaps = exact_duplicate_inventory(points, faces, source_indices, component_ids, max_items=max_items)
    return {
        "stage": stage,
        "summary": topology_snapshot(metrics),
        "repair_route": boundary_classification["route"],
        "boundary_classification": boundary_classification,
        "boundary_regions": region_report(
            "boundary_loop",
            boundary_regions,
            points,
            edge_to_faces,
            component_ids,
            source_indices,
            max_items=max_items,
            requires_policy=False,
            classification="patch_required",
            detector_reason="topology_boundary_defect",
            policy_reason_source="geometry classifier over an ordered boundary loop and exterior-component evidence",
            region_classifications=boundary_classifications,
        ),
        "gap_regions": region_report(
            "gap_or_opening_candidate",
            boundary_regions,
            points,
            edge_to_faces,
            component_ids,
            source_indices,
            max_items=max_items,
            requires_policy=False,
            classification="unclassified_gap_or_opening_candidate",
            detector_reason="unclassified_gap_or_opening_candidate",
            policy_reason_source=(
                "derived from boundary edge detector only; no explicit semantic opening classifier ran"
            ),
        ),
        "non_manifold_regions": region_report(
            "non_manifold_edge_region",
            non_manifold_regions,
            points,
            edge_to_faces,
            component_ids,
            source_indices,
            max_items=max_items,
            requires_policy=False,
            classification="reject_region",
            detector_reason="topology_defect",
            policy_reason_source="non-manifold edge detector",
        ),
        "overlap_regions": exact_overlaps,
        "components": metrics["topology"]["components"],
        "checks": {
            "normal_consistency": {
                "status": "computed",
                "inconsistent_winding_edges": metrics["topology"]["inconsistent_winding_edges"],
                "passed": metrics["topology"]["inconsistent_winding_edges"] == 0,
            },
            "component_orientation": {
                "status": "computed" if metrics["volume"]["reliable"] else "topology_not_closed",
                "orientation_consistent": metrics["volume"]["orientation_consistent"],
            },
            "self_intersection": {
                "status": "deferred_to_closed_hybrid_candidate",
                "method": "vtk_static_cell_locator_triangle_intersection",
            },
        },
        "not_checked": [],
    }


def topology_snapshot(metrics: dict[str, Any]) -> dict[str, Any]:
    topology = metrics["topology"]
    quality = metrics["quality"]
    return {
        "triangles": metrics["triangles"],
        "points": metrics["points"],
        "boundary_edges": topology["boundary_edges"],
        "non_manifold_edges": topology["non_manifold_edges"],
        "non_manifold_vertices": topology["non_manifold_vertices"],
        "inconsistent_winding_edges": topology["inconsistent_winding_edges"],
        "components": topology["components"]["count"],
        "degenerate_faces": quality["degenerate_faces"],
        "volume_reliable": metrics["volume"]["reliable"],
    }


def sample_ints(values: np.ndarray, limit: int = 50) -> list[int]:
    unique = np.unique(values.astype(np.int64, copy=False))
    return [int(value) for value in unique[:limit]]


def connected_edge_regions(
    edge_to_faces: dict[tuple[int, int], list[int]],
    predicate: Any,
    *,
    face_component_ids: np.ndarray | None = None,
) -> list[list[tuple[int, int]]]:
    edges = [edge for edge, face_ids in edge_to_faces.items() if predicate(len(face_ids))]
    edge_components: list[int | None] = []
    for edge in edges:
        incident = edge_to_faces[edge]
        edge_components.append(
            int(face_component_ids[incident[0]])
            if face_component_ids is not None and incident
            else None
        )
    vertex_to_edges: dict[tuple[int, int | None], list[int]] = {}
    for index, (left, right) in enumerate(edges):
        component_id = edge_components[index]
        vertex_to_edges.setdefault((left, component_id), []).append(index)
        vertex_to_edges.setdefault((right, component_id), []).append(index)
    visited: set[int] = set()
    regions = []
    for start in range(len(edges)):
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        region = []
        while stack:
            edge_index = stack.pop()
            region.append(edges[edge_index])
            component_id = edge_components[edge_index]
            for vertex in edges[edge_index]:
                for next_index in vertex_to_edges[(vertex, component_id)]:
                    if next_index not in visited:
                        visited.add(next_index)
                        stack.append(next_index)
        regions.append(region)
    return sorted(regions, key=len, reverse=True)


def face_component_ids(
    faces: np.ndarray,
    edge_to_faces: dict[tuple[int, int], list[int]],
) -> np.ndarray:
    union_find = UnionFind(faces.shape[0])
    for face_ids in edge_to_faces.values():
        for face_id in face_ids[1:]:
            union_find.union(face_ids[0], face_id)
    roots = [union_find.find(index) for index in range(faces.shape[0])]
    labels = {root: index for index, root in enumerate(sorted(set(roots)))}
    return np.asarray([labels[root] for root in roots], dtype=np.int64)


def region_report(
    prefix: str,
    regions: list[list[tuple[int, int]]],
    points: np.ndarray,
    edge_to_faces: dict[tuple[int, int], list[int]],
    component_ids: np.ndarray,
    source_indices: np.ndarray,
    *,
    max_items: int,
    requires_policy: bool,
    classification: str,
    detector_reason: str,
    policy_reason_source: str,
    region_classifications: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    items = [
        region_item(
            prefix,
            index,
            region,
            points,
            edge_to_faces,
            component_ids,
            source_indices,
            requires_policy,
            classification,
            detector_reason,
            policy_reason_source,
            None if region_classifications is None else region_classifications[index - 1],
        )
        for index, region in enumerate(regions, start=1)
    ]
    return {
        "total_regions": len(regions),
        "reported_regions": len(items),
        "truncated": False,
        "geometry_truth_complete": True,
        "requested_report_item_limit": max(0, int(max_items)),
        "report_limit_applied_to_geometry": False,
        "items": items,
    }


def region_item(
    prefix: str,
    index: int,
    region: list[tuple[int, int]],
    points: np.ndarray,
    edge_to_faces: dict[tuple[int, int], list[int]],
    component_ids: np.ndarray,
    source_indices: np.ndarray,
    requires_policy: bool,
    classification: str,
    detector_reason: str,
    policy_reason_source: str,
    region_classification: dict[str, Any] | None,
) -> dict[str, Any]:
    vertices = sorted({vertex for edge in region for vertex in edge})
    face_ids = sorted({face_id for edge in region for face_id in edge_to_faces[edge]})
    degrees: dict[int, int] = {}
    for left, right in region:
        degrees[left] = degrees.get(left, 0) + 1
        degrees[right] = degrees.get(right, 0) + 1
    return {
        "id": f"{prefix}_{index:04d}",
        "type": prefix,
        "edge_count": len(region),
        "vertex_count": len(vertices),
        "edge_vertex_ids": vertices,
        "closed_chain": all(degree == 2 for degree in degrees.values()) if degrees else False,
        "length": edge_length(points, region),
        "bbox": bbox(points[vertices]) if vertices else None,
        # These ids are consumed by downstream geometry operations. They are
        # intentionally complete; report-size limits must never change which
        # faces a repair pass sees.
        "face_ids": [int(face_id) for face_id in face_ids],
        "source_triangle_ids": sample_ints(source_indices[face_ids], limit=max(len(face_ids), 1)),
        "source_triangle_count": len(face_ids),
        "component_id": region_component_id(face_ids, component_ids),
        "local_scale": inventory_local_scale(points, vertices, len(region), edge_length(points, region)),
        "nearby_region_ids": [],
        "classification": classification,
        "detector_reason": detector_reason,
        "requires_policy": requires_policy,
        "policy_reason_source": policy_reason_source,
        "blocking": True,
        "evidence_views": [],
        **(region_classification or {}),
    }


def exact_duplicate_inventory(
    points: np.ndarray,
    faces: np.ndarray,
    source_indices: np.ndarray,
    component_ids: np.ndarray,
    *,
    max_items: int,
) -> dict[str, Any]:
    keys = np.sort(faces, axis=1)
    groups: dict[tuple[int, int, int], list[int]] = {}
    for face_id, key in enumerate(keys):
        key_tuple = tuple(int(value) for value in key)
        groups.setdefault((key_tuple[0], key_tuple[1], key_tuple[2]), []).append(face_id)
    duplicates = [face_ids for face_ids in groups.values() if len(face_ids) > 1]
    items = [
        {
            "id": f"exact_overlap_{index:04d}",
            "type": "exact_duplicate_face_overlap",
            "face_count": len(face_ids),
            "face_ids": [int(face_id) for face_id in face_ids],
            "source_triangle_ids": sample_ints(source_indices[face_ids], limit=max(len(face_ids), 1)),
            "component_id": region_component_id(face_ids, component_ids),
            "bbox": bbox(points[faces[face_ids].ravel()]),
            "local_scale": duplicate_local_scale(points, faces, face_ids),
            "nearby_region_ids": [],
            "classification": "reject_region",
            "detector_reason": "exact_duplicate_face_overlap",
            "requires_policy": False,
            "blocking": True,
        }
        for index, face_ids in enumerate(duplicates, start=1)
    ]
    return {
        "method": "exact_duplicate_cleaned_vertex_sets",
        "total_regions": len(duplicates),
        "reported_regions": len(items),
        "truncated": False,
        "geometry_truth_complete": True,
        "requested_report_item_limit": max(0, int(max_items)),
        "report_limit_applied_to_geometry": False,
        "items": items,
    }


def inventory_truncation_items(inventory: dict[str, Any], *, stage_label: str = "inventory") -> list[dict[str, Any]]:
    items = []
    for section_name in INVENTORY_REGION_SECTIONS:
        section = inventory.get(section_name, {})
        if not section.get("truncated"):
            continue
        total = int(section.get("total_regions") or 0)
        reported = int(section.get("reported_regions") or 0)
        reason_codes = ["inventory_truncated"]
        if section_name in {"gap_regions", "semantic_opening_regions"}:
            reason_codes.append("opening_inventory_unresolved")
        items.append(
            {
                "item": f"{stage_label}.{section_name}",
                "status": "truncated",
                "blocking": True,
                "reason_codes": reason_codes,
                "failure_reason": (
                    f"{section_name} reported {reported} of {total}; "
                    "unreported regions were not audited for hybrid fusion"
                ),
            }
        )
    return items


def region_component_id(face_ids: list[int], component_ids: np.ndarray) -> int | None:
    if not face_ids:
        return None
    first = int(face_ids[0])
    if first < 0 or first >= component_ids.size:
        return None
    return int(component_ids[first])


def inventory_local_scale(points: np.ndarray, vertices: list[int], edge_count: int, length: float) -> float:
    if length > 0.0 and edge_count > 0:
        return float(length / max(edge_count, 1) * 3.0)
    if vertices:
        local = points[vertices]
        return float(np.linalg.norm(local.max(axis=0) - local.min(axis=0)) * 0.01)
    return 0.0


def duplicate_local_scale(points: np.ndarray, faces: np.ndarray, face_ids: list[int]) -> float:
    if not face_ids:
        return 0.0
    local = points[faces[face_ids].ravel()]
    return float(np.linalg.norm(local.max(axis=0) - local.min(axis=0)) * 0.01)


def bbox(points: np.ndarray) -> dict[str, Any]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return {"min": mins.tolist(), "max": maxs.tolist()}


def edge_length(points: np.ndarray, edges: list[tuple[int, int]]) -> float:
    return float(sum(np.linalg.norm(points[left] - points[right]) for left, right in edges))
