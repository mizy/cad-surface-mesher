from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hybrid_proxy_geometry import (
    decisions_by_source,
    expanded_bbox,
    face_centroids_normals,
    first_present,
    graph_components,
    local_scale,
    make_edges,
    nearest_signed_distance,
    points_touch_seam,
    rejection_reason,
    union_bbox,
)
from mesh_io import compact_mesh, write_vtp
from mesh_metrics import edge_topology, mesh_report, triangle_quality


@dataclass(frozen=True)
class ProxyFaceIndex:
    centroids: np.ndarray
    normals: np.ndarray
    axis_orders: tuple[np.ndarray, np.ndarray, np.ndarray]
    axis_values: tuple[np.ndarray, np.ndarray, np.ndarray]

    def query_bbox(self, bbox: dict[str, Any]) -> np.ndarray:
        mins = np.asarray(bbox["min"], dtype=np.float64)
        maxs = np.asarray(bbox["max"], dtype=np.float64)
        best_axis = 0
        best_start = best_end = 0
        best_count: int | None = None
        for axis, values in enumerate(self.axis_values):
            start = int(np.searchsorted(values, mins[axis], side="left"))
            end = int(np.searchsorted(values, maxs[axis], side="right"))
            count = end - start
            if best_count is None or count < best_count:
                best_axis, best_start, best_end, best_count = axis, start, end, count
        if not best_count:
            return np.zeros(0, dtype=np.int64)
        ids = self.axis_orders[best_axis][best_start:best_end]
        centroids = self.centroids[ids]
        inside = np.all((centroids >= mins) & (centroids <= maxs), axis=1)
        return ids[inside]


def build_proxy_face_index(proxy_points: np.ndarray, proxy_faces: np.ndarray) -> ProxyFaceIndex:
    centroids, normals = face_centroids_normals(proxy_points, proxy_faces)
    orders = tuple(np.argsort(centroids[:, axis], kind="mergesort") for axis in range(3))
    values = tuple(centroids[orders[axis], axis] for axis in range(3))
    return ProxyFaceIndex(centroids, normals, orders, values)


def build_patch_graph(
    points: np.ndarray,
    faces: np.ndarray,
    source_indices: np.ndarray,
    component_ids: np.ndarray,
    inventory: dict[str, Any],
    policy_packet: dict[str, Any],
    policy_decisions: list[dict[str, Any]],
    thresholds: Any,
) -> dict[str, Any]:
    decisions = decisions_by_source(policy_packet, policy_decisions)
    nodes = [
        make_node(item, points, faces, source_indices, component_ids, decisions, thresholds)
        for item in fusion_regions(inventory)
    ]
    edges, rejected = make_edges(nodes)
    component_map = graph_components(len(nodes), edges)
    components = []
    for group_id in sorted(set(component_map)):
        group_nodes = [nodes[index] for index, value in enumerate(component_map) if value == group_id]
        components.append(make_component(group_id + 1, group_nodes, edges))
    return {
        "thresholds": asdict(thresholds),
        "nodes": nodes,
        "edges": edges,
        "rejected_edges": rejected[:500],
        "components": components,
    }


def build_patch_region(
    source_points: np.ndarray,
    source_faces: np.ndarray,
    source_indices: np.ndarray,
    adjacency: list[set[int]],
    proxy_points: np.ndarray,
    proxy_faces: np.ndarray,
    proxy_index: ProxyFaceIndex,
    region: dict[str, Any],
    thresholds: Any,
    patches_dir: Path,
) -> dict[str, Any]:
    from hybrid_proxy_geometry import expand_faces

    region_id = int(region["fusion_region_id"])
    source_face_ids = region_source_faces(source_points, source_faces, source_indices, region)
    seam_face_ids = expand_faces(source_face_ids, adjacency, thresholds.seam_belt_rings)
    seam_vertex_ids = defect_vertex_ids(source_faces, seam_face_ids, region, source_points.shape[0])
    seam_points = source_points[seam_vertex_ids] if seam_vertex_ids.size else np.empty((0, 3))
    proxy_face_ids = extract_proxy_faces(
        source_points,
        source_faces,
        source_face_ids,
        proxy_points,
        proxy_faces,
        proxy_index,
        seam_points,
        region,
        thresholds,
    )
    if region["classification"] in {"preserve_opening", "pending_policy", "reject_region"}:
        proxy_face_ids = np.zeros(0, dtype=np.int64)
    artifacts, patch_points, patch_faces = write_patch_artifacts(
        patches_dir,
        region_id,
        source_points,
        source_faces,
        source_indices,
        seam_face_ids,
        proxy_points,
        proxy_faces,
        proxy_face_ids,
    )
    source_report = source_trust(region, source_indices, source_face_ids)
    proxy_report = proxy_trust_report(
        source_points,
        source_faces,
        source_face_ids,
        proxy_points,
        proxy_faces,
        proxy_face_ids,
        seam_points,
        thresholds,
        proxy_index,
        seam_contact_tolerance=region["local_scale"] * thresholds.seam_contact_scale,
        normal_is_blocking=region["classification"] != "patch_required",
    )
    selection = select_action(region, source_report, proxy_report, proxy_face_ids)
    blend_summary = patch_blend_summary(selection, proxy_face_ids)
    report = {
        "id": f"region_{region_id:04d}",
        "component_id": region.get("component_id"),
        "source_region_ids": region["source_region_ids"],
        "nearby_region_ids": region.get("nearby_region_ids", []),
        "local_scale": region["local_scale"],
        "classification": region["classification"],
        "semantic_classifications": region.get("semantic_classifications", []),
        "operator": region.get("operator"),
        "requires_policy": region["requires_policy"],
        "blocking": selection in {"hold_for_policy", "reject_patch"},
        "selection_status": selection,
        "source_trust": source_report,
        "proxy_trust": proxy_report,
        "proxy_weight": blend_summary,
        "sdf_blend_weight": blend_summary,
        "source_weight": {
            "min": 1.0 - blend_summary["max"],
            "mean": 1.0 - blend_summary["mean"],
            "max": 1.0 - blend_summary["min"],
            "p95": 1.0 - blend_summary["min"],
        },
        "policy_decision": region.get("policy_decision"),
        "graph": {"fusion_region_id": region_id, "node_ids": region["source_region_ids"], "edge_reasons": region.get("edge_reasons", [])},
        "artifacts": artifacts,
        "final_provenance": {
            "consumed_by_final": selection == "use_proxy_patch",
            "fusion_region_id": region_id,
            "face_origin_values": ["source"] + (["proxy_patch"] if selection == "use_proxy_patch" else []),
            "source_triangle_index_required_for_source_faces": True,
        },
        "accepted": selection in {"keep_source", "use_proxy_patch"},
        "rejection_reason": rejection_reason(selection, source_report, proxy_report),
        "seam_results": {
            "status": "not_run",
            "method": "conformal_loop_stitch",
            "boundary_edges_after": None,
            "non_manifold_edges_after": None,
        },
    }
    return {
        "report": report,
        "accepted_for_final": selection == "use_proxy_patch",
        "source_face_ids": source_face_ids,
        "remove_source_face_ids": source_face_ids if selection == "use_proxy_patch" else np.zeros(0, dtype=np.int64),
        "proxy_points": patch_points,
        "proxy_faces": patch_faces,
        "seam_vertex_ids": seam_vertex_ids,
        "fusion_region_id": region_id,
        "local_scale": float(region["local_scale"]),
        "ordered_boundary_loops": [
            np.asarray(loop, dtype=np.int64)
            for loop in region.get("ordered_boundary_loops", [])
        ],
        "trace": {
            "step": "region_patch",
            "region_id": report["id"],
            "selection_status": selection,
            "source_face_count": int(source_face_ids.size),
            "proxy_patch_face_count": int(proxy_face_ids.size),
            "artifacts": artifacts,
        },
    }


def fusion_regions(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    regions = [
        item
        for item in inventory.get("boundary_regions", {}).get("items", [])
        if item.get("patch_eligible")
        or item.get("requires_policy")
        or item.get("classification") == "patch_required"
    ]
    regions.extend(inventory.get("non_manifold_regions", {}).get("items", []))
    regions.extend(
        item for item in inventory.get("gap_regions", {}).get("items", [])
        if item.get("requires_policy") or item.get("classification") == "pending_policy"
    )
    seen, result = set(), []
    for region in regions:
        if region["id"] not in seen:
            seen.add(region["id"])
            result.append(region)
    return result


def make_node(
    item: dict[str, Any],
    points: np.ndarray,
    faces: np.ndarray,
    source_indices: np.ndarray,
    component_ids: np.ndarray,
    decisions: dict[str, dict[str, Any]],
    thresholds: Any,
) -> dict[str, Any]:
    face_ids = face_ids_for_item(item, points, faces, source_indices)
    scale = local_scale(item, thresholds)
    decision = decisions.get(item["id"])
    return {
        "id": item["id"],
        "type": item.get("type"),
        "component_id": int(component_ids[face_ids[0]]) if face_ids.size else None,
        "expanded_bbox": expanded_bbox(item.get("bbox"), points, faces, face_ids, scale, thresholds.bbox_expand_scale),
        "source_face_ids": face_ids.astype(int).tolist(),
        "source_triangle_ids": item.get("source_triangle_ids", []),
        "edge_vertex_ids": item.get("edge_vertex_ids", []),
        "local_scale": scale,
        "classification": classify_region(item, decision),
        "semantic_classification": item.get("classification"),
        "operator": item.get("operator"),
        "patch_eligible": bool(item.get("patch_eligible")) or item.get("classification") == "patch_required",
        "paired_region_id": item.get("paired_region_id"),
        "pair_id": item.get("pair_id"),
        "ordered_vertex_ids": item.get("ordered_vertex_ids", []),
        "requires_policy": bool(item.get("requires_policy")) or classify_region(item, decision) == "pending_policy",
        "policy_decision": decision,
    }


def make_component(component_id: int, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
    source_region_ids = [node["id"] for node in nodes]
    classifications = {node["classification"] for node in nodes}
    classification = "patch_required"
    if "pending_policy" in classifications:
        classification = "pending_policy"
    elif "reject_region" in classifications:
        classification = "reject_region"
    elif classifications == {"preserve_opening"}:
        classification = "preserve_opening"
    operators = sorted({str(node["operator"]) for node in nodes if node.get("operator")})
    return {
        "fusion_region_id": component_id,
        "component_id": nodes[0].get("component_id"),
        "source_region_ids": source_region_ids,
        "nearby_region_ids": source_region_ids[1:],
        "bbox": union_bbox([node["expanded_bbox"] for node in nodes]),
        "local_scale": max(float(node["local_scale"]) for node in nodes),
        "classification": classification,
        "semantic_classifications": sorted({str(node.get("semantic_classification")) for node in nodes}),
        "operators": operators,
        "operator": operators[0] if len(operators) == 1 else None,
        "ordered_boundary_loops": [
            node["ordered_vertex_ids"]
            for node in nodes
            if node.get("ordered_vertex_ids")
        ],
        "paired_region_ids": [node.get("paired_region_id") for node in nodes if node.get("paired_region_id")],
        "requires_policy": any(node["requires_policy"] for node in nodes),
        "policy_decision": first_present(node.get("policy_decision") for node in nodes),
        "source_face_ids": sorted({face_id for node in nodes for face_id in node["source_face_ids"]}),
        "edge_vertex_ids": sorted({vertex_id for node in nodes for vertex_id in node.get("edge_vertex_ids", [])}),
        "edge_reasons": [edge for edge in edges if edge["left"] in source_region_ids and edge["right"] in source_region_ids],
    }


def extract_proxy_faces(
    source_points: np.ndarray,
    source_faces: np.ndarray,
    source_face_ids: np.ndarray,
    proxy_points: np.ndarray,
    proxy_faces: np.ndarray,
    proxy_index: ProxyFaceIndex,
    seam_points: np.ndarray,
    region: dict[str, Any],
    thresholds: Any,
) -> np.ndarray:
    ids = proxy_index.query_bbox(region["bbox"])
    if ids.size == 0:
        return ids
    centroids = proxy_index.centroids
    normals = proxy_index.normals
    if source_face_ids.size:
        src_centroids, src_normals = face_centroids_normals(source_points, source_faces[source_face_ids])
        distances, _, nearest = nearest_signed_distance(centroids[ids], src_centroids, src_normals, thresholds)
        dot = np.abs(np.einsum("ij,ij->i", normals[ids], src_normals[nearest]))
        covered = (distances < thresholds.voxel_pitch * 0.6) & (dot > 0.85)
    else:
        covered = np.zeros(ids.size, dtype=bool)
    seam_contact = points_touch_seam(centroids[ids], seam_points, region["local_scale"] * thresholds.seam_contact_scale)
    selected = ids[(~covered) | seam_contact]
    seeds = ids[seam_contact]
    max_faces = min(max(int(source_face_ids.size * 4), 80), 1600)
    return limited_seam_patch(proxy_faces, selected, seeds, max_faces)


def defect_vertex_ids(
    faces: np.ndarray,
    seam_face_ids: np.ndarray,
    region: dict[str, Any],
    point_count: int,
) -> np.ndarray:
    explicit = np.asarray(region.get("edge_vertex_ids") or [], dtype=np.int64)
    explicit = explicit[(explicit >= 0) & (explicit < point_count)]
    if explicit.size:
        return np.unique(explicit)
    if seam_face_ids.size == 0:
        return np.zeros(0, dtype=np.int64)
    return np.unique(faces[seam_face_ids].ravel())


def limited_seam_patch(
    faces: np.ndarray,
    selected: np.ndarray,
    seeds: np.ndarray,
    max_faces: int,
) -> np.ndarray:
    if selected.size == 0 or seeds.size == 0:
        return np.zeros(0, dtype=np.int64)
    selected = np.asarray(sorted(set(int(value) for value in selected)), dtype=np.int64)
    selected_lookup = {int(face_id): index for index, face_id in enumerate(selected)}
    seed_local = [selected_lookup[int(face_id)] for face_id in seeds if int(face_id) in selected_lookup]
    if not seed_local:
        return np.zeros(0, dtype=np.int64)
    seed_local = seed_local[:max_faces]
    local_faces = faces[selected]
    _, edge_faces = edge_topology(local_faces)
    adjacency: list[set[int]] = [set() for _ in range(local_faces.shape[0])]
    for face_ids in edge_faces.values():
        if len(face_ids) < 2:
            continue
        for left in face_ids:
            adjacency[left].update(face_id for face_id in face_ids if face_id != left)
    visited = set(seed_local)
    queue = list(seed_local)
    while queue and len(visited) < max_faces:
        face_id = queue.pop(0)
        for next_id in adjacency[face_id]:
            if next_id in visited:
                continue
            visited.add(next_id)
            queue.append(next_id)
            if len(visited) >= max_faces:
                break
    return selected[np.asarray(sorted(visited), dtype=np.int64)]


def write_patch_artifacts(
    patches_dir: Path,
    region_id: int,
    source_points: np.ndarray,
    source_faces: np.ndarray,
    source_indices: np.ndarray,
    seam_face_ids: np.ndarray,
    proxy_points: np.ndarray,
    proxy_faces: np.ndarray,
    proxy_face_ids: np.ndarray,
) -> tuple[dict[str, str], np.ndarray, np.ndarray]:
    artifacts: dict[str, str] = {}
    patch_points = np.empty((0, 3), dtype=np.float64)
    patch_faces = np.empty((0, 3), dtype=np.int64)
    if proxy_face_ids.size:
        patch_points, patch_faces = compact_mesh(proxy_points, proxy_faces, proxy_face_ids)
        path = patches_dir / f"region_{region_id:04d}_proxy_patch.vtp"
        write_vtp(path, patch_points, patch_faces, {"fusion_region_id": np.full(patch_faces.shape[0], region_id)})
        artifacts["proxy_patch_vtp"] = str(path)
    if seam_face_ids.size:
        seam_points, seam_faces = compact_mesh(source_points, source_faces, seam_face_ids)
        path = patches_dir / f"region_{region_id:04d}_seam_belt.vtp"
        write_vtp(path, seam_points, seam_faces, {"source_triangle_index": source_indices[seam_face_ids], "fusion_region_id": np.full(seam_faces.shape[0], region_id)})
        artifacts["seam_belt_vtp"] = str(path)
    return artifacts, patch_points, patch_faces


def source_trust(region: dict[str, Any], source_indices: np.ndarray, face_ids: np.ndarray) -> dict[str, Any]:
    reasons = []
    if region["classification"] in {"patch_required", "reject_region"}:
        reasons.append("open_boundary" if any("boundary" in rid for rid in region["source_region_ids"]) else "topology_risk")
    if face_ids.size == 0:
        reasons.append("missing_source_faces")
    elif np.any(source_indices[face_ids] < 0):
        reasons.append("missing_source_triangle_index")
    return {"status": "trusted" if not reasons else "untrusted", "reason_codes": reasons, "values": {"source_face_count": int(face_ids.size)}}


def proxy_trust_report(
    source_points: np.ndarray,
    source_faces: np.ndarray,
    source_face_ids: np.ndarray,
    proxy_points: np.ndarray,
    proxy_faces: np.ndarray,
    proxy_face_ids: np.ndarray,
    seam_points: np.ndarray,
    thresholds: Any,
    proxy_index: ProxyFaceIndex | None = None,
    seam_contact_tolerance: float | None = None,
    normal_is_blocking: bool = True,
) -> dict[str, Any]:
    if proxy_face_ids.size == 0:
        return {
            "status": "untrusted",
            "reason_codes": ["proxy_patch_extraction_failed"],
            "failure_reason_codes": ["proxy_patch_extraction_failed"],
            "evidence_codes": [],
            "values": {},
        }
    patch_points, patch_faces = compact_mesh(proxy_points, proxy_faces, proxy_face_ids)
    patch_report = mesh_report(patch_points, patch_faces)
    patch_area = triangle_quality(proxy_points, proxy_faces[proxy_face_ids])["surface_area"]
    source_area = triangle_quality(source_points, source_faces[source_face_ids])["surface_area"] if source_face_ids.size else 0.0
    contact_tolerance = (
        float(seam_contact_tolerance)
        if seam_contact_tolerance is not None
        else thresholds.voxel_pitch * thresholds.seam_contact_scale
    )
    seam_contact = bool(points_touch_seam(proxy_points[proxy_faces[proxy_face_ids]].reshape(-1, 3), seam_points, contact_tolerance).any())
    area_ratio = float(patch_area / max(source_area, thresholds.voxel_pitch ** 2))
    normal_min_dot = normal_compatibility(
        source_points,
        source_faces,
        source_face_ids,
        proxy_points,
        proxy_faces,
        proxy_face_ids,
        thresholds,
        proxy_index,
    )
    evidence_codes = ["seam_contact_passed"] if seam_contact else []
    failure_codes = [] if seam_contact else ["proxy_patch_extraction_failed", "seam_contact_failed"]
    normal_passed = normal_min_dot >= 0.55
    if normal_passed:
        evidence_codes.append("normal_compatibility_passed")
    elif normal_is_blocking:
        failure_codes.append("normal_compatibility_failed")
    else:
        evidence_codes.append("normal_compatibility_low_nonblocking")
    if patch_report["topology"]["non_manifold_edges"] == 0:
        evidence_codes.append("proxy_patch_non_manifold_zero")
    if area_ratio <= thresholds.max_patch_area_ratio:
        evidence_codes.append("patch_area_ratio_within_limit")
    else:
        failure_codes.append("patch_area_ratio_exceeded")
    normal_allowed = normal_passed or not normal_is_blocking
    status = "trusted" if seam_contact and normal_allowed and area_ratio <= thresholds.max_patch_area_ratio else "untrusted"
    return {
        "status": status,
        "reason_codes": failure_codes,
        "failure_reason_codes": failure_codes,
        "evidence_codes": evidence_codes,
        "values": {
            "proxy_face_count": int(proxy_face_ids.size),
            "normal_min_dot": normal_min_dot,
            "normal_is_blocking": bool(normal_is_blocking),
            "patch_area": float(patch_area),
            "source_seam_area": float(source_area),
            "patch_area_ratio": area_ratio,
            "boundary_edges": patch_report["topology"]["boundary_edges"],
            "non_manifold_edges": patch_report["topology"]["non_manifold_edges"],
        },
    }


def normal_compatibility(
    source_points: np.ndarray,
    source_faces: np.ndarray,
    source_face_ids: np.ndarray,
    proxy_points: np.ndarray,
    proxy_faces: np.ndarray,
    proxy_face_ids: np.ndarray,
    thresholds: Any,
    proxy_index: ProxyFaceIndex | None = None,
) -> float:
    if source_face_ids.size == 0 or proxy_face_ids.size == 0:
        return 0.0
    source_centroids, source_normals = face_centroids_normals(source_points, source_faces[source_face_ids])
    if proxy_index is None:
        proxy_centroids, proxy_normals = face_centroids_normals(proxy_points, proxy_faces[proxy_face_ids])
    else:
        proxy_centroids = proxy_index.centroids[proxy_face_ids]
        proxy_normals = proxy_index.normals[proxy_face_ids]
    _, _, nearest = nearest_signed_distance(proxy_centroids, source_centroids, source_normals, thresholds)
    dots = np.abs(np.einsum("ij,ij->i", proxy_normals, source_normals[nearest]))
    return float(np.percentile(dots, 5)) if dots.size else 0.0


def select_action(region: dict[str, Any], source: dict[str, Any], proxy: dict[str, Any], proxy_face_ids: np.ndarray) -> str:
    if region["classification"] == "pending_policy":
        return "hold_for_policy"
    if region["classification"] == "preserve_opening" and source["status"] == "trusted":
        return "keep_source"
    if region["classification"] == "reject_region":
        return "reject_patch"
    return "use_proxy_patch" if proxy_face_ids.size and proxy["status"] == "trusted" else "reject_patch"


def patch_blend_summary(selection: str, proxy_face_ids: np.ndarray) -> dict[str, float]:
    weight = 1.0 if selection == "use_proxy_patch" and proxy_face_ids.size else 0.0
    return {"min": weight, "mean": weight, "max": weight, "p95": weight}


def classify_region(item: dict[str, Any], decision: dict[str, Any] | None) -> str:
    if item.get("requires_policy") and (not decision or decision.get("status") != "decided"):
        return "pending_policy"
    if decision and decision.get("status") == "decided":
        if decision.get("decision") == "preserve":
            return "preserve_opening"
        if decision.get("decision") == "defer":
            return "pending_policy"
        if decision.get("decision") == "reject":
            return "reject_region"
        return "patch_required"
    if item.get("classification") in {"patch_required", "preserve_opening", "pending_policy", "reject_region"}:
        return item["classification"]
    return "reject_region" if item.get("type") == "non_manifold_edge_region" else "patch_required"


def face_ids_for_item(item: dict[str, Any], points: np.ndarray, faces: np.ndarray, source_indices: np.ndarray) -> np.ndarray:
    explicit = np.asarray(item.get("face_ids") or [], dtype=np.int64)
    explicit = explicit[(explicit >= 0) & (explicit < faces.shape[0])]
    source_ids = np.asarray(item.get("source_triangle_ids") or [], dtype=np.int64)
    by_source = np.flatnonzero(np.isin(source_indices, source_ids)) if source_ids.size else np.zeros(0, dtype=np.int64)
    result = np.unique(np.concatenate([explicit, by_source]))
    if result.size or not item.get("bbox"):
        return result
    centroids, _ = face_centroids_normals(points, faces)
    bbox = item["bbox"]
    return np.flatnonzero(np.all((centroids >= np.asarray(bbox["min"])) & (centroids <= np.asarray(bbox["max"])), axis=1))


def region_source_faces(points: np.ndarray, faces: np.ndarray, source_indices: np.ndarray, region: dict[str, Any]) -> np.ndarray:
    from_region = np.asarray(region.get("source_face_ids") or [], dtype=np.int64)
    from_region = from_region[(from_region >= 0) & (from_region < faces.shape[0])]
    return from_region if from_region.size else face_ids_for_item(region, points, faces, source_indices)
