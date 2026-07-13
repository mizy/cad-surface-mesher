from __future__ import annotations

from typing import Any

import numpy as np

from mesh_metrics import UnionFind, connected_components, edge_topology


FACE_ORIGIN = {"source": 0, "proxy_patch": 1, "stitch_band": 2, "hole_fill": 3}


def bidirectional_distance(
    source_points: np.ndarray,
    source_faces: np.ndarray,
    final_points: np.ndarray,
    final_faces: np.ndarray,
    thresholds: Any,
) -> dict[str, Any]:
    source_samples, source_normals = sampled_face_points(source_points, source_faces, thresholds.max_distance_samples)
    final_samples, final_normals = sampled_face_points(final_points, final_faces, thresholds.max_distance_samples)
    diag = max(float(np.linalg.norm(source_points.max(axis=0) - source_points.min(axis=0))), 1e-12)
    threshold = diag * thresholds.source_distance_ratio
    fwd, fwd_signed, _ = nearest_signed_distance(final_samples, source_samples, source_normals, thresholds)
    rev, rev_signed, _ = nearest_signed_distance(source_samples, final_samples, final_normals, thresholds)
    distance = np.concatenate([fwd, rev])
    signed = np.concatenate([fwd_signed, rev_signed])
    report = distance_summary(distance, signed, threshold, "bidirectional_sampled_centroid_signed_distance")
    report["final_to_source"] = distance_summary(fwd, fwd_signed, threshold, "sampled_final_to_source")
    report["source_to_final"] = distance_summary(rev, rev_signed, threshold, "sampled_source_to_final")
    return report


def patch_local_drift(
    source_points: np.ndarray,
    source_faces: np.ndarray,
    final_points: np.ndarray,
    final_faces: np.ndarray,
    patch_faces: np.ndarray,
    thresholds: Any,
) -> dict[str, Any]:
    if patch_faces.size == 0:
        return {"method": "no_proxy_patch_faces", "status": "not_applicable", "passed": True, "max_ratio": 0.0, "threshold": None}
    src_samples, src_normals = sampled_face_points(source_points, source_faces, thresholds.max_target_samples)
    patch_samples, _ = sampled_face_points(final_points, final_faces[patch_faces], thresholds.max_distance_samples)
    distances, signed, _ = nearest_signed_distance(patch_samples, src_samples, src_normals, thresholds)
    diag = max(float(np.linalg.norm(source_points.max(axis=0) - source_points.min(axis=0))), 1e-12)
    max_ratio = float(distances.max() / diag) if distances.size else 0.0
    return {
        "method": "proxy_patch_to_source_sampled_drift",
        "max_ratio": max_ratio,
        "max": float(distances.max()) if distances.size else 0.0,
        "signed": {"min": float(signed.min()) if signed.size else 0.0, "max": float(signed.max()) if signed.size else 0.0},
        "threshold": thresholds.source_distance_ratio,
        "passed": bool(max_ratio <= thresholds.source_distance_ratio),
        "status": "computed",
    }


def nearest_signed_distance(points: np.ndarray, targets: np.ndarray, normals: np.ndarray, thresholds: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points.size == 0 or targets.size == 0:
        return np.zeros(points.shape[0]), np.zeros(points.shape[0]), np.zeros(points.shape[0], dtype=np.int64)
    target_ids = sample_indices(targets.shape[0], thresholds.max_target_samples)
    target_points = targets[target_ids]
    target_normals = normals[target_ids]
    nearest = np.zeros(points.shape[0], dtype=np.int64)
    distances = np.zeros(points.shape[0], dtype=np.float64)
    signed = np.zeros(points.shape[0], dtype=np.float64)
    for start in range(0, points.shape[0], 256):
        chunk = points[start:start + 256]
        diff = chunk[:, None, :] - target_points[None, :, :]
        squared = np.einsum("ijk,ijk->ij", diff, diff)
        local = np.argmin(squared, axis=1)
        nearest[start:start + chunk.shape[0]] = target_ids[local]
        distances[start:start + chunk.shape[0]] = np.sqrt(squared[np.arange(chunk.shape[0]), local])
        signed[start:start + chunk.shape[0]] = np.einsum("ij,ij->i", diff[np.arange(chunk.shape[0]), local], target_normals[local])
    return distances, signed, nearest


def sampled_face_points(points: np.ndarray, faces: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray]:
    centroids, normals = face_centroids_normals(points, faces)
    ids = sample_indices(centroids.shape[0], limit)
    return centroids[ids], normals[ids]


def face_centroids_normals(points: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    triangles = points[faces]
    centroids = triangles.mean(axis=1)
    raw = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(raw, axis=1)
    normals = np.divide(raw, lengths[:, None], out=np.zeros_like(raw), where=lengths[:, None] > 1e-15)
    return centroids, normals


def largest_seam_piece(
    faces: np.ndarray,
    selected: np.ndarray,
    points: np.ndarray,
    seam_points: np.ndarray,
    region: dict[str, Any],
    thresholds: Any,
) -> np.ndarray:
    if selected.size == 0:
        return selected
    local_faces = faces[selected]
    _, edge_faces = edge_topology(local_faces)
    comps = connected_components(local_faces.shape[0], edge_faces)
    labels = face_component_ids(local_faces, edge_faces)
    best, best_score = np.zeros(0, dtype=np.int64), -1.0
    centroids, _ = face_centroids_normals(points, faces[selected])
    contact = points_touch_seam(centroids, seam_points, region["local_scale"] * thresholds.seam_contact_scale)
    for label in np.unique(labels):
        local_ids = np.flatnonzero(labels == label)
        score = float(local_ids.size) + (1000000.0 if np.any(contact[local_ids]) else 0.0)
        if score > best_score:
            best_score = score
            best = selected[local_ids]
    return best if best.size else selected[: comps["largest_faces"][0]]


def points_touch_seam(points: np.ndarray, seam_points: np.ndarray, tolerance: float) -> np.ndarray:
    if points.size == 0 or seam_points.size == 0:
        return np.zeros(points.shape[0], dtype=bool)
    touched = np.zeros(points.shape[0], dtype=bool)
    for start in range(0, points.shape[0], 256):
        chunk = points[start:start + 256]
        diff = chunk[:, None, :] - seam_points[None, :, :]
        touched[start:start + chunk.shape[0]] = np.min(np.einsum("ijk,ijk->ij", diff, diff), axis=1) <= tolerance * tolerance
    return touched


def face_adjacency(faces: np.ndarray) -> list[set[int]]:
    _, edge_faces = edge_topology(faces)
    adjacency: list[set[int]] = [set() for _ in range(faces.shape[0])]
    for face_ids in edge_faces.values():
        for left in face_ids:
            adjacency[left].update(face_id for face_id in face_ids if face_id != left)
    return adjacency


def expand_faces(seed: np.ndarray, adjacency: list[set[int]], rings: int) -> np.ndarray:
    selected = set(int(value) for value in seed)
    frontier = set(selected)
    for _ in range(max(0, rings)):
        next_frontier = set()
        for face_id in frontier:
            next_frontier.update(adjacency[face_id])
        next_frontier -= selected
        selected.update(next_frontier)
        frontier = next_frontier
    return np.asarray(sorted(selected), dtype=np.int64)


def face_component_ids(faces: np.ndarray, edge_faces: dict[tuple[int, int], list[int]]) -> np.ndarray:
    union_find = UnionFind(faces.shape[0])
    for face_ids in edge_faces.values():
        for face_id in face_ids[1:]:
            union_find.union(face_ids[0], face_id)
    roots = [union_find.find(index) for index in range(faces.shape[0])]
    labels = {root: index for index, root in enumerate(sorted(set(roots)))}
    return np.asarray([labels[root] for root in roots], dtype=np.int64)


def graph_components(count: int, edges: list[dict[str, Any]]) -> list[int]:
    parent = list(range(count))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for edge in edges:
        parent[find(int(edge["right_index"]))] = find(int(edge["left_index"]))
    roots = {root: index for index, root in enumerate(sorted({find(index) for index in range(count)}))}
    return [roots[find(index)] for index in range(count)]


def make_edges(nodes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    edges: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for left in range(len(nodes)):
        for right in range(left + 1, len(nodes)):
            reason = edge_reason(nodes[left], nodes[right])
            row = {"left": nodes[left]["id"], "right": nodes[right]["id"], "left_index": left, "right_index": right, "reason": reason}
            (edges if reason else rejected).append(
                row
                if reason
                else {**row, "reason": "not_an_explicit_loop_pair_or_coincident_non_boundary_defect"}
            )
    return edges, rejected


def decisions_by_source(policy_packet: dict[str, Any], decisions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_item = {row.get("item_id") or row.get("id"): row for row in decisions}
    return {
        item.get("source_region"): by_item[item.get("id")]
        for item in policy_packet.get("items", [])
        if item.get("id") in by_item
    }


def local_scale(item: dict[str, Any], thresholds: Any) -> float:
    explicit = float(item.get("local_scale") or 0.0)
    if explicit > 0.0:
        return explicit
    length = float(item.get("length") or 0.0)
    edge_count = max(int(item.get("edge_count") or 1), 1)
    if length > 0.0:
        return max(length / edge_count * 3.0, 1e-12)
    bbox = item.get("bbox")
    if bbox:
        diagonal = float(
            np.linalg.norm(np.asarray(bbox["max"], dtype=np.float64) - np.asarray(bbox["min"], dtype=np.float64))
        )
        if diagonal > 0.0:
            return max(diagonal * 0.05, 1e-12)
    return max(float(thresholds.voxel_pitch), 1e-12)


def expanded_bbox(raw_bbox: dict[str, Any] | None, points: np.ndarray, faces: np.ndarray, face_ids: np.ndarray, scale: float, factor: float) -> dict[str, Any]:
    if raw_bbox:
        mins = np.asarray(raw_bbox["min"], dtype=np.float64)
        maxs = np.asarray(raw_bbox["max"], dtype=np.float64)
    elif face_ids.size:
        local = points[faces[face_ids].ravel()]
        mins, maxs = local.min(axis=0), local.max(axis=0)
    else:
        mins, maxs = points.min(axis=0), points.max(axis=0)
    pad = max(scale * factor, 1e-12)
    return {"min": (mins - pad).tolist(), "max": (maxs + pad).tolist()}


def union_bbox(boxes: list[dict[str, Any]]) -> dict[str, Any]:
    mins = np.vstack([box["min"] for box in boxes]).min(axis=0)
    maxs = np.vstack([box["max"] for box in boxes]).max(axis=0)
    return {"min": mins.tolist(), "max": maxs.tolist()}


def first_present(values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def rejection_reason(selection: str, source: dict[str, Any], proxy: dict[str, Any]) -> str | None:
    if selection in {"keep_source", "use_proxy_patch"}:
        return None
    if selection == "hold_for_policy":
        return "policy_review_pending"
    return first_failure_code(proxy) or first_failure_code(source) or "patch_region_rejected"


def first_failure_code(report: dict[str, Any]) -> str | None:
    for key in ("failure_reason_codes", "reason_codes"):
        for code in report.get(key) or []:
            if is_failure_code(str(code)):
                return str(code)
    return None


def is_failure_code(code: str) -> bool:
    return (
        code == "proxy_patch_extraction_failed"
        or code == "patch_region_rejected"
        or code == "policy_region_rejected"
        or code.endswith("_failed")
        or code.endswith("_exceeded")
        or code.endswith("_missing")
        or code.endswith("_pending")
    )


def topology_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "triangles": metrics["triangles"],
        "boundary_edges": metrics["topology"]["boundary_edges"],
        "non_manifold_edges": metrics["topology"]["non_manifold_edges"],
        "inconsistent_winding_edges": metrics["topology"].get("inconsistent_winding_edges"),
        "components": metrics["topology"]["components"]["count"],
        "degenerate_faces": metrics["quality"]["degenerate_faces"],
        "volume_reliable": metrics["volume"]["reliable"],
    }


def provenance_summary(provenance: dict[str, np.ndarray]) -> dict[str, Any]:
    origins = sorted(int(value) for value in np.unique(provenance["face_origin"]))
    return {
        "present": all(name in provenance for name in ("face_origin", "source_triangle_index", "fusion_region_id")),
        "face_count": int(provenance["face_origin"].size),
        "face_origin_values": origins,
        "face_origin_labels": {str(value): label for label, value in FACE_ORIGIN.items()},
        "source_faces_missing_source_triangle_index": int(np.count_nonzero((provenance["face_origin"] == FACE_ORIGIN["source"]) & (provenance["source_triangle_index"] < 0))),
    }


def artifact_paths(patch_regions: list[dict[str, Any]]) -> list[str]:
    return [path for region in patch_regions for path in region.get("artifacts", {}).values()]


def distance_summary(distance: np.ndarray, signed: np.ndarray, threshold: float, method: str) -> dict[str, Any]:
    return {
        "method": method,
        "max": float(distance.max()) if distance.size else 0.0,
        "p95": float(np.percentile(distance, 95)) if distance.size else 0.0,
        "mean": float(distance.mean()) if distance.size else 0.0,
        "signed": {
            "min": float(signed.min()) if signed.size else 0.0,
            "max": float(signed.max()) if signed.size else 0.0,
            "mean": float(signed.mean()) if signed.size else 0.0,
        },
        "threshold": threshold,
        "passed": bool(distance.size == 0 or distance.max() <= threshold),
        "status": "computed",
    }


def sample_indices(size: int, limit: int) -> np.ndarray:
    return np.arange(size, dtype=np.int64) if size <= limit else np.linspace(0, size - 1, limit, dtype=np.int64)


def edge_reason(left: dict[str, Any], right: dict[str, Any]) -> str | None:
    if left.get("paired_region_id") == right.get("id") or right.get("paired_region_id") == left.get("id"):
        return "explicit_compatible_boundary_loop_pair"
    # Boundary loops are repair transactions, not proximity clusters.  Merging a
    # small hole or zipper loop into a nearby policy opening would make one bad
    # region roll back unrelated geometry and would change semantic truth.
    if left.get("type") == "boundary_loop" or right.get("type") == "boundary_loop":
        return None
    if (
        left.get("component_id") is not None
        and left.get("component_id") == right.get("component_id")
        and bbox_intersects(left["expanded_bbox"], right["expanded_bbox"])
    ):
        return "coincident_non_boundary_defect_evidence"
    return None


def bbox_intersects(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return bool(np.all(np.asarray(left["min"]) <= np.asarray(right["max"])) and np.all(np.asarray(right["min"]) <= np.asarray(left["max"])))


# The functions below are deliberately independent from the current proxy-fusion
# orchestration.  They form the deterministic geometry primitive that the
# orchestrator can call once a source boundary and a proxy boundary have been
# classified as a genuine patch pair.


def extract_ordered_boundary_loops(
    points: np.ndarray,
    faces: np.ndarray,
    *,
    geometric_tolerance: float | None = None,
) -> dict[str, Any]:
    """Extract consistently oriented, simple boundary loops from a triangle mesh.

    Loops follow the directed boundary edges induced by face winding.  A branched
    boundary, an open chain, inconsistent boundary winding, repeated vertices, or
    a geometric self-intersection is a hard failure instead of being silently
    converted into a patch candidate.
    """

    validation = _mesh_input_diagnostics(points, faces)
    if validation:
        return _loop_failure(validation[0], "mesh_validation", validation[1])
    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    tolerance = _geometric_tolerance(points, geometric_tolerance)
    _, edge_faces = edge_topology(faces)
    boundary_edges = sorted(edge for edge, face_ids in edge_faces.items() if len(face_ids) == 1)
    if not boundary_edges:
        return {
            "success": True,
            "failure_reason_codes": [],
            "loops": [],
            "diagnostics": {
                "stage": "boundary_loop_extraction",
                "boundary_edge_count": 0,
                "loop_count": 0,
                "geometric_tolerance": tolerance,
            },
        }

    adjacency: dict[int, list[int]] = {}
    directed: dict[tuple[int, int], tuple[int, int]] = {}
    for edge in boundary_edges:
        left, right = edge
        adjacency.setdefault(left, []).append(right)
        adjacency.setdefault(right, []).append(left)
        face_id = edge_faces[edge][0]
        if _triangle_double_area(points, faces[face_id]) <= tolerance * tolerance:
            return _loop_failure(
                "boundary_adjacent_face_degenerate",
                "boundary_loop_extraction",
                "A boundary loop cannot be oriented from a degenerate incident triangle.",
                edge=list(edge),
                face_id=int(face_id),
            )
        directed_edge = _face_directed_edge(faces[face_id], edge)
        if directed_edge is None:
            return _loop_failure(
                "boundary_edge_not_present_in_face",
                "boundary_loop_extraction",
                "A topology boundary edge could not be recovered from its incident face.",
                edge=list(edge),
                face_id=int(face_id),
            )
        directed[edge] = directed_edge

    invalid_degrees = {
        int(vertex_id): len(neighbours)
        for vertex_id, neighbours in adjacency.items()
        if len(neighbours) != 2
    }
    if invalid_degrees:
        return _loop_failure(
            "boundary_graph_not_simple",
            "boundary_loop_extraction",
            "Boundary vertices must have degree two; branches and open chains cannot be stitched.",
            invalid_vertex_degrees=dict(list(sorted(invalid_degrees.items()))[:100]),
            invalid_vertex_count=len(invalid_degrees),
        )

    unvisited = set(boundary_edges)
    loops: list[np.ndarray] = []
    loop_diagnostics: list[dict[str, Any]] = []
    while unvisited:
        component_vertices = _boundary_component_vertices(min(unvisited), adjacency)
        start = min(component_vertices)
        outgoing = [
            neighbour
            for neighbour in adjacency[start]
            if directed[tuple(sorted((start, neighbour)))] == (start, neighbour)
        ]
        if len(outgoing) != 1:
            return _loop_failure(
                "boundary_orientation_inconsistent",
                "boundary_loop_extraction",
                "Each oriented boundary loop must have exactly one outgoing edge per vertex.",
                vertex_id=int(start),
                outgoing_boundary_edges=len(outgoing),
            )
        loop = [start]
        previous = start
        current = outgoing[0]
        while current != start:
            if current in loop:
                return _loop_failure(
                    "boundary_loop_repeats_vertex",
                    "boundary_loop_extraction",
                    "Boundary traversal repeated a vertex before closing.",
                    vertex_id=int(current),
                )
            loop.append(current)
            next_vertices = [value for value in adjacency[current] if value != previous]
            if len(next_vertices) != 1:
                return _loop_failure(
                    "boundary_graph_not_simple",
                    "boundary_loop_extraction",
                    "Boundary traversal encountered a branch or open chain.",
                    vertex_id=int(current),
                )
            previous, current = current, next_vertices[0]
            if len(loop) > len(component_vertices):
                return _loop_failure(
                    "boundary_loop_traversal_failed",
                    "boundary_loop_extraction",
                    "Boundary traversal did not close deterministically.",
                )

        loop_array = np.asarray(loop, dtype=np.int64)
        orientation_conflicts = _loop_orientation_conflicts(loop_array, directed)
        if orientation_conflicts:
            return _loop_failure(
                "boundary_orientation_inconsistent",
                "boundary_loop_extraction",
                "Boundary edge directions do not form one consistently oriented cycle.",
                conflicting_edges=orientation_conflicts[:100],
                conflict_count=len(orientation_conflicts),
            )
        simplicity = _simple_loop_diagnostics(points, loop_array, tolerance)
        if not simplicity["passed"]:
            return _loop_failure(
                simplicity["failure_reason"],
                "boundary_loop_extraction",
                simplicity["message"],
                **simplicity.get("details", {}),
            )
        for index, vertex_id in enumerate(loop):
            unvisited.discard(tuple(sorted((vertex_id, loop[(index + 1) % len(loop)]))))
        lengths = _loop_arc_lengths(points, loop_array)[1]
        loops.append(loop_array)
        loop_diagnostics.append(
            {
                "start_vertex_id": int(loop_array[0]),
                "vertex_count": int(loop_array.size),
                "edge_count": int(loop_array.size),
                "length": float(lengths.sum()),
                "simple": True,
                "orientation": "face_winding_induced",
            }
        )

    order = sorted(range(len(loops)), key=lambda index: (int(loops[index][0]), int(loops[index].size)))
    loops = [loops[index] for index in order]
    loop_diagnostics = [loop_diagnostics[index] for index in order]
    return {
        "success": True,
        "failure_reason_codes": [],
        "loops": loops,
        "diagnostics": {
            "stage": "boundary_loop_extraction",
            "boundary_edge_count": len(boundary_edges),
            "loop_count": len(loops),
            "geometric_tolerance": tolerance,
            "loops": loop_diagnostics,
        },
    }


def triangulate_simple_boundary_loop(
    points: np.ndarray,
    loop: np.ndarray,
    *,
    source_faces: np.ndarray | None = None,
    geometric_tolerance: float | None = None,
) -> dict[str, Any]:
    """Constrained ear-clipping fill that reuses every source boundary edge.

    ``loop`` must follow the incident source-face direction.  Generated cap faces
    use the opposite traversal, so every repaired boundary edge has two incident
    faces with opposite directed edge occurrences.
    """
    points = np.asarray(points, dtype=np.float64)
    loop = np.asarray(loop, dtype=np.int64).reshape(-1)
    if loop.size < 3 or np.unique(loop).size != loop.size:
        return _hole_fill_failure("hole_fill_loop_not_simple", "a fill loop needs at least three distinct vertices")
    if np.any(loop < 0) or np.any(loop >= points.shape[0]):
        return _hole_fill_failure("hole_fill_loop_vertex_out_of_range", "fill loop vertex ids are out of range")
    orientation_diagnostics = None
    if source_faces is not None:
        validated = _validate_explicit_boundary_loop(
            points,
            np.asarray(source_faces, dtype=np.int64),
            loop,
            geometric_tolerance,
        )
        if not validated["success"]:
            return _hole_fill_failure(
                validated["failure_reason_codes"][0],
                "fill loop is not a valid oriented boundary of the current source mesh",
                boundary_validation=validated["diagnostics"],
            )
        loop = validated["loop"]
        orientation_diagnostics = validated["diagnostics"]
    tolerance = _geometric_tolerance(points[loop], geometric_tolerance)
    simplicity = _simple_loop_diagnostics(points, loop, tolerance)
    if not simplicity["passed"]:
        return _hole_fill_failure(
            "hole_fill_loop_self_intersection",
            "fill loop is geometrically self-intersecting",
            simplicity=simplicity,
        )

    cap_loop = loop[::-1].copy()
    loop_points = points[cap_loop]
    centered = loop_points - loop_points.mean(axis=0)
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ vh[:2].T
    signed_area = polygon_signed_area(projected)
    scale = max(float(np.linalg.norm(np.ptp(projected, axis=0))), tolerance)
    area_tolerance = max(scale * scale * 1e-14, tolerance * tolerance)
    if abs(signed_area) <= area_tolerance:
        return _hole_fill_failure("hole_fill_loop_zero_area", "fill loop has zero projected area")
    orientation = 1.0 if signed_area > 0.0 else -1.0
    remaining = list(range(cap_loop.size))
    triangles: list[list[int]] = []
    guard = 0
    while len(remaining) > 3:
        ear_found = False
        for position, current in enumerate(remaining):
            previous = remaining[position - 1]
            following = remaining[(position + 1) % len(remaining)]
            cross = cross_2d(projected[current] - projected[previous], projected[following] - projected[current])
            if orientation * cross <= area_tolerance:
                continue
            if any(
                point_strictly_inside_triangle(
                    projected[candidate],
                    projected[previous],
                    projected[current],
                    projected[following],
                    orientation,
                    area_tolerance,
                )
                for candidate in remaining
                if candidate not in {previous, current, following}
            ):
                continue
            triangles.append([
                int(cap_loop[previous]),
                int(cap_loop[current]),
                int(cap_loop[following]),
            ])
            del remaining[position]
            ear_found = True
            break
        guard += 1
        if not ear_found or guard > cap_loop.size * cap_loop.size:
            return _hole_fill_failure(
                "hole_fill_constrained_triangulation_failed",
                "ear clipping could not find a valid constrained triangle",
                remaining_vertices=len(remaining),
            )
    triangles.append([int(cap_loop[index]) for index in remaining])
    faces = np.asarray(triangles, dtype=np.int64)
    raw_normals = np.cross(points[faces[:, 1]] - points[faces[:, 0]], points[faces[:, 2]] - points[faces[:, 0]])
    areas = np.linalg.norm(raw_normals, axis=1) * 0.5
    if np.any(areas <= area_tolerance):
        return _hole_fill_failure(
            "hole_fill_degenerate_triangle",
            "constrained triangulation generated a degenerate face",
            degenerate_face_ids=np.flatnonzero(areas <= area_tolerance).astype(int).tolist(),
        )
    return {
        "success": True,
        "failure_reason_codes": [],
        "faces": faces,
        "diagnostics": {
            "stage": "constrained_boundary_loop_triangulation",
            "method": "best_fit_plane_deterministic_ear_clipping",
            "boundary_vertices": int(loop.size),
            "generated_faces": int(faces.shape[0]),
            "projected_signed_area": float(signed_area),
            "planarity_ratio": float(singular[-1] / max(singular[0], 1e-30)),
            "minimum_triangle_area": float(areas.min()),
            "area_tolerance": float(area_tolerance),
            "source_boundary_traversal_reversed": True,
            "source_boundary_validation": orientation_diagnostics,
        },
    }


def polygon_signed_area(points_2d: np.ndarray) -> float:
    return float(
        0.5
        * np.sum(
            points_2d[:, 0] * np.roll(points_2d[:, 1], -1)
            - np.roll(points_2d[:, 0], -1) * points_2d[:, 1]
        )
    )


def cross_2d(left: np.ndarray, right: np.ndarray) -> float:
    return float(left[0] * right[1] - left[1] * right[0])


def point_strictly_inside_triangle(
    point: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    orientation: float,
    tolerance: float,
) -> bool:
    values = (
        orientation * cross_2d(b - a, point - a),
        orientation * cross_2d(c - b, point - b),
        orientation * cross_2d(a - c, point - c),
    )
    return all(value > tolerance for value in values)


def _hole_fill_failure(code: str, message: str, **diagnostics: Any) -> dict[str, Any]:
    return {
        "success": False,
        "failure_reason_codes": [code],
        "faces": np.empty((0, 3), dtype=np.int64),
        "diagnostics": {
            "stage": "constrained_boundary_loop_triangulation",
            "message": message,
            **diagnostics,
        },
    }


def plan_loop_arc_length_correspondence(
    source_points: np.ndarray,
    source_faces: np.ndarray,
    source_loop: np.ndarray,
    proxy_points: np.ndarray,
    proxy_faces: np.ndarray,
    proxy_loop: np.ndarray,
    *,
    parameter_tolerance: float = 1e-10,
    max_ring_vertices: int = 8192,
    phase_samples: int = 128,
    max_correspondence_distance: float | None = None,
    normal_score_weight: float = 0.25,
    allow_proxy_face_flip: bool = True,
) -> dict[str, Any]:
    """Plan deterministic orientation, phase, and paired arc-length samples.

    The common ring parameters are the union of both loops' normalized arc-length
    breakpoints.  Consequently every original loop vertex is retained and only
    the opposite loop edges are split.  If using the proxy loop's induced
    direction gives the best pairing, the plan explicitly requests a proxy face
    flip so the eventual annular band remains consistently oriented.
    """

    source_loop_result = _validate_explicit_boundary_loop(source_points, source_faces, source_loop)
    if not source_loop_result["success"]:
        return source_loop_result
    proxy_loop_result = _validate_explicit_boundary_loop(proxy_points, proxy_faces, proxy_loop)
    if not proxy_loop_result["success"]:
        result = dict(proxy_loop_result)
        result["diagnostics"] = {**result["diagnostics"], "mesh_role": "proxy"}
        return result
    source_loop = source_loop_result["loop"]
    proxy_loop = proxy_loop_result["loop"]
    source_points = np.asarray(source_points, dtype=np.float64)
    proxy_points = np.asarray(proxy_points, dtype=np.float64)
    source_faces = np.asarray(source_faces, dtype=np.int64)
    proxy_faces = np.asarray(proxy_faces, dtype=np.int64)
    source_parameters, source_lengths = _loop_arc_lengths(source_points, source_loop)
    source_normals = _loop_vertex_normals(source_points, source_faces, source_loop)
    proxy_normals_induced = _loop_vertex_normals(proxy_points, proxy_faces, proxy_loop)
    scale = max(
        float(source_lengths.sum()),
        float(_loop_arc_lengths(proxy_points, proxy_loop)[1].sum()),
        1e-12,
    )
    candidates = []
    for reversed_from_induced in (True, False):
        candidate_loop = _reverse_cycle(proxy_loop) if reversed_from_induced else proxy_loop.copy()
        candidate_normals = (
            _reverse_cycle(proxy_normals_induced)
            if reversed_from_induced
            else proxy_normals_induced.copy()
        )
        proxy_faces_flipped = not reversed_from_induced
        if proxy_faces_flipped and not allow_proxy_face_flip:
            continue
        if proxy_faces_flipped:
            candidate_normals = -candidate_normals
        candidate = _best_loop_phase(
            source_points,
            source_loop,
            source_normals,
            proxy_points,
            candidate_loop,
            candidate_normals,
            scale,
            phase_samples,
            normal_score_weight,
        )
        candidate.update(
            {
                "proxy_loop": candidate_loop,
                "proxy_faces_flipped": proxy_faces_flipped,
                "proxy_traversal_reversed_from_induced": reversed_from_induced,
            }
        )
        candidates.append(candidate)
    candidates.sort(
        key=lambda row: (
            row["score"],
            bool(row["proxy_faces_flipped"]),
            row["phase"],
        )
    )
    selected = candidates[0]
    if max_correspondence_distance is not None and selected["distance_max"] > max_correspondence_distance:
        return _stitch_failure(
            "loop_correspondence_distance_exceeded",
            "arc_length_correspondence",
            "The best source-to-proxy loop pairing exceeds the allowed distance.",
            distance_max=selected["distance_max"],
            threshold=float(max_correspondence_distance),
        )

    proxy_loop_selected = selected["proxy_loop"]
    proxy_parameters, _ = _loop_arc_lengths(proxy_points, proxy_loop_selected)
    phase = float(selected["phase"])
    proxy_breakpoints_in_source_frame = np.mod(proxy_parameters - phase, 1.0)
    common_parameters = _union_circular_parameters(
        source_parameters,
        proxy_breakpoints_in_source_frame,
        max(float(parameter_tolerance), 1e-14),
    )
    if common_parameters.size > max_ring_vertices:
        return _stitch_failure(
            "resampled_ring_vertex_limit_exceeded",
            "arc_length_correspondence",
            "The paired ring would exceed the configured vertex limit.",
            target_ring_vertices=int(common_parameters.size),
            max_ring_vertices=int(max_ring_vertices),
        )
    if common_parameters.size < 3:
        return _stitch_failure(
            "resampled_ring_too_small",
            "arc_length_correspondence",
            "A conformal annulus requires at least three paired ring vertices.",
            target_ring_vertices=int(common_parameters.size),
        )
    spacing = np.diff(np.r_[common_parameters, common_parameters[0] + 1.0])
    if np.any(spacing <= max(float(parameter_tolerance), 1e-14) * 0.1):
        return _stitch_failure(
            "resampling_parameter_spacing_too_small",
            "arc_length_correspondence",
            "Arc-length breakpoints are too close to form stable triangles.",
            minimum_parameter_spacing=float(spacing.min()),
        )
    return {
        "success": True,
        "failure_reason_codes": [],
        "source_loop": source_loop,
        "proxy_loop": proxy_loop_selected,
        "common_parameters": common_parameters,
        "source_query_parameters": common_parameters,
        "proxy_query_parameters": np.mod(common_parameters + phase, 1.0),
        "proxy_faces_flipped": bool(selected["proxy_faces_flipped"]),
        "diagnostics": {
            "stage": "arc_length_correspondence",
            "method": "normalized_arc_length_breakpoint_union",
            "orientation_pairing": {
                "proxy_faces_flipped": bool(selected["proxy_faces_flipped"]),
                "proxy_traversal_reversed_from_induced": bool(selected["proxy_traversal_reversed_from_induced"]),
                "phase": phase,
                "score": float(selected["score"]),
                "distance_rms": float(selected["distance_rms"]),
                "distance_max": float(selected["distance_max"]),
                "normal_dot_mean": float(selected["normal_dot_mean"]),
                "alternative_score": (
                    float(candidates[1]["score"])
                    if len(candidates) > 1
                    else None
                ),
                "proxy_face_flip_allowed": bool(allow_proxy_face_flip),
            },
            "resampling": {
                "source_original_vertices": int(source_loop.size),
                "proxy_original_vertices": int(proxy_loop.size),
                "target_ring_vertices": int(common_parameters.size),
                "parameter_tolerance": float(parameter_tolerance),
                "minimum_parameter_spacing": float(spacing.min()),
            },
        },
    }


def conformal_same_mesh_loop_stitch(
    points: np.ndarray,
    faces: np.ndarray,
    *,
    source_loop: np.ndarray,
    target_loop: np.ndarray,
    geometric_tolerance: float | None = None,
    parameter_tolerance: float = 1e-10,
    max_ring_vertices: int = 8192,
    phase_samples: int = 128,
    max_correspondence_distance: float | None = None,
    min_triangle_area: float | None = None,
    min_adjacent_normal_dot: float = -0.5,
    normal_score_weight: float = 0.25,
) -> dict[str, Any]:
    """Conformally zipper two disjoint boundary loops on one source mesh.

    Unlike :func:`conformal_loop_stitch`, this primitive never duplicates or
    flips a mesh component.  Both boundary loops are validated against the same
    topology, both rings are split at the union of their normalized arc-length
    breakpoints, and every retained/split source face carries its original face
    parent.  The result is transactional and may only be consumed on success.
    """

    input_failure = _mesh_input_diagnostics(points, faces)
    if input_failure:
        return _stitch_failure(input_failure[0], "source_mesh_validation", input_failure[1])
    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    tolerance = _geometric_tolerance(points, geometric_tolerance)
    source_selection = _select_stitch_loop(points, faces, source_loop, "source", tolerance)
    if not source_selection["success"]:
        return source_selection
    target_selection = _select_stitch_loop(points, faces, target_loop, "target", tolerance)
    if not target_selection["success"]:
        return target_selection
    source_loop_oriented = source_selection["loop"]
    target_loop_oriented = target_selection["loop"]
    shared_vertices = np.intersect1d(source_loop_oriented, target_loop_oriented)
    if shared_vertices.size:
        return _stitch_failure(
            "same_mesh_boundary_loops_overlap",
            "boundary_loop_pair_validation",
            "Two same-mesh stitch loops must be vertex-disjoint boundary cycles.",
            shared_vertex_ids=shared_vertices[:100].astype(int).tolist(),
            shared_vertex_count=int(shared_vertices.size),
        )

    plan = plan_loop_arc_length_correspondence(
        points,
        faces,
        source_loop_oriented,
        points,
        faces,
        target_loop_oriented,
        parameter_tolerance=parameter_tolerance,
        max_ring_vertices=max_ring_vertices,
        phase_samples=phase_samples,
        max_correspondence_distance=max_correspondence_distance,
        normal_score_weight=normal_score_weight,
        allow_proxy_face_flip=False,
    )
    if not plan["success"]:
        return plan
    source_split = _split_boundary_loop_at_parameters(
        points,
        faces,
        plan["source_loop"],
        plan["source_query_parameters"],
        parameter_tolerance,
    )
    if not source_split["success"]:
        return source_split
    target_split = _split_boundary_loop_at_parameters(
        source_split["points"],
        source_split["faces"],
        plan["proxy_loop"],
        plan["proxy_query_parameters"],
        parameter_tolerance,
    )
    if not target_split["success"]:
        result = dict(target_split)
        result["diagnostics"] = {**result["diagnostics"], "mesh_role": "target"}
        return result

    source_ring = source_split["ring_vertex_ids"]
    target_ring = target_split["ring_vertex_ids"]
    stitch_faces = _annular_bridge_faces(source_ring, target_ring)
    source_face_count = int(target_split["faces"].shape[0])
    output_faces = np.vstack([target_split["faces"], stitch_faces])
    stitch_face_ids = np.arange(source_face_count, output_faces.shape[0], dtype=np.int64)
    source_parent = source_split["face_parent"][target_split["face_parent"]]
    source_face_parent = np.concatenate(
        [source_parent, np.full(stitch_faces.shape[0], -1, dtype=np.int64)]
    )
    face_origin = np.concatenate(
        [
            np.full(source_face_count, FACE_ORIGIN["source"], dtype=np.int16),
            np.full(stitch_faces.shape[0], FACE_ORIGIN["stitch_band"], dtype=np.int16),
        ]
    )
    first_generated_parent = np.isin(
        target_split["face_parent"],
        source_split["generated_face_ids"],
    )
    second_generated = np.zeros(source_face_count, dtype=bool)
    second_generated[target_split["generated_face_ids"]] = True
    generated_source_face_ids = np.flatnonzero(first_generated_parent | second_generated)
    generated_face_ids = np.concatenate([generated_source_face_ids, stitch_face_ids])
    scale = max(
        float(np.linalg.norm(target_split["points"].max(axis=0) - target_split["points"].min(axis=0))),
        1e-12,
    )
    area_tolerance = (
        float(min_triangle_area)
        if min_triangle_area is not None
        else max(scale * scale * 1e-16, 1e-24)
    )
    validation = _validate_conformal_stitch(
        target_split["points"],
        output_faces,
        face_origin,
        source_ring,
        target_ring,
        stitch_face_ids,
        generated_face_ids,
        area_tolerance,
        min_adjacent_normal_dot,
        target_seam_origin=FACE_ORIGIN["source"],
    )
    if not validation["passed"]:
        return _stitch_failure(
            validation["failure_reason_codes"][0],
            "stitch_validation",
            "The same-mesh annular bridge failed conformal seam validation.",
            validation=validation,
            correspondence=plan["diagnostics"],
            source_split=source_split["diagnostics"],
            target_split=target_split["diagnostics"],
        )
    return {
        "success": True,
        "failure_reason_codes": [],
        "points": target_split["points"],
        "faces": output_faces,
        "face_origin": face_origin,
        "source_face_parent": source_face_parent,
        "proxy_face_parent": np.full(output_faces.shape[0], -1, dtype=np.int64),
        "source_original_point_count": int(points.shape[0]),
        "source_ring_vertex_ids": source_ring,
        "target_ring_vertex_ids": target_ring,
        "stitch_face_ids": stitch_face_ids,
        "diagnostics": {
            "stage": "conformal_same_mesh_loop_stitch",
            "method": "same_mesh_paired_arc_length_edge_split_annular_bridge",
            "source_loop_extraction": source_selection["diagnostics"],
            "target_loop_extraction": target_selection["diagnostics"],
            "correspondence": plan["diagnostics"],
            "source_split": source_split["diagnostics"],
            "target_split": target_split["diagnostics"],
            "validation": validation,
            "output": {
                "points": int(target_split["points"].shape[0]),
                "triangles": int(output_faces.shape[0]),
                "source_faces": source_face_count,
                "stitch_faces": int(stitch_faces.shape[0]),
            },
        },
    }


def conformal_loop_stitch(
    source_points: np.ndarray,
    source_faces: np.ndarray,
    proxy_points: np.ndarray,
    proxy_faces: np.ndarray,
    *,
    source_loop: np.ndarray | None = None,
    proxy_loop: np.ndarray | None = None,
    geometric_tolerance: float | None = None,
    parameter_tolerance: float = 1e-10,
    max_ring_vertices: int = 8192,
    phase_samples: int = 128,
    max_correspondence_distance: float | None = None,
    min_triangle_area: float | None = None,
    min_adjacent_normal_dot: float = -0.5,
    normal_score_weight: float = 0.25,
) -> dict[str, Any]:
    """Split two boundary rings and build a validated conformal annular bridge.

    This function is transactional: callers must only consume ``points`` and
    ``faces`` when ``success`` is true.  It never falls back to concatenating two
    disconnected meshes.
    """

    source_input_failure = _mesh_input_diagnostics(source_points, source_faces)
    if source_input_failure:
        return _stitch_failure(source_input_failure[0], "source_mesh_validation", source_input_failure[1])
    proxy_input_failure = _mesh_input_diagnostics(proxy_points, proxy_faces)
    if proxy_input_failure:
        return _stitch_failure(proxy_input_failure[0], "proxy_mesh_validation", proxy_input_failure[1])
    source_points = np.asarray(source_points, dtype=np.float64)
    source_faces = np.asarray(source_faces, dtype=np.int64)
    proxy_points = np.asarray(proxy_points, dtype=np.float64)
    proxy_faces = np.asarray(proxy_faces, dtype=np.int64)
    tolerance = max(
        _geometric_tolerance(source_points, geometric_tolerance),
        _geometric_tolerance(proxy_points, geometric_tolerance),
    )

    source_selection = _select_stitch_loop(source_points, source_faces, source_loop, "source", tolerance)
    if not source_selection["success"]:
        return source_selection
    proxy_selection = _select_stitch_loop(proxy_points, proxy_faces, proxy_loop, "proxy", tolerance)
    if not proxy_selection["success"]:
        return proxy_selection
    source_loop_oriented = source_selection["loop"]
    proxy_loop_oriented = proxy_selection["loop"]

    plan = plan_loop_arc_length_correspondence(
        source_points,
        source_faces,
        source_loop_oriented,
        proxy_points,
        proxy_faces,
        proxy_loop_oriented,
        parameter_tolerance=parameter_tolerance,
        max_ring_vertices=max_ring_vertices,
        phase_samples=phase_samples,
        max_correspondence_distance=max_correspondence_distance,
        normal_score_weight=normal_score_weight,
    )
    if not plan["success"]:
        return plan
    oriented_proxy_faces = proxy_faces[:, [0, 2, 1]] if plan["proxy_faces_flipped"] else proxy_faces.copy()
    source_split = _split_boundary_loop_at_parameters(
        source_points,
        source_faces,
        plan["source_loop"],
        plan["source_query_parameters"],
        parameter_tolerance,
    )
    if not source_split["success"]:
        return source_split
    proxy_split = _split_boundary_loop_at_parameters(
        proxy_points,
        oriented_proxy_faces,
        plan["proxy_loop"],
        plan["proxy_query_parameters"],
        parameter_tolerance,
    )
    if not proxy_split["success"]:
        result = dict(proxy_split)
        result["diagnostics"] = {**result["diagnostics"], "mesh_role": "proxy"}
        return result

    point_offset = source_split["points"].shape[0]
    points = np.vstack([source_split["points"], proxy_split["points"]])
    source_face_count = source_split["faces"].shape[0]
    proxy_face_count = proxy_split["faces"].shape[0]
    proxy_faces_merged = proxy_split["faces"] + point_offset
    source_ring = source_split["ring_vertex_ids"]
    proxy_ring = proxy_split["ring_vertex_ids"] + point_offset
    stitch_faces = _annular_bridge_faces(source_ring, proxy_ring)
    faces = np.vstack([source_split["faces"], proxy_faces_merged, stitch_faces])
    stitch_face_ids = np.arange(
        source_face_count + proxy_face_count,
        faces.shape[0],
        dtype=np.int64,
    )
    face_origin = np.concatenate(
        [
            np.full(source_face_count, FACE_ORIGIN["source"], dtype=np.int16),
            np.full(proxy_face_count, FACE_ORIGIN["proxy_patch"], dtype=np.int16),
            np.full(stitch_faces.shape[0], FACE_ORIGIN["stitch_band"], dtype=np.int16),
        ]
    )
    source_parent = np.concatenate(
        [
            source_split["face_parent"],
            np.full(proxy_face_count + stitch_faces.shape[0], -1, dtype=np.int64),
        ]
    )
    proxy_parent = np.concatenate(
        [
            np.full(source_face_count, -1, dtype=np.int64),
            proxy_split["face_parent"],
            np.full(stitch_faces.shape[0], -1, dtype=np.int64),
        ]
    )
    scale = max(float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))), 1e-12)
    area_tolerance = (
        float(min_triangle_area)
        if min_triangle_area is not None
        else max(scale * scale * 1e-16, 1e-24)
    )
    generated_face_ids = np.unique(
        np.concatenate(
            [
                source_split["generated_face_ids"],
                proxy_split["generated_face_ids"] + source_face_count,
                stitch_face_ids,
            ]
        )
    )
    validation = _validate_conformal_stitch(
        points,
        faces,
        face_origin,
        source_ring,
        proxy_ring,
        stitch_face_ids,
        generated_face_ids,
        area_tolerance,
        min_adjacent_normal_dot,
    )
    if not validation["passed"]:
        return _stitch_failure(
            validation["failure_reason_codes"][0],
            "stitch_validation",
            "The generated annular bridge failed conformal seam validation.",
            validation=validation,
            correspondence=plan["diagnostics"],
            source_split=source_split["diagnostics"],
            proxy_split=proxy_split["diagnostics"],
        )
    return {
        "success": True,
        "failure_reason_codes": [],
        "points": points,
        "faces": faces,
        "face_origin": face_origin,
        "source_face_parent": source_parent,
        "proxy_face_parent": proxy_parent,
        "source_point_count": int(point_offset),
        "proxy_point_offset": int(point_offset),
        "source_original_point_count": int(source_points.shape[0]),
        "proxy_original_point_count": int(proxy_points.shape[0]),
        "source_ring_vertex_ids": source_ring,
        "proxy_ring_vertex_ids": proxy_ring,
        "stitch_face_ids": stitch_face_ids,
        "diagnostics": {
            "stage": "conformal_loop_stitch",
            "method": "paired_arc_length_edge_split_annular_bridge",
            "source_loop_extraction": source_selection["diagnostics"],
            "proxy_loop_extraction": proxy_selection["diagnostics"],
            "correspondence": plan["diagnostics"],
            "source_split": source_split["diagnostics"],
            "proxy_split": proxy_split["diagnostics"],
            "validation": validation,
            "output": {
                "points": int(points.shape[0]),
                "triangles": int(faces.shape[0]),
                "source_faces": int(source_face_count),
                "proxy_faces": int(proxy_face_count),
                "stitch_faces": int(stitch_faces.shape[0]),
            },
        },
    }


def _select_stitch_loop(
    points: np.ndarray,
    faces: np.ndarray,
    explicit_loop: np.ndarray | None,
    role: str,
    tolerance: float,
) -> dict[str, Any]:
    if explicit_loop is not None:
        result = _validate_explicit_boundary_loop(points, faces, explicit_loop, tolerance)
        if not result["success"]:
            result["diagnostics"] = {**result["diagnostics"], "mesh_role": role}
            return result
        return {
            "success": True,
            "failure_reason_codes": [],
            "loop": result["loop"],
            "diagnostics": {
                "stage": "boundary_loop_selection",
                "mesh_role": role,
                "method": "explicit_boundary_loop",
                "loop_count": 1,
                "vertex_count": int(result["loop"].size),
            },
        }
    extraction = extract_ordered_boundary_loops(points, faces, geometric_tolerance=tolerance)
    if not extraction["success"]:
        diagnostics = {**extraction["diagnostics"], "mesh_role": role}
        stage = str(diagnostics.pop("stage", "boundary_loop_selection"))
        message = str(diagnostics.pop("message", "Boundary loop extraction failed."))
        return _stitch_failure(
            extraction["failure_reason_codes"][0],
            stage,
            message,
            **diagnostics,
        )
    if len(extraction["loops"]) != 1:
        return _stitch_failure(
            f"{role}_boundary_loop_count_mismatch",
            "boundary_loop_selection",
            "Automatic stitching requires exactly one boundary loop per input mesh.",
            mesh_role=role,
            loop_count=len(extraction["loops"]),
            boundary_edge_count=extraction["diagnostics"]["boundary_edge_count"],
        )
    return {
        "success": True,
        "failure_reason_codes": [],
        "loop": extraction["loops"][0],
        "diagnostics": {
            **extraction["diagnostics"],
            "mesh_role": role,
            "method": "single_extracted_boundary_loop",
        },
    }


def _validate_explicit_boundary_loop(
    points: np.ndarray,
    faces: np.ndarray,
    loop: np.ndarray,
    geometric_tolerance: float | None = None,
) -> dict[str, Any]:
    mesh_failure = _mesh_input_diagnostics(points, faces)
    if mesh_failure:
        return _stitch_failure(mesh_failure[0], "boundary_loop_validation", mesh_failure[1])
    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    raw = np.asarray(loop, dtype=np.int64).reshape(-1)
    if raw.size > 1 and raw[0] == raw[-1]:
        raw = raw[:-1]
    if raw.size < 3:
        return _stitch_failure(
            "boundary_loop_too_small",
            "boundary_loop_validation",
            "A boundary loop requires at least three distinct vertices.",
            vertex_count=int(raw.size),
        )
    if np.any(raw < 0) or np.any(raw >= points.shape[0]):
        return _stitch_failure(
            "boundary_loop_vertex_out_of_range",
            "boundary_loop_validation",
            "Boundary loop vertex IDs must reference the input point array.",
        )
    if np.unique(raw).size != raw.size:
        return _stitch_failure(
            "boundary_loop_repeats_vertex",
            "boundary_loop_validation",
            "A simple boundary loop cannot repeat vertices.",
        )
    tolerance = _geometric_tolerance(points, geometric_tolerance)
    _, edge_faces = edge_topology(faces)
    directions = []
    for index, left in enumerate(raw):
        right = int(raw[(index + 1) % raw.size])
        edge = tuple(sorted((int(left), right)))
        incident = edge_faces.get(edge, [])
        if len(incident) != 1:
            return _stitch_failure(
                "loop_edge_is_not_boundary",
                "boundary_loop_validation",
                "Every selected loop edge must have exactly one incident triangle.",
                edge=list(edge),
                incidence=len(incident),
            )
        if _triangle_double_area(points, faces[incident[0]]) <= tolerance * tolerance:
            return _stitch_failure(
                "boundary_adjacent_face_degenerate",
                "boundary_loop_validation",
                "A boundary loop cannot be oriented from a degenerate incident triangle.",
                edge=list(edge),
                face_id=int(incident[0]),
            )
        directed = _face_directed_edge(faces[incident[0]], edge)
        directions.append(1 if directed == (int(left), right) else -1)
    if all(direction == -1 for direction in directions):
        raw = _reverse_cycle(raw)
    elif not all(direction == 1 for direction in directions):
        return _stitch_failure(
            "boundary_orientation_inconsistent",
            "boundary_loop_validation",
            "The selected loop mixes face-induced edge directions.",
        )
    raw = _rotate_cycle_to_smallest(raw)
    simplicity = _simple_loop_diagnostics(points, raw, tolerance)
    if not simplicity["passed"]:
        return _stitch_failure(
            simplicity["failure_reason"],
            "boundary_loop_validation",
            simplicity["message"],
            **simplicity.get("details", {}),
        )
    return {
        "success": True,
        "failure_reason_codes": [],
        "loop": raw,
        "diagnostics": {
            "stage": "boundary_loop_validation",
            "vertex_count": int(raw.size),
            "simple": True,
            "orientation": "face_winding_induced",
        },
    }


def _split_boundary_loop_at_parameters(
    points: np.ndarray,
    faces: np.ndarray,
    loop: np.ndarray,
    query_parameters: np.ndarray,
    parameter_tolerance: float,
) -> dict[str, Any]:
    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    loop = np.asarray(loop, dtype=np.int64)
    query = np.mod(np.asarray(query_parameters, dtype=np.float64), 1.0)
    loop_parameters, edge_lengths = _loop_arc_lengths(points, loop)
    if edge_lengths.sum() <= 0.0:
        return _stitch_failure(
            "boundary_loop_zero_length",
            "boundary_edge_split",
            "Cannot split a boundary loop with zero total length.",
        )
    point_rows = points.tolist()
    edge_insertions: dict[tuple[int, int], dict[str, Any]] = {}
    ring_ids = []
    parameter_tolerance = max(float(parameter_tolerance), 1e-14)
    for value in query:
        location = _loop_parameter_location(loop_parameters, value, parameter_tolerance)
        if location["vertex_index"] is not None:
            ring_ids.append(int(loop[location["vertex_index"]]))
            continue
        edge_index = int(location["edge_index"])
        fraction = float(location["fraction"])
        left = int(loop[edge_index])
        right = int(loop[(edge_index + 1) % loop.size])
        key = tuple(sorted((left, right)))
        record = edge_insertions.setdefault(
            key,
            {"left": left, "right": right, "samples": []},
        )
        existing = next(
            (
                row
                for row in record["samples"]
                if abs(float(row["fraction"]) - fraction) <= parameter_tolerance
            ),
            None,
        )
        if existing is None:
            point = points[left] * (1.0 - fraction) + points[right] * fraction
            vertex_id = len(point_rows)
            point_rows.append(point.tolist())
            existing = {"fraction": fraction, "vertex_id": vertex_id, "parameter": float(value)}
            record["samples"].append(existing)
        ring_ids.append(int(existing["vertex_id"]))

    ring_ids_array = np.asarray(ring_ids, dtype=np.int64)
    if np.unique(ring_ids_array).size != ring_ids_array.size:
        return _stitch_failure(
            "resampled_ring_repeats_vertex",
            "boundary_edge_split",
            "The resampling plan mapped multiple ring positions to one vertex.",
        )
    _, edge_faces = edge_topology(faces)
    for edge, record in edge_insertions.items():
        if len(edge_faces.get(edge, [])) != 1:
            return _stitch_failure(
                "split_edge_is_not_boundary",
                "boundary_edge_split",
                "Only a single-incidence boundary edge can be split by this primitive.",
                edge=list(edge),
                incidence=len(edge_faces.get(edge, [])),
            )

    new_faces: list[list[int]] = []
    face_parent: list[int] = []
    generated_face_ids: list[int] = []
    split_face_count = 0
    for face_id, face in enumerate(faces):
        a, b, c = (int(value) for value in face)
        directed_edges = ((a, b), (b, c), (c, a))
        has_split = any(tuple(sorted(edge)) in edge_insertions for edge in directed_edges)
        if not has_split:
            new_faces.append([a, b, c])
            face_parent.append(face_id)
            continue
        split_face_count += 1
        cycle: list[int] = []
        for left, right in directed_edges:
            cycle.append(left)
            record = edge_insertions.get(tuple(sorted((left, right))))
            if record is None:
                continue
            samples = sorted(record["samples"], key=lambda row: float(row["fraction"]))
            if (left, right) != (record["left"], record["right"]):
                samples = list(reversed(samples))
            cycle.extend(int(row["vertex_id"]) for row in samples)
        center_id = len(point_rows)
        point_rows.append(points[face].mean(axis=0).tolist())
        for index, left in enumerate(cycle):
            new_faces.append([left, cycle[(index + 1) % len(cycle)], center_id])
            face_parent.append(face_id)
            generated_face_ids.append(len(new_faces) - 1)

    split_plan = []
    for edge in sorted(edge_insertions):
        record = edge_insertions[edge]
        samples = sorted(record["samples"], key=lambda row: float(row["fraction"]))
        split_plan.append(
            {
                "edge": [int(record["left"]), int(record["right"])],
                "fractions": [float(row["fraction"]) for row in samples],
                "inserted_vertex_ids": [int(row["vertex_id"]) for row in samples],
            }
        )
    return {
        "success": True,
        "failure_reason_codes": [],
        "points": np.asarray(point_rows, dtype=np.float64),
        "faces": np.asarray(new_faces, dtype=np.int64),
        "face_parent": np.asarray(face_parent, dtype=np.int64),
        "ring_vertex_ids": ring_ids_array,
        "generated_face_ids": np.asarray(generated_face_ids, dtype=np.int64),
        "diagnostics": {
            "stage": "boundary_edge_split",
            "method": "boundary_edge_insertion_with_incident_triangle_fan",
            "original_ring_vertices": int(loop.size),
            "resampled_ring_vertices": int(query.size),
            "inserted_boundary_vertices": sum(len(row["samples"]) for row in edge_insertions.values()),
            "split_boundary_edges": len(edge_insertions),
            "split_incident_faces": split_face_count,
            "split_plan": split_plan,
        },
    }


def _annular_bridge_faces(source_ring: np.ndarray, proxy_ring: np.ndarray) -> np.ndarray:
    if source_ring.size != proxy_ring.size:
        raise ValueError("Paired conformal rings must have equal vertex counts.")
    faces = []
    for index, source_left in enumerate(source_ring):
        next_index = (index + 1) % source_ring.size
        source_right = int(source_ring[next_index])
        proxy_left = int(proxy_ring[index])
        proxy_right = int(proxy_ring[next_index])
        # This winding opposes the source boundary direction and follows the
        # selected proxy traversal, which itself opposes the effective proxy
        # face boundary direction.
        faces.append([int(source_left), proxy_left, source_right])
        faces.append([source_right, proxy_left, proxy_right])
    return np.asarray(faces, dtype=np.int64)


def _validate_conformal_stitch(
    points: np.ndarray,
    faces: np.ndarray,
    face_origin: np.ndarray,
    source_ring: np.ndarray,
    proxy_ring: np.ndarray,
    stitch_face_ids: np.ndarray,
    generated_face_ids: np.ndarray,
    area_tolerance: float,
    min_adjacent_normal_dot: float,
    *,
    target_seam_origin: int = FACE_ORIGIN["proxy_patch"],
) -> dict[str, Any]:
    occurrences = _directed_edge_occurrences(faces)
    stitch_edges = {
        tuple(sorted((int(left), int(right))))
        for face_id in stitch_face_ids
        for left, right in _face_directed_edges(faces[face_id])
    }
    source_seam_edges = {
        tuple(sorted((int(source_ring[index]), int(source_ring[(index + 1) % source_ring.size]))))
        for index in range(source_ring.size)
    }
    proxy_seam_edges = {
        tuple(sorted((int(proxy_ring[index]), int(proxy_ring[(index + 1) % proxy_ring.size]))))
        for index in range(proxy_ring.size)
    }
    incidence_failures = []
    orientation_conflicts = []
    origin_conflicts = []
    for edge in sorted(stitch_edges):
        rows = occurrences.get(edge, [])
        if len(rows) != 2:
            incidence_failures.append({"edge": list(edge), "incidence": len(rows)})
            continue
        if rows[0][1:] != (rows[1][2], rows[1][1]):
            orientation_conflicts.append({"edge": list(edge), "faces": [rows[0][0], rows[1][0]]})
        origins = [int(face_origin[row[0]]) for row in rows]
        if edge in source_seam_edges and sorted(origins) != sorted([FACE_ORIGIN["source"], FACE_ORIGIN["stitch_band"]]):
            origin_conflicts.append({"edge": list(edge), "origins": origins, "expected": [FACE_ORIGIN["source"], FACE_ORIGIN["stitch_band"]]})
        elif edge in proxy_seam_edges and sorted(origins) != sorted([target_seam_origin, FACE_ORIGIN["stitch_band"]]):
            origin_conflicts.append({"edge": list(edge), "origins": origins, "expected": [target_seam_origin, FACE_ORIGIN["stitch_band"]]})
        elif edge not in source_seam_edges and edge not in proxy_seam_edges and origins != [FACE_ORIGIN["stitch_band"], FACE_ORIGIN["stitch_band"]]:
            origin_conflicts.append({"edge": list(edge), "origins": origins, "expected": [FACE_ORIGIN["stitch_band"], FACE_ORIGIN["stitch_band"]]})

    generated_triangles = points[faces[generated_face_ids]]
    generated_areas = np.linalg.norm(
        np.cross(
            generated_triangles[:, 1] - generated_triangles[:, 0],
            generated_triangles[:, 2] - generated_triangles[:, 0],
        ),
        axis=1,
    ) * 0.5
    repeated_vertex = np.asarray(
        [len(set(int(value) for value in faces[face_id])) != 3 for face_id in generated_face_ids],
        dtype=bool,
    )
    degenerate_mask = (generated_areas <= area_tolerance) | repeated_vertex
    degenerate_faces = generated_face_ids[degenerate_mask].astype(int).tolist()

    face_normals = _unit_face_normals(points, faces)
    normal_dots = []
    sharp_pairs = []
    for edge in sorted(stitch_edges):
        rows = occurrences.get(edge, [])
        if len(rows) != 2:
            continue
        dot = float(np.dot(face_normals[rows[0][0]], face_normals[rows[1][0]]))
        normal_dots.append(dot)
        if dot < min_adjacent_normal_dot:
            sharp_pairs.append({"edge": list(edge), "faces": [rows[0][0], rows[1][0]], "normal_dot": dot})

    failures = []
    if incidence_failures:
        failures.append("stitch_edge_incidence_failed")
    if orientation_conflicts or origin_conflicts:
        failures.append("stitch_orientation_failed")
    if degenerate_faces:
        failures.append("stitch_degenerate_faces_detected")
    if sharp_pairs:
        failures.append("stitch_normal_transition_failed")
    return {
        "passed": not failures,
        "failure_reason_codes": failures,
        "seam_edge_count": len(source_seam_edges) + len(proxy_seam_edges),
        "stitch_edge_count": len(stitch_edges),
        "edge_incidence_two": not incidence_failures,
        "opposite_directed_edge_pairs": not orientation_conflicts,
        "face_origin_pairing_valid": not origin_conflicts,
        "incidence_failures": incidence_failures[:100],
        "orientation_conflicts": orientation_conflicts[:100],
        "origin_conflicts": origin_conflicts[:100],
        "generated_face_count": int(generated_face_ids.size),
        "degenerate_generated_faces": degenerate_faces[:100],
        "minimum_generated_area": float(generated_areas.min()) if generated_areas.size else None,
        "area_tolerance": float(area_tolerance),
        "adjacent_normal_dot_min": min(normal_dots) if normal_dots else None,
        "minimum_adjacent_normal_dot": float(min_adjacent_normal_dot),
        "sharp_or_inverted_pairs": sharp_pairs[:100],
    }


def _best_loop_phase(
    source_points: np.ndarray,
    source_loop: np.ndarray,
    source_normals: np.ndarray,
    proxy_points: np.ndarray,
    proxy_loop: np.ndarray,
    proxy_normals: np.ndarray,
    scale: float,
    phase_samples: int,
    normal_score_weight: float,
) -> dict[str, float]:
    evaluation_count = min(max(max(source_loop.size, proxy_loop.size) * 2, 32), 256)
    evaluation_parameters = np.linspace(0.0, 1.0, evaluation_count, endpoint=False)
    source_arc_parameters, _ = _loop_arc_lengths(source_points, source_loop)
    proxy_arc_parameters, _ = _loop_arc_lengths(proxy_points, proxy_loop)
    source_samples = _sample_loop_values(
        source_points[source_loop],
        evaluation_parameters,
        vertex_parameters=source_arc_parameters,
    )
    source_normal_samples = _sample_loop_values(
        source_normals,
        evaluation_parameters,
        vertex_parameters=source_arc_parameters,
        normalize=True,
    )
    phase_count = min(max(int(phase_samples), max(source_loop.size, proxy_loop.size), 16), 512)
    phases = np.linspace(0.0, 1.0, phase_count, endpoint=False)
    best: dict[str, float] | None = None
    for phase in phases:
        proxy_parameters = np.mod(evaluation_parameters + phase, 1.0)
        proxy_samples = _sample_loop_values(
            proxy_points[proxy_loop],
            proxy_parameters,
            vertex_parameters=proxy_arc_parameters,
        )
        proxy_normal_samples = _sample_loop_values(
            proxy_normals,
            proxy_parameters,
            vertex_parameters=proxy_arc_parameters,
            normalize=True,
        )
        distances = np.linalg.norm(source_samples - proxy_samples, axis=1)
        normal_dots = np.einsum("ij,ij->i", source_normal_samples, proxy_normal_samples)
        distance_rms = float(np.sqrt(np.mean(distances * distances)))
        normal_dot_mean = float(np.mean(normal_dots))
        score = distance_rms / scale + float(normal_score_weight) * (
            1.0 - np.clip(normal_dot_mean, -1.0, 1.0)
        )
        row = {
            "phase": float(phase),
            "score": float(score),
            "distance_rms": distance_rms,
            "distance_max": float(distances.max()),
            "normal_dot_mean": normal_dot_mean,
        }
        if best is None or (row["score"], row["phase"]) < (best["score"], best["phase"]):
            best = row
    assert best is not None
    return best


def _sample_loop_values(
    values: np.ndarray,
    parameters: np.ndarray,
    *,
    vertex_parameters: np.ndarray | None = None,
    normalize: bool = False,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    count = values.shape[0]
    vertex_parameters = (
        np.arange(count, dtype=np.float64) / count
        if vertex_parameters is None
        else np.asarray(vertex_parameters, dtype=np.float64)
    )
    query = np.mod(np.asarray(parameters, dtype=np.float64), 1.0)
    left = np.searchsorted(vertex_parameters, query, side="right") - 1
    left[left < 0] = count - 1
    right = (left + 1) % count
    left_parameters = vertex_parameters[left]
    right_parameters = vertex_parameters[right]
    wrapped = right == 0
    right_parameters = right_parameters + wrapped.astype(np.float64)
    adjusted_query = query + (wrapped & (query < left_parameters)).astype(np.float64)
    fraction = np.divide(
        adjusted_query - left_parameters,
        right_parameters - left_parameters,
        out=np.zeros_like(query),
        where=(right_parameters - left_parameters) > 1e-15,
    )
    result = values[left] * (1.0 - fraction[:, None]) + values[right] * fraction[:, None]
    if normalize:
        lengths = np.linalg.norm(result, axis=1)
        result = np.divide(result, lengths[:, None], out=np.zeros_like(result), where=lengths[:, None] > 1e-15)
    return result


def _loop_arc_lengths(points: np.ndarray, loop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    loop_points = np.asarray(points, dtype=np.float64)[np.asarray(loop, dtype=np.int64)]
    edge_lengths = np.linalg.norm(np.roll(loop_points, -1, axis=0) - loop_points, axis=1)
    total = float(edge_lengths.sum())
    if total <= 0.0:
        return np.zeros(loop_points.shape[0], dtype=np.float64), edge_lengths
    parameters = np.r_[0.0, np.cumsum(edge_lengths[:-1])] / total
    return parameters, edge_lengths


def _loop_parameter_location(parameters: np.ndarray, value: float, tolerance: float) -> dict[str, Any]:
    circular_distance = np.minimum(np.abs(parameters - value), 1.0 - np.abs(parameters - value))
    closest = int(np.argmin(circular_distance))
    if circular_distance[closest] <= tolerance:
        return {"vertex_index": closest, "edge_index": None, "fraction": 0.0}
    edge_index = int(np.searchsorted(parameters, value, side="right") - 1)
    if edge_index < 0:
        edge_index = parameters.size - 1
    left = float(parameters[edge_index])
    right = float(parameters[(edge_index + 1) % parameters.size])
    adjusted = float(value)
    if edge_index == parameters.size - 1:
        right += 1.0
        if adjusted < left:
            adjusted += 1.0
    fraction = (adjusted - left) / max(right - left, 1e-15)
    return {"vertex_index": None, "edge_index": edge_index, "fraction": fraction}


def _union_circular_parameters(left: np.ndarray, right: np.ndarray, tolerance: float) -> np.ndarray:
    values = np.mod(np.r_[left, right], 1.0)
    values[np.abs(values - 1.0) <= tolerance] = 0.0
    values.sort()
    merged = []
    for value in values:
        if not merged or value - merged[-1] > tolerance:
            merged.append(float(value))
    if len(merged) > 1 and (merged[0] + 1.0 - merged[-1]) <= tolerance:
        merged.pop()
    return np.asarray(merged, dtype=np.float64)


def _loop_vertex_normals(points: np.ndarray, faces: np.ndarray, loop: np.ndarray) -> np.ndarray:
    _, edge_faces = edge_topology(faces)
    face_normals = _unit_face_normals(points, faces)
    edge_normals = []
    for index, left in enumerate(loop):
        right = int(loop[(index + 1) % loop.size])
        face_id = edge_faces[tuple(sorted((int(left), right)))][0]
        edge_normals.append(face_normals[face_id])
    edge_normals_array = np.asarray(edge_normals, dtype=np.float64)
    normals = edge_normals_array + np.roll(edge_normals_array, 1, axis=0)
    lengths = np.linalg.norm(normals, axis=1)
    fallback = lengths <= 1e-15
    normals[fallback] = edge_normals_array[fallback]
    lengths = np.linalg.norm(normals, axis=1)
    return np.divide(normals, lengths[:, None], out=np.zeros_like(normals), where=lengths[:, None] > 1e-15)


def _unit_face_normals(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = np.asarray(points, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    return np.divide(normals, lengths[:, None], out=np.zeros_like(normals), where=lengths[:, None] > 1e-15)


def _triangle_double_area(points: np.ndarray, face: np.ndarray) -> float:
    triangle = np.asarray(points, dtype=np.float64)[np.asarray(face, dtype=np.int64)]
    return float(np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])))


def _simple_loop_diagnostics(points: np.ndarray, loop: np.ndarray, tolerance: float) -> dict[str, Any]:
    loop_points = points[loop]
    segments = np.roll(loop_points, -1, axis=0) - loop_points
    lengths = np.linalg.norm(segments, axis=1)
    if np.any(lengths <= tolerance):
        edge_id = int(np.flatnonzero(lengths <= tolerance)[0])
        return {
            "passed": False,
            "failure_reason": "boundary_loop_zero_length_edge",
            "message": "A simple boundary loop cannot contain zero-length edges.",
            "details": {
                "edge_vertex_ids": [int(loop[edge_id]), int(loop[(edge_id + 1) % loop.size])],
                "edge_length": float(lengths[edge_id]),
                "geometric_tolerance": float(tolerance),
            },
        }
    intersections = _loop_self_intersections(loop_points, tolerance)
    if intersections:
        return {
            "passed": False,
            "failure_reason": "boundary_loop_self_intersection",
            "message": "A selected boundary cycle intersects itself geometrically.",
            "details": {"segment_pairs": intersections[:50], "intersection_count": len(intersections)},
        }
    return {"passed": True}


def _loop_self_intersections(loop_points: np.ndarray, tolerance: float) -> list[list[int]]:
    count = loop_points.shape[0]
    starts = loop_points
    ends = np.roll(loop_points, -1, axis=0)
    mins = np.minimum(starts, ends) - tolerance
    maxs = np.maximum(starts, ends) + tolerance
    sweep_axis = int(np.argmax(np.ptp(loop_points, axis=0)))
    order = np.argsort(mins[:, sweep_axis], kind="mergesort")
    active: list[int] = []
    intersections = []
    for current_value in order:
        current = int(current_value)
        active = [
            candidate
            for candidate in active
            if maxs[candidate, sweep_axis] >= mins[current, sweep_axis]
        ]
        for candidate in active:
            left, right = sorted((candidate, current))
            if right == left + 1 or (left == 0 and right == count - 1):
                continue
            if not np.all(mins[left] <= maxs[right]) or not np.all(mins[right] <= maxs[left]):
                continue
            distance = _segment_distance(starts[left], ends[left], starts[right], ends[right])
            if distance <= tolerance:
                intersections.append([left, right])
        active.append(current)
    intersections.sort()
    return intersections


def _segment_distance(p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray) -> float:
    # Closest distance between finite 3-D segments, including parallel segments.
    u = p1 - p0
    v = q1 - q0
    w = p0 - q0
    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))
    denominator = a * c - b * b
    small = 1e-30
    if denominator < small:
        s_numerator, s_denominator = 0.0, 1.0
        t_numerator, t_denominator = e, c
    else:
        s_numerator = b * e - c * d
        t_numerator = a * e - b * d
        s_denominator = t_denominator = denominator
        if s_numerator < 0.0:
            s_numerator = 0.0
            t_numerator, t_denominator = e, c
        elif s_numerator > s_denominator:
            s_numerator = s_denominator
            t_numerator, t_denominator = e + b, c
    if t_numerator < 0.0:
        t_numerator = 0.0
        if -d < 0.0:
            s_numerator = 0.0
        elif -d > a:
            s_numerator = s_denominator
        else:
            s_numerator, s_denominator = -d, a
    elif t_numerator > t_denominator:
        t_numerator = t_denominator
        if -d + b < 0.0:
            s_numerator = 0.0
        elif -d + b > a:
            s_numerator = s_denominator
        else:
            s_numerator, s_denominator = -d + b, a
    sc = 0.0 if abs(s_numerator) < small else s_numerator / max(s_denominator, small)
    tc = 0.0 if abs(t_numerator) < small else t_numerator / max(t_denominator, small)
    return float(np.linalg.norm(w + sc * u - tc * v))


def _boundary_component_vertices(seed_edge: tuple[int, int], adjacency: dict[int, list[int]]) -> set[int]:
    visited = set(seed_edge)
    stack = list(seed_edge)
    while stack:
        vertex = stack.pop()
        for neighbour in adjacency[vertex]:
            if neighbour not in visited:
                visited.add(neighbour)
                stack.append(neighbour)
    return visited


def _loop_orientation_conflicts(
    loop: np.ndarray,
    directed: dict[tuple[int, int], tuple[int, int]],
) -> list[list[int]]:
    conflicts = []
    for index, left in enumerate(loop):
        right = int(loop[(index + 1) % loop.size])
        if directed[tuple(sorted((int(left), right)))] != (int(left), right):
            conflicts.append([int(left), right])
    return conflicts


def _face_directed_edge(face: np.ndarray, edge: tuple[int, int]) -> tuple[int, int] | None:
    for left, right in _face_directed_edges(face):
        if tuple(sorted((left, right))) == edge:
            return left, right
    return None


def _face_directed_edges(face: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    a, b, c = (int(value) for value in face)
    return (a, b), (b, c), (c, a)


def _directed_edge_occurrences(faces: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int, int]]]:
    result: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for face_id, face in enumerate(faces):
        for left, right in _face_directed_edges(face):
            result.setdefault(tuple(sorted((left, right))), []).append((face_id, left, right))
    return result


def _reverse_cycle(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    return np.concatenate([values[:1], values[:0:-1]], axis=0)


def _rotate_cycle_to_smallest(loop: np.ndarray) -> np.ndarray:
    start = int(np.argmin(loop))
    return np.roll(loop, -start)


def _geometric_tolerance(points: np.ndarray, requested: float | None) -> float:
    if requested is not None:
        return max(float(requested), 1e-15)
    if np.asarray(points).size == 0:
        return 1e-12
    points = np.asarray(points, dtype=np.float64)
    diagonal = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    return max(diagonal * 1e-10, 1e-12)


def _mesh_input_diagnostics(points: np.ndarray, faces: np.ndarray) -> tuple[str, str] | None:
    points = np.asarray(points)
    faces = np.asarray(faces)
    if points.ndim != 2 or points.shape[1:] != (3,):
        return "mesh_points_shape_invalid", "Mesh points must have shape (N, 3)."
    if faces.ndim != 2 or faces.shape[1:] != (3,):
        return "mesh_faces_shape_invalid", "Mesh faces must have shape (M, 3)."
    if points.shape[0] < 3 or faces.shape[0] < 1:
        return "mesh_empty", "A stitch input must contain points and triangle faces."
    if not np.all(np.isfinite(points)):
        return "mesh_points_non_finite", "Mesh points must be finite."
    if not np.issubdtype(faces.dtype, np.integer):
        return "mesh_faces_not_integer", "Triangle vertex IDs must be integers."
    if np.any(faces < 0) or np.any(faces >= points.shape[0]):
        return "mesh_face_vertex_out_of_range", "Triangle vertex IDs must reference the point array."
    repeated = (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 2] == faces[:, 0])
    if np.any(repeated):
        return "mesh_face_repeats_vertex", "Input triangles may not repeat a vertex ID."
    return None


def _loop_failure(code: str, stage: str, message: str, **details: Any) -> dict[str, Any]:
    return {
        "success": False,
        "failure_reason_codes": [code],
        "loops": [],
        "diagnostics": {"stage": stage, "message": message, **details},
    }


def _stitch_failure(code: str, stage: str, message: str, **details: Any) -> dict[str, Any]:
    return {
        "success": False,
        "failure_reason_codes": [code],
        "points": np.empty((0, 3), dtype=np.float64),
        "faces": np.empty((0, 3), dtype=np.int64),
        "face_origin": np.empty(0, dtype=np.int16),
        "diagnostics": {"stage": stage, "message": message, **details},
    }
