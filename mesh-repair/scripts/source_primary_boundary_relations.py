from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def classify_loop_relation(
    points: np.ndarray,
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> dict[str, Any] | None:
    left_geometry, right_geometry = left["geometry"], right["geometry"]
    if (
        left_geometry["planes"]["pca"]["status"] != "computed"
        or right_geometry["planes"]["pca"]["status"] != "computed"
    ):
        return None
    left_scale, right_scale = left_geometry["scale"], right_geometry["scale"]
    scale = max(left_scale["major_extent"], right_scale["major_extent"], thresholds["absolute_tolerance"])
    left_normal = np.asarray(left_geometry["planes"]["pca"]["normal"])
    right_normal = np.asarray(right_geometry["planes"]["pca"]["normal"])
    left_center = np.asarray(left_geometry["stable_center"]["value"])
    right_center = np.asarray(right_geometry["stable_center"]["value"])
    center_delta = right_center - left_center
    center_distance = float(np.linalg.norm(center_delta))
    plane_offset = float(max(abs(np.dot(center_delta, left_normal)), abs(np.dot(center_delta, right_normal))))
    shared_components = sorted(set(left["component_ids"]) & set(right["component_ids"]))
    evidence = {
        "left_loop_id": left["loop_id"],
        "right_loop_id": right["loop_id"],
        "perimeter_similarity": _ratio(left_scale["perimeter"], right_scale["perimeter"]),
        "major_extent_similarity": _ratio(left_scale["major_extent"], right_scale["major_extent"]),
        "normal_abs_dot": float(abs(np.dot(left_normal, right_normal))),
        "center_distance": center_distance,
        "center_distance_scale_ratio": center_distance / scale,
        "plane_offset": plane_offset,
        "plane_offset_scale_ratio": plane_offset / scale,
        "axial_fraction": plane_offset / max(center_distance, thresholds["absolute_tolerance"]),
        "topology_relation": "same_face_component" if shared_components else "distinct_face_components",
        "shared_component_ids": shared_components,
    }
    if evidence["center_distance_scale_ratio"] > max(
        thresholds["pair_max_center_distance_scale_ratio"],
        thresholds["nested_max_center_distance_scale_ratio"],
    ):
        return None
    if (
        min(evidence["perimeter_similarity"], evidence["major_extent_similarity"])
        >= thresholds["pair_min_similarity"]
        and evidence["normal_abs_dot"] >= thresholds["pair_min_normal_abs_dot"]
        and evidence["center_distance_scale_ratio"] <= thresholds["pair_max_center_distance_scale_ratio"]
    ):
        paired = _paired_relation(points, left, right, scale, evidence, thresholds)
        if paired is not None:
            return paired
    return _nesting_relation(points, left, right, evidence, thresholds)


def find_loop_self_intersections(
    points: np.ndarray,
    ordered_vertex_ids: Sequence[int],
    tolerance: float,
) -> list[list[int]]:
    ids = [int(value) for value in ordered_vertex_ids]
    starts = points[np.asarray(ids, dtype=np.int64)]
    ends = np.roll(starts, -1, axis=0)
    minimum = np.minimum(starts, ends) - tolerance
    maximum = np.maximum(starts, ends) + tolerance
    order = np.argsort(minimum[:, 0], kind="mergesort")
    pairs: list[list[int]] = []
    for position, left in enumerate(order):
        for right in order[position + 1:]:
            if minimum[right, 0] > maximum[left, 0]:
                break
            if right == left + 1 or (left == 0 and right == len(ids) - 1):
                continue
            if left == right + 1 or (right == 0 and left == len(ids) - 1):
                continue
            if np.all(minimum[left, 1:] <= maximum[right, 1:]) and np.all(minimum[right, 1:] <= maximum[left, 1:]):
                if _segment_distance(starts[left], ends[left], starts[right], ends[right]) <= tolerance:
                    pairs.append(sorted((int(left), int(right))))
    pairs.sort()
    return pairs


def select_unambiguous_pairs(
    relations: Sequence[dict[str, Any]],
    config: Mapping[str, float],
) -> tuple[list[dict[str, Any]], set[str], list[dict[str, Any]]]:
    candidates = [row for row in relations if row["kind"] != "nested_loops"]
    by_loop: dict[str, list[dict[str, Any]]] = {}
    priority = {"near_coincident_part_seam": 0, "compatible_paired_boundary_loops": 1}
    for row in candidates:
        by_loop.setdefault(row["left_loop_id"], []).append(row)
        by_loop.setdefault(row["right_loop_id"], []).append(row)
    best, ambiguous, diagnostics = {}, set(), []
    for loop_id, rows in by_loop.items():
        rows.sort(key=lambda item: (priority[item["kind"]], item["score"], item["left_loop_id"], item["right_loop_id"]))
        if len(rows) > 1 and priority[rows[0]["kind"]] == priority[rows[1]["kind"]] and rows[1]["score"] - rows[0]["score"] <= config["pair_score_ambiguity_margin"]:
            ambiguous.add(loop_id)
            diagnostics.append({"reason_code": "ambiguous_loop_pairing", "loop_id": loop_id, "candidate_loop_ids": sorted({_other_loop(row, loop_id) for row in rows})})
        else:
            best[loop_id] = rows[0]
    accepted = []
    for row in candidates:
        left, right = row["left_loop_id"], row["right_loop_id"]
        if left not in ambiguous and right not in ambiguous and best.get(left) is row and best.get(right) is row:
            accepted.append(row)
    matched = {loop_id for row in accepted for loop_id in (row["left_loop_id"], row["right_loop_id"])}
    for loop_id in sorted(set(by_loop) - matched - ambiguous):
        ambiguous.add(loop_id)
        diagnostics.append({
            "reason_code": "non_mutual_loop_pairing",
            "loop_id": loop_id,
            "candidate_loop_ids": sorted({_other_loop(row, loop_id) for row in by_loop[loop_id]}),
        })
    return accepted, ambiguous, sorted(diagnostics, key=lambda item: item["loop_id"])


def select_nesting(
    relations: Sequence[dict[str, Any]],
    consumed: set[str],
) -> tuple[list[dict[str, Any]], set[str]]:
    candidates = [row for row in relations if row["kind"] == "nested_loops" and row["left_loop_id"] not in consumed and row["right_loop_id"] not in consumed]
    by_inner: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        by_inner.setdefault(row["inner_loop_id"], []).append(row)
    ambiguous_seeds: set[str] = set()
    nesting_pairs = {frozenset((row["left_loop_id"], row["right_loop_id"])) for row in candidates}
    for inner, rows in by_inner.items():
        outers = [row["outer_loop_id"] for row in rows]
        if len(outers) > 1 and any(frozenset((left, right)) not in nesting_pairs for index, left in enumerate(outers) for right in outers[index + 1:]):
            ambiguous_seeds.add(inner)
    adjacency: dict[str, set[str]] = {}
    for row in candidates:
        left, right = row["left_loop_id"], row["right_loop_id"]
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    ambiguous, stack = set(ambiguous_seeds), list(ambiguous_seeds)
    while stack:
        for neighbor in adjacency.get(stack.pop(), set()):
            if neighbor not in ambiguous:
                ambiguous.add(neighbor)
                stack.append(neighbor)
    accepted = [row for row in candidates if row["left_loop_id"] not in ambiguous and row["right_loop_id"] not in ambiguous]
    return accepted, ambiguous


def nested_groups(
    loops: Sequence[dict[str, Any]],
    nesting: Sequence[dict[str, Any]],
    excluded: set[str],
) -> list[set[str]]:
    parent = {loop["loop_id"]: loop["loop_id"] for loop in loops if loop["loop_id"] not in excluded}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for row in nesting:
        left, right = row["left_loop_id"], row["right_loop_id"]
        if left in parent and right in parent:
            left_root, right_root = find(left), find(right)
            parent[max(left_root, right_root)] = min(left_root, right_root)
    groups: dict[str, set[str]] = {}
    for loop_id in parent:
        groups.setdefault(find(loop_id), set()).add(loop_id)
    return sorted((group for group in groups.values() if len(group) > 1), key=lambda group: sorted(group))


def _paired_relation(
    points: np.ndarray,
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    scale: float,
    evidence: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> dict[str, Any] | None:
    hausdorff = _polyline_hausdorff(points, left["ordered_vertex_ids"], right["ordered_vertex_ids"])
    local_edge = max(
        left["geometry"]["scale"]["median_edge_length"],
        right["geometry"]["scale"]["median_edge_length"],
        thresholds["absolute_tolerance"],
    )
    distances = {
        "polyline_hausdorff": hausdorff,
        "polyline_distance_method": "symmetric_vertices_to_opposite_polyline_segments",
        "hausdorff_local_edge_ratio": hausdorff / local_edge,
        "hausdorff_scale_ratio": hausdorff / scale,
    }
    near_limit = max(
        thresholds["absolute_tolerance"],
        min(
            local_edge * thresholds["near_seam_max_gap_edge_ratio"],
            scale * thresholds["near_seam_max_gap_scale_ratio"],
        ),
    )
    if hausdorff <= near_limit:
        kind = (
            "near_coincident_part_seam"
            if evidence["topology_relation"] == "distinct_face_components"
            else "compatible_paired_boundary_loops"
        )
        return {
            "kind": kind,
            "score": hausdorff / near_limit,
            **evidence,
            **distances,
        }
    pair_limit = min(
        local_edge * thresholds["pair_max_gap_edge_ratio"],
        scale * thresholds["pair_max_gap_scale_ratio"],
    )
    if (
        hausdorff <= pair_limit
        and evidence["center_distance_scale_ratio"] <= thresholds["pair_max_center_distance_scale_ratio"]
        and evidence["axial_fraction"] >= thresholds["pair_min_axial_fraction"]
    ):
        return {
            "kind": "compatible_paired_boundary_loops",
            "score": hausdorff / max(pair_limit, 1e-30),
            **evidence,
            **distances,
        }
    return None


def _nesting_relation(
    points: np.ndarray,
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    evidence: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> dict[str, Any] | None:
    if evidence["normal_abs_dot"] < thresholds["nested_min_normal_abs_dot"]:
        return None
    if evidence["center_distance_scale_ratio"] > thresholds["nested_max_center_distance_scale_ratio"]:
        return None
    if max(
        left["geometry"]["planarity"]["max_distance_scale_ratio"],
        right["geometry"]["planarity"]["max_distance_scale_ratio"],
    ) > thresholds["nested_max_planarity_scale_ratio"]:
        return None
    left_area = left["geometry"]["scale"]["footprint_area"]
    right_area = right["geometry"]["scale"]["footprint_area"]
    outer, inner = (left, right) if left_area >= right_area else (right, left)
    if (
        outer["geometry"]["normals"]["boundary_side"]["status"]
        != "missing_surface_inside_loop"
        or inner["geometry"]["normals"]["boundary_side"]["status"]
        != "open_component_perimeter_or_inner_island"
    ):
        return None
    outer_scale = max(outer["geometry"]["scale"]["major_extent"], thresholds["absolute_tolerance"])
    if evidence["plane_offset"] > outer_scale * thresholds["nested_max_plane_offset_scale_ratio"]:
        return None
    pca = outer["geometry"]["planes"]["pca"]
    origin = np.asarray(pca["origin"])
    axes = np.vstack((pca["major_axis"], pca["minor_axis"])).T
    outer_polygon = (points[np.asarray(outer["ordered_vertex_ids"])] - origin) @ axes
    inner_polygon = (points[np.asarray(inner["ordered_vertex_ids"])] - origin) @ axes
    tolerance = outer_scale * thresholds["nested_boundary_clearance_scale_ratio"]
    if not all(_point_in_polygon(point, outer_polygon, tolerance) for point in inner_polygon):
        return None
    if _polygons_intersect(outer_polygon, inner_polygon, tolerance):
        return None
    return {
        "kind": "nested_loops",
        "score": float(inner["geometry"]["scale"]["major_extent"] / outer_scale),
        "outer_loop_id": outer["loop_id"],
        "inner_loop_id": inner["loop_id"],
        **evidence,
    }


def _polyline_hausdorff(points: np.ndarray, left_ids: Sequence[int], right_ids: Sequence[int]) -> float:
    left = points[np.asarray(left_ids, dtype=np.int64)]
    right = points[np.asarray(right_ids, dtype=np.int64)]
    return max(_vertices_to_polyline(left, right), _vertices_to_polyline(right, left))


def _vertices_to_polyline(vertices: np.ndarray, polyline: np.ndarray) -> float:
    starts = polyline
    directions = np.roll(polyline, -1, axis=0) - starts
    denominator = np.einsum("ij,ij->i", directions, directions)
    maximum = 0.0
    for offset in range(0, vertices.shape[0], 32):
        chunk = vertices[offset:offset + 32]
        delta = chunk[:, None, :] - starts[None, :, :]
        parameter = np.divide(
            np.einsum("bsi,si->bs", delta, directions),
            denominator[None, :],
            out=np.zeros((chunk.shape[0], starts.shape[0])),
            where=denominator[None, :] > 0.0,
        )
        closest = starts[None, :, :] + np.clip(parameter, 0.0, 1.0)[:, :, None] * directions[None, :, :]
        squared = np.sum((chunk[:, None, :] - closest) ** 2, axis=2)
        maximum = max(maximum, float(np.sqrt(squared.min(axis=1)).max(initial=0.0)))
    return maximum


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray, tolerance: float) -> bool:
    edges = zip(polygon, np.roll(polygon, -1, axis=0), strict=True)
    if min(_point_segment_distance(point, start, end) for start, end in edges) <= tolerance:
        return False
    inside = False
    x, y = float(point[0]), float(point[1])
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        if (start[1] > y) != (end[1] > y):
            crossing = float((end[0] - start[0]) * (y - start[1]) / (end[1] - start[1]) + start[0])
            inside ^= x < crossing
    return inside


def _polygons_intersect(left: np.ndarray, right: np.ndarray, tolerance: float) -> bool:
    for a0, a1 in zip(left, np.roll(left, -1, axis=0), strict=True):
        for b0, b1 in zip(right, np.roll(right, -1, axis=0), strict=True):
            if _segment_distance(np.pad(a0, (0, 1)), np.pad(a1, (0, 1)), np.pad(b0, (0, 1)), np.pad(b1, (0, 1))) <= tolerance:
                return True
    return False


def _segment_distance(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> float:
    first, second, offset = a1 - a0, b1 - b0, a0 - b0
    aa, ab, bb = float(np.dot(first, first)), float(np.dot(first, second)), float(np.dot(second, second))
    ac, bc = float(np.dot(first, offset)), float(np.dot(second, offset))
    denominator = aa * bb - ab * ab
    left = 0.0 if denominator <= 1e-30 else np.clip((ab * bc - bb * ac) / denominator, 0.0, 1.0)
    right = np.clip((ab * left + bc) / max(bb, 1e-30), 0.0, 1.0)
    left = np.clip((ab * right - ac) / max(aa, 1e-30), 0.0, 1.0)
    return float(np.linalg.norm(offset + left * first - right * second))


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    direction = end - start
    denominator = float(np.dot(direction, direction))
    parameter = 0.0 if denominator <= 0.0 else float(np.clip(np.dot(point - start, direction) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + parameter * direction)))


def _ratio(left: float, right: float) -> float:
    return float(min(left, right) / max(left, right, 1e-30))


def _other_loop(row: Mapping[str, Any], loop_id: str) -> str:
    return str(row["right_loop_id"] if row["left_loop_id"] == loop_id else row["left_loop_id"])
