from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree


DEFAULT_COMPONENT_THRESHOLDS = {
    "exterior_support_threshold": 0.25,
    "sealed_exterior_support_threshold": 0.01,
    "internal_max_face_ratio": 0.01,
    "internal_max_diameter_ratio": 0.05,
    "internal_max_projected_bbox_area_ratio": 0.01,
    "floating_max_face_ratio": 1.0e-4,
    "floating_max_diameter_ratio": 0.01,
    "shell_envelope_min_component_face_ratio": 1.0e-3,
    "shell_envelope_margin_ratio": 0.005,
}


# Ratios use the global/component bbox diagonal as the reference length and its
# square as the reference area.  Boundary-edge median length is the local mesh
# scale.  These defaults deliberately make semantic openings a narrow class:
# geometry that looks like a part perimeter stays unresolved rather than being
# sent to the opening-policy packet.
DEFAULT_BOUNDARY_THRESHOLDS = {
    "small_min_exterior_confidence": 0.50,
    "small_max_diameter_bbox_ratio": 0.005,
    "small_max_projected_area_bbox_ratio": 2.5e-5,
    "small_max_diameter_local_edge_ratio": 32.0,
    "small_max_diameter_component_ratio": 0.45,
    "small_max_planarity_ratio": 0.08,
    "small_min_compactness": 0.10,
    "small_max_component_boundary_count": 8.0,
    "small_polygon_max_edge_count": 4.0,
    "small_polygon_max_diameter_bbox_ratio": 0.015,
    "small_polygon_max_projected_area_bbox_ratio": 5.0e-5,
    "small_polygon_max_diameter_local_edge_ratio": 2.5,
    "part_perimeter_min_diameter_component_ratio": 0.80,
    "part_perimeter_min_projected_area_component_ratio": 0.18,
    "part_perimeter_single_boundary_min_diameter_component_ratio": 0.55,
    "single_shell_semantic_min_exterior_confidence": 0.50,
    "semantic_min_exterior_confidence": 0.75,
    "semantic_min_diameter_bbox_ratio": 0.005,
    "semantic_min_projected_area_bbox_ratio": 2.5e-5,
    "semantic_max_diameter_component_ratio": 0.70,
    "semantic_max_planarity_ratio": 0.30,
    "semantic_min_compactness": 0.15,
    "semantic_max_component_boundary_count": 4.0,
    "pair_min_perimeter_ratio": 0.50,
    "pair_max_perimeter_ratio": 2.00,
    "pair_min_normal_abs_dot": 0.65,
    "pair_max_gap_local_edge_ratio": 3.0,
    "pair_max_gap_loop_diameter_ratio": 0.05,
    "pair_gap_bbox_ratio_floor": 0.001,
    "pair_max_centroid_gap_limits": 4.0,
    "pair_max_centroid_loop_diameter_ratio": 0.25,
    "pair_search_neighbors": 32.0,
    "pair_interface_min_perimeter_similarity": 0.55,
    "pair_interface_min_diameter_similarity": 0.85,
    "pair_interface_min_normal_abs_dot": 0.95,
    "pair_interface_max_centroid_loop_diameter_ratio": 0.12,
    "pair_interface_max_gap_local_edge_ratio": 4.0,
    "pair_interface_max_gap_loop_diameter_ratio": 0.14,
}


BOUNDARY_THRESHOLD_DESCRIPTIONS = {
    "small_min_exterior_confidence": "minimum exterior support for deterministic small-hole fill",
    "small_max_diameter_bbox_ratio": "maximum loop diameter divided by global bbox diagonal",
    "small_max_projected_area_bbox_ratio": "maximum projected loop area divided by global bbox diagonal squared",
    "small_max_diameter_local_edge_ratio": "maximum loop diameter divided by median boundary-edge length",
    "small_max_diameter_component_ratio": "maximum loop diameter divided by component bbox diagonal",
    "small_max_planarity_ratio": "maximum smallest-to-largest loop-point singular-value ratio",
    "small_min_compactness": "minimum 4*pi*projected_area/perimeter^2 for small-hole fill",
    "small_max_component_boundary_count": "maximum boundary-region count on a component for automatic small-hole fill",
    "small_polygon_max_edge_count": "maximum edge count for an isolated missing-polygon fill exception",
    "small_polygon_max_diameter_bbox_ratio": "maximum bbox-normalized diameter for a missing-polygon fill exception",
    "small_polygon_max_projected_area_bbox_ratio": "maximum bbox-normalized area for a missing-polygon fill exception",
    "small_polygon_max_diameter_local_edge_ratio": "maximum diameter in boundary-edge lengths for a missing-polygon fill exception",
    "part_perimeter_min_diameter_component_ratio": "component-scale loop diameter above which the loop is treated as a part perimeter",
    "part_perimeter_min_projected_area_component_ratio": "component-scale projected area above which the loop is treated as a part perimeter",
    "part_perimeter_single_boundary_min_diameter_component_ratio": "single-loop component-scale diameter treated as a dominant open perimeter",
    "single_shell_semantic_min_exterior_confidence": "minimum bbox-only exterior support retained for near-single-shell policy compatibility",
    "semantic_min_exterior_confidence": "minimum exterior support for semantic-opening policy routing",
    "semantic_min_diameter_bbox_ratio": "minimum global bbox-normalized diameter for a large semantic opening",
    "semantic_min_projected_area_bbox_ratio": "minimum global bbox-normalized projected area for a large semantic opening",
    "semantic_max_diameter_component_ratio": "maximum component bbox-normalized diameter for an interior opening candidate",
    "semantic_max_planarity_ratio": "maximum loop non-planarity accepted for semantic-opening routing",
    "semantic_min_compactness": "minimum projected compactness for semantic-opening routing",
    "semantic_max_component_boundary_count": "maximum component boundary complexity accepted for semantic-opening routing",
    "pair_min_perimeter_ratio": "minimum smaller/larger loop-perimeter ratio for seam pairing",
    "pair_max_perimeter_ratio": "maximum left/right loop-perimeter ratio retained for backward-compatible reporting",
    "pair_min_normal_abs_dot": "minimum absolute loop-normal dot product for seam pairing",
    "pair_max_gap_local_edge_ratio": "maximum symmetric Hausdorff gap in median boundary-edge lengths",
    "pair_max_gap_loop_diameter_ratio": "maximum symmetric Hausdorff gap relative to the smaller loop diameter",
    "pair_gap_bbox_ratio_floor": "global bbox-normalized numerical floor for the seam gap tolerance",
    "pair_max_centroid_gap_limits": "maximum centroid distance in accepted Hausdorff gap limits",
    "pair_max_centroid_loop_diameter_ratio": "maximum centroid distance relative to the larger loop diameter",
    "pair_search_neighbors": "number of centroid-nearest loop candidates considered for pairing",
    "pair_interface_min_perimeter_similarity": "minimum smaller/larger perimeter ratio for a concentric component interface",
    "pair_interface_min_diameter_similarity": "minimum smaller/larger diameter ratio for a concentric component interface",
    "pair_interface_min_normal_abs_dot": "minimum absolute normal dot for a concentric component interface",
    "pair_interface_max_centroid_loop_diameter_ratio": "maximum centroid separation for a concentric component interface",
    "pair_interface_max_gap_local_edge_ratio": "maximum Hausdorff gap in local edges for a concentric component interface",
    "pair_interface_max_gap_loop_diameter_ratio": "maximum Hausdorff gap relative to loop diameter for a concentric component interface",
}


def classify_boundary_regions(
    points: np.ndarray,
    faces: np.ndarray,
    regions: list[list[tuple[int, int]]],
    edge_to_faces: dict[tuple[int, int], list[int]],
    component_ids: np.ndarray,
    *,
    exterior_face_score: np.ndarray | None = None,
    sealed_exterior_face_mask: np.ndarray | None = None,
    component_thresholds: dict[str, float] | None = None,
    boundary_thresholds: dict[str, float] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Classify topology boundaries without assuming that every free edge is a hole.

    The classifier intentionally returns an unresolved category when geometry alone
    cannot distinguish a functional opening, a part perimeter, and a missing wall.
    Only simple loops with exterior evidence, or a compatible loop pair, are marked
    eligible for an automatic geometry operator.
    """
    global_diag = bbox_diagonal(points)
    component_config = normalized_component_thresholds(component_thresholds)
    boundary_config = normalized_boundary_thresholds(boundary_thresholds)
    component_stats = build_component_stats(
        points,
        faces,
        component_ids,
        exterior_face_score,
        sealed_exterior_face_mask,
        component_config,
    )
    descriptors = [
        describe_loop(points, region, component_id_for_region(region, edge_to_faces, component_ids))
        for region in regions
    ]
    boundary_counts: dict[int, int] = {}
    for descriptor in descriptors:
        component_id = descriptor.get("component_id")
        if component_id is not None:
            boundary_counts[int(component_id)] = boundary_counts.get(int(component_id), 0) + 1
    for component_id, stats in component_stats.items():
        stats["boundary_region_count"] = boundary_counts.get(component_id, 0)
    for descriptor in descriptors:
        component = component_stats.get(descriptor.get("component_id"), empty_component_stats())
        descriptor.update(normalized_loop_features(descriptor, component, global_diag))
    pairs = compatible_loop_pairs(points, descriptors, global_diag, boundary_config)
    paired_by_index: dict[int, tuple[int, dict[str, Any]]] = {}
    for pair_id, pair in enumerate(pairs, start=1):
        paired_by_index[int(pair["left"])] = (pair_id, pair)
        paired_by_index[int(pair["right"])] = (pair_id, pair)

    rows = []
    for index, descriptor in enumerate(descriptors):
        stats = component_stats.get(descriptor.get("component_id"), empty_component_stats())
        classification = classify_one(
            descriptor,
            stats,
            paired_by_index.get(index),
            len(regions),
            len(component_stats),
            component_config,
            boundary_config,
        )
        rows.append({**descriptor, **classification})

    assembly_route = len(component_stats) > 12 or len(regions) > 64
    return rows, {
        "route": "assembly_exterior_reconstruction" if assembly_route else "near_single_shell_repair",
        "component_count": len(component_stats),
        "boundary_region_count": len(regions),
        "global_bbox_diagonal": global_diag,
        "loop_pair_count": len(pairs),
        "exterior_evidence": {
            "visibility_face_score": exterior_face_score is not None,
            "sealed_far_field_face_mask": sealed_exterior_face_mask is not None,
            "bbox_and_component_geometry": True,
        },
        "component_thresholds": component_config,
        "boundary_thresholds": boundary_config,
        "boundary_threshold_definitions": BOUNDARY_THRESHOLD_DESCRIPTIONS,
        "dimensionless_feature_normalization": {
            "bbox_length": "global bbox diagonal",
            "bbox_area": "global bbox diagonal squared",
            "component_length": "component bbox diagonal",
            "component_area": "component bbox diagonal squared",
            "local_edge": "median boundary-edge length",
            "pair_gap": "symmetric loop Hausdorff distance divided by local edge scale",
        },
        "component_evidence": [
            {"component_id": component_id, **stats}
            for component_id, stats in sorted(component_stats.items())
        ],
        "rules": [
            "a boundary graph is not a hole unless it is a simple ordered loop",
            "near-coincident compatible loop pairs route to zipper stitching",
            "small-hole fill requires simple-loop geometry plus exterior-component evidence",
            "in assembly routing, component-dominant, long-thin, low-compactness, and boundary-complex loops stay unresolved rather than becoming large openings",
            "large semantic openings require strong exterior support, sufficient projected area, and compact interior-loop geometry",
            "multi-view first-hit support is a hard keep that sealed exterior evidence cannot override",
            "contained removal requires direct and sealed negative evidence plus small bbox/local-edge normalized scale",
            "tiny unseen components outside the robust shell envelope may be floating-fragment candidates",
            "internal or fragment component perimeters are never sent to a hole filler",
        ],
    }


def classify_one(
    loop: dict[str, Any],
    component: dict[str, Any],
    paired: tuple[int, dict[str, Any]] | None,
    region_count: int,
    component_count: int,
    component_thresholds: dict[str, float],
    boundary_thresholds: dict[str, float],
) -> dict[str, Any]:
    features = dict(loop["dimensionless_features"])
    base = {
        "patch_eligible": False,
        "requires_policy": False,
        "blocking": True,
        "paired_region_index": None,
        "pair_id": None,
        "operator": None,
        "dimensionless_features": features,
    }
    component_removal = component.get("removal_classification")
    if component_removal in {
        "internal_or_fragment_component_perimeter",
        "isolated_floating_fragment_perimeter",
    }:
        return {
            **base,
            "classification": component_removal,
            "detector_reason": component.get("removal_reason"),
            "operator": "remove_component_candidate",
        }
    if not loop["simple_closed_loop"]:
        return {
            **base,
            "classification": "non_simple_boundary_graph",
            "detector_reason": "boundary_vertex_degree_is_not_two_or_chain_is_disconnected",
        }
    if paired is not None:
        pair_id, evidence = paired
        other = int(evidence["right"] if evidence["left"] == loop["region_index"] else evidence["left"])
        features.update(
            {
                "pair_gap": float(evidence["normalized_hausdorff"]),
                "pair_gap_bbox_ratio": float(evidence["gap_bbox_ratio"]),
                "pair_gap_loop_diameter_ratio": float(evidence["gap_loop_diameter_ratio"]),
            }
        )
        return {
            **base,
            "dimensionless_features": features,
            "classification": "near_coincident_part_seam",
            "detector_reason": "compatible_boundary_loop_pair",
            "patch_eligible": True,
            "operator": "loop_pair_zipper",
            "paired_region_index": other,
            "pair_id": pair_id,
            "pair_evidence": evidence,
        }

    exterior_confidence = float(component.get("exterior_confidence", 0.0))
    assembly_case = component_count > 12 or region_count > 64

    component_boundary_count = int(features["component_boundary_count"])
    component_perimeter_checks = {
        "diameter_dominates_component": (
            features["diameter_component_bbox_ratio"]
            >= boundary_thresholds["part_perimeter_min_diameter_component_ratio"]
        ),
        "projected_area_dominates_component": (
            features["projected_area_component_bbox_ratio"]
            >= boundary_thresholds["part_perimeter_min_projected_area_component_ratio"]
        ),
        "single_boundary_dominates_component": (
            component_boundary_count == 1
            and features["diameter_component_bbox_ratio"]
            >= boundary_thresholds["part_perimeter_single_boundary_min_diameter_component_ratio"]
        ),
    }
    # A lone open shell historically represents a user-visible opening and must
    # still reach the policy gate.  Component-perimeter suppression is aimed at
    # assembly inventories, where hundreds of open part sheets otherwise flood
    # the semantic-opening packet.
    tiny_open_component = int(component.get("face_count", 0)) <= 4
    component_perimeter = (assembly_case or tiny_open_component) and any(
        component_perimeter_checks.values()
    )

    small_checks = {
        "not_component_perimeter": not component_perimeter,
        "exterior_support": exterior_confidence >= boundary_thresholds["small_min_exterior_confidence"],
        "diameter_bbox": (
            features["diameter_bbox_ratio"]
            <= boundary_thresholds["small_max_diameter_bbox_ratio"]
        ),
        "projected_area_bbox": (
            features["projected_area_bbox_ratio"]
            <= boundary_thresholds["small_max_projected_area_bbox_ratio"]
        ),
        "diameter_local_edge": (
            features["diameter_local_edge_ratio"]
            <= boundary_thresholds["small_max_diameter_local_edge_ratio"]
        ),
        "diameter_component": (
            features["diameter_component_bbox_ratio"]
            <= boundary_thresholds["small_max_diameter_component_ratio"]
        ),
        "planarity": features["planarity"] <= boundary_thresholds["small_max_planarity_ratio"],
        "compactness": features["compactness"] >= boundary_thresholds["small_min_compactness"],
        "component_boundary_count": (
            component_boundary_count <= boundary_thresholds["small_max_component_boundary_count"]
        ),
    }
    missing_polygon_checks = {
        "not_component_perimeter": not component_perimeter,
        "exterior_support": exterior_confidence >= boundary_thresholds["small_min_exterior_confidence"],
        "edge_count": len(loop["ordered_vertex_ids"]) <= boundary_thresholds["small_polygon_max_edge_count"],
        "diameter_bbox": (
            features["diameter_bbox_ratio"]
            <= boundary_thresholds["small_polygon_max_diameter_bbox_ratio"]
        ),
        "projected_area_bbox": (
            features["projected_area_bbox_ratio"]
            <= boundary_thresholds["small_polygon_max_projected_area_bbox_ratio"]
        ),
        "diameter_local_edge": (
            features["diameter_local_edge_ratio"]
            <= boundary_thresholds["small_polygon_max_diameter_local_edge_ratio"]
        ),
        "planarity": features["planarity"] <= boundary_thresholds["small_max_planarity_ratio"],
        "component_boundary_count": (
            component_boundary_count <= boundary_thresholds["small_max_component_boundary_count"]
        ),
    }
    strict_small = all(small_checks.values())
    missing_polygon = all(missing_polygon_checks.values())
    if strict_small or missing_polygon:
        return {
            **base,
            "classification": "small_exterior_hole",
            "detector_reason": (
                "isolated_missing_polygon_boundary"
                if missing_polygon and not strict_small
                else "bbox_and_local_scale_small_planar_compact_exterior_loop"
            ),
            "patch_eligible": True,
            "operator": "constrained_loop_triangulation",
            "decision_evidence": {
                "assembly_guard_active": assembly_case,
                "component_perimeter_checks": component_perimeter_checks,
                "small_hole_checks": small_checks,
                "missing_polygon_checks": missing_polygon_checks,
            },
        }

    semantic_exterior_threshold = boundary_thresholds[
        "semantic_min_exterior_confidence"
        if assembly_case
        else "single_shell_semantic_min_exterior_confidence"
    ]
    semantic_checks = {
        "not_component_perimeter": not component_perimeter,
        "exterior_support": exterior_confidence >= semantic_exterior_threshold,
        "diameter_bbox": (
            features["diameter_bbox_ratio"]
            >= boundary_thresholds["semantic_min_diameter_bbox_ratio"]
        ),
        "projected_area_bbox": (
            features["projected_area_bbox_ratio"]
            >= boundary_thresholds["semantic_min_projected_area_bbox_ratio"]
        ),
        "diameter_component": (
            not assembly_case
            or features["diameter_component_bbox_ratio"]
            <= boundary_thresholds["semantic_max_diameter_component_ratio"]
        ),
        "planarity": features["planarity"] <= boundary_thresholds["semantic_max_planarity_ratio"],
        "compactness": features["compactness"] >= boundary_thresholds["semantic_min_compactness"],
        "component_boundary_count": (
            component_boundary_count <= boundary_thresholds["semantic_max_component_boundary_count"]
        ),
    }
    decision_evidence = {
        "assembly_guard_active": assembly_case,
        "semantic_exterior_threshold": semantic_exterior_threshold,
        "component_perimeter_checks": component_perimeter_checks,
        "small_hole_checks": small_checks,
        "missing_polygon_checks": missing_polygon_checks,
        "semantic_opening_checks": semantic_checks,
    }
    if all(semantic_checks.values()):
        return {
            **base,
            "classification": "large_opening_or_missing_surface",
            "detector_reason": "exterior_loop_requires_semantic_opening_policy",
            "requires_policy": True,
            "operator": "proxy_conformal_patch_after_cap_decision",
            "decision_evidence": decision_evidence,
        }

    if component_perimeter and component_perimeter_checks["diameter_dominates_component"]:
        reason = "loop_diameter_spans_component_bbox"
    elif component_perimeter and component_perimeter_checks["projected_area_dominates_component"]:
        reason = "projected_loop_area_spans_component_bbox"
    elif component_perimeter and component_perimeter_checks["single_boundary_dominates_component"]:
        reason = "single_boundary_loop_dominates_open_component"
    elif not semantic_checks["component_boundary_count"]:
        reason = "component_has_too_many_boundaries_for_semantic_opening"
    elif not semantic_checks["compactness"]:
        reason = "long_narrow_or_low_compactness_boundary"
    elif not semantic_checks["projected_area_bbox"]:
        reason = "insufficient_bbox_normalized_projected_area_for_semantic_opening"
    elif not semantic_checks["exterior_support"]:
        reason = "insufficient_exterior_evidence_for_hole_semantics"
    elif not semantic_checks["planarity"]:
        reason = "boundary_nonplanarity_exceeds_semantic_geometry_limit"
    elif not semantic_checks["diameter_component"]:
        reason = "loop_scale_is_ambiguous_with_component_perimeter"
    elif not semantic_checks["diameter_bbox"]:
        reason = "subsemantic_scale_loop_not_safe_for_automatic_fill"
    else:
        reason = "geometry_is_ambiguous_for_opening_semantics"
    return {
        **base,
        "classification": "part_perimeter_or_opening_unknown",
        "detector_reason": reason,
        "operator": "exterior_atlas_review",
        "decision_evidence": decision_evidence,
    }


def describe_loop(
    points: np.ndarray,
    edges: list[tuple[int, int]],
    component_id: int | None,
) -> dict[str, Any]:
    ordered = order_simple_loop(edges)
    vertex_ids = ordered[:-1] if ordered else sorted({vertex for edge in edges for vertex in edge})
    loop_points = points[np.asarray(vertex_ids, dtype=np.int64)] if vertex_ids else np.empty((0, 3))
    edge_lengths = np.asarray(
        [np.linalg.norm(points[left] - points[right]) for left, right in edges],
        dtype=np.float64,
    )
    centroid = loop_points.mean(axis=0) if loop_points.size else np.zeros(3)
    extents = loop_points.max(axis=0) - loop_points.min(axis=0) if loop_points.size else np.zeros(3)
    diameter = float(np.linalg.norm(extents))
    normal, area, planarity = loop_plane_metrics(loop_points)
    perimeter = float(edge_lengths.sum())
    compactness = float(4.0 * np.pi * area / max(perimeter * perimeter, 1e-30))
    return {
        "region_index": -1,
        "component_id": component_id,
        "ordered_vertex_ids": [int(value) for value in vertex_ids],
        "simple_closed_loop": ordered is not None,
        "centroid": centroid.tolist(),
        "normal": normal.tolist(),
        "diameter": diameter,
        "perimeter": perimeter,
        "projected_area": area,
        "compactness": compactness,
        "planarity_ratio": planarity,
        "mean_edge_length": float(edge_lengths.mean()) if edge_lengths.size else 0.0,
        "median_edge_length": float(np.median(edge_lengths)) if edge_lengths.size else 0.0,
    }


def normalized_loop_features(
    loop: dict[str, Any],
    component: dict[str, Any],
    global_diag: float,
) -> dict[str, Any]:
    """Return scale-free evidence used by deterministic and policy routing."""
    component_diag = float(component.get("bbox_diagonal", 0.0))
    local_edge = float(loop.get("median_edge_length", loop.get("mean_edge_length", 0.0)))
    diameter = float(loop.get("diameter", 0.0))
    area = float(loop.get("projected_area", 0.0))
    global_length_scale = max(global_diag, 1e-30)
    component_length_scale = max(component_diag, global_length_scale * 1e-12, 1e-30)
    local_edge_scale = max(local_edge, global_length_scale * 1e-12, 1e-30)
    features = {
        "diameter_bbox_ratio": diameter / global_length_scale,
        "diameter_local_edge_ratio": diameter / local_edge_scale,
        "projected_area_bbox_ratio": area / (global_length_scale * global_length_scale),
        "diameter_component_bbox_ratio": diameter / component_length_scale,
        "projected_area_component_bbox_ratio": area / (component_length_scale * component_length_scale),
        "planarity": float(loop.get("planarity_ratio", float("inf"))),
        "compactness": float(loop.get("compactness", 0.0)),
        "component_boundary_count": int(component.get("boundary_region_count", 0)),
        "pair_gap": None,
        "pair_gap_bbox_ratio": None,
        "pair_gap_loop_diameter_ratio": None,
    }
    return {
        "diameter_bbox_ratio": features["diameter_bbox_ratio"],
        "diameter_local_edge_ratio": features["diameter_local_edge_ratio"],
        "projected_area_bbox_ratio": features["projected_area_bbox_ratio"],
        "diameter_component_bbox_ratio": features["diameter_component_bbox_ratio"],
        "projected_area_component_bbox_ratio": features["projected_area_component_bbox_ratio"],
        "component_boundary_count": features["component_boundary_count"],
        "dimensionless_features": features,
    }


def order_simple_loop(edges: list[tuple[int, int]]) -> list[int] | None:
    if len(edges) < 3:
        return None
    neighbors: dict[int, list[int]] = {}
    edge_set: set[tuple[int, int]] = set()
    for left, right in edges:
        left_i, right_i = int(left), int(right)
        if left_i == right_i:
            return None
        key = tuple(sorted((left_i, right_i)))
        if key in edge_set:
            return None
        edge_set.add(key)
        neighbors.setdefault(left_i, []).append(right_i)
        neighbors.setdefault(right_i, []).append(left_i)
    if any(len(values) != 2 for values in neighbors.values()):
        return None
    start = min(neighbors)
    ordered = [start]
    previous: int | None = None
    current = start
    for _ in range(len(edge_set)):
        candidates = neighbors[current]
        next_vertex = candidates[0] if candidates[0] != previous else candidates[1]
        ordered.append(next_vertex)
        previous, current = current, next_vertex
        if current == start:
            break
    traversed = {
        tuple(sorted((ordered[index], ordered[index + 1])))
        for index in range(len(ordered) - 1)
    }
    if ordered[-1] != start or len(ordered) != len(edge_set) + 1 or traversed != edge_set:
        return None
    return ordered


def loop_plane_metrics(points: np.ndarray) -> tuple[np.ndarray, float, float]:
    if points.shape[0] < 3:
        return np.zeros(3), 0.0, float("inf")
    centered = points - points.mean(axis=0)
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    polygon_vector = np.sum(np.cross(centered, np.roll(centered, -1, axis=0)), axis=0) * 0.5
    if np.dot(normal, polygon_vector) < 0.0:
        normal = -normal
    area = float(abs(np.dot(polygon_vector, normal)))
    planarity = float(singular[-1] / max(singular[0], 1e-30))
    return normal, area, planarity


def compatible_loop_pairs(
    points: np.ndarray,
    loops: list[dict[str, Any]],
    global_diag: float,
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    candidates = [index for index, loop in enumerate(loops) if loop["simple_closed_loop"]]
    if len(candidates) < 2:
        for index, loop in enumerate(loops):
            loop["region_index"] = index
        return []
    for index, loop in enumerate(loops):
        loop["region_index"] = index
    centroids = np.asarray([loops[index]["centroid"] for index in candidates], dtype=np.float64)
    tree = cKDTree(centroids)
    proposed = []
    for local_index, loop_index in enumerate(candidates):
        neighbor_count = min(int(thresholds["pair_search_neighbors"]), len(candidates))
        _, neighbors = tree.query(centroids[local_index], k=neighbor_count)
        for neighbor_local in np.atleast_1d(neighbors):
            other_index = candidates[int(neighbor_local)]
            if other_index <= loop_index:
                continue
            evidence = loop_pair_evidence(
                points,
                loops[loop_index],
                loops[other_index],
                global_diag,
                thresholds,
            )
            if evidence["compatible"]:
                proposed.append({"left": loop_index, "right": other_index, **evidence})
    proposed.sort(key=lambda row: (row["normalized_hausdorff"], row["centroid_distance"]))
    used: set[int] = set()
    accepted = []
    for row in proposed:
        if row["left"] in used or row["right"] in used:
            continue
        used.update((int(row["left"]), int(row["right"])))
        accepted.append(row)
    return accepted


def loop_pair_evidence(
    points: np.ndarray,
    left: dict[str, Any],
    right: dict[str, Any],
    global_diag: float,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    left_points = points[np.asarray(left["ordered_vertex_ids"], dtype=np.int64)]
    right_points = points[np.asarray(right["ordered_vertex_ids"], dtype=np.int64)]
    left_tree, right_tree = cKDTree(left_points), cKDTree(right_points)
    left_to_right = right_tree.query(left_points, k=1)[0]
    right_to_left = left_tree.query(right_points, k=1)[0]
    hausdorff = float(max(left_to_right.max(initial=0.0), right_to_left.max(initial=0.0)))
    global_scale = max(global_diag, 1e-30)
    local_edge = max(
        float(left["median_edge_length"]),
        float(right["median_edge_length"]),
        global_scale * 1e-12,
    )
    centroid_distance = float(np.linalg.norm(np.asarray(left["centroid"]) - np.asarray(right["centroid"])))
    perimeter_ratio = float(left["perimeter"] / max(right["perimeter"], 1e-30))
    perimeter_similarity = float(
        min(left["perimeter"], right["perimeter"])
        / max(left["perimeter"], right["perimeter"], 1e-30)
    )
    normal_dot = float(abs(np.dot(np.asarray(left["normal"]), np.asarray(right["normal"]))))
    diameter_similarity = float(
        min(left["diameter"], right["diameter"])
        / max(left["diameter"], right["diameter"], 1e-30)
    )
    loop_scale = max(
        min(float(left["diameter"]), float(right["diameter"])),
        global_scale * 1e-12,
    )
    distance_limit = max(
        global_scale * thresholds["pair_gap_bbox_ratio_floor"],
        min(
            local_edge * thresholds["pair_max_gap_local_edge_ratio"],
            loop_scale * thresholds["pair_max_gap_loop_diameter_ratio"],
        ),
    )
    near_coincident = bool(
        thresholds["pair_min_perimeter_ratio"]
        <= perimeter_ratio
        <= thresholds["pair_max_perimeter_ratio"]
        and perimeter_similarity >= thresholds["pair_min_perimeter_ratio"]
        and normal_dot >= thresholds["pair_min_normal_abs_dot"]
        and centroid_distance
        <= max(
            distance_limit * thresholds["pair_max_centroid_gap_limits"],
            thresholds["pair_max_centroid_loop_diameter_ratio"]
            * max(left["diameter"], right["diameter"]),
        )
        and hausdorff <= distance_limit
    )
    distinct_components = left.get("component_id") != right.get("component_id")
    interface_compatible = bool(
        distinct_components
        and perimeter_similarity >= thresholds["pair_interface_min_perimeter_similarity"]
        and diameter_similarity >= thresholds["pair_interface_min_diameter_similarity"]
        and normal_dot >= thresholds["pair_interface_min_normal_abs_dot"]
        and centroid_distance
        <= thresholds["pair_interface_max_centroid_loop_diameter_ratio"]
        * max(left["diameter"], right["diameter"])
        and hausdorff
        <= local_edge * thresholds["pair_interface_max_gap_local_edge_ratio"]
        and hausdorff
        <= loop_scale * thresholds["pair_interface_max_gap_loop_diameter_ratio"]
    )
    compatible = near_coincident or interface_compatible
    return {
        "compatible": compatible,
        "pair_mode": (
            "near_coincident_edge"
            if near_coincident
            else "concentric_component_interface"
            if interface_compatible
            else None
        ),
        "centroid_distance": centroid_distance,
        "hausdorff_distance": hausdorff,
        "normalized_hausdorff": hausdorff / local_edge,
        "gap_bbox_ratio": hausdorff / global_scale,
        "gap_loop_diameter_ratio": hausdorff / loop_scale,
        "perimeter_ratio": perimeter_ratio,
        "perimeter_similarity": perimeter_similarity,
        "diameter_similarity": diameter_similarity,
        "normal_abs_dot": normal_dot,
        "distance_limit": distance_limit,
        "distance_limit_bbox_ratio": distance_limit / global_scale,
        "distance_limit_local_edge_ratio": distance_limit / local_edge,
    }


def build_component_stats(
    points: np.ndarray,
    faces: np.ndarray,
    component_ids: np.ndarray,
    exterior_face_score: np.ndarray | None,
    sealed_exterior_face_mask: np.ndarray | None,
    thresholds: dict[str, float],
) -> dict[int, dict[str, Any]]:
    if component_ids.size == 0:
        return {}
    if sealed_exterior_face_mask is not None:
        sealed_exterior_face_mask = np.asarray(sealed_exterior_face_mask, dtype=bool)
        if sealed_exterior_face_mask.shape != (faces.shape[0],):
            raise ValueError("sealed_exterior_face_mask must contain one value per face")
    global_diag = bbox_diagonal(points)
    raw: dict[int, dict[str, Any]] = {}
    for component_id in np.unique(component_ids):
        face_ids = np.flatnonzero(component_ids == component_id)
        local_points = points[np.unique(faces[face_ids].ravel())]
        local_min, local_max = local_points.min(axis=0), local_points.max(axis=0)
        local_extents = local_max - local_min
        raw[int(component_id)] = {
            "face_ids": face_ids,
            "bbox_min": local_min,
            "bbox_max": local_max,
            "bbox_diagonal": float(np.linalg.norm(local_max - local_min)),
            "projected_bbox_area": float(
                max(
                    local_extents[0] * local_extents[1],
                    local_extents[0] * local_extents[2],
                    local_extents[1] * local_extents[2],
                )
            ),
            "face_count": int(face_ids.size),
            "face_ratio": float(face_ids.size / max(faces.shape[0], 1)),
        }
    core = [
        row
        for row in raw.values()
        if row["face_ratio"] >= thresholds["shell_envelope_min_component_face_ratio"]
    ]
    if not core:
        core = [max(raw.values(), key=lambda row: row["face_count"])]
    shell_min = np.vstack([row["bbox_min"] for row in core]).min(axis=0)
    shell_max = np.vstack([row["bbox_max"] for row in core]).max(axis=0)
    shell_diag = max(float(np.linalg.norm(shell_max - shell_min)), global_diag * 1e-12, 1e-15)
    shell_extents = shell_max - shell_min
    shell_projected_area = max(
        float(shell_extents[0] * shell_extents[1]),
        float(shell_extents[0] * shell_extents[2]),
        float(shell_extents[1] * shell_extents[2]),
        shell_diag * shell_diag * 1e-12,
        1e-30,
    )
    margin = max(shell_diag * thresholds["shell_envelope_margin_ratio"], 1e-15)
    result: dict[int, dict[str, Any]] = {}
    for component_id, raw_row in raw.items():
        face_ids = raw_row["face_ids"]
        local_min = raw_row["bbox_min"]
        local_max = raw_row["bbox_max"]
        touches_bbox = bool(np.any(local_min <= shell_min + margin) or np.any(local_max >= shell_max - margin))
        strictly_contained = bool(np.all(local_min > shell_min + margin) and np.all(local_max < shell_max - margin))
        envelope_axis_gap = np.maximum(
            np.maximum(shell_min - local_max, local_min - shell_max),
            0.0,
        )
        shell_envelope_distance = float(np.linalg.norm(envelope_axis_gap))
        outside_shell_envelope = shell_envelope_distance > margin
        visible_fraction = None
        if exterior_face_score is not None and exterior_face_score.shape[0] == faces.shape[0]:
            visible_fraction = float(np.count_nonzero(exterior_face_score[face_ids] > 0) / max(face_ids.size, 1))
        sealed_exterior_fraction = None
        if sealed_exterior_face_mask is not None:
            sealed_exterior_fraction = float(
                np.count_nonzero(sealed_exterior_face_mask[face_ids]) / max(face_ids.size, 1)
            )
        exterior_confidence = max(0.55 if touches_bbox else 0.0, visible_fraction or 0.0)
        visible_hard_keep = bool(visible_fraction is not None and visible_fraction > 0.0)
        bbox_diagonal_ratio = raw_row["bbox_diagonal"] / shell_diag
        projected_bbox_area_ratio = raw_row["projected_bbox_area"] / shell_projected_area
        physical_scale_small = bool(
            bbox_diagonal_ratio <= thresholds["internal_max_diameter_ratio"]
            and projected_bbox_area_ratio <= thresholds["internal_max_projected_bbox_area_ratio"]
        )
        visibility_contained_internal = bool(
            visible_fraction is not None
            and not visible_hard_keep
            and strictly_contained
            and exterior_confidence < thresholds["exterior_support_threshold"]
            and raw_row["face_ratio"] < thresholds["internal_max_face_ratio"]
        )
        sealed_contained_internal = bool(
            sealed_exterior_fraction is not None
            and strictly_contained
            and sealed_exterior_fraction <= thresholds["sealed_exterior_support_threshold"]
            and raw_row["face_ratio"] < thresholds["internal_max_face_ratio"]
        )
        contained_internal = bool(
            visibility_contained_internal
            and sealed_contained_internal
            and physical_scale_small
        )
        floating_fragment = bool(
            not visible_hard_keep
            and outside_shell_envelope
            and raw_row["face_ratio"] <= thresholds["floating_max_face_ratio"]
            and raw_row["bbox_diagonal"] <= shell_diag * thresholds["floating_max_diameter_ratio"]
        )
        removal_classification = None
        removal_reason = None
        if floating_fragment:
            removal_classification = "isolated_floating_fragment_perimeter"
            removal_reason = "tiny_component_outside_robust_shell_envelope"
        elif contained_internal:
            removal_classification = "internal_or_fragment_component_perimeter"
            removal_reason = (
                "contained_outside_sealed_exterior_band"
                if sealed_contained_internal
                else "contained_low_visibility_small_component"
            )
        hard_keep_reasons = []
        if visible_hard_keep:
            hard_keep_reasons.append("multi_view_first_hit")
        if visible_fraction is None:
            hard_keep_reasons.append("visibility_evidence_unavailable")
        if sealed_exterior_fraction is None:
            hard_keep_reasons.append("sealed_exterior_evidence_unavailable")
        if bbox_diagonal_ratio > thresholds["internal_max_diameter_ratio"]:
            hard_keep_reasons.append("physical_diameter_exceeds_internal_limit")
        if projected_bbox_area_ratio > thresholds["internal_max_projected_bbox_area_ratio"]:
            hard_keep_reasons.append("projected_bbox_area_exceeds_internal_limit")
        result[component_id] = {
            "face_count": raw_row["face_count"],
            "face_ratio": raw_row["face_ratio"],
            "touches_global_bbox": touches_bbox,
            "strictly_contained": strictly_contained,
            "outside_shell_envelope": outside_shell_envelope,
            "shell_envelope_distance": shell_envelope_distance,
            "visible_face_fraction": visible_fraction,
            "visibility_label": (
                "visible_hard_keep"
                if visible_hard_keep
                else "unseen_candidate"
                if visible_fraction is not None
                else "visibility_unavailable"
            ),
            "visible_hard_keep": visible_hard_keep,
            "sealed_exterior_face_fraction": sealed_exterior_fraction,
            "exterior_confidence": exterior_confidence,
            "bbox_min": local_min.tolist(),
            "bbox_max": local_max.tolist(),
            "bbox_diagonal": raw_row["bbox_diagonal"],
            "bbox_diagonal_ratio": bbox_diagonal_ratio,
            "projected_bbox_area": raw_row["projected_bbox_area"],
            "projected_bbox_area_ratio": projected_bbox_area_ratio,
            "physical_scale_small": physical_scale_small,
            "direct_and_sealed_internal_consensus": bool(
                visibility_contained_internal and sealed_contained_internal
            ),
            "shell_envelope_min": shell_min.tolist(),
            "shell_envelope_max": shell_max.tolist(),
            "shell_envelope_diagonal": shell_diag,
            "removal_classification": removal_classification,
            "removal_reason": removal_reason,
            "automatic_remove_candidate": removal_classification is not None,
            "protected_from_automatic_removal": removal_classification is None,
            "hard_keep_reasons": hard_keep_reasons,
            "boundary_region_count": 0,
        }
    return result


def normalized_component_thresholds(values: dict[str, float] | None) -> dict[str, float]:
    result = dict(DEFAULT_COMPONENT_THRESHOLDS)
    for key, value in (values or {}).items():
        if key not in result:
            continue
        numeric = float(value)
        if numeric < 0.0:
            raise ValueError(f"component threshold {key} must be non-negative")
        result[key] = numeric
    return result


def normalized_boundary_thresholds(values: dict[str, float] | None) -> dict[str, float]:
    result = dict(DEFAULT_BOUNDARY_THRESHOLDS)
    unknown = sorted(set(values or {}) - set(result))
    if unknown:
        raise ValueError(f"unknown boundary threshold(s): {', '.join(unknown)}")
    for key, value in (values or {}).items():
        numeric = float(value)
        if not np.isfinite(numeric) or numeric < 0.0:
            raise ValueError(f"boundary threshold {key} must be finite and non-negative")
        result[key] = numeric
    if result["pair_min_perimeter_ratio"] > result["pair_max_perimeter_ratio"]:
        raise ValueError("pair_min_perimeter_ratio must not exceed pair_max_perimeter_ratio")
    for key in (
        "small_max_component_boundary_count",
        "small_polygon_max_edge_count",
        "semantic_max_component_boundary_count",
        "pair_search_neighbors",
    ):
        if result[key] < 1.0:
            raise ValueError(f"boundary threshold {key} must be at least one")
        if not result[key].is_integer():
            raise ValueError(f"boundary threshold {key} must be an integer count")
    return result


def component_id_for_region(
    edges: list[tuple[int, int]],
    edge_to_faces: dict[tuple[int, int], list[int]],
    component_ids: np.ndarray,
) -> int | None:
    for edge in edges:
        face_ids = edge_to_faces.get(tuple(sorted(edge)), [])
        if face_ids:
            return int(component_ids[face_ids[0]])
    return None


def empty_component_stats() -> dict[str, Any]:
    return {
        "face_count": 0,
        "face_ratio": 0.0,
        "touches_global_bbox": False,
        "strictly_contained": False,
        "outside_shell_envelope": False,
        "shell_envelope_distance": 0.0,
        "visible_face_fraction": None,
        "visibility_label": "visibility_unavailable",
        "visible_hard_keep": False,
        "sealed_exterior_face_fraction": None,
        "exterior_confidence": 0.0,
        "bbox_min": [],
        "bbox_max": [],
        "bbox_diagonal": 0.0,
        "bbox_diagonal_ratio": 0.0,
        "projected_bbox_area": 0.0,
        "projected_bbox_area_ratio": 0.0,
        "physical_scale_small": False,
        "direct_and_sealed_internal_consensus": False,
        "removal_classification": None,
        "removal_reason": None,
        "automatic_remove_candidate": False,
        "protected_from_automatic_removal": True,
        "hard_keep_reasons": [
            "visibility_evidence_unavailable",
            "sealed_exterior_evidence_unavailable",
        ],
        "boundary_region_count": 0,
    }


def bbox_diagonal(points: np.ndarray) -> float:
    if points.size == 0:
        return 0.0
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
