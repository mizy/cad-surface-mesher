from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import numpy as np

from source_primary_boundary_geometry import (
    build_local_curvature_samples,
    compute_face_geometry,
    describe_boundary_geometry,
)
from source_primary_boundary_relations import (
    classify_loop_relation,
    nested_groups,
    select_nesting,
    select_unambiguous_pairs,
)
from source_primary_boundary_topology import (
    SCHEMA_VERSION,
    build_edge_topology,
    canonical_cycle,
    extract_boundary_graphs,
    face_components,
    geometry_fingerprint,
    mesh_fingerprint,
    normalized_thresholds,
    stable_id,
    topology_diagnostics,
    validated_component_ids,
    validated_inputs,
)


# @entry
def build_source_primary_boundary_inventory(
    points: np.ndarray,
    faces: np.ndarray,
    source_triangle_indices: np.ndarray | None = None,
    *,
    source_vertex_ids: np.ndarray | None = None,
    face_component_ids: np.ndarray | None = None,
    face_external_directions: np.ndarray | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Build a complete, deterministic inventory without mutating source geometry."""
    supplied = {
        "source_triangle_indices": source_triangle_indices is not None,
        "source_vertex_ids": source_vertex_ids is not None,
        "face_component_ids": face_component_ids is not None,
        "face_external_directions": face_external_directions is not None,
    }
    points, faces, source_triangles, source_vertices, external = validated_inputs(
        points,
        faces,
        source_triangle_indices,
        source_vertex_ids,
        face_external_directions,
    )
    config = normalized_thresholds(points, thresholds)
    topology = build_edge_topology(faces)
    components = (
        validated_component_ids(face_component_ids, faces.shape[0])
        if face_component_ids is not None
        else face_components(faces.shape[0], topology)
    )
    _, face_normals, face_areas = compute_face_geometry(points, faces)
    boundary_rows, graph_diagnostics = extract_boundary_graphs(
        points,
        topology,
        source_triangles,
        source_vertices,
        components,
        config["absolute_tolerance"],
    )
    boundary_faces = np.asarray(
        sorted({face_id for row in boundary_rows for face_id in row["incident_face_ids"]}),
        dtype=np.int64,
    )
    curvature = build_local_curvature_samples(
        points,
        face_normals,
        topology["manifold_edges"],
        topology["manifold_face_pairs"],
        boundary_faces,
    )
    loops, anomalies = _describe_graphs(
        points,
        boundary_rows,
        graph_diagnostics,
        face_normals,
        face_areas,
        curvature,
        external,
        config,
    )
    regions, relationships, relationship_diagnostics = _aggregate_regions(
        points,
        loops,
        anomalies,
        face_normals,
        face_areas,
        curvature,
        external,
        config,
    )
    topology_diagnostics_result = topology_diagnostics(topology, source_triangles, source_vertices)
    return {
        "schema_version": SCHEMA_VERSION,
        "method": "incidence_one_face_oriented_boundary_graph_and_explicit_hole_relations",
        "geometry_truth_complete": True,
        "input_evidence": supplied,
        "source_mesh_fingerprint": mesh_fingerprint(points, faces, source_triangles, source_vertices),
        "id_contract": {
            "algorithm": "sha256",
            "classification_in_identity": False,
            "basis": "canonical source vertex cycle or edge set plus adjacent source triangle ids",
        },
        "mutation_policy": {
            "source_points_mutated": False,
            "source_faces_removed": False,
            "boundary_vertex_policy": "fixed_and_reuse_original_indices",
            "allowed_future_change": "append_vertices_and_faces_inside_inventory_region_only",
        },
        "parameters": config,
        "summary": {
            "point_count": int(points.shape[0]),
            "face_count": int(faces.shape[0]),
            "boundary_edge_count": int(topology["boundary_edges"].shape[0]),
            "ordered_loop_count": len(loops),
            "anomalous_boundary_graph_count": len(anomalies),
            "boundary_region_count": len(regions),
            "hole_region_count": sum(
                region["classification"]
                in {"single_loop_hole", "narrow_slit", "multi_loop_or_inner_island_hole"}
                for region in regions
            ),
            "paired_boundary_region_count": sum(
                region["classification"]
                in {"near_coincident_part_seam", "compatible_paired_boundary_loops"}
                for region in regions
            ),
            "non_manifold_edge_count": len(topology_diagnostics_result["non_manifold_edges"]),
            "inconsistent_winding_edge_count": len(topology_diagnostics_result["inconsistent_winding_edges"]),
        },
        "topology_diagnostics": topology_diagnostics_result,
        "boundary_graph_diagnostics": graph_diagnostics,
        "loops": loops,
        "relationships": relationships,
        "relationship_diagnostics": relationship_diagnostics,
        "regions": regions,
    }


def serialize_source_primary_boundary_inventory(inventory: Mapping[str, Any]) -> str:
    """Return the canonical durable JSON representation; file ownership stays with the caller."""
    return json.dumps(inventory, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _describe_graphs(
    points: np.ndarray,
    rows: Sequence[dict[str, Any]],
    diagnostics: Sequence[dict[str, Any]],
    face_normals: np.ndarray,
    face_areas: np.ndarray,
    curvature: Mapping[int, Sequence[tuple[int, float, float]]],
    external: np.ndarray | None,
    config: Mapping[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    invalid_ids = {row["graph_id"] for row in diagnostics}
    diagnostic_by_id = {row["graph_id"]: row for row in diagnostics}
    loops, anomalies = [], []
    for row in rows:
        edges = [tuple(edge) for edge in row["boundary_edges"]]
        if row["graph_id"] in invalid_ids:
            anomalies.append({
                **row,
                "reason_codes": diagnostic_by_id[row["graph_id"]]["reason_codes"],
                "geometry": describe_boundary_geometry(points, edges, [], row["incident_face_ids"], face_normals, face_areas, curvature, external, "anomalous", config),
            })
            continue
        ordered = [int(value) for value in row["ordered_vertex_ids"]]
        canonical_source = canonical_cycle(row["ordered_source_vertex_ids"])
        source_to_vertex = dict(zip(row["ordered_source_vertex_ids"], ordered, strict=True))
        canonical = [source_to_vertex[value] for value in canonical_source]
        loop_id = stable_id("boundary_loop", {
            "source_vertices": canonical_source,
            "source_triangles": sorted(row["source_triangle_ids"]),
        })
        loops.append({
            **row,
            "loop_id": loop_id,
            "simple_closed_loop": True,
            "edge_count": len(edges),
            "vertex_count": len(ordered),
            "incident_face_count": len(row["incident_face_ids"]),
            "source_triangle_count": len(row["source_triangle_ids"]),
            "source_winding_vertex_ids": ordered,
            "canonical_vertex_ids": canonical,
            "canonical_source_vertex_ids": canonical_source,
            "patch_boundary_vertex_ids": list(reversed(ordered)),
            "orientation_status": "source_face_winding_consistent",
            "geometry_fingerprint": geometry_fingerprint(points[np.asarray(canonical)], config["absolute_tolerance"]),
            "geometry": describe_boundary_geometry(points, edges, [ordered], row["incident_face_ids"], face_normals, face_areas, curvature, external, "single", config),
        })
    return sorted(loops, key=lambda item: item["loop_id"]), sorted(anomalies, key=lambda item: item["graph_id"])


def _aggregate_regions(
    points: np.ndarray,
    loops: Sequence[dict[str, Any]],
    anomalies: Sequence[dict[str, Any]],
    face_normals: np.ndarray,
    face_areas: np.ndarray,
    curvature: Mapping[int, Sequence[tuple[int, float, float]]],
    external: np.ndarray | None,
    config: Mapping[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    relations = []
    for left in range(len(loops)):
        for right in range(left + 1, len(loops)):
            relation = classify_loop_relation(points, loops[left], loops[right], config)
            if relation is not None:
                relations.append(relation)
    relations.sort(key=lambda item: (item["kind"], item["score"], item["left_loop_id"], item["right_loop_id"]))
    pairs, ambiguous, pair_diagnostics = select_unambiguous_pairs(relations, config)
    consumed = {loop_id for pair in pairs for loop_id in (pair["left_loop_id"], pair["right_loop_id"])}
    nesting, nesting_ambiguous = select_nesting(relations, consumed)
    ambiguous.update(nesting_ambiguous)
    nesting_groups = nested_groups(loops, nesting, consumed | ambiguous)
    grouped_ids = consumed | {loop_id for group in nesting_groups for loop_id in group}
    by_id = {loop["loop_id"]: loop for loop in loops}
    groups: list[tuple[list[dict[str, Any]], str, list[dict[str, Any]]]] = []
    for pair in pairs:
        groups.append(([by_id[pair["left_loop_id"]], by_id[pair["right_loop_id"]]], pair["kind"], [pair]))
    for group in nesting_groups:
        evidence = [row for row in nesting if row["left_loop_id"] in group and row["right_loop_id"] in group]
        groups.append(([by_id[loop_id] for loop_id in sorted(group)], "nested_loops", evidence))
    for loop in loops:
        if loop["loop_id"] not in grouped_ids:
            kind = "ambiguous_boundary_relationship" if loop["loop_id"] in ambiguous else "single"
            groups.append(([loop], kind, []))
    regions = [
        _make_region(points, group, kind, evidence, face_normals, face_areas, curvature, external, config)
        for group, kind, evidence in groups
    ]
    regions.extend(_make_anomaly_region(row) for row in anomalies)
    regions.sort(key=lambda item: item["region_id"])
    for index, region in enumerate(regions, start=1):
        region["inventory_index"] = index
    relationship_diagnostics = pair_diagnostics + [
        {"reason_code": "ambiguous_nested_loop_ownership", "loop_id": loop_id}
        for loop_id in sorted(nesting_ambiguous)
    ]
    accepted_keys = {(row["kind"], row["left_loop_id"], row["right_loop_id"]) for row in pairs + nesting}
    exported = [{**row, "accepted": (row["kind"], row["left_loop_id"], row["right_loop_id"]) in accepted_keys} for row in relations]
    return regions, exported, relationship_diagnostics


def _make_region(
    points: np.ndarray,
    loops: Sequence[dict[str, Any]],
    relation_kind: str,
    relations: Sequence[dict[str, Any]],
    face_normals: np.ndarray,
    face_areas: np.ndarray,
    curvature: Mapping[int, Sequence[tuple[int, float, float]]],
    external: np.ndarray | None,
    config: Mapping[str, float],
) -> dict[str, Any]:
    loop_ids = sorted(loop["loop_id"] for loop in loops)
    edges = sorted({tuple(edge) for loop in loops for edge in loop["boundary_edges"]})
    face_ids = sorted({value for loop in loops for value in loop["incident_face_ids"]})
    source_by_face = {
        row["face_id"]: row["source_triangle_id"]
        for loop in loops
        for row in loop["source_face_provenance"]
    }
    source_ids = [source_by_face[value] for value in face_ids]
    edge_records = {
        tuple(row["edge"]): row
        for loop in loops
        for row in loop["boundary_edge_records"]
    }
    ordered = [loop["ordered_vertex_ids"] for loop in loops]
    geometry = describe_boundary_geometry(points, edges, ordered, face_ids, face_normals, face_areas, curvature, external, relation_kind, config)
    classification = _region_classification(relation_kind, geometry, config)
    blockers = _repair_blockers(classification, geometry, config)
    if classification == "multi_loop_or_inner_island_hole" and len(loops) != 2:
        blockers.append("multi_loop_requires_exactly_two_nested_loops")
    if (
        classification in {"near_coincident_part_seam", "compatible_paired_boundary_loops"}
        and relations
        and min(float(row.get("polyline_hausdorff", 0.0)) for row in relations)
        <= config["absolute_tolerance"]
    ):
        blockers.append("zero_width_pair_requires_forbidden_source_connectivity_change")
    blockers = sorted(set(blockers))
    return {
        "region_id": stable_id("hole_region", {"loop_ids": loop_ids}),
        "region_id_basis": "sorted durable loop ids",
        "classification": classification,
        "loop_count": len(loops),
        "edge_count": len(edges),
        "loop_ids": loop_ids,
        "ordered_boundary_loops": ordered,
        "patch_boundary_loops": [loop["patch_boundary_vertex_ids"] for loop in loops],
        "boundary_edges": [list(edge) for edge in edges],
        "boundary_edge_records": [edge_records[edge] for edge in edges],
        "boundary_vertex_ids": sorted({value for edge in edges for value in edge}),
        "incident_face_ids": face_ids,
        "incident_face_count": len(face_ids),
        "source_triangle_ids": source_ids,
        "source_triangle_count": len(source_ids),
        "source_face_provenance": [
            {"face_id": face_id, "source_triangle_id": source_by_face[face_id]}
            for face_id in face_ids
        ],
        "component_ids": sorted({value for loop in loops for value in loop["component_ids"]}),
        "relations": list(relations),
        "recommended_operator": _recommended_operator(classification, geometry, config),
        "patch_eligible": not blockers,
        "blocking_reason_codes": blockers,
        "source_boundary_policy": "fixed_vertices_reuse_source_indices_no_exterior_face_deletion",
        **geometry,
    }


def _make_anomaly_region(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "region_id": stable_id("hole_region", {"boundary_graph_id": row["graph_id"]}),
        "region_id_basis": "durable anomalous boundary graph id",
        "classification": "anomalous_boundary_graph",
        "loop_count": 0,
        "edge_count": len(row["boundary_edges"]),
        "loop_ids": [],
        "ordered_boundary_loops": [],
        "patch_boundary_loops": [],
        "boundary_edges": row["boundary_edges"],
        "boundary_edge_records": row["boundary_edge_records"],
        "boundary_vertex_ids": sorted({value for edge in row["boundary_edges"] for value in edge}),
        "incident_face_ids": row["incident_face_ids"],
        "incident_face_count": len(row["incident_face_ids"]),
        "source_triangle_ids": row["source_triangle_ids"],
        "source_triangle_count": len(row["source_triangle_ids"]),
        "source_face_provenance": row["source_face_provenance"],
        "component_ids": row["component_ids"],
        "relations": [],
        "recommended_operator": None,
        "patch_eligible": False,
        "blocking_reason_codes": row["reason_codes"],
        "source_boundary_policy": "diagnose_only_no_surface_deletion",
        **row["geometry"],
    }


def _region_classification(
    relation_kind: str,
    geometry: Mapping[str, Any],
    config: Mapping[str, float],
) -> str:
    if relation_kind in {"near_coincident_part_seam", "compatible_paired_boundary_loops", "ambiguous_boundary_relationship"}:
        return relation_kind
    if relation_kind == "nested_loops":
        return "multi_loop_or_inner_island_hole"
    if geometry["normals"]["boundary_side"]["status"] != "missing_surface_inside_loop":
        return "unresolved_open_perimeter"
    scale = geometry["scale"]
    width_ratio = scale["estimated_width"] / max(scale["major_extent"], 1e-30)
    major_edge_ratio = scale["major_extent"] / max(scale["median_edge_length"], 1e-30)
    if width_ratio <= config["slit_max_width_major_ratio"] and scale["aspect_ratio"] >= config["slit_min_aspect_ratio"] and major_edge_ratio >= config["slit_min_major_edge_ratio"]:
        return "narrow_slit"
    return "single_loop_hole"


def _repair_blockers(
    classification: str,
    geometry: Mapping[str, Any],
    config: Mapping[str, float],
) -> list[str]:
    blockers = []
    if classification == "ambiguous_boundary_relationship":
        blockers.append("ambiguous_boundary_relationship")
    if classification == "unresolved_open_perimeter":
        blockers.append("boundary_not_proven_to_enclose_missing_surface")
    if geometry["normals"]["oriented"]["resolved"] is not True:
        blockers.append(str(geometry["normals"]["oriented"]["status"]))
    source_normal = geometry["normals"]["area_weighted_source"]
    if not source_normal["consistent"] or source_normal["resultant_ratio"] < config["normal_consistency_min_resultant"]:
        blockers.append("adjacent_source_face_normal_inconsistency")
    if geometry["scale"]["perimeter"] <= config["absolute_tolerance"]:
        blockers.append("zero_length_boundary")
    if geometry["planes"]["pca"]["status"] != "computed":
        blockers.append("pca_plane_normal_degenerate")
    if (
        classification not in {"near_coincident_part_seam", "compatible_paired_boundary_loops"}
        and geometry["planes"]["newell"]["status"] != "computed"
    ):
        blockers.append("newell_plane_degenerate")
    return sorted(set(blockers))


def _recommended_operator(
    classification: str,
    geometry: Mapping[str, Any],
    config: Mapping[str, float],
) -> str | None:
    if classification == "near_coincident_part_seam":
        return "fixed_boundary_seam_bridge"
    if classification == "compatible_paired_boundary_loops":
        return "paired_loop_zipper"
    if classification == "narrow_slit":
        return "fixed_boundary_slit_bridge"
    if classification == "multi_loop_or_inner_island_hole":
        return "multi_loop_constrained_patch"
    if classification != "single_loop_hole":
        return None
    curvature = geometry["local_curvature"]
    planar = (
        geometry["planarity"]["max_distance_scale_ratio"]
        <= config["planar_max_distance_scale_ratio"]
        and curvature["status"] == "computed"
        and curvature["dimensionless_rms"]
        <= config["planar_max_curvature_dimensionless_rms"]
    )
    return "planar_cap" if planar else "curved_conformal_patch"
