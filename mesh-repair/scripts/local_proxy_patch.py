from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from hybrid_proxy_geometry import (
    FACE_ORIGIN,
    conformal_loop_stitch,
    extract_ordered_boundary_loops,
)
from mesh_metrics import (
    edge_topology,
    face_component_labels,
    inconsistent_winding_edges,
    self_intersection_report,
    triangle_quality,
)


@dataclass(frozen=True)
class LocalProxyPatchConfig:
    """Conservative limits for a source-locked local proxy patch.

    The proxy is treated as the zero surface of the closure SDF.  Selection is
    topological: complete, edge-connected proxy triangles are cropped by the
    source loop's local prism.  No unordered nearest-point cloud is ever built.
    """

    inset_fraction: float = 0.10
    min_inset_fraction: float = 0.025
    inset_attempts: int = 4
    max_normal_distance_ratio: float = 0.35
    max_source_nonplanarity_ratio: float = 0.35
    min_projected_area_ratio: float = 0.02
    max_projected_area_ratio: float = 0.98
    min_normal_alignment: float = 0.20
    min_projected_orientation_consistency: float = 0.95
    min_inner_ring_clearance_ratio: float = 0.005
    max_correspondence_distance_ratio: float = 0.80
    ambiguity_score_margin: float = 0.02
    max_candidate_components: int = 128
    max_ring_vertices: int = 8192
    phase_samples: int = 128
    min_adjacent_normal_dot: float = -0.5
    normal_score_weight: float = 0.25
    check_self_intersections: bool = True
    max_self_intersection_candidate_pairs: int = 250_000


# @entry
def build_source_locked_proxy_patch(
    source_points: np.ndarray,
    source_faces: np.ndarray,
    source_loop: np.ndarray,
    proxy_points: np.ndarray,
    proxy_faces: np.ndarray,
    *,
    config: LocalProxyPatchConfig | None = None,
    source_triangle_indices: np.ndarray | None = None,
    region_id: int = 0,
) -> dict[str, Any]:
    """Extract and stitch one local closure-proxy patch into a source loop.

    ``source_loop`` is an ordered outer boundary of ``source_faces``.  The
    function accepts the *complete* closure proxy, crops a connected disk from
    its local SDF-zero-surface neighbourhood, extracts exactly one ordered inner
    ring, and invokes the existing deterministic arc-length/conformal annulus
    primitive.  It is transactional: every failure returns empty consumable
    geometry.
    """

    cfg = config or LocalProxyPatchConfig()
    configuration_failure = _validate_config(cfg)
    if configuration_failure is not None:
        return _failure(
            "local_proxy_patch_config_invalid",
            "configuration",
            configuration_failure,
            config=asdict(cfg),
        )
    source_validation = _validate_mesh(source_points, source_faces, "source")
    if source_validation is not None:
        return _failure(source_validation[0], "input_validation", source_validation[1])
    proxy_validation = _validate_mesh(proxy_points, proxy_faces, "proxy")
    if proxy_validation is not None:
        return _failure(proxy_validation[0], "input_validation", proxy_validation[1])

    source_points = np.asarray(source_points, dtype=np.float64)
    source_faces = np.asarray(source_faces, dtype=np.int64)
    proxy_points = np.asarray(proxy_points, dtype=np.float64)
    proxy_faces = np.asarray(proxy_faces, dtype=np.int64)
    loop_validation = _validate_source_loop(source_points, source_faces, source_loop)
    if not loop_validation["success"]:
        return _failure(
            loop_validation["failure_reason_codes"][0],
            "source_loop_validation",
            loop_validation["diagnostics"]["message"],
            source_loop_validation=loop_validation["diagnostics"],
        )
    source_loop = loop_validation["loop"]

    source_indices = _source_triangle_indices(
        source_triangle_indices, source_faces.shape[0]
    )
    if source_indices is None:
        return _failure(
            "source_triangle_indices_shape_invalid",
            "input_validation",
            "source_triangle_indices must have one integer value per source face.",
        )

    extraction = extract_local_proxy_patch(
        source_points,
        source_loop,
        proxy_points,
        proxy_faces,
        config=cfg,
    )
    if not extraction["success"]:
        return _failure(
            extraction["failure_reason_codes"][0],
            "proxy_patch_extraction",
            extraction["diagnostics"]["message"],
            extraction=extraction["diagnostics"],
        )

    characteristic_length = float(
        extraction["diagnostics"]["local_frame"]["footprint_diagonal"]
    )
    correspondence_limit = max(
        characteristic_length * cfg.max_correspondence_distance_ratio,
        extraction["diagnostics"]["geometric_tolerance"] * 10.0,
    )
    stitch = conformal_loop_stitch(
        source_points,
        source_faces,
        extraction["proxy_patch_points"],
        extraction["proxy_patch_faces"],
        source_loop=source_loop,
        proxy_loop=extraction["proxy_inner_loop"],
        geometric_tolerance=extraction["diagnostics"]["geometric_tolerance"],
        max_ring_vertices=cfg.max_ring_vertices,
        phase_samples=cfg.phase_samples,
        max_correspondence_distance=correspondence_limit,
        min_adjacent_normal_dot=cfg.min_adjacent_normal_dot,
        normal_score_weight=cfg.normal_score_weight,
    )
    if not stitch["success"]:
        return _failure(
            stitch["failure_reason_codes"][0],
            "source_locked_local_remesh",
            stitch["diagnostics"].get("message", "Conformal local remesh failed."),
            extraction=extraction["diagnostics"],
            stitch=stitch["diagnostics"],
        )

    source_locked = _source_lock_report(source_points, source_loop, stitch["points"])
    if not source_locked["passed"]:
        return _failure(
            "source_boundary_lock_failed",
            "source_locked_local_remesh",
            "The local remesh moved at least one original source boundary vertex.",
            extraction=extraction["diagnostics"],
            stitch=stitch["diagnostics"],
            source_boundary_lock=source_locked,
        )

    topology_gate = _topology_gate(
        source_faces,
        stitch["faces"],
        source_loop_edge_count=int(source_loop.size),
    )
    if not topology_gate["passed"]:
        return _failure(
            topology_gate["failure_reason_codes"][0],
            "source_locked_local_remesh",
            "The stitched local proxy patch failed the topology transaction gate.",
            extraction=extraction["diagnostics"],
            stitch=stitch["diagnostics"],
            source_boundary_lock=source_locked,
            topology_gate=topology_gate,
        )

    generated_face_ids = np.flatnonzero(stitch["face_origin"] != FACE_ORIGIN["source"])
    local_self_intersection: dict[str, Any]
    if cfg.check_self_intersections:
        try:
            local_self_intersection = self_intersection_report(
                stitch["points"],
                stitch["faces"],
                focus_face_ids=generated_face_ids,
                max_candidate_pairs=cfg.max_self_intersection_candidate_pairs,
            )
        # Keep optional VTK/runtime failures transactional as well.
        except Exception as exc:  # pragma: no cover
            return _failure(
                "local_self_intersection_check_failed",
                "source_locked_local_remesh",
                f"Local self-intersection validation could not be completed: {type(exc).__name__}: {exc}",
                extraction=extraction["diagnostics"],
                stitch=stitch["diagnostics"],
                source_boundary_lock=source_locked,
                topology_gate=topology_gate,
            )
        if not local_self_intersection["passed"]:
            code = (
                "local_proxy_patch_self_intersection_detected"
                if local_self_intersection.get("status") == "computed"
                else "local_proxy_patch_self_intersection_check_incomplete"
            )
            return _failure(
                code,
                "source_locked_local_remesh",
                "The generated proxy patch or annular bridge intersects non-adjacent geometry.",
                extraction=extraction["diagnostics"],
                stitch=stitch["diagnostics"],
                source_boundary_lock=source_locked,
                topology_gate=topology_gate,
                local_self_intersection=local_self_intersection,
            )
    else:
        local_self_intersection = {
            "method": "vtk_static_cell_locator_triangle_intersection",
            "status": "disabled_by_config",
            "passed": True,
        }

    proxy_face_parent = _map_proxy_face_parents(
        stitch["proxy_face_parent"],
        extraction["proxy_face_ids"],
    )
    provenance = _build_provenance(
        stitch,
        source_indices,
        proxy_face_parent,
        int(region_id),
    )
    diagnostics = {
        "stage": "source_locked_local_proxy_patch",
        "method": "local_proxy_disk_crop_arc_length_resample_annular_bridge",
        "config": asdict(cfg),
        "selection_contract": {
            "proxy_role": "closure_sdf_zero_surface",
            "selection_primitive": "edge_connected_proxy_faces_in_source_loop_local_prism",
            "nearest_point_scatter_used": False,
            "requires_exactly_one_simple_inner_loop": True,
        },
        "extraction": extraction["diagnostics"],
        "stitch": stitch["diagnostics"],
        "source_boundary_lock": source_locked,
        "topology_gate": topology_gate,
        "local_self_intersection": local_self_intersection,
        "provenance": {
            "source_triangle_indices_provided": source_triangle_indices is not None,
            "source_parent_faces": int(
                np.count_nonzero(stitch["source_face_parent"] >= 0)
            ),
            "proxy_parent_faces": int(np.count_nonzero(proxy_face_parent >= 0)),
            "stitch_faces": int(stitch["stitch_face_ids"].size),
            "region_id": int(region_id),
        },
    }
    return {
        "success": True,
        "accepted": True,
        "failure_reason_codes": [],
        "points": stitch["points"],
        "faces": stitch["faces"],
        "face_origin": stitch["face_origin"],
        "source_face_parent": stitch["source_face_parent"],
        "proxy_face_parent": proxy_face_parent,
        "stitch_face_ids": stitch["stitch_face_ids"],
        "source_ring_vertex_ids": stitch["source_ring_vertex_ids"],
        "proxy_ring_vertex_ids": stitch["proxy_ring_vertex_ids"],
        "proxy_face_ids": extraction["proxy_face_ids"],
        "proxy_point_ids": extraction["proxy_point_ids"],
        "proxy_patch_points": extraction["proxy_patch_points"],
        "proxy_patch_faces": extraction["proxy_patch_faces"],
        "proxy_inner_loop": extraction["proxy_inner_loop"],
        "proxy_inner_loop_original_point_ids": extraction[
            "proxy_inner_loop_original_point_ids"
        ],
        "provenance": provenance,
        "diagnostics": diagnostics,
    }


def extract_local_proxy_patch(
    source_points: np.ndarray,
    source_loop: np.ndarray,
    proxy_points: np.ndarray,
    proxy_faces: np.ndarray,
    *,
    config: LocalProxyPatchConfig | None = None,
) -> dict[str, Any]:
    """Crop one connected proxy disk from a complete closure proxy surface."""

    cfg = config or LocalProxyPatchConfig()
    source_points = np.asarray(source_points, dtype=np.float64)
    source_loop = np.asarray(source_loop, dtype=np.int64).reshape(-1)
    proxy_points = np.asarray(proxy_points, dtype=np.float64)
    proxy_faces = np.asarray(proxy_faces, dtype=np.int64)
    frame = _source_loop_frame(source_points[source_loop], cfg)
    if not frame["success"]:
        return _extraction_failure(
            frame["failure_reason_codes"][0],
            frame["diagnostics"]["message"],
            local_frame=frame["diagnostics"],
        )

    center = frame["center"]
    basis = frame["basis"]
    proxy_local = (proxy_points - center) @ basis.T
    triangles_local = proxy_local[proxy_faces]
    face_centroids_local = triangles_local.mean(axis=1)
    face_depth_max = np.max(np.abs(triangles_local[:, :, 2]), axis=1)
    polygon = frame["polygon"]
    uv_min = polygon.min(axis=0)
    uv_max = polygon.max(axis=0)
    geometric_tolerance = float(frame["diagnostics"]["geometric_tolerance"])
    depth_limit = float(frame["depth_limit"])
    local_bbox_mask = (
        np.all(face_centroids_local[:, :2] >= uv_min - geometric_tolerance, axis=1)
        & np.all(face_centroids_local[:, :2] <= uv_max + geometric_tolerance, axis=1)
        & (face_depth_max <= depth_limit)
    )
    bbox_face_ids = np.flatnonzero(local_bbox_mask)
    _, global_proxy_edge_faces = edge_topology(proxy_faces)
    schedule = _inset_schedule(cfg)
    attempts: list[dict[str, Any]] = []
    observed_rejections: list[str] = []
    selected: dict[str, Any] | None = None

    for inset_fraction in schedule:
        inset_distance = float(frame["footprint_min_span"] * inset_fraction)
        face_ids = _faces_in_polygon_prism(
            proxy_local,
            proxy_faces,
            face_centroids_local,
            bbox_face_ids,
            polygon,
            inset_distance,
            geometric_tolerance,
        )
        attempt: dict[str, Any] = {
            "inset_fraction": float(inset_fraction),
            "inset_distance": inset_distance,
            "bbox_candidate_faces": int(bbox_face_ids.size),
            "cropped_face_count": int(face_ids.size),
            "component_count": 0,
            "valid_component_count": 0,
            "components": [],
        }
        if face_ids.size == 0:
            attempt["reason_codes"] = ["proxy_patch_local_neighborhood_empty"]
            observed_rejections.append("proxy_patch_local_neighborhood_empty")
            attempts.append(attempt)
            continue

        components = _selected_face_components(proxy_faces, face_ids)
        attempt["component_count"] = len(components)
        if len(components) > cfg.max_candidate_components:
            attempt["reason_codes"] = ["proxy_patch_component_limit_exceeded"]
            observed_rejections.append("proxy_patch_component_limit_exceeded")
            attempts.append(attempt)
            continue

        valid_components = []
        for component_index, component_face_ids in enumerate(components):
            component = _evaluate_proxy_component(
                proxy_points,
                proxy_faces,
                component_face_ids,
                proxy_local,
                polygon,
                frame,
                global_proxy_edge_faces,
                cfg,
                component_index,
            )
            attempt["components"].append(component["diagnostics"])
            observed_rejections.extend(component["diagnostics"]["reason_codes"])
            if component["valid"]:
                valid_components.append(component)
        attempt["valid_component_count"] = len(valid_components)
        attempts.append(attempt)
        if not valid_components:
            continue

        valid_components.sort(
            key=lambda row: (
                -float(row["score"]),
                int(row["face_ids"][0]),
            )
        )
        if (
            len(valid_components) > 1
            and float(valid_components[0]["score"] - valid_components[1]["score"])
            <= cfg.ambiguity_score_margin
        ):
            return _extraction_failure(
                "proxy_patch_component_ambiguous",
                "Multiple local proxy disks have indistinguishable deterministic scores.",
                local_frame=frame["diagnostics"],
                local_prism={
                    "depth_limit": depth_limit,
                    "bbox_candidate_faces": int(bbox_face_ids.size),
                },
                attempts=attempts,
                ambiguity={
                    "score_margin": float(
                        valid_components[0]["score"] - valid_components[1]["score"]
                    ),
                    "threshold": cfg.ambiguity_score_margin,
                    "component_face_ids": [
                        int(valid_components[0]["face_ids"][0]),
                        int(valid_components[1]["face_ids"][0]),
                    ],
                },
            )
        selected = valid_components[0]
        attempt["selected_component_index"] = int(selected["component_index"])
        attempt["selected_score"] = float(selected["score"])
        break

    if selected is None:
        failure_code = _aggregate_extraction_failure(observed_rejections)
        return _extraction_failure(
            failure_code,
            "No cropped proxy component passed the single-disk and geometric trust gates.",
            local_frame=frame["diagnostics"],
            local_prism={
                "method": "source_loop_oriented_bbox_and_normal_depth_gate",
                "depth_limit": depth_limit,
                "bbox_candidate_faces": int(bbox_face_ids.size),
                "proxy_surface_role": "closure_sdf_zero_surface",
                "nearest_point_scatter_used": False,
            },
            attempts=attempts,
        )

    diagnostics = {
        "stage": "local_proxy_patch_extraction",
        "message": "One edge-connected proxy disk was selected from the source-loop local prism.",
        "method": "source_loop_local_prism_connected_face_crop",
        "geometric_tolerance": geometric_tolerance,
        "local_frame": frame["diagnostics"],
        "local_prism": {
            "method": "source_loop_oriented_bbox_and_normal_depth_gate",
            "depth_limit": depth_limit,
            "bbox_candidate_faces": int(bbox_face_ids.size),
            "proxy_surface_role": "closure_sdf_zero_surface",
            "nearest_point_scatter_used": False,
        },
        "attempts": attempts,
        "selected_component": selected["diagnostics"],
        "ordered_inner_loop": {
            "loop_count": 1,
            "vertex_count": int(selected["inner_loop"].size),
            "simple": True,
            "face_winding_induced": True,
        },
    }
    return {
        "success": True,
        "failure_reason_codes": [],
        "proxy_face_ids": selected["face_ids"],
        "proxy_point_ids": selected["point_ids"],
        "proxy_patch_points": selected["points"],
        "proxy_patch_faces": selected["faces"],
        "proxy_inner_loop": selected["inner_loop"],
        "proxy_inner_loop_original_point_ids": selected["point_ids"][
            selected["inner_loop"]
        ],
        "diagnostics": diagnostics,
    }


def _source_loop_frame(
    loop_points: np.ndarray, config: LocalProxyPatchConfig
) -> dict[str, Any]:
    edges = np.roll(loop_points, -1, axis=0) - loop_points
    lengths = np.linalg.norm(edges, axis=1)
    diagonal = float(np.linalg.norm(loop_points.max(axis=0) - loop_points.min(axis=0)))
    tolerance = max(diagonal * 1e-10, 1e-12)
    if np.any(lengths <= tolerance):
        return _frame_failure(
            "source_loop_zero_length_edge",
            "The ordered source loop contains a zero-length geometric edge.",
            geometric_tolerance=tolerance,
        )
    perimeter = float(lengths.sum())
    center = (
        np.sum(
            ((loop_points + np.roll(loop_points, -1, axis=0)) * 0.5) * lengths[:, None],
            axis=0,
        )
        / perimeter
    )
    relative = loop_points - center
    area_vector = np.cross(relative, np.roll(relative, -1, axis=0)).sum(axis=0)
    area_vector_length = float(np.linalg.norm(area_vector))
    if area_vector_length <= tolerance * tolerance:
        return _frame_failure(
            "source_loop_local_frame_degenerate",
            "The source loop has no stable area vector for a deterministic local frame.",
            geometric_tolerance=tolerance,
        )
    normal = area_vector / area_vector_length
    reference_axis = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(normal)))]
    u_axis = np.cross(reference_axis, normal)
    u_axis /= np.linalg.norm(u_axis)
    v_axis = np.cross(normal, u_axis)
    basis = np.vstack([u_axis, v_axis, normal])
    local = relative @ basis.T
    polygon = local[:, :2]
    intersections = _polygon_self_intersections(polygon, tolerance)
    if intersections:
        return _frame_failure(
            "source_loop_projection_self_intersects",
            "The source loop is not simple in its deterministic local projection.",
            geometric_tolerance=tolerance,
            segment_pairs=intersections[:100],
        )
    signed_area = _polygon_signed_area(polygon)
    uv_spans = np.ptp(polygon, axis=0)
    footprint_diagonal = float(np.linalg.norm(uv_spans))
    footprint_min_span = float(np.min(uv_spans))
    if abs(signed_area) <= tolerance * tolerance or footprint_min_span <= tolerance:
        return _frame_failure(
            "source_loop_projected_area_degenerate",
            "The source loop has insufficient projected area for local proxy extraction.",
            geometric_tolerance=tolerance,
            projected_signed_area=float(signed_area),
        )
    max_source_depth = float(np.max(np.abs(local[:, 2])))
    nonplanarity_ratio = max_source_depth / max(footprint_diagonal, tolerance)
    if nonplanarity_ratio > config.max_source_nonplanarity_ratio:
        return _frame_failure(
            "source_loop_nonplanarity_exceeded",
            "The source loop is too folded for one safe local footprint projection.",
            geometric_tolerance=tolerance,
            nonplanarity_ratio=nonplanarity_ratio,
            threshold=config.max_source_nonplanarity_ratio,
        )
    depth_limit = (
        max_source_depth + footprint_diagonal * config.max_normal_distance_ratio
    )
    return {
        "success": True,
        "failure_reason_codes": [],
        "center": center,
        "basis": basis,
        "polygon": polygon,
        "footprint_min_span": footprint_min_span,
        "depth_limit": depth_limit,
        "diagnostics": {
            "stage": "source_loop_local_frame",
            "method": "perimeter_centroid_newell_normal_world_axis_tangent",
            "center": center.tolist(),
            "u_axis": u_axis.tolist(),
            "v_axis": v_axis.tolist(),
            "normal": normal.tolist(),
            "source_loop_vertices": int(loop_points.shape[0]),
            "perimeter": perimeter,
            "projected_signed_area": float(signed_area),
            "projected_area": float(abs(signed_area)),
            "projected_bbox_min": polygon.min(axis=0).tolist(),
            "projected_bbox_max": polygon.max(axis=0).tolist(),
            "footprint_diagonal": footprint_diagonal,
            "footprint_min_span": footprint_min_span,
            "max_source_normal_depth": max_source_depth,
            "source_nonplanarity_ratio": nonplanarity_ratio,
            "max_source_nonplanarity_ratio": config.max_source_nonplanarity_ratio,
            "geometric_tolerance": tolerance,
        },
    }


def _faces_in_polygon_prism(
    proxy_local: np.ndarray,
    proxy_faces: np.ndarray,
    face_centroids_local: np.ndarray,
    bbox_face_ids: np.ndarray,
    polygon: np.ndarray,
    inset_distance: float,
    tolerance: float,
) -> np.ndarray:
    if bbox_face_ids.size == 0:
        return np.zeros(0, dtype=np.int64)
    centroid_uv = face_centroids_local[bbox_face_ids, :2]
    inside = _points_in_polygon(centroid_uv, polygon, tolerance)
    clearance = _distance_to_polygon_edges(centroid_uv, polygon)
    local_face_ids = bbox_face_ids[inside & (clearance >= inset_distance - tolerance)]
    if local_face_ids.size == 0:
        return np.zeros(0, dtype=np.int64)

    candidate_faces = proxy_faces[local_face_ids]
    unique_vertices = np.unique(candidate_faces)
    vertex_inside_values = _points_in_polygon(
        proxy_local[unique_vertices, :2], polygon, tolerance
    )
    vertex_inside = np.zeros(proxy_local.shape[0], dtype=bool)
    vertex_inside[unique_vertices] = vertex_inside_values
    keep_vertices = np.all(vertex_inside[candidate_faces], axis=1)
    local_face_ids = local_face_ids[keep_vertices]
    if local_face_ids.size == 0:
        return np.zeros(0, dtype=np.int64)

    triangles = proxy_local[proxy_faces[local_face_ids], :2]
    midpoints = np.stack(
        [
            (triangles[:, 0] + triangles[:, 1]) * 0.5,
            (triangles[:, 1] + triangles[:, 2]) * 0.5,
            (triangles[:, 2] + triangles[:, 0]) * 0.5,
        ],
        axis=1,
    )
    keep_midpoints = np.all(
        _points_in_polygon(midpoints.reshape(-1, 2), polygon, tolerance).reshape(-1, 3),
        axis=1,
    )
    return np.asarray(local_face_ids[keep_midpoints], dtype=np.int64)


def _selected_face_components(
    proxy_faces: np.ndarray, face_ids: np.ndarray
) -> list[np.ndarray]:
    selected_faces = proxy_faces[face_ids]
    _, edge_faces = edge_topology(selected_faces)
    labels = face_component_labels(selected_faces.shape[0], edge_faces)
    components = [
        face_ids[np.flatnonzero(labels == label)] for label in np.unique(labels)
    ]
    components.sort(key=lambda values: int(values.min()))
    return components


def _evaluate_proxy_component(
    proxy_points: np.ndarray,
    proxy_faces: np.ndarray,
    face_ids: np.ndarray,
    proxy_local: np.ndarray,
    source_polygon: np.ndarray,
    frame: dict[str, Any],
    global_proxy_edge_faces: dict[tuple[int, int], list[int]],
    config: LocalProxyPatchConfig,
    component_index: int,
) -> dict[str, Any]:
    point_ids, points, faces = _compact_mesh_with_ids(
        proxy_points, proxy_faces, face_ids
    )
    topology, local_edge_faces = edge_topology(faces)
    extraction = extract_ordered_boundary_loops(
        points,
        faces,
        geometric_tolerance=frame["diagnostics"]["geometric_tolerance"],
    )
    reasons: list[str] = []
    loops = extraction.get("loops", [])
    if not extraction["success"]:
        reasons.extend(extraction["failure_reason_codes"])
    elif len(loops) != 1:
        reasons.append("proxy_patch_boundary_loop_count_mismatch")
    if topology["non_manifold_edges"]:
        reasons.append("proxy_patch_non_manifold")
    euler_characteristic = int(points.shape[0] - topology["edges"] + faces.shape[0])
    if euler_characteristic != 1:
        reasons.append("proxy_patch_not_topological_disk")

    boundary_touches_input_boundary = 0
    for edge, incident in local_edge_faces.items():
        if len(incident) != 1:
            continue
        global_edge = tuple(sorted((int(point_ids[edge[0]]), int(point_ids[edge[1]]))))
        if len(global_proxy_edge_faces.get(global_edge, [])) != 2:
            boundary_touches_input_boundary += 1
    if boundary_touches_input_boundary:
        reasons.append("proxy_patch_touches_input_boundary")

    quality = triangle_quality(points, faces)
    if quality["degenerate_faces"]:
        reasons.append("proxy_patch_degenerate_faces")
    local_points = proxy_local[point_ids]
    triangles_uv = local_points[faces, :2]
    signed_double_area = (triangles_uv[:, 1, 0] - triangles_uv[:, 0, 0]) * (
        triangles_uv[:, 2, 1] - triangles_uv[:, 0, 1]
    ) - (triangles_uv[:, 1, 1] - triangles_uv[:, 0, 1]) * (
        triangles_uv[:, 2, 0] - triangles_uv[:, 0, 0]
    )
    area_tolerance = max(
        float(frame["diagnostics"]["projected_area"]) * 1e-14,
        float(frame["diagnostics"]["geometric_tolerance"]) ** 2,
    )
    nonzero_projected = np.abs(signed_double_area) > area_tolerance
    if not np.all(nonzero_projected):
        reasons.append("proxy_patch_projected_triangle_degenerate")
    positive = int(np.count_nonzero(signed_double_area[nonzero_projected] > 0.0))
    negative = int(np.count_nonzero(signed_double_area[nonzero_projected] < 0.0))
    orientation_consistency = max(positive, negative) / max(positive + negative, 1)
    if orientation_consistency < config.min_projected_orientation_consistency:
        reasons.append("proxy_patch_projected_orientation_inconsistent")
    projected_area = float(np.abs(signed_double_area).sum() * 0.5)
    coverage_ratio = projected_area / max(
        float(frame["diagnostics"]["projected_area"]), 1e-30
    )
    if coverage_ratio < config.min_projected_area_ratio:
        reasons.append("proxy_patch_projected_area_too_small")
    if coverage_ratio > config.max_projected_area_ratio:
        reasons.append("proxy_patch_projected_area_too_large")

    triangles = points[faces]
    raw_normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    normal_lengths = np.linalg.norm(raw_normals, axis=1)
    normal_alignment = float(
        np.mean(
            np.divide(
                np.abs(raw_normals @ frame["basis"][2]),
                normal_lengths,
                out=np.zeros_like(normal_lengths),
                where=normal_lengths > 1e-30,
            )
        )
    )
    if normal_alignment < config.min_normal_alignment:
        reasons.append("proxy_patch_normal_alignment_failed")

    inner_loop = (
        loops[0]
        if extraction["success"] and len(loops) == 1
        else np.zeros(0, dtype=np.int64)
    )
    if inner_loop.size:
        boundary_uv = local_points[inner_loop, :2]
        min_clearance = float(
            _distance_to_polygon_edges(boundary_uv, source_polygon).min()
        )
    else:
        min_clearance = 0.0
    required_clearance = float(
        frame["diagnostics"]["footprint_min_span"]
        * config.min_inner_ring_clearance_ratio
    )
    if min_clearance < required_clearance:
        reasons.append("proxy_inner_ring_clearance_too_small")

    centroids_depth = local_points[faces, 2].mean(axis=1)
    depth_center_abs = float(abs(np.median(centroids_depth)))
    depth_span = float(np.ptp(local_points[:, 2]))
    depth_score = depth_center_abs / max(float(frame["depth_limit"]), 1e-30)
    score = float(coverage_ratio + 0.15 * normal_alignment - 0.40 * depth_score)
    reasons = list(dict.fromkeys(reasons))
    diagnostics = {
        "component_index": int(component_index),
        "first_proxy_face_id": int(face_ids[0]),
        "proxy_face_count": int(face_ids.size),
        "proxy_point_count": int(point_ids.size),
        "valid": not reasons,
        "reason_codes": reasons,
        "topology": {
            "boundary_edges": topology["boundary_edges"],
            "non_manifold_edges": topology["non_manifold_edges"],
            "euler_characteristic": euler_characteristic,
            "boundary_loop_count": len(loops) if extraction["success"] else None,
            "boundary_touches_input_boundary_edges": boundary_touches_input_boundary,
        },
        "geometry": {
            "surface_area": float(quality["surface_area"]),
            "projected_area": projected_area,
            "projected_area_ratio": coverage_ratio,
            "normal_alignment_mean_abs_dot": normal_alignment,
            "projected_orientation_consistency": orientation_consistency,
            "inner_ring_min_clearance": min_clearance,
            "inner_ring_required_clearance": required_clearance,
            "normal_depth_center_abs": depth_center_abs,
            "normal_depth_span": depth_span,
        },
        "score": score,
        "loop_extraction": extraction["diagnostics"],
    }
    return {
        "valid": not reasons,
        "score": score,
        "component_index": component_index,
        "face_ids": np.asarray(face_ids, dtype=np.int64),
        "point_ids": point_ids,
        "points": points,
        "faces": faces,
        "inner_loop": inner_loop,
        "diagnostics": diagnostics,
    }


def _compact_mesh_with_ids(
    points: np.ndarray,
    faces: np.ndarray,
    face_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = faces[face_ids]
    point_ids, inverse = np.unique(selected.ravel(), return_inverse=True)
    compact_points = points[point_ids]
    compact_faces = inverse.reshape((-1, 3)).astype(np.int64, copy=False)
    return point_ids.astype(np.int64, copy=False), compact_points, compact_faces


def _source_lock_report(
    source_points: np.ndarray,
    source_loop: np.ndarray,
    output_points: np.ndarray,
) -> dict[str, Any]:
    retained = output_points[: source_points.shape[0]]
    deltas = np.linalg.norm(retained[source_loop] - source_points[source_loop], axis=1)
    exact = bool(np.array_equal(retained[source_loop], source_points[source_loop]))
    all_source_exact = bool(np.array_equal(retained, source_points))
    return {
        "passed": exact and all_source_exact,
        "method": "exact_coordinate_identity_for_all_original_source_vertices",
        "source_boundary_vertex_count": int(source_loop.size),
        "all_original_source_vertex_count": int(source_points.shape[0]),
        "source_boundary_exactly_locked": exact,
        "all_original_source_vertices_exactly_locked": all_source_exact,
        "source_boundary_max_displacement": float(deltas.max()) if deltas.size else 0.0,
        "source_boundary_tolerance": 0.0,
    }


def _topology_gate(
    source_faces: np.ndarray,
    output_faces: np.ndarray,
    *,
    source_loop_edge_count: int,
) -> dict[str, Any]:
    before, _ = edge_topology(source_faces)
    after, _ = edge_topology(output_faces)
    before_winding = inconsistent_winding_edges(source_faces)
    after_winding = inconsistent_winding_edges(output_faces)
    expected_boundary_edges = before["boundary_edges"] - source_loop_edge_count
    failures = []
    if after["boundary_edges"] != expected_boundary_edges:
        failures.append("source_loop_not_closed_by_proxy_patch")
    if after["non_manifold_edges"] > before["non_manifold_edges"]:
        failures.append("local_proxy_patch_non_manifold_edges_increased")
    if after_winding > before_winding:
        failures.append("local_proxy_patch_inconsistent_winding_increased")
    return {
        "passed": not failures,
        "failure_reason_codes": failures,
        "before": {**before, "inconsistent_winding_edges": before_winding},
        "after": {**after, "inconsistent_winding_edges": after_winding},
        "expected_boundary_edges_after": int(expected_boundary_edges),
        "boundary_edges_removed": int(
            before["boundary_edges"] - after["boundary_edges"]
        ),
        "selected_source_loop_edges": int(source_loop_edge_count),
    }


def _map_proxy_face_parents(
    local_parents: np.ndarray, proxy_face_ids: np.ndarray
) -> np.ndarray:
    mapped = np.full(local_parents.shape[0], -1, dtype=np.int64)
    mask = local_parents >= 0
    mapped[mask] = proxy_face_ids[local_parents[mask]]
    return mapped


def _build_provenance(
    stitch: dict[str, Any],
    source_triangle_indices: np.ndarray,
    proxy_face_parent: np.ndarray,
    region_id: int,
) -> dict[str, np.ndarray]:
    face_origin = np.asarray(stitch["face_origin"], dtype=np.int16)
    source_parent = np.asarray(stitch["source_face_parent"], dtype=np.int64)
    source_index = np.full(face_origin.size, -1, dtype=np.int64)
    source_mask = source_parent >= 0
    source_index[source_mask] = source_triangle_indices[source_parent[source_mask]]
    generated_mask = ~source_mask
    proxy_mask = proxy_face_parent >= 0
    stitch_mask = generated_mask & ~proxy_mask
    fusion_region_id = np.zeros(face_origin.size, dtype=np.int32)
    fusion_region_id[generated_mask] = int(region_id)
    proxy_weight = np.zeros(face_origin.size, dtype=np.float32)
    proxy_weight[proxy_mask] = 1.0
    proxy_weight[stitch_mask] = 0.5
    return {
        "face_origin": face_origin.copy(),
        "source_triangle_index": source_index,
        "source_face_parent": source_parent.copy(),
        "proxy_face_parent": proxy_face_parent.copy(),
        "proxy_triangle_index": proxy_face_parent.copy(),
        "fusion_region_id": fusion_region_id,
        "proxy_weight": proxy_weight,
        "sdf_blend_weight": proxy_weight.copy(),
    }


def _source_triangle_indices(
    values: np.ndarray | None, face_count: int
) -> np.ndarray | None:
    if values is None:
        return np.full(face_count, -1, dtype=np.int64)
    raw = np.asarray(values)
    if (
        raw.ndim != 1
        or raw.size != face_count
        or not np.issubdtype(raw.dtype, np.integer)
    ):
        return None
    return raw.astype(np.int64, copy=True)


def _validate_mesh(
    points: np.ndarray, faces: np.ndarray, role: str
) -> tuple[str, str] | None:
    raw_points = np.asarray(points)
    raw_faces = np.asarray(faces)
    if raw_points.ndim != 2 or raw_points.shape[1:] != (3,):
        return (
            f"{role}_mesh_points_shape_invalid",
            f"{role} points must have shape (N, 3).",
        )
    if raw_faces.ndim != 2 or raw_faces.shape[1:] != (3,):
        return (
            f"{role}_mesh_faces_shape_invalid",
            f"{role} faces must have shape (M, 3).",
        )
    if raw_points.shape[0] < 3 or raw_faces.shape[0] < 1:
        return (
            f"{role}_mesh_empty",
            f"{role} mesh must contain points and triangle faces.",
        )
    if not np.all(np.isfinite(raw_points)):
        return f"{role}_mesh_points_non_finite", f"{role} mesh points must be finite."
    if not np.issubdtype(raw_faces.dtype, np.integer):
        return (
            f"{role}_mesh_faces_not_integer",
            f"{role} triangle vertex IDs must be integers.",
        )
    if np.any(raw_faces < 0) or np.any(raw_faces >= raw_points.shape[0]):
        return (
            f"{role}_mesh_face_vertex_out_of_range",
            f"{role} triangle vertex IDs are out of range.",
        )
    repeated = (
        (raw_faces[:, 0] == raw_faces[:, 1])
        | (raw_faces[:, 1] == raw_faces[:, 2])
        | (raw_faces[:, 2] == raw_faces[:, 0])
    )
    if np.any(repeated):
        return (
            f"{role}_mesh_face_repeats_vertex",
            f"{role} triangles may not repeat vertex IDs.",
        )
    return None


def _validate_source_loop(
    points: np.ndarray,
    faces: np.ndarray,
    loop: np.ndarray,
) -> dict[str, Any]:
    raw_input = np.asarray(loop)
    if raw_input.ndim != 1 or not np.issubdtype(raw_input.dtype, np.integer):
        return _loop_validation_failure(
            "source_loop_shape_invalid",
            "source_loop must be a one-dimensional integer vertex-ID array.",
        )
    raw = raw_input.astype(np.int64, copy=True)
    if raw.size > 1 and raw[0] == raw[-1]:
        raw = raw[:-1]
    if raw.size < 3 or np.unique(raw).size != raw.size:
        return _loop_validation_failure(
            "source_loop_not_simple",
            "source_loop must contain at least three distinct vertices.",
        )
    if np.any(raw < 0) or np.any(raw >= points.shape[0]):
        return _loop_validation_failure(
            "source_loop_vertex_out_of_range",
            "source_loop vertex IDs are out of range.",
        )
    _, edge_faces = edge_topology(faces)
    invalid_edges = []
    for index, left in enumerate(raw):
        right = int(raw[(index + 1) % raw.size])
        edge = tuple(sorted((int(left), right)))
        incidence = len(edge_faces.get(edge, []))
        if incidence != 1:
            invalid_edges.append({"edge": list(edge), "incidence": incidence})
    if invalid_edges:
        return _loop_validation_failure(
            "source_loop_edge_is_not_boundary",
            "Every source_loop edge must have exactly one incident source face.",
            invalid_edges=invalid_edges[:100],
        )
    return {
        "success": True,
        "failure_reason_codes": [],
        "loop": raw,
        "diagnostics": {
            "stage": "source_loop_validation",
            "message": "source_loop is an ordered topological boundary cycle.",
            "vertex_count": int(raw.size),
        },
    }


def _validate_config(config: LocalProxyPatchConfig) -> str | None:
    finite_values = {
        "inset_fraction": config.inset_fraction,
        "min_inset_fraction": config.min_inset_fraction,
        "max_normal_distance_ratio": config.max_normal_distance_ratio,
        "max_source_nonplanarity_ratio": config.max_source_nonplanarity_ratio,
        "min_projected_area_ratio": config.min_projected_area_ratio,
        "max_projected_area_ratio": config.max_projected_area_ratio,
        "min_normal_alignment": config.min_normal_alignment,
        "min_projected_orientation_consistency": config.min_projected_orientation_consistency,
        "min_inner_ring_clearance_ratio": config.min_inner_ring_clearance_ratio,
        "max_correspondence_distance_ratio": config.max_correspondence_distance_ratio,
        "ambiguity_score_margin": config.ambiguity_score_margin,
        "min_adjacent_normal_dot": config.min_adjacent_normal_dot,
        "normal_score_weight": config.normal_score_weight,
    }
    if not all(np.isfinite(value) for value in finite_values.values()):
        return "All floating-point config values must be finite."
    if not 0.0 < config.min_inset_fraction <= config.inset_fraction < 0.5:
        return "Inset fractions must satisfy 0 < min_inset_fraction <= inset_fraction < 0.5."
    if config.inset_attempts < 1:
        return "inset_attempts must be positive."
    if (
        config.max_normal_distance_ratio <= 0.0
        or config.max_correspondence_distance_ratio <= 0.0
    ):
        return "Distance ratios must be positive."
    if not 0.0 <= config.max_source_nonplanarity_ratio <= 1.0:
        return "max_source_nonplanarity_ratio must be in [0, 1]."
    if (
        not 0.0
        < config.min_projected_area_ratio
        < config.max_projected_area_ratio
        <= 1.5
    ):
        return "Projected area ratios are inconsistent."
    if not 0.0 <= config.min_normal_alignment <= 1.0:
        return "min_normal_alignment must be in [0, 1]."
    if not 0.0 <= config.min_projected_orientation_consistency <= 1.0:
        return "min_projected_orientation_consistency must be in [0, 1]."
    if config.max_candidate_components < 1 or config.max_ring_vertices < 3:
        return "Component and ring limits are too small."
    if config.phase_samples < 1 or config.max_self_intersection_candidate_pairs < 1:
        return "Sampling limits must be positive."
    return None


def _inset_schedule(config: LocalProxyPatchConfig) -> list[float]:
    if config.inset_attempts == 1 or config.inset_fraction == config.min_inset_fraction:
        return [float(config.inset_fraction)]
    values = np.geomspace(
        config.inset_fraction,
        config.min_inset_fraction,
        num=config.inset_attempts,
    )
    return [float(value) for value in values]


def _aggregate_extraction_failure(reason_codes: list[str]) -> str:
    precedence = (
        "proxy_patch_boundary_loop_count_mismatch",
        "proxy_patch_not_topological_disk",
        "proxy_patch_touches_input_boundary",
        "proxy_patch_non_manifold",
        "proxy_patch_component_limit_exceeded",
        "proxy_patch_geometric_gate_failed",
        "proxy_patch_local_neighborhood_empty",
    )
    reasons = set(reason_codes)
    for code in precedence:
        if code in reasons:
            return code
    return (
        "proxy_patch_geometric_gate_failed"
        if reasons
        else "proxy_patch_local_neighborhood_empty"
    )


def _points_in_polygon(
    points: np.ndarray, polygon: np.ndarray, tolerance: float
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return np.zeros(points.shape[0], dtype=bool)
    x = points[:, 0]
    y = points[:, 1]
    inside = np.zeros(points.shape[0], dtype=bool)
    for index, left in enumerate(polygon):
        right = polygon[(index + 1) % polygon.shape[0]]
        crosses = (left[1] > y) != (right[1] > y)
        denominator = right[1] - left[1]
        intersection_x = left[0] + (y - left[1]) * (right[0] - left[0]) / (
            denominator
            if abs(denominator) > 1e-30
            else np.copysign(1e-30, denominator or 1.0)
        )
        inside ^= crosses & (x < intersection_x)
    on_boundary = _distance_to_polygon_edges(points, polygon) <= tolerance
    return inside | on_boundary


def _distance_to_polygon_edges(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return np.zeros(points.shape[0], dtype=np.float64)
    minimum_squared = np.full(points.shape[0], np.inf, dtype=np.float64)
    for index, left in enumerate(polygon):
        right = polygon[(index + 1) % polygon.shape[0]]
        edge = right - left
        squared_length = float(np.dot(edge, edge))
        if squared_length <= 1e-30:
            squared = np.einsum("ij,ij->i", points - left, points - left)
        else:
            parameter = np.clip(((points - left) @ edge) / squared_length, 0.0, 1.0)
            delta = points - (left + parameter[:, None] * edge)
            squared = np.einsum("ij,ij->i", delta, delta)
        minimum_squared = np.minimum(minimum_squared, squared)
    return np.sqrt(minimum_squared)


def _polygon_signed_area(points: np.ndarray) -> float:
    return float(
        0.5
        * np.sum(
            points[:, 0] * np.roll(points[:, 1], -1)
            - np.roll(points[:, 0], -1) * points[:, 1]
        )
    )


def _polygon_self_intersections(
    polygon: np.ndarray, tolerance: float
) -> list[list[int]]:
    intersections: list[list[int]] = []
    count = polygon.shape[0]
    for left_index in range(count):
        left_next = (left_index + 1) % count
        for right_index in range(left_index + 1, count):
            right_next = (right_index + 1) % count
            if len({left_index, left_next, right_index, right_next}) < 4:
                continue
            if _segments_intersect_2d(
                polygon[left_index],
                polygon[left_next],
                polygon[right_index],
                polygon[right_next],
                tolerance,
            ):
                intersections.append([left_index, right_index])
    return intersections


def _segments_intersect_2d(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    tolerance: float,
) -> bool:
    def orientation(left: np.ndarray, right: np.ndarray, point: np.ndarray) -> float:
        edge = right - left
        delta = point - left
        return float(edge[0] * delta[1] - edge[1] * delta[0])

    values = (
        orientation(a, b, c),
        orientation(a, b, d),
        orientation(c, d, a),
        orientation(c, d, b),
    )
    if values[0] * values[1] < -(tolerance * tolerance) and values[2] * values[3] < -(
        tolerance * tolerance
    ):
        return True
    for point, left, right, value in (
        (c, a, b, values[0]),
        (d, a, b, values[1]),
        (a, c, d, values[2]),
        (b, c, d, values[3]),
    ):
        if (
            abs(value) <= tolerance
            and np.all(point >= np.minimum(left, right) - tolerance)
            and np.all(point <= np.maximum(left, right) + tolerance)
        ):
            return True
    return False


def _frame_failure(code: str, message: str, **diagnostics: Any) -> dict[str, Any]:
    return {
        "success": False,
        "failure_reason_codes": [code],
        "diagnostics": {
            "stage": "source_loop_local_frame",
            "message": message,
            **diagnostics,
        },
    }


def _loop_validation_failure(
    code: str, message: str, **diagnostics: Any
) -> dict[str, Any]:
    return {
        "success": False,
        "failure_reason_codes": [code],
        "loop": np.zeros(0, dtype=np.int64),
        "diagnostics": {
            "stage": "source_loop_validation",
            "message": message,
            **diagnostics,
        },
    }


def _extraction_failure(code: str, message: str, **diagnostics: Any) -> dict[str, Any]:
    return {
        "success": False,
        "failure_reason_codes": [code],
        "proxy_face_ids": np.zeros(0, dtype=np.int64),
        "proxy_point_ids": np.zeros(0, dtype=np.int64),
        "proxy_patch_points": np.empty((0, 3), dtype=np.float64),
        "proxy_patch_faces": np.empty((0, 3), dtype=np.int64),
        "proxy_inner_loop": np.zeros(0, dtype=np.int64),
        "proxy_inner_loop_original_point_ids": np.zeros(0, dtype=np.int64),
        "diagnostics": {
            "stage": "local_proxy_patch_extraction",
            "message": message,
            **diagnostics,
        },
    }


def _empty_provenance() -> dict[str, np.ndarray]:
    return {
        "face_origin": np.zeros(0, dtype=np.int16),
        "source_triangle_index": np.zeros(0, dtype=np.int64),
        "source_face_parent": np.zeros(0, dtype=np.int64),
        "proxy_face_parent": np.zeros(0, dtype=np.int64),
        "proxy_triangle_index": np.zeros(0, dtype=np.int64),
        "fusion_region_id": np.zeros(0, dtype=np.int32),
        "proxy_weight": np.zeros(0, dtype=np.float32),
        "sdf_blend_weight": np.zeros(0, dtype=np.float32),
    }


def _failure(code: str, stage: str, message: str, **diagnostics: Any) -> dict[str, Any]:
    return {
        "success": False,
        "accepted": False,
        "failure_reason_codes": [code],
        "points": np.empty((0, 3), dtype=np.float64),
        "faces": np.empty((0, 3), dtype=np.int64),
        "face_origin": np.zeros(0, dtype=np.int16),
        "source_face_parent": np.zeros(0, dtype=np.int64),
        "proxy_face_parent": np.zeros(0, dtype=np.int64),
        "stitch_face_ids": np.zeros(0, dtype=np.int64),
        "source_ring_vertex_ids": np.zeros(0, dtype=np.int64),
        "proxy_ring_vertex_ids": np.zeros(0, dtype=np.int64),
        "proxy_face_ids": np.zeros(0, dtype=np.int64),
        "proxy_point_ids": np.zeros(0, dtype=np.int64),
        "proxy_patch_points": np.empty((0, 3), dtype=np.float64),
        "proxy_patch_faces": np.empty((0, 3), dtype=np.int64),
        "proxy_inner_loop": np.zeros(0, dtype=np.int64),
        "proxy_inner_loop_original_point_ids": np.zeros(0, dtype=np.int64),
        "provenance": _empty_provenance(),
        "diagnostics": {"stage": stage, "message": message, **diagnostics},
    }
