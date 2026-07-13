from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from hybrid_proxy_geometry import (
    FACE_ORIGIN,
    _split_boundary_loop_at_parameters,
    plan_loop_arc_length_correspondence,
)
from mesh_metrics import (
    edge_topology,
    inconsistent_winding_edges,
    self_intersection_report,
)


@dataclass(frozen=True)
class CoincidentLoopWeldConfig:
    """Conservative transaction limits for welding two coincident open rings."""

    parameter_tolerance: float = 1e-10
    max_ring_vertices: int = 8192
    phase_samples: int = 128
    max_target_displacement: float | None = None
    max_target_displacement_ratio: float = 0.05
    min_triangle_area: float | None = None
    min_adjacent_normal_dot: float = 0.0
    max_same_side_dot: float = 0.25
    normal_score_weight: float = 0.25
    allow_target_face_flip: bool = False
    check_self_intersections: bool = True
    max_self_intersection_candidate_pairs: int = 250_000


# @entry
def weld_coincident_boundary_loops(
    source_points: np.ndarray,
    source_faces: np.ndarray,
    source_loop: np.ndarray,
    target_points: np.ndarray,
    target_faces: np.ndarray,
    target_loop: np.ndarray,
    *,
    config: CoincidentLoopWeldConfig | None = None,
    source_triangle_indices: np.ndarray | None = None,
    target_triangle_indices: np.ndarray | None = None,
    region_id: int = 0,
) -> dict[str, Any]:
    """Weld two near-coincident component boundaries without an annular band.

    The first component is the coordinate authority.  Both boundary rings are
    split at the union of their normalized arc-length breakpoints, every point
    on the second ring is mapped to its paired first-ring point, and the two
    triangle soups are then merged through the shared point IDs.  Original
    first-component points are never moved.

    The operation is transactional.  A failed result contains diagnostics and
    empty consumable geometry/provenance, so callers cannot accidentally commit
    a partially split or partially welded mesh.
    """

    cfg = config or CoincidentLoopWeldConfig()
    configuration_failure = _validate_config(cfg)
    if configuration_failure is not None:
        return _failure(
            "coincident_weld_config_invalid",
            "configuration",
            configuration_failure,
            config=asdict(cfg),
        )

    input_failure = _validate_mesh(source_points, source_faces, "source")
    if input_failure is not None:
        return _failure(input_failure[0], "input_validation", input_failure[1])
    input_failure = _validate_mesh(target_points, target_faces, "target")
    if input_failure is not None:
        return _failure(input_failure[0], "input_validation", input_failure[1])
    loop_failure = _validate_loop_array(source_loop, "source")
    if loop_failure is not None:
        return _failure(loop_failure[0], "input_validation", loop_failure[1])
    loop_failure = _validate_loop_array(target_loop, "target")
    if loop_failure is not None:
        return _failure(loop_failure[0], "input_validation", loop_failure[1])

    # Work exclusively on owned copies.  This makes rollback independent of
    # whether the caller supplied writable views or shared arrays.
    source_points = np.asarray(source_points, dtype=np.float64).copy()
    source_faces = np.asarray(source_faces, dtype=np.int64).copy()
    target_points = np.asarray(target_points, dtype=np.float64).copy()
    target_faces = np.asarray(target_faces, dtype=np.int64).copy()
    source_loop = np.asarray(source_loop, dtype=np.int64).copy()
    target_loop = np.asarray(target_loop, dtype=np.int64).copy()

    source_indices = _triangle_indices(
        source_triangle_indices, source_faces.shape[0]
    )
    if source_indices is None:
        return _failure(
            "source_triangle_indices_shape_invalid",
            "input_validation",
            "source_triangle_indices must have one integer value per source face.",
        )
    target_indices = _triangle_indices(
        target_triangle_indices, target_faces.shape[0]
    )
    if target_indices is None:
        return _failure(
            "target_triangle_indices_shape_invalid",
            "input_validation",
            "target_triangle_indices must have one integer value per target face.",
        )

    plan = plan_loop_arc_length_correspondence(
        source_points,
        source_faces,
        source_loop,
        target_points,
        target_faces,
        target_loop,
        parameter_tolerance=cfg.parameter_tolerance,
        max_ring_vertices=cfg.max_ring_vertices,
        phase_samples=cfg.phase_samples,
        max_correspondence_distance=cfg.max_target_displacement,
        normal_score_weight=cfg.normal_score_weight,
        allow_proxy_face_flip=cfg.allow_target_face_flip,
    )
    if not plan["success"]:
        return _failure(
            plan["failure_reason_codes"][0],
            "arc_length_correspondence",
            plan["diagnostics"].get(
                "message", "Coincident-loop correspondence planning failed."
            ),
            correspondence=plan["diagnostics"],
        )

    oriented_target_faces = (
        target_faces[:, [0, 2, 1]]
        if plan["proxy_faces_flipped"]
        else target_faces.copy()
    )
    source_split = _split_boundary_loop_at_parameters(
        source_points,
        source_faces,
        plan["source_loop"],
        plan["source_query_parameters"],
        cfg.parameter_tolerance,
    )
    if not source_split["success"]:
        return _failure(
            source_split["failure_reason_codes"][0],
            "source_boundary_split",
            source_split["diagnostics"].get(
                "message", "The authoritative boundary could not be split."
            ),
            correspondence=plan["diagnostics"],
            source_split=source_split["diagnostics"],
        )
    target_split = _split_boundary_loop_at_parameters(
        target_points,
        oriented_target_faces,
        plan["proxy_loop"],
        plan["proxy_query_parameters"],
        cfg.parameter_tolerance,
    )
    if not target_split["success"]:
        return _failure(
            target_split["failure_reason_codes"][0],
            "target_boundary_split",
            target_split["diagnostics"].get(
                "message", "The target boundary could not be split."
            ),
            correspondence=plan["diagnostics"],
            source_split=source_split["diagnostics"],
            target_split=target_split["diagnostics"],
        )

    source_ring = np.asarray(source_split["ring_vertex_ids"], dtype=np.int64)
    target_ring = np.asarray(target_split["ring_vertex_ids"], dtype=np.int64)
    if source_ring.size != target_ring.size:
        return _failure(
            "coincident_ring_bijection_size_mismatch",
            "boundary_bijection",
            "The split boundary rings do not contain the same number of vertices.",
            correspondence=plan["diagnostics"],
            source_split=source_split["diagnostics"],
            target_split=target_split["diagnostics"],
            source_ring_vertices=int(source_ring.size),
            target_ring_vertices=int(target_ring.size),
        )

    displacement_threshold = _target_displacement_threshold(
        source_split["points"][source_ring], cfg
    )
    paired_displacements = np.linalg.norm(
        source_split["points"][source_ring] - target_split["points"][target_ring],
        axis=1,
    )
    displacement = {
        "passed": bool(
            paired_displacements.size == 0
            or float(paired_displacements.max()) <= displacement_threshold
        ),
        "method": "paired_resampled_target_ring_to_fixed_source_ring",
        "ring_vertex_count": int(source_ring.size),
        "maximum": (
            float(paired_displacements.max())
            if paired_displacements.size
            else 0.0
        ),
        "rms": (
            float(np.sqrt(np.mean(paired_displacements * paired_displacements)))
            if paired_displacements.size
            else 0.0
        ),
        "threshold": float(displacement_threshold),
        "threshold_mode": (
            "absolute" if cfg.max_target_displacement is not None else "ring_scale_ratio"
        ),
    }
    if not displacement["passed"]:
        return _failure(
            "target_boundary_displacement_exceeded",
            "boundary_bijection",
            "Welding the target ring would exceed its maximum displacement.",
            correspondence=plan["diagnostics"],
            source_split=source_split["diagnostics"],
            target_split=target_split["diagnostics"],
            target_displacement=displacement,
        )

    merge = _merge_split_components(
        source_split,
        target_split,
        source_ring,
        target_ring,
        source_original_point_count=source_points.shape[0],
        target_original_point_count=target_points.shape[0],
    )
    output_points = merge["points"]
    output_faces = merge["faces"]
    source_face_count = int(source_split["faces"].shape[0])

    source_lock = _source_lock_report(source_points, output_points)
    if not source_lock["passed"]:
        return _failure(
            "source_coordinate_lock_failed",
            "coordinate_lock_validation",
            "At least one original source point moved during coincident welding.",
            correspondence=plan["diagnostics"],
            source_split=source_split["diagnostics"],
            target_split=target_split["diagnostics"],
            target_displacement=displacement,
            source_coordinate_lock=source_lock,
        )

    seam_validation = _validate_welded_seam(
        output_points,
        output_faces,
        source_ring,
        source_face_count,
        min_triangle_area=cfg.min_triangle_area,
        min_adjacent_normal_dot=cfg.min_adjacent_normal_dot,
        max_same_side_dot=cfg.max_same_side_dot,
    )
    if not seam_validation["passed"]:
        return _failure(
            seam_validation["failure_reason_codes"][0],
            "welded_seam_validation",
            "The merged ring failed conformal weld validation.",
            correspondence=plan["diagnostics"],
            source_split=source_split["diagnostics"],
            target_split=target_split["diagnostics"],
            target_displacement=displacement,
            source_coordinate_lock=source_lock,
            seam_validation=seam_validation,
        )

    topology = _topology_gate(
        source_faces,
        target_faces,
        output_faces,
        source_loop_vertices=int(plan["source_loop"].size),
        target_loop_vertices=int(plan["proxy_loop"].size),
    )
    if not topology["passed"]:
        return _failure(
            topology["failure_reason_codes"][0],
            "welded_topology_validation",
            "The coincident weld failed its topology transaction gate.",
            correspondence=plan["diagnostics"],
            source_split=source_split["diagnostics"],
            target_split=target_split["diagnostics"],
            target_displacement=displacement,
            source_coordinate_lock=source_lock,
            seam_validation=seam_validation,
            topology_gate=topology,
        )

    focus_face_ids = np.unique(
        np.concatenate(
            [
                np.asarray(source_split["generated_face_ids"], dtype=np.int64),
                np.asarray(target_split["generated_face_ids"], dtype=np.int64)
                + source_face_count,
                np.asarray(seam_validation["seam_face_ids"], dtype=np.int64),
            ]
        )
    )
    if cfg.check_self_intersections:
        try:
            local_self_intersection = self_intersection_report(
                output_points,
                output_faces,
                focus_face_ids=focus_face_ids,
                max_candidate_pairs=cfg.max_self_intersection_candidate_pairs,
            )
        except Exception as exc:  # pragma: no cover - optional VTK/runtime failure
            return _failure(
                "coincident_weld_self_intersection_check_failed",
                "focused_local_self_intersection",
                "Focused local self-intersection validation could not complete: "
                f"{type(exc).__name__}: {exc}",
                correspondence=plan["diagnostics"],
                source_split=source_split["diagnostics"],
                target_split=target_split["diagnostics"],
                target_displacement=displacement,
                source_coordinate_lock=source_lock,
                seam_validation=seam_validation,
                topology_gate=topology,
                focused_face_ids=focus_face_ids.astype(int).tolist(),
            )
        if not local_self_intersection["passed"]:
            failure_code = (
                "coincident_weld_local_self_intersection_detected"
                if local_self_intersection.get("status") == "computed"
                else "coincident_weld_self_intersection_check_incomplete"
            )
            return _failure(
                failure_code,
                "focused_local_self_intersection",
                "The welded seam intersects non-adjacent local geometry.",
                correspondence=plan["diagnostics"],
                source_split=source_split["diagnostics"],
                target_split=target_split["diagnostics"],
                target_displacement=displacement,
                source_coordinate_lock=source_lock,
                seam_validation=seam_validation,
                topology_gate=topology,
                local_self_intersection=local_self_intersection,
            )
    else:
        local_self_intersection = {
            "method": "vtk_static_cell_locator_triangle_intersection",
            "scope": "focused_faces",
            "focus_face_count": int(focus_face_ids.size),
            "status": "disabled_by_config",
            "passed": True,
        }

    original_target_displacements = np.linalg.norm(
        output_points[merge["target_original_point_to_output"]] - target_points,
        axis=1,
    )
    displacement["all_original_target_points_maximum"] = float(
        original_target_displacements.max()
    )
    displacement["all_original_target_points_rms"] = float(
        np.sqrt(np.mean(original_target_displacements**2))
    )

    provenance = _build_provenance(
        source_split,
        target_split,
        merge,
        source_indices,
        target_indices,
        int(region_id),
    )
    diagnostics = {
        "stage": "transactional_coincident_loop_weld",
        "method": "arc_length_union_edge_split_fixed_source_ring_index_weld",
        "config": asdict(cfg),
        "contract": {
            "annular_bridge_generated": False,
            "source_ring_is_coordinate_authority": True,
            "target_ring_uses_source_ring_point_ids": True,
            "transactional_empty_geometry_on_failure": True,
        },
        "correspondence": plan["diagnostics"],
        "source_split": source_split["diagnostics"],
        "target_split": target_split["diagnostics"],
        "target_displacement": displacement,
        "source_coordinate_lock": source_lock,
        "seam_validation": {
            key: value
            for key, value in seam_validation.items()
            if key != "seam_face_ids"
        },
        "topology_gate": topology,
        "local_self_intersection": local_self_intersection,
        "provenance": {
            "source_parent_faces": int(
                np.count_nonzero(provenance["source_face_parent"] >= 0)
            ),
            "target_parent_faces": int(
                np.count_nonzero(provenance["target_face_parent"] >= 0)
            ),
            "unmapped_faces": int(
                np.count_nonzero(provenance["component_face_parent"] < 0)
            ),
            "region_id": int(region_id),
        },
        "output": {
            "points": int(output_points.shape[0]),
            "triangles": int(output_faces.shape[0]),
            "source_faces": source_face_count,
            "target_faces": int(target_split["faces"].shape[0]),
            "stitch_faces": 0,
        },
    }
    return {
        "success": True,
        "accepted": True,
        "failure_reason_codes": [],
        "points": output_points,
        "faces": output_faces,
        "source_ring_vertex_ids": source_ring,
        "target_ring_vertex_ids_before_merge": target_ring,
        "target_ring_vertex_ids": source_ring.copy(),
        "welded_seam_edges": np.asarray(
            [
                [int(source_ring[index]), int(source_ring[(index + 1) % source_ring.size])]
                for index in range(source_ring.size)
            ],
            dtype=np.int64,
        ),
        "stitch_face_ids": np.zeros(0, dtype=np.int64),
        "source_face_parent": provenance["source_face_parent"],
        "target_face_parent": provenance["target_face_parent"],
        "source_original_point_to_output": merge[
            "source_original_point_to_output"
        ],
        "target_original_point_to_output": merge[
            "target_original_point_to_output"
        ],
        "provenance": provenance,
        "diagnostics": diagnostics,
    }


def _merge_split_components(
    source_split: dict[str, Any],
    target_split: dict[str, Any],
    source_ring: np.ndarray,
    target_ring: np.ndarray,
    *,
    source_original_point_count: int,
    target_original_point_count: int,
) -> dict[str, np.ndarray]:
    source_split_points = np.asarray(source_split["points"], dtype=np.float64)
    target_split_points = np.asarray(target_split["points"], dtype=np.float64)
    target_to_output = np.full(target_split_points.shape[0], -1, dtype=np.int64)
    target_to_output[target_ring] = source_ring
    retained_target_point_ids = np.flatnonzero(target_to_output < 0)
    retained_output_ids = np.arange(
        source_split_points.shape[0],
        source_split_points.shape[0] + retained_target_point_ids.size,
        dtype=np.int64,
    )
    target_to_output[retained_target_point_ids] = retained_output_ids
    output_points = np.vstack(
        [source_split_points, target_split_points[retained_target_point_ids]]
    )
    target_faces_merged = target_to_output[
        np.asarray(target_split["faces"], dtype=np.int64)
    ]
    output_faces = np.vstack(
        [np.asarray(source_split["faces"], dtype=np.int64), target_faces_merged]
    )
    return {
        "points": output_points,
        "faces": output_faces,
        "target_split_point_to_output": target_to_output,
        "retained_target_point_ids": retained_target_point_ids,
        "source_original_point_to_output": np.arange(
            source_original_point_count, dtype=np.int64
        ),
        "target_original_point_to_output": target_to_output[
            :target_original_point_count
        ].copy(),
    }


def _validate_welded_seam(
    points: np.ndarray,
    faces: np.ndarray,
    source_ring: np.ndarray,
    source_face_count: int,
    *,
    min_triangle_area: float | None,
    min_adjacent_normal_dot: float,
    max_same_side_dot: float,
) -> dict[str, Any]:
    occurrences = _directed_edge_occurrences(faces)
    seam_edges = [
        tuple(
            sorted(
                (
                    int(source_ring[index]),
                    int(source_ring[(index + 1) % source_ring.size]),
                )
            )
        )
        for index in range(source_ring.size)
    ]
    incidence_failures: list[dict[str, Any]] = []
    orientation_conflicts: list[dict[str, Any]] = []
    parent_role_conflicts: list[dict[str, Any]] = []
    normal_failures: list[dict[str, Any]] = []
    same_side_failures: list[dict[str, Any]] = []
    seam_face_ids: list[int] = []
    normals = _unit_face_normals(points, faces)
    normal_dots: list[float] = []
    side_dots: list[float] = []

    for edge in seam_edges:
        rows = occurrences.get(edge, [])
        if len(rows) != 2:
            incidence_failures.append({"edge": list(edge), "incidence": len(rows)})
            continue
        seam_face_ids.extend([rows[0][0], rows[1][0]])
        if not (rows[0][1] == rows[1][2] and rows[0][2] == rows[1][1]):
            orientation_conflicts.append(
                {"edge": list(edge), "faces": [rows[0][0], rows[1][0]]}
            )
        roles = [
            "source" if row[0] < source_face_count else "target" for row in rows
        ]
        if sorted(roles) != ["source", "target"]:
            parent_role_conflicts.append(
                {"edge": list(edge), "faces": [rows[0][0], rows[1][0]], "roles": roles}
            )

        dot = float(np.dot(normals[rows[0][0]], normals[rows[1][0]]))
        normal_dots.append(dot)
        if dot < min_adjacent_normal_dot:
            normal_failures.append(
                {
                    "edge": list(edge),
                    "faces": [rows[0][0], rows[1][0]],
                    "normal_dot": dot,
                }
            )

        side_dot = _incident_side_dot(points, faces, edge, rows)
        if side_dot is not None:
            side_dots.append(side_dot)
            if side_dot > max_same_side_dot:
                same_side_failures.append(
                    {
                        "edge": list(edge),
                        "faces": [rows[0][0], rows[1][0]],
                        "side_dot": side_dot,
                    }
                )

    scale = max(float(np.linalg.norm(np.ptp(points, axis=0))), 1e-12)
    area_tolerance = (
        float(min_triangle_area)
        if min_triangle_area is not None
        else max(scale * scale * 1e-16, 1e-24)
    )
    triangles = points[faces]
    areas = (
        np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            ),
            axis=1,
        )
        * 0.5
    )
    repeated = (
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 2] == faces[:, 0])
    )
    degenerate_face_ids = np.flatnonzero((areas <= area_tolerance) | repeated)

    failures = []
    if incidence_failures:
        failures.append("weld_seam_edge_incidence_failed")
    if orientation_conflicts or parent_role_conflicts:
        failures.append("weld_seam_orientation_failed")
    if degenerate_face_ids.size:
        failures.append("weld_degenerate_faces_detected")
    if normal_failures:
        failures.append("weld_normal_transition_failed")
    if same_side_failures:
        failures.append("weld_same_side_overlap_detected")
    return {
        "passed": not failures,
        "failure_reason_codes": failures,
        "seam_edge_count": len(seam_edges),
        "seam_face_ids": np.unique(np.asarray(seam_face_ids, dtype=np.int64)),
        "edge_incidence_two": not incidence_failures,
        "opposite_directed_edge_pairs": not orientation_conflicts,
        "one_source_and_one_target_face": not parent_role_conflicts,
        "incidence_failures": incidence_failures[:100],
        "orientation_conflicts": orientation_conflicts[:100],
        "parent_role_conflicts": parent_role_conflicts[:100],
        "degenerate_faces": degenerate_face_ids[:100].astype(int).tolist(),
        "minimum_area": float(areas.min()) if areas.size else None,
        "area_tolerance": float(area_tolerance),
        "adjacent_normal_dot_min": min(normal_dots) if normal_dots else None,
        "minimum_adjacent_normal_dot": float(min_adjacent_normal_dot),
        "normal_transition_failures": normal_failures[:100],
        "incident_side_dot_max": max(side_dots) if side_dots else None,
        "maximum_same_side_dot": float(max_same_side_dot),
        "same_side_failures": same_side_failures[:100],
    }


def _topology_gate(
    source_faces: np.ndarray,
    target_faces: np.ndarray,
    output_faces: np.ndarray,
    *,
    source_loop_vertices: int,
    target_loop_vertices: int,
) -> dict[str, Any]:
    source_topology, _ = edge_topology(source_faces)
    target_topology, _ = edge_topology(target_faces)
    after, _ = edge_topology(output_faces)
    before_boundary_edges = (
        source_topology["boundary_edges"] + target_topology["boundary_edges"]
    )
    expected_boundary_edges = (
        before_boundary_edges - source_loop_vertices - target_loop_vertices
    )
    before_non_manifold = (
        source_topology["non_manifold_edges"]
        + target_topology["non_manifold_edges"]
    )
    before_winding = inconsistent_winding_edges(source_faces) + inconsistent_winding_edges(
        target_faces
    )
    after_winding = inconsistent_winding_edges(output_faces)
    failures = []
    if after["boundary_edges"] != expected_boundary_edges:
        failures.append("coincident_weld_boundary_count_unexpected")
    if after["non_manifold_edges"] > before_non_manifold:
        failures.append("coincident_weld_non_manifold_edges_increased")
    if after_winding > before_winding:
        failures.append("coincident_weld_inconsistent_winding_increased")
    return {
        "passed": not failures,
        "failure_reason_codes": failures,
        "before": {
            "source": source_topology,
            "target": target_topology,
            "boundary_edges_total": int(before_boundary_edges),
            "non_manifold_edges_total": int(before_non_manifold),
            "inconsistent_winding_edges_total": int(before_winding),
        },
        "after": {**after, "inconsistent_winding_edges": int(after_winding)},
        "expected_boundary_edges_after": int(expected_boundary_edges),
        "selected_source_loop_edges": int(source_loop_vertices),
        "selected_target_loop_edges": int(target_loop_vertices),
    }


def _incident_side_dot(
    points: np.ndarray,
    faces: np.ndarray,
    edge: tuple[int, int],
    rows: list[tuple[int, int, int]],
) -> float | None:
    left, right = edge
    edge_vector = points[right] - points[left]
    edge_length = float(np.linalg.norm(edge_vector))
    if edge_length <= 1e-15:
        return None
    edge_direction = edge_vector / edge_length
    midpoint = (points[left] + points[right]) * 0.5
    side_vectors = []
    for face_id, _, _ in rows:
        face = faces[face_id]
        third = next(
            int(vertex_id)
            for vertex_id in face
            if int(vertex_id) not in {left, right}
        )
        vector = points[third] - midpoint
        vector = vector - float(np.dot(vector, edge_direction)) * edge_direction
        length = float(np.linalg.norm(vector))
        if length <= 1e-15:
            return None
        side_vectors.append(vector / length)
    return float(np.dot(side_vectors[0], side_vectors[1]))


def _directed_edge_occurrences(
    faces: np.ndarray,
) -> dict[tuple[int, int], list[tuple[int, int, int]]]:
    occurrences: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for face_id, (a, b, c) in enumerate(np.asarray(faces, dtype=np.int64)):
        for left, right in ((a, b), (b, c), (c, a)):
            key = tuple(sorted((int(left), int(right))))
            occurrences.setdefault(key, []).append(
                (face_id, int(left), int(right))
            )
    return occurrences


def _unit_face_normals(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = points[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    lengths = np.linalg.norm(normals, axis=1)
    return np.divide(
        normals,
        lengths[:, None],
        out=np.zeros_like(normals),
        where=lengths[:, None] > 1e-15,
    )


def _source_lock_report(
    source_points: np.ndarray, output_points: np.ndarray
) -> dict[str, Any]:
    deltas = np.linalg.norm(
        output_points[: source_points.shape[0]] - source_points, axis=1
    )
    exact = bool(np.array_equal(output_points[: source_points.shape[0]], source_points))
    return {
        "passed": exact,
        "method": "bitwise_coordinate_identity_for_all_original_source_points",
        "original_source_point_count": int(source_points.shape[0]),
        "exactly_locked": exact,
        "maximum_displacement": float(deltas.max()) if deltas.size else 0.0,
        "tolerance": 0.0,
    }


def _target_displacement_threshold(
    source_ring_points: np.ndarray, config: CoincidentLoopWeldConfig
) -> float:
    if config.max_target_displacement is not None:
        return float(config.max_target_displacement)
    diagonal = float(np.linalg.norm(np.ptp(source_ring_points, axis=0)))
    edge_lengths = np.linalg.norm(
        np.roll(source_ring_points, -1, axis=0) - source_ring_points, axis=1
    )
    characteristic_length = max(diagonal, float(edge_lengths.sum()) / np.pi, 1e-12)
    return characteristic_length * float(config.max_target_displacement_ratio)


def _build_provenance(
    source_split: dict[str, Any],
    target_split: dict[str, Any],
    merge: dict[str, np.ndarray],
    source_triangle_indices: np.ndarray,
    target_triangle_indices: np.ndarray,
    region_id: int,
) -> dict[str, np.ndarray]:
    source_parent_local = np.asarray(source_split["face_parent"], dtype=np.int64)
    target_parent_local = np.asarray(target_split["face_parent"], dtype=np.int64)
    source_face_count = source_parent_local.size
    target_face_count = target_parent_local.size
    face_count = source_face_count + target_face_count
    source_face_parent = np.concatenate(
        [source_parent_local, np.full(target_face_count, -1, dtype=np.int64)]
    )
    target_face_parent = np.concatenate(
        [np.full(source_face_count, -1, dtype=np.int64), target_parent_local]
    )
    component_role = np.concatenate(
        [
            np.zeros(source_face_count, dtype=np.int8),
            np.ones(target_face_count, dtype=np.int8),
        ]
    )
    component_parent = np.concatenate([source_parent_local, target_parent_local])
    source_triangle_index = np.concatenate(
        [
            source_triangle_indices[source_parent_local],
            target_triangle_indices[target_parent_local],
        ]
    )
    output_source_point_parent = np.full(
        merge["points"].shape[0], -1, dtype=np.int64
    )
    output_source_point_parent[
        merge["source_original_point_to_output"]
    ] = np.arange(merge["source_original_point_to_output"].size, dtype=np.int64)
    output_target_point_parent = np.full(
        merge["points"].shape[0], -1, dtype=np.int64
    )
    output_target_point_parent[
        merge["target_original_point_to_output"]
    ] = np.arange(merge["target_original_point_to_output"].size, dtype=np.int64)
    return {
        "face_origin": np.full(face_count, FACE_ORIGIN["source"], dtype=np.int16),
        "source_component_role": component_role,
        "component_face_parent": component_parent,
        "source_face_parent": source_face_parent,
        "target_face_parent": target_face_parent,
        "source_triangle_index": source_triangle_index,
        "fusion_region_id": np.full(face_count, region_id, dtype=np.int32),
        "proxy_weight": np.zeros(face_count, dtype=np.float32),
        "sdf_blend_weight": np.zeros(face_count, dtype=np.float32),
        "output_source_point_parent": output_source_point_parent,
        "output_target_point_parent": output_target_point_parent,
        "source_original_point_to_output": merge[
            "source_original_point_to_output"
        ].copy(),
        "target_original_point_to_output": merge[
            "target_original_point_to_output"
        ].copy(),
    }


def _validate_mesh(
    points: np.ndarray, faces: np.ndarray, role: str
) -> tuple[str, str] | None:
    raw_points = np.asarray(points)
    raw_faces = np.asarray(faces)
    if raw_points.ndim != 2 or raw_points.shape[1:] != (3,):
        return f"{role}_mesh_points_shape_invalid", f"{role} points must have shape (N, 3)."
    if raw_faces.ndim != 2 or raw_faces.shape[1:] != (3,):
        return f"{role}_mesh_faces_shape_invalid", f"{role} faces must have shape (M, 3)."
    if raw_points.shape[0] < 3 or raw_faces.shape[0] < 1:
        return f"{role}_mesh_empty", f"{role} mesh must contain triangle faces."
    if not np.all(np.isfinite(raw_points)):
        return f"{role}_mesh_points_non_finite", f"{role} mesh points must be finite."
    if not np.issubdtype(raw_faces.dtype, np.integer):
        return f"{role}_mesh_faces_not_integer", f"{role} face IDs must be integers."
    if np.any(raw_faces < 0) or np.any(raw_faces >= raw_points.shape[0]):
        return (
            f"{role}_mesh_face_vertex_out_of_range",
            f"{role} triangle vertex IDs are out of range.",
        )
    return None


def _triangle_indices(
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


def _validate_loop_array(loop: np.ndarray, role: str) -> tuple[str, str] | None:
    raw = np.asarray(loop)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
        return (
            f"{role}_loop_shape_invalid",
            f"{role}_loop must be a one-dimensional integer vertex-ID array.",
        )
    return None


def _validate_config(config: CoincidentLoopWeldConfig) -> str | None:
    if config.parameter_tolerance <= 0.0:
        return "parameter_tolerance must be positive."
    if config.max_ring_vertices < 3:
        return "max_ring_vertices must be at least three."
    if config.phase_samples < 1:
        return "phase_samples must be positive."
    if config.max_target_displacement is not None and config.max_target_displacement < 0.0:
        return "max_target_displacement cannot be negative."
    if config.max_target_displacement_ratio <= 0.0:
        return "max_target_displacement_ratio must be positive."
    if config.min_triangle_area is not None and config.min_triangle_area < 0.0:
        return "min_triangle_area cannot be negative."
    if not -1.0 <= config.min_adjacent_normal_dot <= 1.0:
        return "min_adjacent_normal_dot must be in [-1, 1]."
    if not -1.0 <= config.max_same_side_dot <= 1.0:
        return "max_same_side_dot must be in [-1, 1]."
    if config.normal_score_weight < 0.0:
        return "normal_score_weight cannot be negative."
    if config.max_self_intersection_candidate_pairs < 1:
        return "max_self_intersection_candidate_pairs must be positive."
    return None


def _empty_provenance() -> dict[str, np.ndarray]:
    return {
        "face_origin": np.zeros(0, dtype=np.int16),
        "source_component_role": np.zeros(0, dtype=np.int8),
        "component_face_parent": np.zeros(0, dtype=np.int64),
        "source_face_parent": np.zeros(0, dtype=np.int64),
        "target_face_parent": np.zeros(0, dtype=np.int64),
        "source_triangle_index": np.zeros(0, dtype=np.int64),
        "fusion_region_id": np.zeros(0, dtype=np.int32),
        "proxy_weight": np.zeros(0, dtype=np.float32),
        "sdf_blend_weight": np.zeros(0, dtype=np.float32),
        "output_source_point_parent": np.zeros(0, dtype=np.int64),
        "output_target_point_parent": np.zeros(0, dtype=np.int64),
        "source_original_point_to_output": np.zeros(0, dtype=np.int64),
        "target_original_point_to_output": np.zeros(0, dtype=np.int64),
    }


def _failure(code: str, stage: str, message: str, **diagnostics: Any) -> dict[str, Any]:
    return {
        "success": False,
        "accepted": False,
        "failure_reason_codes": [code],
        "points": np.empty((0, 3), dtype=np.float64),
        "faces": np.empty((0, 3), dtype=np.int64),
        "source_ring_vertex_ids": np.zeros(0, dtype=np.int64),
        "target_ring_vertex_ids_before_merge": np.zeros(0, dtype=np.int64),
        "target_ring_vertex_ids": np.zeros(0, dtype=np.int64),
        "welded_seam_edges": np.empty((0, 2), dtype=np.int64),
        "stitch_face_ids": np.zeros(0, dtype=np.int64),
        "source_face_parent": np.zeros(0, dtype=np.int64),
        "target_face_parent": np.zeros(0, dtype=np.int64),
        "source_original_point_to_output": np.zeros(0, dtype=np.int64),
        "target_original_point_to_output": np.zeros(0, dtype=np.int64),
        "provenance": _empty_provenance(),
        "diagnostics": _json_compatible(
            {
                "stage": stage,
                "message": message,
                **diagnostics,
                "rollback": {
                    "applied": True,
                    "input_arrays_mutated": False,
                    "consumable_geometry_empty": True,
                    "consumable_provenance_empty": True,
                },
            },
        ),
    }


def _json_compatible(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value
