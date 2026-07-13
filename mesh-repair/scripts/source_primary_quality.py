from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from deterministic_hole_fill import validate_local_patch_intersections
from source_primary_prefix_audit import audit_source_prefix
from source_primary_quality_inputs import validate_quality_inputs
from source_primary_quality_geometry import (
    audit_boundary_sharing,
    audit_interior_footprint,
    audit_patch_internal_continuity,
    audit_topology_delta,
    build_directed_edge_map,
    build_gate,
    calculate_area_weighted_normal,
    calculate_coordinate_tolerance,
    calculate_normal_angle,
    calculate_triangle_geometry,
    collect_incident_source_faces,
    collect_loop_edges,
    describe_local_geometry,
    normalize_open_loop,
    summarize_values,
)


__all__ = [
    "PatchQualityLimits",
    "REQUIRED_PATCH_QUALITY_GATES",
    "audit_source_prefix",
    "audit_source_primary_patch",
    "validate_transaction_local_quality",
]


REQUIRED_PATCH_QUALITY_GATES = frozenset(
    {
        "boundary_index_sharing",
        "interior_points_inside_repair_footprint",
        "source_local_normal_consistent",
        "triangle_minimum_area",
        "triangle_maximum_aspect_ratio",
        "patch_orientation",
        "boundary_normal_transition",
        "boundary_curvature_continuity",
        "patch_internal_normal_transition",
        "patch_internal_curvature_continuity",
        "patch_non_adjacent_intersection",
        "patch_topology_delta",
    }
)


@dataclass(frozen=True)
class PatchQualityLimits:
    """Fail-closed limits for one append-only source-primary patch."""

    minimum_triangle_area: float = 1e-18
    minimum_area_ratio: float = 1e-10
    maximum_aspect_ratio: float = 25.0
    minimum_orientation_dot: float = 0.0
    minimum_source_normal_dot: float = 0.0
    minimum_boundary_normal_dot: float = 0.25
    maximum_normal_transition_degrees: float = 75.0
    maximum_curvature_jump: float = 0.75
    coordinate_tolerance_ratio: float = 1e-12
    maximum_intersection_candidate_pairs: int = 250_000


# @entry
def audit_source_primary_patch(
    current_points: np.ndarray,
    current_faces: np.ndarray,
    interior_points: np.ndarray,
    patch_faces: np.ndarray,
    boundary_loops: tuple[np.ndarray, ...],
    *,
    expected_face_normals: np.ndarray | None = None,
    limits: PatchQualityLimits | None = None,
) -> dict[str, Any]:
    """Audit one mapped patch against the complete current mesh.

    ``patch_faces`` uses global point IDs in ``current_points + interior_points``.
    Existing point IDs may only be the declared source boundary IDs. Missing or
    incomplete evidence is a failed gate, never an implicit pass.
    """

    cfg = limits or PatchQualityLimits()
    validation = validate_quality_inputs(
        current_points,
        current_faces,
        interior_points,
        patch_faces,
        boundary_loops,
        cfg,
    )
    if validation is not None:
        return _failed_audit("patch_delta_invalid", validation, cfg)

    current_points = np.asarray(current_points)
    current_faces = np.asarray(current_faces, dtype=np.int64)
    interior_points = np.asarray(interior_points, dtype=current_points.dtype)
    patch_faces = np.asarray(patch_faces, dtype=np.int64)
    trial_points = np.concatenate([current_points, interior_points], axis=0)
    loops = tuple(normalize_open_loop(loop) for loop in boundary_loops)
    boundary_edges = collect_loop_edges(loops)
    current_edges = build_directed_edge_map(current_faces)
    patch_edges = build_directed_edge_map(patch_faces)
    boundary_vertices = np.unique(np.concatenate(loops))
    tolerance = calculate_coordinate_tolerance(
        current_points, cfg.coordinate_tolerance_ratio
    )

    boundary = audit_boundary_sharing(
        current_points,
        interior_points,
        patch_faces,
        boundary_edges,
        boundary_vertices,
        current_edges,
        patch_edges,
        tolerance,
    )
    footprint = audit_interior_footprint(
        current_points,
        interior_points,
        loops,
        tolerance,
    )
    geometry = calculate_triangle_geometry(trial_points, patch_faces)
    incident_face_ids = collect_incident_source_faces(boundary_edges, current_edges)
    source_geometry = calculate_triangle_geometry(
        current_points, current_faces[incident_face_ids]
    )
    local_normal, source_normal_dots = calculate_area_weighted_normal(
        source_geometry["normals"],
        source_geometry["areas"],
    )
    local_geometry = describe_local_geometry(
        current_points,
        loops,
        boundary_edges,
        current_edges,
        local_normal,
    )

    area_threshold = max(
        float(cfg.minimum_triangle_area),
        float(np.median(source_geometry["areas"])) * float(cfg.minimum_area_ratio),
    )
    repeated_mask = (
        (patch_faces[:, 0] == patch_faces[:, 1])
        | (patch_faces[:, 1] == patch_faces[:, 2])
        | (patch_faces[:, 2] == patch_faces[:, 0])
    )
    below_area = geometry["areas"] < area_threshold
    area_gate = build_gate(
        not np.any(repeated_mask) and not np.any(below_area),
        "patch_triangle_area_below_minimum",
        {
            "minimum": float(geometry["areas"].min()),
            "repeated_index_face_count": int(np.count_nonzero(repeated_mask)),
            "repeated_index_face_ids": np.flatnonzero(repeated_mask)[:100]
            .astype(int)
            .tolist(),
            "below_minimum_face_count": int(np.count_nonzero(below_area)),
            "below_minimum_face_ids": np.flatnonzero(below_area)[:100]
            .astype(int)
            .tolist(),
        },
        {"minimum": area_threshold},
    )
    aspect_gate = build_gate(
        bool(np.all(np.isfinite(geometry["aspect"])))
        and bool(np.all(geometry["aspect"] <= cfg.maximum_aspect_ratio)),
        "patch_triangle_aspect_exceeded",
        {"maximum": float(geometry["aspect"].max())},
        {"maximum": float(cfg.maximum_aspect_ratio)},
    )
    source_normal_gate = build_gate(
        local_normal is not None
        and source_normal_dots.size > 0
        and float(source_normal_dots.min()) >= cfg.minimum_source_normal_dot,
        "source_local_normal_conflict",
        {
            "area_weighted_normal": local_normal.tolist()
            if local_normal is not None
            else None,
            "minimum_incident_normal_dot": (
                float(source_normal_dots.min()) if source_normal_dots.size else None
            ),
        },
        {"minimum_dot": float(cfg.minimum_source_normal_dot)},
    )
    orientation = _orientation_gate(
        geometry["normals"],
        local_geometry.get("patch_orientation_normal"),
        expected_face_normals,
        cfg,
    )
    transition, curvature = _continuity_gates(
        trial_points,
        current_faces,
        patch_faces,
        boundary_edges,
        current_edges,
        patch_edges,
        cfg,
    )
    internal_transition, internal_curvature = audit_patch_internal_continuity(
        trial_points,
        patch_faces,
        boundary_edges,
        patch_edges,
        cfg.minimum_boundary_normal_dot,
        cfg.maximum_normal_transition_degrees,
        cfg.maximum_curvature_jump,
    )
    intersection = _intersection_gate(
        trial_points,
        current_faces,
        patch_faces,
        cfg.maximum_intersection_candidate_pairs,
    )
    topology = audit_topology_delta(
        current_faces,
        np.vstack([current_faces, patch_faces]),
        len(boundary_edges),
    )
    gates = {
        "boundary_index_sharing": boundary,
        "interior_points_inside_repair_footprint": footprint,
        "source_local_normal_consistent": source_normal_gate,
        "triangle_minimum_area": area_gate,
        "triangle_maximum_aspect_ratio": aspect_gate,
        "patch_orientation": orientation,
        "boundary_normal_transition": transition,
        "boundary_curvature_continuity": curvature,
        "patch_internal_normal_transition": internal_transition,
        "patch_internal_curvature_continuity": internal_curvature,
        "patch_non_adjacent_intersection": intersection,
        "patch_topology_delta": topology,
    }
    failed_codes = [
        gate["reason_code"] for gate in gates.values() if not gate["passed"]
    ]
    return {
        "status": "computed",
        "passed": not failed_codes,
        "reason_codes": list(dict.fromkeys(failed_codes)),
        "limits": asdict(cfg),
        "gates": gates,
        "local_geometry": local_geometry,
        "triangle_quality": {
            "face_count": int(patch_faces.shape[0]),
            "area": summarize_values(geometry["areas"]),
            "aspect_ratio": summarize_values(geometry["aspect"]),
        },
        "repair_mask_area": float(geometry["areas"].sum()),
    }


def validate_transaction_local_quality(
    transaction_index: int,
    transaction_row: dict[str, Any],
    current_mesh: dict[str, Any],
    current_points: np.ndarray,
    current_faces: np.ndarray,
    interior_points: np.ndarray,
    patch_faces: np.ndarray,
) -> list[str]:
    """Independently recompute local patch quality from geometry, not report booleans."""
    loops = tuple(
        np.asarray(row.get("source_vertex_ids", []), dtype=np.int64)
        for row in transaction_row.get("boundary_loops", [])
    )
    loops = tuple(loop for loop in loops if loop.size >= 3)
    if not loops:
        return [f"transaction_independent_local_quality_failed:{transaction_index}"]
    external = (
        current_mesh.get("cell_data", {}).get("external_direction")
        if isinstance(current_mesh, dict)
        else None
    )
    expected = None
    if external is not None:
        directions = np.asarray(external, dtype=np.float64)
        if directions.shape == (np.asarray(current_faces).shape[0], 3):
            vector = directions.sum(axis=0)
            length = float(np.linalg.norm(vector))
            if length > 0.0:
                expected = np.repeat(
                    (vector / length)[None, :],
                    np.asarray(patch_faces).shape[0],
                    axis=0,
                )
    audit = audit_source_primary_patch(
        current_points,
        current_faces,
        interior_points,
        patch_faces,
        loops,
        expected_face_normals=expected,
    )
    return (
        []
        if audit.get("passed") is True
        else [f"transaction_independent_local_quality_failed:{transaction_index}"]
    )


def _orientation_gate(
    patch_normals: np.ndarray,
    patch_orientation_normal: list[float] | None,
    expected_face_normals: np.ndarray | None,
    cfg: PatchQualityLimits,
) -> dict[str, Any]:
    expected = expected_face_normals
    if expected is None and patch_orientation_normal is not None:
        local_normal = np.asarray(patch_orientation_normal, dtype=np.float64)
        expected = np.repeat(local_normal[None, :], patch_normals.shape[0], axis=0)
    if expected is None:
        return build_gate(
            False,
            "patch_winding_evidence_missing",
            None,
            "oriented local normal evidence",
        )
    expected = np.asarray(expected, dtype=np.float64)
    if expected.shape != patch_normals.shape:
        return build_gate(
            False,
            "patch_winding_evidence_missing",
            list(expected.shape),
            list(patch_normals.shape),
        )
    lengths = np.linalg.norm(expected, axis=1)
    if np.any(~np.isfinite(expected)) or np.any(lengths <= 1e-15):
        return build_gate(
            False,
            "patch_winding_evidence_missing",
            "invalid expected normals",
            "finite nonzero normals",
        )
    expected = expected / lengths[:, None]
    dots = np.einsum("ij,ij->i", patch_normals, expected)
    return build_gate(
        bool(np.all(dots >= cfg.minimum_orientation_dot)),
        "patch_winding_flipped",
        {
            "minimum_signed_dot": float(dots.min()),
            "negative_face_count": int(np.count_nonzero(dots < 0.0)),
        },
        {"minimum_signed_dot": float(cfg.minimum_orientation_dot)},
    )


def _continuity_gates(
    points: np.ndarray,
    source_faces: np.ndarray,
    patch_faces: np.ndarray,
    boundary_edges: set[tuple[int, int]],
    source_edges: dict[tuple[int, int], list[tuple[int, int]]],
    patch_edges: dict[tuple[int, int], list[tuple[int, int]]],
    cfg: PatchQualityLimits,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_normals = calculate_triangle_geometry(points, source_faces)["normals"]
    patch_normals = calculate_triangle_geometry(points, patch_faces)["normals"]
    transition_angles = []
    patch_curvatures = []
    source_curvatures = []
    source_edge_references = []
    boundary_lengths = []
    for edge in sorted(boundary_edges):
        source_occurrences = source_edges.get(edge, [])
        patch_occurrences = patch_edges.get(edge, [])
        if len(source_occurrences) != 1 or len(patch_occurrences) != 1:
            continue
        source_face_id = source_occurrences[0][0]
        patch_face_id = patch_occurrences[0][0]
        angle = calculate_normal_angle(
            source_normals[source_face_id], patch_normals[patch_face_id]
        )
        edge_length = float(np.linalg.norm(points[edge[0]] - points[edge[1]]))
        transition_angles.append(angle)
        boundary_lengths.append(edge_length)
        patch_curvatures.append(angle / max(edge_length, 1e-30))
        face = source_faces[source_face_id]
        edge_source_curvatures = []
        for local_index in range(3):
            neighbor_left = int(face[local_index])
            neighbor_right = int(face[(local_index + 1) % 3])
            neighbor_edge = (
                (neighbor_left, neighbor_right)
                if neighbor_left < neighbor_right
                else (neighbor_right, neighbor_left)
            )
            if neighbor_edge == edge:
                continue
            neighbors = source_edges.get(neighbor_edge, [])
            if len(neighbors) != 2:
                continue
            neighbor_id = (
                neighbors[0][0]
                if neighbors[1][0] == source_face_id
                else neighbors[1][0]
            )
            neighbor_length = float(
                np.linalg.norm(points[neighbor_edge[0]] - points[neighbor_edge[1]])
            )
            value = calculate_normal_angle(
                source_normals[source_face_id], source_normals[neighbor_id]
            ) / max(neighbor_length, 1e-30)
            source_curvatures.append(value)
            edge_source_curvatures.append(value)
        source_edge_references.append(
            float(np.median(edge_source_curvatures))
            if edge_source_curvatures
            else np.nan
        )
    angles = np.asarray(transition_angles, dtype=np.float64)
    angle_degrees = np.degrees(angles)
    normal_gate = build_gate(
        angles.size == len(boundary_edges)
        and bool(np.all(np.cos(angles) >= cfg.minimum_boundary_normal_dot))
        and bool(np.all(angle_degrees <= cfg.maximum_normal_transition_degrees)),
        "patch_normal_transition_failed",
        {
            "checked_edges": int(angles.size),
            "maximum_degrees": float(angle_degrees.max()) if angles.size else None,
            "minimum_dot": float(np.cos(angles).min()) if angles.size else None,
        },
        {
            "required_edges": len(boundary_edges),
            "maximum_degrees": float(cfg.maximum_normal_transition_degrees),
            "minimum_dot": float(cfg.minimum_boundary_normal_dot),
        },
    )
    source_values = np.asarray(source_curvatures, dtype=np.float64)
    source_references = np.asarray(source_edge_references, dtype=np.float64)
    patch_values = np.asarray(patch_curvatures, dtype=np.float64)
    edge_scales = np.asarray(boundary_lengths, dtype=np.float64)
    jumps = np.abs(patch_values - source_references) * edge_scales
    curvature_gate = build_gate(
        source_values.size > 0
        and patch_values.size == len(boundary_edges)
        and source_references.size == patch_values.size
        and bool(np.all(np.isfinite(source_references)))
        and bool(np.all(jumps <= cfg.maximum_curvature_jump)),
        "patch_curvature_continuity_failed",
        {
            "status": "computed"
            if source_values.size
            else "incomplete_source_neighborhood",
            "source_sample_count": int(source_values.size),
            "source_per_edge_reference": summarize_values(source_references),
            "patch": summarize_values(patch_values),
            "normalized_jump": summarize_values(jumps),
            "normalization_length": summarize_values(edge_scales),
        },
        {"maximum_normalized_jump": float(cfg.maximum_curvature_jump)},
    )
    return normal_gate, curvature_gate


def _intersection_gate(
    points: np.ndarray,
    current_faces: np.ndarray,
    patch_faces: np.ndarray,
    max_candidate_pairs: int,
) -> dict[str, Any]:
    try:
        report = validate_local_patch_intersections(
            points,
            current_faces,
            patch_faces,
            max_candidate_pairs=max_candidate_pairs,
        )
    except Exception as exc:  # Geometry backends must fail only this transaction.
        report = {
            "status": "check_failed",
            "passed": False,
            "failure_reason": f"{type(exc).__name__}: {exc}",
        }
    boundary_only_contacts = int(report.get("ignored_boundary_contacts", 0))
    degenerate_pairs = int(report.get("degenerate_candidate_pairs", 0))
    complete = report.get("status") == "computed" and degenerate_pairs == 0
    passed = bool(complete and report.get("passed")) and boundary_only_contacts == 0
    if not complete:
        reason = "patch_self_intersection_check_incomplete"
    elif boundary_only_contacts:
        reason = "patch_non_topological_boundary_contact"
    else:
        reason = "patch_self_intersection_detected"
    return build_gate(
        passed,
        reason,
        {
            **report,
            "non_topological_boundary_contacts_rejected": boundary_only_contacts,
            "degenerate_candidate_pairs_rejected": degenerate_pairs,
        },
        {
            "status": "computed",
            "intersection_pairs": 0,
            "non_topological_boundary_contacts": 0,
            "degenerate_candidate_pairs": 0,
        },
    )


def _failed_audit(code: str, message: str, cfg: PatchQualityLimits) -> dict[str, Any]:
    gate = build_gate(False, code, message, "valid append-only patch delta")
    return {
        "status": "not_computed_invalid_input",
        "passed": False,
        "reason_codes": [code],
        "limits": asdict(cfg),
        "gates": {"patch_delta_schema": gate},
        "local_geometry": {},
        "triangle_quality": {},
        "repair_mask_area": 0.0,
    }
