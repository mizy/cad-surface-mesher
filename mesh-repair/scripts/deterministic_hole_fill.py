from __future__ import annotations

from typing import Any

import numpy as np


__all__ = [
    "classify_triangle_pair_contact",
    "run_deterministic_hole_fill",
    "triangulate_fixed_boundary_loop",
    "validate_local_patch_intersections",
]


SOURCE_FACE_ORIGIN = 0
HOLE_FILL_FACE_ORIGIN = 3


def triangulate_fixed_boundary_loop(
    points: np.ndarray,
    source_faces: np.ndarray,
    boundary_loop: np.ndarray | list[int],
    *,
    geometric_tolerance: float | None = None,
    beam_width: int = 16,
    max_candidates: int = 32,
    minimum_normal_dot: float = -0.25,
) -> dict[str, Any]:
    """Triangulate one source boundary without moving or duplicating its vertices.

    The selected loop is re-oriented from the incident source faces.  Generated
    faces traverse every source boundary edge in the opposite direction.  Mildly
    non-planar loops are handled by trying several deterministic local charts and
    ranking constrained ear-clipping triangulations in the original 3-D geometry.
    A loop for which no simple chart and non-folded triangulation can be certified
    is rejected instead of being flattened destructively.
    """

    prepared = _prepare_boundary(
        points,
        source_faces,
        boundary_loop,
        geometric_tolerance=geometric_tolerance,
    )
    if not prepared["success"]:
        return {
            "success": False,
            "generated_faces": _empty_faces(),
            "oriented_boundary_loop": np.empty(0, dtype=np.int64),
            "failure_reason_codes": prepared["failure_reason_codes"],
            "diagnostics": prepared["diagnostics"],
        }

    candidates, construction = _triangulation_candidates(
        prepared["points"],
        prepared["oriented_loop"],
        prepared["tolerance"],
        beam_width=max(int(beam_width), 1),
        max_candidates=max(int(max_candidates), 1),
        minimum_normal_dot=float(minimum_normal_dot),
    )
    diagnostics = {
        "stage": "fixed_source_boundary_triangulation",
        "method": "multi_chart_constrained_ear_clipping",
        "boundary": prepared["diagnostics"],
        **construction,
    }
    if not candidates:
        return {
            "success": False,
            "generated_faces": _empty_faces(),
            "oriented_boundary_loop": prepared["oriented_loop"],
            "failure_reason_codes": ["hole_fill_constrained_triangulation_failed"],
            "diagnostics": diagnostics,
        }

    selected = candidates[0]
    diagnostics["selected_candidate"] = selected["diagnostics"]
    diagnostics["generated_face_count"] = int(selected["faces"].shape[0])
    diagnostics["source_points_modified"] = False
    diagnostics["new_points_added"] = 0
    return {
        "success": True,
        "generated_faces": selected["faces"].copy(),
        "oriented_boundary_loop": prepared["oriented_loop"].copy(),
        "failure_reason_codes": [],
        "diagnostics": diagnostics,
    }


def classify_triangle_pair_contact(
    points: np.ndarray,
    left_face: np.ndarray,
    right_face: np.ndarray,
    *,
    geometric_tolerance: float,
) -> dict[str, str]:
    """Classify contact while allowing only a genuinely shared topological feature."""
    return _triangle_pair_contact(
        np.asarray(points, dtype=np.float64),
        np.asarray(left_face, dtype=np.int64),
        np.asarray(right_face, dtype=np.int64),
        set(),
        float(geometric_tolerance),
        other_is_patch=True,
    )


def validate_local_patch_intersections(
    points: np.ndarray,
    source_faces: np.ndarray,
    generated_faces: np.ndarray,
    *,
    geometric_tolerance: float | None = None,
    max_candidate_pairs: int = 250_000,
    max_reported_pairs: int = 200,
) -> dict[str, Any]:
    """Check generated faces for true local penetration against the current mesh.

    Triangle pairs are not discarded merely because they share a vertex.  Their
    geometric contact is classified instead: contact confined to a shared
    topological feature, or to the generated patch's outer boundary, is allowed;
    contact entering a patch interior and coplanar area overlap are intersections.
    This avoids the common VTK false positive for a conformal cap while retaining
    detection of a triangle that shares one vertex and penetrates elsewhere.
    """

    validation = _validate_mesh_arrays(points, source_faces, allow_empty_faces=True)
    if validation is not None:
        return _intersection_failure("invalid_source_mesh", validation)
    validation = _validate_mesh_arrays(points, generated_faces, allow_empty_faces=True)
    if validation is not None:
        return _intersection_failure("invalid_generated_patch", validation)

    points = np.asarray(points, dtype=np.float64)
    source_faces = np.asarray(source_faces, dtype=np.int64).reshape(-1, 3)
    generated_faces = np.asarray(generated_faces, dtype=np.int64).reshape(-1, 3)
    if generated_faces.shape[0] == 0:
        return {
            "method": "focused_proper_triangle_intersection",
            "status": "computed",
            "passed": True,
            "intersection_pairs": 0,
            "candidate_pairs_tested": 0,
            "reported_pairs": [],
            "ignored_topological_contacts": 0,
            "ignored_boundary_contacts": 0,
            "max_candidate_pairs": int(max_candidate_pairs),
            "contact_policy": _contact_policy(),
        }

    tolerance = _geometric_tolerance(points, geometric_tolerance)
    patch_boundary_edges = _boundary_edges(generated_faces)
    source_triangles = points[source_faces]
    patch_triangles = points[generated_faces]
    source_min = source_triangles.min(axis=1) - tolerance
    source_max = source_triangles.max(axis=1) + tolerance
    patch_min = patch_triangles.min(axis=1) - tolerance
    patch_max = patch_triangles.max(axis=1) + tolerance

    tested = 0
    proper = 0
    ignored_topological = 0
    ignored_boundary = 0
    degenerate_candidates = 0
    reported: list[dict[str, Any]] = []

    def consume(
        patch_id: int,
        other_id: int,
        other_role: str,
        other_face: np.ndarray,
    ) -> dict[str, Any] | None:
        nonlocal tested, proper, ignored_topological, ignored_boundary, degenerate_candidates
        tested += 1
        if tested > max_candidate_pairs:
            return {
                "method": "focused_proper_triangle_intersection",
                "status": "incomplete_candidate_pair_limit_exceeded",
                "passed": False,
                "intersection_pairs": proper,
                "candidate_pairs_tested": tested - 1,
                "reported_pairs": reported,
                "ignored_topological_contacts": ignored_topological,
                "ignored_boundary_contacts": ignored_boundary,
                "degenerate_candidate_pairs": degenerate_candidates,
                "max_candidate_pairs": int(max_candidate_pairs),
                "failure_reason": "local intersection broad phase exceeded the configured candidate-pair limit",
                "contact_policy": _contact_policy(),
            }
        contact = _triangle_pair_contact(
            points,
            generated_faces[patch_id],
            other_face,
            patch_boundary_edges,
            tolerance,
            other_is_patch=other_role == "patch",
        )
        if contact["classification"] == "degenerate":
            degenerate_candidates += 1
            return None
        if contact["classification"] == "proper_intersection":
            proper += 1
            if len(reported) < max_reported_pairs:
                reported.append(
                    {
                        "patch_face_id": int(patch_id),
                        "other_role": other_role,
                        "other_face_id": int(other_id),
                        "contact": contact["contact"],
                    }
                )
        elif contact["classification"] == "allowed_topological_contact":
            ignored_topological += 1
        elif contact["classification"] == "allowed_boundary_contact":
            ignored_boundary += 1
        return None

    for patch_id in range(generated_faces.shape[0]):
        overlaps = np.all(source_min <= patch_max[patch_id], axis=1) & np.all(
            patch_min[patch_id] <= source_max,
            axis=1,
        )
        for source_id_value in np.flatnonzero(overlaps):
            limit_result = consume(
                patch_id,
                int(source_id_value),
                "source",
                source_faces[int(source_id_value)],
            )
            if limit_result is not None:
                return limit_result
        if patch_id + 1 < generated_faces.shape[0]:
            later_min = patch_min[patch_id + 1 :]
            later_max = patch_max[patch_id + 1 :]
            overlaps = np.all(later_min <= patch_max[patch_id], axis=1) & np.all(
                patch_min[patch_id] <= later_max,
                axis=1,
            )
            for relative_value in np.flatnonzero(overlaps):
                other_id = patch_id + 1 + int(relative_value)
                limit_result = consume(
                    patch_id,
                    other_id,
                    "patch",
                    generated_faces[other_id],
                )
                if limit_result is not None:
                    return limit_result

    return {
        "method": "focused_proper_triangle_intersection",
        "status": "computed",
        "passed": proper == 0,
        "intersection_pairs": proper,
        "candidate_pairs_tested": tested,
        "reported_pairs": reported,
        "truncated": proper > len(reported),
        "ignored_topological_contacts": ignored_topological,
        "ignored_boundary_contacts": ignored_boundary,
        "degenerate_candidate_pairs": degenerate_candidates,
        "max_candidate_pairs": int(max_candidate_pairs),
        "geometric_tolerance": float(tolerance),
        "contact_policy": _contact_policy(),
    }


def run_deterministic_hole_fill(
    points: np.ndarray,
    source_faces: np.ndarray,
    boundary_loop: np.ndarray | list[int],
    provenance: dict[str, np.ndarray] | None = None,
    *,
    fusion_region_id: int = 0,
    face_origin_value: int = HOLE_FILL_FACE_ORIGIN,
    geometric_tolerance: float | None = None,
    beam_width: int = 16,
    max_candidates: int = 32,
    minimum_normal_dot: float = -0.25,
    max_candidate_pairs: int = 250_000,
) -> dict[str, Any]:
    """Run a failure-safe deterministic small-hole transaction.

    On commit, only faces and per-face provenance are appended; ``points`` are
    byte-for-byte unchanged.  On any failure, returned geometry and provenance
    are copies of the input state, ``generated_faces`` is empty, and any rejected
    triangulation is available only under ``candidate`` for diagnostics.
    """

    input_points = np.asarray(points)
    input_faces = np.asarray(source_faces)
    input_validation = _validate_mesh_arrays(input_points, input_faces, allow_empty_faces=False)
    if input_validation is not None:
        return _rolled_back(
            input_points,
            input_faces,
            provenance,
            ["invalid_source_mesh"],
            {"stage": "transaction_precondition", "message": input_validation},
        )
    points_array = np.asarray(points, dtype=np.float64)
    faces_array = np.asarray(source_faces, dtype=np.int64).reshape(-1, 3)
    provenance_validation = _validate_provenance(provenance, faces_array.shape[0])
    if provenance_validation is not None:
        return _rolled_back(
            input_points,
            input_faces,
            provenance,
            ["invalid_source_provenance"],
            {"stage": "transaction_precondition", "message": provenance_validation},
        )

    prepared = _prepare_boundary(
        points_array,
        faces_array,
        boundary_loop,
        geometric_tolerance=geometric_tolerance,
    )
    if not prepared["success"]:
        return _rolled_back(
            input_points,
            input_faces,
            provenance,
            prepared["failure_reason_codes"],
            prepared["diagnostics"],
        )

    candidates, construction = _triangulation_candidates(
        points_array,
        prepared["oriented_loop"],
        prepared["tolerance"],
        beam_width=max(int(beam_width), 1),
        max_candidates=max(int(max_candidates), 1),
        minimum_normal_dot=float(minimum_normal_dot),
    )
    if not candidates:
        return _rolled_back(
            input_points,
            input_faces,
            provenance,
            ["hole_fill_constrained_triangulation_failed"],
            {
                "stage": "fixed_source_boundary_triangulation",
                "boundary": prepared["diagnostics"],
                **construction,
            },
        )

    rejected_candidates = []
    selected: dict[str, Any] | None = None
    selected_intersection: dict[str, Any] | None = None
    for candidate_index, candidate in enumerate(candidates):
        local_intersection = validate_local_patch_intersections(
            points_array,
            faces_array,
            candidate["faces"],
            geometric_tolerance=prepared["tolerance"],
            max_candidate_pairs=max_candidate_pairs,
        )
        if local_intersection["passed"]:
            selected = candidate
            selected_intersection = local_intersection
            break
        rejected_candidates.append(
            {
                "candidate_index": candidate_index,
                "reason": (
                    "local_self_intersection_detected"
                    if local_intersection.get("status") == "computed"
                    else "local_self_intersection_check_incomplete"
                ),
                "intersection_pairs": int(local_intersection.get("intersection_pairs", 0)),
                "status": local_intersection.get("status"),
            }
        )

    if selected is None or selected_intersection is None:
        best_candidate = candidates[0]
        local_intersection = validate_local_patch_intersections(
            points_array,
            faces_array,
            best_candidate["faces"],
            geometric_tolerance=prepared["tolerance"],
            max_candidate_pairs=max_candidate_pairs,
        )
        reason = (
            "local_self_intersection_detected"
            if local_intersection.get("status") == "computed"
            else "local_self_intersection_check_incomplete"
        )
        return _rolled_back(
            input_points,
            input_faces,
            provenance,
            [reason],
            {
                "stage": "local_patch_acceptance",
                "boundary": prepared["diagnostics"],
                "triangulation": construction,
                "rejected_candidates": rejected_candidates,
            },
            candidate={"generated_faces": best_candidate["faces"].copy()},
            local_intersection=local_intersection,
        )

    generated_faces = selected["faces"]
    topology = _patch_topology_diagnostics(
        points_array,
        faces_array,
        generated_faces,
        prepared["oriented_loop"],
        prepared["tolerance"],
    )
    if not topology["passed"]:
        return _rolled_back(
            input_points,
            input_faces,
            provenance,
            topology["reason_codes"],
            {
                "stage": "patch_topology_acceptance",
                "boundary": prepared["diagnostics"],
                "triangulation": construction,
                "topology": topology,
            },
            candidate={"generated_faces": generated_faces.copy()},
            local_intersection=selected_intersection,
        )

    base_provenance = _normalized_source_provenance(provenance, faces_array.shape[0])
    generated_provenance = _generated_provenance(
        base_provenance,
        generated_faces.shape[0],
        fusion_region_id=int(fusion_region_id),
        face_origin_value=int(face_origin_value),
    )
    final_provenance = {
        name: np.concatenate([values, generated_provenance[name]], axis=0)
        for name, values in base_provenance.items()
    }
    final_faces = np.vstack([faces_array, generated_faces])
    generated_face_ids = np.arange(
        faces_array.shape[0],
        final_faces.shape[0],
        dtype=np.int64,
    )
    diagnostics = {
        "stage": "deterministic_hole_fill_transaction",
        "method": "fixed_source_boundary_multi_chart_constrained_ear_clipping",
        "boundary": prepared["diagnostics"],
        "triangulation": {
            **construction,
            "selected_candidate": selected["diagnostics"],
            "rejected_candidate_count": len(rejected_candidates),
            "rejected_candidates": rejected_candidates,
        },
        "topology": topology,
        "provenance": {
            "fields": sorted(final_provenance),
            "generated_face_origin": int(face_origin_value),
            "generated_source_triangle_index": -1,
            "fusion_region_id": int(fusion_region_id),
        },
        "source_points_modified": False,
        "new_points_added": 0,
    }
    return {
        "success": True,
        "committed": True,
        "status": "committed",
        "points": input_points.copy(),
        "faces": final_faces,
        "provenance": final_provenance,
        "generated_faces": generated_faces.copy(),
        "generated_face_ids": generated_face_ids,
        "generated_provenance": generated_provenance,
        "oriented_boundary_loop": prepared["oriented_loop"].copy(),
        "candidate": {"generated_faces": generated_faces.copy()},
        "diagnostics": diagnostics,
        "local_intersection": selected_intersection,
        "failure_reason_codes": [],
    }


def _prepare_boundary(
    points: np.ndarray,
    source_faces: np.ndarray,
    boundary_loop: np.ndarray | list[int],
    *,
    geometric_tolerance: float | None,
) -> dict[str, Any]:
    validation = _validate_mesh_arrays(points, source_faces, allow_empty_faces=False)
    if validation is not None:
        return _boundary_failure("invalid_source_mesh", validation)
    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(source_faces, dtype=np.int64).reshape(-1, 3)
    loop = np.asarray(boundary_loop, dtype=np.int64).reshape(-1)
    if loop.size > 1 and loop[0] == loop[-1]:
        loop = loop[:-1]
    if loop.size < 3:
        return _boundary_failure("boundary_loop_too_small", "a boundary loop needs at least three vertices")
    if np.unique(loop).size != loop.size:
        return _boundary_failure("boundary_loop_repeats_vertex", "a simple boundary loop cannot repeat vertices")
    if np.any((loop < 0) | (loop >= points.shape[0])):
        return _boundary_failure("boundary_loop_vertex_out_of_range", "boundary loop vertex IDs are out of range")

    try:
        tolerance = _geometric_tolerance(points[loop], geometric_tolerance)
    except (TypeError, ValueError):
        return _boundary_failure(
            "invalid_geometric_tolerance",
            "geometric_tolerance must be finite and positive",
        )
    loop_points = points[loop]
    edge_lengths = np.linalg.norm(np.roll(loop_points, -1, axis=0) - loop_points, axis=1)
    if np.any(edge_lengths <= tolerance):
        edge_id = int(np.flatnonzero(edge_lengths <= tolerance)[0])
        return _boundary_failure(
            "boundary_loop_zero_length_edge",
            "a boundary loop cannot contain a zero-length edge",
            edge_vertex_ids=[int(loop[edge_id]), int(loop[(edge_id + 1) % loop.size])],
            edge_length=float(edge_lengths[edge_id]),
            geometric_tolerance=float(tolerance),
        )

    edge_faces = _edge_face_map(faces)
    directions = []
    incident_face_ids = []
    for index, left_value in enumerate(loop):
        left = int(left_value)
        right = int(loop[(index + 1) % loop.size])
        edge = tuple(sorted((left, right)))
        incident = edge_faces.get(edge, [])
        if len(incident) != 1:
            return _boundary_failure(
                "loop_edge_is_not_boundary",
                "every selected loop edge must have exactly one incident source face",
                edge=list(edge),
                incidence=len(incident),
            )
        face_id = incident[0]
        directed = _directed_edge(faces[face_id], edge)
        directions.append(1 if directed == (left, right) else -1)
        incident_face_ids.append(face_id)
    if all(value == -1 for value in directions):
        loop = loop[::-1].copy()
        incident_face_ids = incident_face_ids[::-1]
    elif not all(value == 1 for value in directions):
        return _boundary_failure(
            "boundary_orientation_inconsistent",
            "the selected loop mixes source-face-induced edge directions",
        )
    loop = _rotate_to_smallest(loop)

    centered = points[loop] - points[loop].mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    scale = max(float(np.linalg.norm(np.ptp(points[loop], axis=0))), tolerance)
    planarity_ratio = float(singular[-1] / max(singular[0], tolerance))
    return {
        "success": True,
        "points": points,
        "faces": faces,
        "oriented_loop": loop,
        "tolerance": tolerance,
        "failure_reason_codes": [],
        "diagnostics": {
            "stage": "source_boundary_validation",
            "vertex_count": int(loop.size),
            "edge_count": int(loop.size),
            "incident_source_face_count": len(set(incident_face_ids)),
            "orientation": "source_face_winding_induced",
            "patch_boundary_traversal": "reversed",
            "boundary_vertices_fixed": True,
            "geometric_tolerance": float(tolerance),
            "local_scale": float(scale),
            "planarity_ratio": planarity_ratio,
            "max_plane_deviation_ratio": _max_plane_deviation_ratio(points[loop], scale),
        },
    }


def _triangulation_candidates(
    points: np.ndarray,
    oriented_loop: np.ndarray,
    tolerance: float,
    *,
    beam_width: int,
    max_candidates: int,
    minimum_normal_dot: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cap_loop = oriented_loop[::-1].copy()
    loop_points = points[cap_loop]
    charts = _projection_charts(loop_points, tolerance)
    all_candidates: dict[tuple[tuple[int, int, int], ...], dict[str, Any]] = {}
    chart_reports = []
    rejection_counts: dict[str, int] = {}
    for chart in charts:
        report = {name: value for name, value in chart.items() if name != "projected"}
        if not chart["valid"]:
            chart_reports.append(report)
            rejection_counts[chart["reason"]] = rejection_counts.get(chart["reason"], 0) + 1
            continue
        triangulations = _ear_clip_beam(
            points,
            cap_loop,
            chart["projected"],
            tolerance,
            beam_width=beam_width,
            max_candidates=max_candidates,
        )
        report["raw_candidate_count"] = len(triangulations)
        chart_reports.append(report)
        for faces in triangulations:
            assessed = _assess_triangulation(
                points,
                faces,
                cap_loop,
                tolerance,
                minimum_normal_dot,
            )
            if not assessed["passed"]:
                code = assessed["reason"]
                rejection_counts[code] = rejection_counts.get(code, 0) + 1
                continue
            key = tuple(sorted(tuple(sorted(int(value) for value in face)) for face in faces))
            candidate = {
                "faces": faces,
                "score": assessed["score"],
                "diagnostics": {
                    **assessed,
                    "projection_method": chart["method"],
                    "projected_signed_area": chart["signed_area"],
                },
            }
            existing = all_candidates.get(key)
            if existing is None or candidate["score"] > existing["score"]:
                all_candidates[key] = candidate
    candidates = sorted(
        all_candidates.values(),
        key=lambda item: (
            -item["score"][0],
            -item["score"][1],
            item["score"][2],
            tuple(item["faces"].ravel().tolist()),
        ),
    )[:max_candidates]
    return candidates, {
        "projection_candidates": chart_reports,
        "valid_triangulation_candidates": len(candidates),
        "rejection_counts": rejection_counts,
        "beam_width": int(beam_width),
        "maximum_retained_candidates": int(max_candidates),
        "minimum_normal_dot_threshold": float(minimum_normal_dot),
    }


def _projection_charts(loop_points: np.ndarray, tolerance: float) -> list[dict[str, Any]]:
    centered = loop_points - loop_points.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normals: list[tuple[str, np.ndarray]] = [("pca_best_fit_plane", vh[-1])]
    newell = np.sum(np.cross(loop_points, np.roll(loop_points, -1, axis=0)), axis=0)
    if np.linalg.norm(newell) > tolerance * tolerance:
        normals.append(("newell_boundary_normal", newell))
    axis_names = ("drop_x", "drop_y", "drop_z")
    for axis, name in enumerate(axis_names):
        normal = np.zeros(3, dtype=np.float64)
        normal[axis] = 1.0
        normals.append((name, normal))

    charts = []
    accepted_normals: list[np.ndarray] = []
    scale = max(float(np.linalg.norm(np.ptp(loop_points, axis=0))), tolerance)
    area_tolerance = max(tolerance * scale * 8.0, scale * scale * 1e-14)
    for method, normal_value in normals:
        normal = np.asarray(normal_value, dtype=np.float64)
        length = float(np.linalg.norm(normal))
        if length <= tolerance:
            continue
        normal /= length
        if any(abs(float(np.dot(normal, other))) >= 1.0 - 1e-10 for other in accepted_normals):
            continue
        accepted_normals.append(normal)
        basis_u, basis_v = _plane_basis(normal, centered)
        projected = np.column_stack([centered @ basis_u, centered @ basis_v])
        area = _polygon_area(projected)
        edge_3d = np.linalg.norm(np.roll(loop_points, -1, axis=0) - loop_points, axis=1)
        edge_2d = np.linalg.norm(np.roll(projected, -1, axis=0) - projected, axis=1)
        minimum_edge_ratio = float(np.min(edge_2d / np.maximum(edge_3d, tolerance)))
        intersections = _polygon_self_intersections_2d(projected, tolerance)
        reason = None
        if abs(area) <= area_tolerance:
            reason = "zero_projected_area"
        elif intersections:
            reason = "projected_boundary_self_intersection"
        elif minimum_edge_ratio <= 1e-8:
            reason = "projected_boundary_edge_collapsed"
        charts.append(
            {
                "method": method,
                "valid": reason is None,
                "reason": reason,
                "signed_area": float(area),
                "minimum_projected_edge_ratio": minimum_edge_ratio,
                "projected_self_intersection_count": len(intersections),
                "projected": projected,
            }
        )
    return charts


def _ear_clip_beam(
    points: np.ndarray,
    cap_loop: np.ndarray,
    projected: np.ndarray,
    tolerance: float,
    *,
    beam_width: int,
    max_candidates: int,
) -> list[np.ndarray]:
    signed_area = _polygon_area(projected)
    orientation = 1.0 if signed_area > 0.0 else -1.0
    projected_scale = max(float(np.linalg.norm(np.ptp(projected, axis=0))), tolerance)
    area_tolerance = max(tolerance * projected_scale * 8.0, projected_scale**2 * 1e-14)
    states = [
        {
            "remaining": tuple(range(cap_loop.size)),
            "triangles": tuple(),
            "min_quality": 1.0,
            "quality_sum": 0.0,
        }
    ]
    for _ in range(max(cap_loop.size - 3, 0)):
        expanded = []
        for state in states:
            remaining = state["remaining"]
            for position, current in enumerate(remaining):
                previous = remaining[position - 1]
                following = remaining[(position + 1) % len(remaining)]
                if not _is_valid_ear(
                    projected,
                    remaining,
                    previous,
                    current,
                    following,
                    orientation,
                    area_tolerance,
                ):
                    continue
                face = np.asarray(
                    [cap_loop[previous], cap_loop[current], cap_loop[following]],
                    dtype=np.int64,
                )
                quality = _triangle_quality(points[face], tolerance)
                if quality <= 0.0:
                    continue
                next_remaining = remaining[:position] + remaining[position + 1 :]
                expanded.append(
                    {
                        "remaining": next_remaining,
                        "triangles": state["triangles"] + (tuple(int(value) for value in face),),
                        "min_quality": min(float(state["min_quality"]), quality),
                        "quality_sum": float(state["quality_sum"]) + quality,
                    }
                )
        if not expanded:
            return []
        expanded.sort(
            key=lambda state: (
                -state["min_quality"],
                -state["quality_sum"],
                state["remaining"],
                state["triangles"],
            )
        )
        states = expanded[:beam_width]

    completed: dict[tuple[tuple[int, int, int], ...], np.ndarray] = {}
    for state in states:
        remaining = state["remaining"]
        if len(remaining) != 3:
            continue
        final_face = tuple(int(cap_loop[index]) for index in remaining)
        faces = np.asarray(state["triangles"] + (final_face,), dtype=np.int64)
        if _triangle_quality(points[faces[-1]], tolerance) <= 0.0:
            continue
        key = tuple(sorted(tuple(sorted(int(value) for value in face)) for face in faces))
        completed.setdefault(key, faces)
    return list(completed.values())[:max_candidates]


def _is_valid_ear(
    projected: np.ndarray,
    remaining: tuple[int, ...],
    previous: int,
    current: int,
    following: int,
    orientation: float,
    tolerance: float,
) -> bool:
    a, b, c = projected[[previous, current, following]]
    if orientation * _cross_2d(b - a, c - b) <= tolerance:
        return False
    for candidate in remaining:
        if candidate in (previous, current, following):
            continue
        if _point_in_triangle_2d(projected[candidate], a, b, c, orientation, tolerance):
            return False
    for edge_index, left in enumerate(remaining):
        right = remaining[(edge_index + 1) % len(remaining)]
        if left in (previous, following) or right in (previous, following):
            continue
        if _segments_contact_2d(
            projected[previous],
            projected[following],
            projected[left],
            projected[right],
            tolerance,
        ):
            return False
    return True


def _assess_triangulation(
    points: np.ndarray,
    faces: np.ndarray,
    cap_loop: np.ndarray,
    tolerance: float,
    minimum_normal_dot: float,
) -> dict[str, Any]:
    triangles = points[faces]
    raw_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_areas = np.linalg.norm(raw_normals, axis=1)
    if np.any(double_areas <= tolerance * tolerance):
        return {"passed": False, "reason": "degenerate_generated_triangle"}
    normals = raw_normals / double_areas[:, None]
    reference = np.sum(np.cross(points[cap_loop], np.roll(points[cap_loop], -1, axis=0)), axis=0)
    reference_length = float(np.linalg.norm(reference))
    reference_dots = np.ones(faces.shape[0], dtype=np.float64)
    if reference_length > tolerance * tolerance:
        reference /= reference_length
        reference_dots = normals @ reference

    internal_dots = []
    edge_occurrences: dict[tuple[int, int], list[tuple[int, tuple[int, int]]]] = {}
    for face_id, face in enumerate(faces):
        for index, left_value in enumerate(face):
            left = int(left_value)
            right = int(face[(index + 1) % 3])
            edge_occurrences.setdefault(tuple(sorted((left, right))), []).append((face_id, (left, right)))
    for occurrences in edge_occurrences.values():
        if len(occurrences) == 2:
            internal_dots.append(float(np.dot(normals[occurrences[0][0]], normals[occurrences[1][0]])))
    minimum_internal_dot = min(internal_dots, default=1.0)
    minimum_reference = float(reference_dots.min())
    if minimum_internal_dot < minimum_normal_dot or minimum_reference < minimum_normal_dot:
        return {
            "passed": False,
            "reason": "folded_non_planar_triangulation",
            "minimum_internal_normal_dot": minimum_internal_dot,
            "minimum_reference_normal_dot": minimum_reference,
        }
    qualities = np.asarray([_triangle_quality(triangle, tolerance) for triangle in triangles])
    diagonal_length = _internal_edge_length(points, edge_occurrences)
    return {
        "passed": True,
        "reason": None,
        "minimum_triangle_quality": float(qualities.min()),
        "mean_triangle_quality": float(qualities.mean()),
        "minimum_internal_normal_dot": minimum_internal_dot,
        "minimum_reference_normal_dot": minimum_reference,
        "maximum_plane_deviation": float(np.max(np.abs((points[cap_loop] - points[cap_loop].mean(axis=0)) @ reference)))
        if reference_length > tolerance * tolerance
        else None,
        "internal_diagonal_length": diagonal_length,
        "score": (float(qualities.min()), min(minimum_internal_dot, minimum_reference), diagonal_length),
    }


def _patch_topology_diagnostics(
    points: np.ndarray,
    source_faces: np.ndarray,
    generated_faces: np.ndarray,
    oriented_loop: np.ndarray,
    tolerance: float,
) -> dict[str, Any]:
    before = _topology_counts(source_faces)
    combined = np.vstack([source_faces, generated_faces])
    after = _topology_counts(combined)
    occurrences = _directed_edge_occurrences(combined)
    seam_edges = [
        tuple(sorted((int(left), int(oriented_loop[(index + 1) % oriented_loop.size]))))
        for index, left in enumerate(oriented_loop)
    ]
    seam_incidence_two = all(len(occurrences.get(edge, [])) == 2 for edge in seam_edges)
    seam_opposite = all(
        len(occurrences.get(edge, [])) == 2
        and occurrences[edge][0] == occurrences[edge][1][::-1]
        for edge in seam_edges
    )
    patch_edges = _edge_face_map(generated_faces)
    internal_edges = [edge for edge, face_ids in patch_edges.items() if len(face_ids) == 2]
    generated_occurrences = _directed_edge_occurrences(generated_faces)
    internal_opposite = all(
        len(generated_occurrences.get(edge, [])) == 2
        and generated_occurrences[edge][0] == generated_occurrences[edge][1][::-1]
        for edge in internal_edges
    )
    raw_normals = np.cross(
        points[generated_faces[:, 1]] - points[generated_faces[:, 0]],
        points[generated_faces[:, 2]] - points[generated_faces[:, 0]],
    )
    degenerate = np.flatnonzero(np.linalg.norm(raw_normals, axis=1) <= tolerance * tolerance)
    reason_codes = []
    if not seam_incidence_two:
        reason_codes.append("hole_fill_seam_incidence_invalid")
    if not seam_opposite or not internal_opposite:
        reason_codes.append("hole_fill_orientation_invalid")
    if degenerate.size:
        reason_codes.append("hole_fill_degenerate_triangle")
    if after["boundary_edges"] != before["boundary_edges"] - oriented_loop.size:
        reason_codes.append("hole_fill_boundary_count_not_reduced_as_expected")
    if after["non_manifold_edges"] > before["non_manifold_edges"]:
        reason_codes.append("hole_fill_created_non_manifold_edge")
    return {
        "passed": not reason_codes,
        "reason_codes": reason_codes,
        "before": before,
        "after": after,
        "expected_boundary_edge_reduction": int(oriented_loop.size),
        "actual_boundary_edge_reduction": int(before["boundary_edges"] - after["boundary_edges"]),
        "seam_edge_incidence_two": seam_incidence_two,
        "seam_edge_opposite_directions": seam_opposite,
        "internal_edge_opposite_directions": internal_opposite,
        "degenerate_generated_face_ids": degenerate.astype(int).tolist(),
        "source_boundary_vertices_unchanged": True,
    }


def _triangle_pair_contact(
    points: np.ndarray,
    patch_face: np.ndarray,
    other_face: np.ndarray,
    patch_boundary_edges: set[tuple[int, int]],
    tolerance: float,
    *,
    other_is_patch: bool,
) -> dict[str, str]:
    patch_triangle = points[patch_face]
    other_triangle = points[other_face]
    normal_a = np.cross(patch_triangle[1] - patch_triangle[0], patch_triangle[2] - patch_triangle[0])
    normal_b = np.cross(other_triangle[1] - other_triangle[0], other_triangle[2] - other_triangle[0])
    length_a = float(np.linalg.norm(normal_a))
    length_b = float(np.linalg.norm(normal_b))
    if length_a <= tolerance * tolerance or length_b <= tolerance * tolerance:
        return {"classification": "degenerate", "contact": "degenerate_triangle"}
    unit_a = normal_a / length_a
    unit_b = normal_b / length_b
    distances_a_to_b = (patch_triangle - other_triangle[0]) @ unit_b
    distances_b_to_a = (other_triangle - patch_triangle[0]) @ unit_a
    if np.all(distances_a_to_b > tolerance) or np.all(distances_a_to_b < -tolerance):
        return {"classification": "none", "contact": "disjoint_planes"}
    if np.all(distances_b_to_a > tolerance) or np.all(distances_b_to_a < -tolerance):
        return {"classification": "none", "contact": "disjoint_planes"}

    cross_normals = np.cross(unit_a, unit_b)
    cross_length = float(np.linalg.norm(cross_normals))
    if cross_length <= 1e-9:
        if max(float(np.max(np.abs(distances_a_to_b))), float(np.max(np.abs(distances_b_to_a)))) > tolerance:
            return {"classification": "none", "contact": "parallel_planes"}
        return _coplanar_triangle_contact(
            points,
            patch_face,
            other_face,
            patch_boundary_edges,
            unit_a,
            tolerance,
            other_is_patch=other_is_patch,
        )

    segment_a = _triangle_plane_segment(patch_triangle, distances_a_to_b, tolerance)
    segment_b = _triangle_plane_segment(other_triangle, distances_b_to_a, tolerance)
    if not segment_a or not segment_b:
        return {"classification": "none", "contact": "separated_intersection_line"}
    direction = cross_normals / cross_length
    scalars_a = np.asarray([float(np.dot(point, direction)) for point in segment_a])
    scalars_b = np.asarray([float(np.dot(point, direction)) for point in segment_b])
    low = max(float(scalars_a.min()), float(scalars_b.min()))
    high = min(float(scalars_a.max()), float(scalars_b.max()))
    if high < low - tolerance:
        return {"classification": "none", "contact": "separated_intersection_segments"}
    base = segment_a[0]
    base_scalar = float(np.dot(base, direction))
    sample_scalars = [low, high, 0.5 * (low + high)] if high - low > tolerance else [0.5 * (low + high)]
    contact_points = [base + direction * (value - base_scalar) for value in sample_scalars]
    return _classify_contact_points(
        points,
        patch_face,
        other_face,
        patch_boundary_edges,
        contact_points,
        tolerance,
        other_is_patch=other_is_patch,
        contact="non_coplanar_penetration",
    )


def _coplanar_triangle_contact(
    points: np.ndarray,
    patch_face: np.ndarray,
    other_face: np.ndarray,
    patch_boundary_edges: set[tuple[int, int]],
    normal: np.ndarray,
    tolerance: float,
    *,
    other_is_patch: bool,
) -> dict[str, str]:
    drop_axis = int(np.argmax(np.abs(normal)))
    patch_2d = np.delete(points[patch_face], drop_axis, axis=1)
    other_2d = np.delete(points[other_face], drop_axis, axis=1)
    intersection_polygon = _convex_polygon_intersection(patch_2d, other_2d, tolerance)
    scale = max(
        float(np.linalg.norm(np.ptp(np.vstack([patch_2d, other_2d]), axis=0))),
        tolerance,
    )
    overlap_area = abs(_polygon_area(intersection_polygon)) if len(intersection_polygon) >= 3 else 0.0
    if overlap_area > max(tolerance * scale * 8.0, scale * scale * 1e-14):
        return {"classification": "proper_intersection", "contact": "coplanar_area_overlap"}

    contact_points = []
    for patch_edge in range(3):
        a0 = patch_2d[patch_edge]
        a1 = patch_2d[(patch_edge + 1) % 3]
        for other_edge in range(3):
            b0 = other_2d[other_edge]
            b1 = other_2d[(other_edge + 1) % 3]
            parameters = _segment_contact_parameters_2d(a0, a1, b0, b1, tolerance)
            for parameter in parameters:
                contact_points.append(
                    points[patch_face[patch_edge]]
                    + parameter
                    * (points[patch_face[(patch_edge + 1) % 3]] - points[patch_face[patch_edge]])
                )
    if not contact_points:
        return {"classification": "none", "contact": "coplanar_disjoint"}
    return _classify_contact_points(
        points,
        patch_face,
        other_face,
        patch_boundary_edges,
        _unique_points(contact_points, tolerance),
        tolerance,
        other_is_patch=other_is_patch,
        contact="coplanar_boundary_contact",
    )


def _classify_contact_points(
    points: np.ndarray,
    patch_face: np.ndarray,
    other_face: np.ndarray,
    patch_boundary_edges: set[tuple[int, int]],
    contact_points: list[np.ndarray],
    tolerance: float,
    *,
    other_is_patch: bool,
    contact: str,
) -> dict[str, str]:
    shared_vertices = sorted(set(int(value) for value in patch_face) & set(int(value) for value in other_face))
    if shared_vertices and all(
        _point_on_shared_feature(point, points, shared_vertices, tolerance)
        for point in contact_points
    ):
        return {"classification": "allowed_topological_contact", "contact": "shared_topological_feature"}
    if not other_is_patch and all(
        _point_on_patch_boundary(point, points, patch_face, patch_boundary_edges, tolerance)
        for point in contact_points
    ):
        return {"classification": "allowed_boundary_contact", "contact": "patch_outer_boundary_only"}
    return {"classification": "proper_intersection", "contact": contact}


def _triangle_plane_segment(
    triangle: np.ndarray,
    signed_distances: np.ndarray,
    tolerance: float,
) -> list[np.ndarray]:
    points = [triangle[index] for index in range(3) if abs(float(signed_distances[index])) <= tolerance]
    for index in range(3):
        following = (index + 1) % 3
        left = float(signed_distances[index])
        right = float(signed_distances[following])
        if (left < -tolerance and right > tolerance) or (left > tolerance and right < -tolerance):
            fraction = left / (left - right)
            points.append(triangle[index] + fraction * (triangle[following] - triangle[index]))
    return _unique_points(points, tolerance)


def _convex_polygon_intersection(
    subject: np.ndarray,
    clip: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    output = [np.asarray(point, dtype=np.float64) for point in subject]
    orientation = 1.0 if _polygon_area(clip) >= 0.0 else -1.0
    for edge_index in range(clip.shape[0]):
        if not output:
            break
        edge_start = clip[edge_index]
        edge_end = clip[(edge_index + 1) % clip.shape[0]]
        input_points = output
        output = []
        previous = input_points[-1]
        previous_inside = orientation * _cross_2d(edge_end - edge_start, previous - edge_start) >= -tolerance
        for current in input_points:
            current_inside = orientation * _cross_2d(edge_end - edge_start, current - edge_start) >= -tolerance
            if current_inside != previous_inside:
                output.append(_line_intersection_2d(previous, current, edge_start, edge_end, tolerance))
            if current_inside:
                output.append(current)
            previous = current
            previous_inside = current_inside
    if not output:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(_unique_points(output, tolerance), dtype=np.float64)


def _line_intersection_2d(
    left_start: np.ndarray,
    left_end: np.ndarray,
    right_start: np.ndarray,
    right_end: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    left = left_end - left_start
    right = right_end - right_start
    denominator = _cross_2d(left, right)
    if abs(denominator) <= tolerance:
        return 0.5 * (left_start + left_end)
    fraction = _cross_2d(right_start - left_start, right) / denominator
    return left_start + fraction * left


def _segment_contact_parameters_2d(
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
    tolerance: float,
) -> list[float]:
    a = a1 - a0
    b = b1 - b0
    denominator = _cross_2d(a, b)
    if abs(denominator) > tolerance:
        t = _cross_2d(b0 - a0, b) / denominator
        u = _cross_2d(b0 - a0, a) / denominator
        if -tolerance <= t <= 1.0 + tolerance and -tolerance <= u <= 1.0 + tolerance:
            return [float(np.clip(t, 0.0, 1.0))]
        return []
    if abs(_cross_2d(b0 - a0, a)) > tolerance:
        return []
    denominator_a = float(np.dot(a, a))
    if denominator_a <= tolerance * tolerance:
        return [0.0] if np.linalg.norm(a0 - b0) <= tolerance else []
    values = sorted((float(np.dot(b0 - a0, a) / denominator_a), float(np.dot(b1 - a0, a) / denominator_a)))
    low = max(0.0, values[0])
    high = min(1.0, values[1])
    if high < low - tolerance:
        return []
    if high - low <= tolerance:
        return [float(np.clip(0.5 * (low + high), 0.0, 1.0))]
    return [float(low), float(high), float(0.5 * (low + high))]


def _point_on_shared_feature(
    point: np.ndarray,
    points: np.ndarray,
    shared_vertices: list[int],
    tolerance: float,
) -> bool:
    if len(shared_vertices) == 1:
        return bool(np.linalg.norm(point - points[shared_vertices[0]]) <= tolerance * 8.0)
    shared_points = points[np.asarray(shared_vertices, dtype=np.int64)]
    for left_index in range(shared_points.shape[0]):
        for right_index in range(left_index + 1, shared_points.shape[0]):
            if _point_segment_distance(point, shared_points[left_index], shared_points[right_index]) <= tolerance * 8.0:
                return True
    return False


def _point_on_patch_boundary(
    point: np.ndarray,
    points: np.ndarray,
    patch_face: np.ndarray,
    patch_boundary_edges: set[tuple[int, int]],
    tolerance: float,
) -> bool:
    for index, left_value in enumerate(patch_face):
        left = int(left_value)
        right = int(patch_face[(index + 1) % 3])
        if tuple(sorted((left, right))) not in patch_boundary_edges:
            continue
        if _point_segment_distance(point, points[left], points[right]) <= tolerance * 8.0:
            return True
    return False


def _point_segment_distance(point: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    direction = right - left
    squared = float(np.dot(direction, direction))
    if squared <= 1e-30:
        return float(np.linalg.norm(point - left))
    fraction = float(np.clip(np.dot(point - left, direction) / squared, 0.0, 1.0))
    return float(np.linalg.norm(point - (left + fraction * direction)))


def _polygon_self_intersections_2d(points: np.ndarray, tolerance: float) -> list[list[int]]:
    intersections = []
    for left in range(points.shape[0]):
        left_next = (left + 1) % points.shape[0]
        for right in range(left + 1, points.shape[0]):
            right_next = (right + 1) % points.shape[0]
            if right == left_next or right_next == left:
                continue
            if _segments_contact_2d(points[left], points[left_next], points[right], points[right_next], tolerance):
                intersections.append([left, right])
    return intersections


def _segments_contact_2d(
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
    tolerance: float,
) -> bool:
    return bool(_segment_contact_parameters_2d(a0, a1, b0, b1, tolerance))


def _point_in_triangle_2d(
    point: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    orientation: float,
    tolerance: float,
) -> bool:
    values = (
        orientation * _cross_2d(b - a, point - a),
        orientation * _cross_2d(c - b, point - b),
        orientation * _cross_2d(a - c, point - c),
    )
    return all(value >= -tolerance for value in values)


def _plane_basis(normal: np.ndarray, centered: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.linalg.norm(centered, axis=1)
    for index in np.argsort(-lengths, kind="stable"):
        direction = centered[int(index)] - normal * float(np.dot(centered[int(index)], normal))
        length = float(np.linalg.norm(direction))
        if length > 1e-15:
            first = direction / length
            second = np.cross(normal, first)
            return first, second / np.linalg.norm(second)
    axis = np.eye(3)[int(np.argmin(np.abs(normal)))]
    first = np.cross(normal, axis)
    first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    return first, second / np.linalg.norm(second)


def _polygon_area(points: np.ndarray | list[np.ndarray]) -> float:
    array = np.asarray(points, dtype=np.float64)
    if array.shape[0] < 3:
        return 0.0
    return float(
        0.5
        * np.sum(
            array[:, 0] * np.roll(array[:, 1], -1)
            - np.roll(array[:, 0], -1) * array[:, 1]
        )
    )


def _cross_2d(left: np.ndarray, right: np.ndarray) -> float:
    return float(left[0] * right[1] - left[1] * right[0])


def _triangle_quality(triangle: np.ndarray, tolerance: float) -> float:
    lengths_squared = np.sum((triangle - np.roll(triangle, -1, axis=0)) ** 2, axis=1)
    double_area = float(np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])))
    if double_area <= tolerance * tolerance or float(lengths_squared.sum()) <= tolerance * tolerance:
        return 0.0
    return float(2.0 * np.sqrt(3.0) * double_area / lengths_squared.sum())


def _internal_edge_length(
    points: np.ndarray,
    occurrences: dict[tuple[int, int], list[tuple[int, tuple[int, int]]]],
) -> float:
    return float(
        sum(
            np.linalg.norm(points[left] - points[right])
            for (left, right), values in occurrences.items()
            if len(values) == 2
        )
    )


def _max_plane_deviation_ratio(points: np.ndarray, scale: float) -> float:
    centered = points - points.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return float(np.max(np.abs(centered @ vh[-1])) / max(scale, 1e-30))


def _edge_face_map(faces: np.ndarray) -> dict[tuple[int, int], list[int]]:
    result: dict[tuple[int, int], list[int]] = {}
    for face_id, face in enumerate(np.asarray(faces, dtype=np.int64).reshape(-1, 3)):
        for index, left_value in enumerate(face):
            left = int(left_value)
            right = int(face[(index + 1) % 3])
            result.setdefault(tuple(sorted((left, right))), []).append(face_id)
    return result


def _directed_edge_occurrences(faces: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    result: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for face in np.asarray(faces, dtype=np.int64).reshape(-1, 3):
        for index, left_value in enumerate(face):
            left = int(left_value)
            right = int(face[(index + 1) % 3])
            result.setdefault(tuple(sorted((left, right))), []).append((left, right))
    return result


def _directed_edge(face: np.ndarray, edge: tuple[int, int]) -> tuple[int, int] | None:
    for index, left_value in enumerate(face):
        directed = (int(left_value), int(face[(index + 1) % 3]))
        if tuple(sorted(directed)) == edge:
            return directed
    return None


def _boundary_edges(faces: np.ndarray) -> set[tuple[int, int]]:
    return {edge for edge, face_ids in _edge_face_map(faces).items() if len(face_ids) == 1}


def _topology_counts(faces: np.ndarray) -> dict[str, int]:
    edge_faces = _edge_face_map(faces)
    return {
        "edges": len(edge_faces),
        "boundary_edges": sum(len(face_ids) == 1 for face_ids in edge_faces.values()),
        "non_manifold_edges": sum(len(face_ids) > 2 for face_ids in edge_faces.values()),
    }


def _rotate_to_smallest(loop: np.ndarray) -> np.ndarray:
    offset = int(np.argmin(loop))
    return np.roll(loop, -offset).copy()


def _unique_points(points: list[np.ndarray], tolerance: float) -> list[np.ndarray]:
    unique = []
    for point in points:
        point = np.asarray(point, dtype=np.float64)
        if not any(np.linalg.norm(point - existing) <= tolerance * 4.0 for existing in unique):
            unique.append(point)
    return unique


def _geometric_tolerance(points: np.ndarray, requested: float | None) -> float:
    if requested is not None:
        value = float(requested)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("geometric_tolerance must be finite and positive")
        return value
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return 1e-15
    scale = max(float(np.linalg.norm(np.ptp(points, axis=0))), float(np.max(np.abs(points))), 1.0)
    return max(scale * 1e-12, np.finfo(np.float64).eps * scale * 128.0)


def _validate_mesh_arrays(
    points: np.ndarray,
    faces: np.ndarray,
    *,
    allow_empty_faces: bool,
) -> str | None:
    points_array = np.asarray(points)
    faces_array = np.asarray(faces)
    if points_array.ndim != 2 or points_array.shape[1:] != (3,):
        return "points must have shape (N, 3)"
    if not np.issubdtype(points_array.dtype, np.number) or not np.all(np.isfinite(points_array)):
        return "points must be finite numeric values"
    if faces_array.size == 0:
        return None if allow_empty_faces else "source_faces cannot be empty"
    if faces_array.ndim != 2 or faces_array.shape[1:] != (3,):
        return "faces must have shape (M, 3)"
    if not np.issubdtype(faces_array.dtype, np.integer):
        return "faces must contain integer vertex IDs"
    if np.any((faces_array < 0) | (faces_array >= points_array.shape[0])):
        return "faces contain out-of-range vertex IDs"
    if np.any(
        (faces_array[:, 0] == faces_array[:, 1])
        | (faces_array[:, 1] == faces_array[:, 2])
        | (faces_array[:, 2] == faces_array[:, 0])
    ):
        return "faces contain repeated vertex IDs"
    return None


def _validate_provenance(provenance: dict[str, np.ndarray] | None, face_count: int) -> str | None:
    if provenance is None:
        return None
    if not isinstance(provenance, dict):
        return "provenance must be a mapping of per-face arrays"
    for name, values in provenance.items():
        array = np.asarray(values)
        if array.ndim == 0 or array.shape[0] != face_count:
            return f"provenance field {name!r} must have first dimension {face_count}"
    return None


def _normalized_source_provenance(
    provenance: dict[str, np.ndarray] | None,
    face_count: int,
) -> dict[str, np.ndarray]:
    result = {} if provenance is None else {name: np.asarray(values).copy() for name, values in provenance.items()}
    result.setdefault("face_origin", np.full(face_count, SOURCE_FACE_ORIGIN, dtype=np.int16))
    result.setdefault("source_triangle_index", np.arange(face_count, dtype=np.int64))
    result.setdefault("fusion_region_id", np.zeros(face_count, dtype=np.int32))
    return result


def _generated_provenance(
    source_provenance: dict[str, np.ndarray],
    count: int,
    *,
    fusion_region_id: int,
    face_origin_value: int,
) -> dict[str, np.ndarray]:
    result = {}
    for name, source_values in source_provenance.items():
        shape = (count,) + source_values.shape[1:]
        if name == "face_origin":
            values = np.full(shape, face_origin_value, dtype=source_values.dtype)
        elif name == "source_triangle_index":
            values = np.full(shape, -1, dtype=source_values.dtype)
        elif name == "fusion_region_id":
            values = np.full(shape, fusion_region_id, dtype=source_values.dtype)
        elif np.issubdtype(source_values.dtype, np.bool_):
            values = np.zeros(shape, dtype=source_values.dtype)
        elif np.issubdtype(source_values.dtype, np.number):
            values = np.zeros(shape, dtype=source_values.dtype)
        elif np.issubdtype(source_values.dtype, np.str_):
            values = np.full(shape, "", dtype=source_values.dtype)
        else:
            values = np.full(shape, None, dtype=source_values.dtype)
        result[name] = values
    return result


def _rolled_back(
    points: np.ndarray,
    faces: np.ndarray,
    provenance: dict[str, np.ndarray] | None,
    reason_codes: list[str],
    diagnostics: dict[str, Any],
    *,
    candidate: dict[str, Any] | None = None,
    local_intersection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    faces_array = np.asarray(faces)
    face_count = faces_array.shape[0] if faces_array.ndim == 2 else 0
    fallback_provenance = (
        _normalized_source_provenance(None, face_count)
        if provenance is None
        else {name: np.asarray(values).copy() for name, values in provenance.items()}
    )
    return {
        "success": False,
        "committed": False,
        "status": "rolled_back",
        "points": np.asarray(points).copy(),
        "faces": faces_array.copy(),
        "provenance": fallback_provenance,
        "generated_faces": _empty_faces(),
        "generated_face_ids": np.empty(0, dtype=np.int64),
        "generated_provenance": {},
        "oriented_boundary_loop": np.empty(0, dtype=np.int64),
        "candidate": candidate or {"generated_faces": _empty_faces()},
        "diagnostics": diagnostics,
        "local_intersection": local_intersection or {
            "method": "focused_proper_triangle_intersection",
            "status": "not_computed",
            "passed": False,
            "intersection_pairs": None,
        },
        "failure_reason_codes": list(reason_codes),
    }


def _boundary_failure(code: str, message: str, **details: Any) -> dict[str, Any]:
    return {
        "success": False,
        "failure_reason_codes": [code],
        "diagnostics": {
            "stage": "source_boundary_validation",
            "message": message,
            **details,
        },
    }


def _intersection_failure(code: str, message: str) -> dict[str, Any]:
    return {
        "method": "focused_proper_triangle_intersection",
        "status": "invalid_input",
        "passed": False,
        "intersection_pairs": None,
        "candidate_pairs_tested": 0,
        "reported_pairs": [],
        "failure_reason": message,
        "failure_reason_code": code,
        "contact_policy": _contact_policy(),
    }


def _contact_policy() -> str:
    return (
        "allow_shared_topological_features_and_patch_outer_boundary_contact; "
        "reject_interior_penetration_and_coplanar_area_overlap"
    )


def _empty_faces() -> np.ndarray:
    return np.empty((0, 3), dtype=np.int64)
