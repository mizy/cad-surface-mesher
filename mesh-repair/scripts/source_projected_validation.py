from __future__ import annotations
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
from scipy.sparse import csr_matrix

from mesh_io import read_surface, triangle_faces
from mesh_metrics import edge_topology, face_component_labels, mesh_report, self_intersection_report
from source_projected_closure import REJECTION_BITS, rollback_reported_self_intersections


def repair_closure_proxy_self_intersections(
    points: np.ndarray,
    faces: np.ndarray,
    voxel_pitch: float,
    *,
    max_certification_rounds: int = 4,
) -> tuple[np.ndarray, dict[str, Any]]:
    baseline = np.asarray(points, dtype=np.float64)
    current = baseline.copy()
    incidence = _vertex_face_incidence(faces, current.shape[0])
    certification = component_self_intersection_report(current, faces)
    initial = certification
    rounds = []
    for certification_round in range(1, max_certification_rounds + 1):
        pairs = np.asarray(certification.get("reported_pairs", []), dtype=np.int64).reshape(-1, 2)
        if certification.get("passed") or pairs.size == 0:
            break
        local_rows = []
        for local_round in range(1, 9):
            trial, moved, movement = _separate_intersection_pairs(
                current,
                faces,
                incidence,
                pairs,
                voxel_pitch * 0.01 * min(local_round, 5),
            )
            unsafe = _relative_triangle_issue_mask(baseline, trial, faces)
            if np.any(unsafe):
                reverted = np.unique(faces[unsafe].ravel())
                trial[reverted] = current[reverted]
                moved = np.setdiff1d(moved, reverted, assume_unique=False)
            local = _local_intersection_report(trial, faces, incidence, moved)
            local_rows.append({
                "round": local_round,
                **movement,
                "unsafe_faces_reverted": int(np.count_nonzero(unsafe)),
                "local_recheck": local,
            })
            current = trial
            pairs = np.asarray(local.get("reported_pairs", []), dtype=np.int64).reshape(-1, 2)
            if local.get("passed"):
                break
        certification = component_self_intersection_report(current, faces)
        rounds.append({
            "certification_round": certification_round,
            "local_iterations": local_rows,
            "full_recheck": certification,
        })
        if certification.get("passed"):
            break
    displacement = np.linalg.norm(current - baseline, axis=1)
    moved = displacement[displacement > 0.0]
    return current, {
        "method": "connectivity_preserving_local_normal_separation",
        "initial_self_intersection": initial,
        "rounds": rounds,
        "final_self_intersection": certification,
        "passed": certification.get("status") == "computed" and certification.get("passed") is True,
        "connectivity_changed": False,
        "moved_vertices": int(moved.size),
        "displacement": _percentiles(moved),
        "displacement_in_voxel_pitch": _percentiles(moved / max(voxel_pitch, 1e-15)),
    }


def component_self_intersection_report(
    points: np.ndarray,
    faces: np.ndarray,
    *,
    largest_component_pair_limit: int = 100_000_000,
    other_component_pair_limit: int = 10_000_000,
) -> dict[str, Any]:
    _, edge_faces = edge_topology(faces)
    labels = face_component_labels(faces.shape[0], edge_faces)
    sizes = np.bincount(labels)
    largest = int(np.argmax(sizes))
    rows = []
    pairs: set[tuple[int, int]] = set()
    complete = True
    for component_id in range(sizes.size):
        focus = np.flatnonzero(labels == component_id)
        limit = largest_component_pair_limit if component_id == largest else other_component_pair_limit
        row = self_intersection_report(
            points,
            faces,
            focus_face_ids=focus,
            max_candidate_pairs=limit,
            max_reported_pairs=100_000,
        )
        rows.append({"component_id": component_id, "face_count": int(focus.size), **row})
        complete &= row.get("status") == "computed"
        pairs.update(tuple(sorted(map(int, pair))) for pair in row.get("reported_pairs", []))
    ordered = sorted(pairs)
    return {
        "method": "component_focused_global_vtk_triangle_intersection",
        "status": "computed" if complete else "incomplete_component_check",
        "passed": complete and not ordered,
        "intersection_pairs": len(ordered),
        "reported_pairs": [list(pair) for pair in ordered],
        "component_checks": rows,
        "failure_reason": None if complete and not ordered else "one or more global component checks are incomplete or intersecting",
    }


def _vertex_face_incidence(faces: np.ndarray, point_count: int) -> csr_matrix:
    face_ids = np.repeat(np.arange(faces.shape[0], dtype=np.int64), 3)
    return csr_matrix(
        (np.ones(faces.size, dtype=np.uint8), (faces.ravel(), face_ids)),
        shape=(point_count, faces.shape[0]),
    )


def _incident_faces(incidence: csr_matrix, vertices: np.ndarray) -> np.ndarray:
    return np.unique(incidence[np.asarray(vertices, dtype=np.int64)].indices)


def _separate_intersection_pairs(
    points: np.ndarray,
    faces: np.ndarray,
    incidence: csr_matrix,
    pairs: np.ndarray,
    step: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    delta = np.zeros_like(points)
    weight = np.zeros(points.shape[0], dtype=np.float64)
    for left_value, right_value in pairs:
        left, right = int(left_value), int(right_value)
        axis = _separation_axis(points, faces, left, right)
        for face_id, direction in ((left, axis), (right, -axis)):
            core = np.unique(faces[face_id])
            ring_faces = _incident_faces(incidence, core)
            ring = np.setdiff1d(np.unique(faces[ring_faces].ravel()), core, assume_unique=False)
            for vertices, taper in ((core, 1.0), (ring, 0.25)):
                delta[vertices] += direction[None, :] * step * taper
                weight[vertices] += 1.0
    moved = weight > 0.0
    result = points.copy()
    result[moved] += delta[moved] / weight[moved, None]
    return result, np.flatnonzero(moved), {
        "pair_count": int(pairs.shape[0]),
        "moved_vertices": int(np.count_nonzero(moved)),
        "step": float(step),
    }


def _separation_axis(points: np.ndarray, faces: np.ndarray, left: int, right: int) -> np.ndarray:
    triangles = points[faces[[left, right]]]
    raw = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(raw, axis=1)
    normals = np.divide(raw, lengths[:, None], out=np.zeros_like(raw), where=lengths[:, None] > 1e-15)
    aligned_right = normals[1] if np.dot(normals[0], normals[1]) >= 0.0 else -normals[1]
    axis = normals[0] + aligned_right
    if np.linalg.norm(axis) <= 1e-12:
        axis = triangles[0].mean(axis=0) - triangles[1].mean(axis=0)
    if np.linalg.norm(axis) <= 1e-12:
        axis = normals[0]
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    centroid_delta = triangles[0].mean(axis=0) - triangles[1].mean(axis=0)
    return axis if np.dot(centroid_delta, axis) >= 0.0 else -axis


def _relative_triangle_issue_mask(
    baseline_points: np.ndarray,
    candidate_points: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    baseline = _triangle_geometry(baseline_points, faces)
    candidate = _triangle_geometry(candidate_points, faces)
    area_ratio = np.divide(candidate["area"], baseline["area"], out=np.ones_like(candidate["area"]), where=baseline["area"] > 1e-30)
    edge_ratio = np.divide(candidate["edges"], baseline["edges"], out=np.ones_like(candidate["edges"]), where=baseline["edges"] > 1e-15)
    normal_dot = np.einsum("ij,ij->i", baseline["normals"], candidate["normals"])
    return (
        (normal_dot <= 0.0)
        | (area_ratio < 0.2)
        | (area_ratio > 5.0)
        | np.any(edge_ratio > 2.0, axis=1)
        | (candidate["aspect"] > 12.0)
    )


def _triangle_geometry(points: np.ndarray, faces: np.ndarray) -> dict[str, np.ndarray]:
    triangles = points[faces]
    vectors = np.stack(
        [triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 1], triangles[:, 0] - triangles[:, 2]],
        axis=1,
    )
    edges = np.linalg.norm(vectors, axis=2)
    raw = np.cross(vectors[:, 0], -vectors[:, 2])
    double_area = np.linalg.norm(raw, axis=1)
    normals = np.divide(raw, double_area[:, None], out=np.zeros_like(raw), where=double_area[:, None] > 1e-15)
    max_edge = edges.max(axis=1)
    altitude = np.divide(double_area, max_edge, out=np.zeros_like(max_edge), where=max_edge > 1e-15)
    aspect = np.divide(max_edge, altitude, out=np.full_like(max_edge, np.inf), where=altitude > 1e-15)
    return {"area": 0.5 * double_area, "edges": edges, "normals": normals, "aspect": aspect}


def _local_intersection_report(
    points: np.ndarray,
    faces: np.ndarray,
    incidence: csr_matrix,
    moved_vertices: np.ndarray,
) -> dict[str, Any]:
    focus = _incident_faces(incidence, moved_vertices)
    return self_intersection_report(
        points,
        faces,
        focus_face_ids=focus,
        max_candidate_pairs=10_000_000,
        max_reported_pairs=100_000,
    )


def repair_projected_self_intersections(
    points: np.ndarray,
    proxy_points: np.ndarray,
    faces: np.ndarray,
    provenance: dict[str, dict[str, np.ndarray]],
    *,
    certified_proxy_report: dict[str, Any] | None = None,
    max_rounds: int = 16,
) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    current = np.asarray(points, dtype=np.float64).copy()
    current_provenance = provenance
    rounds = []
    for round_index in range(1, max_rounds + 1):
        intersections = projected_delta_self_intersection_report(
            current,
            faces,
            current_provenance,
            certified_proxy_report,
        )
        if intersections.get("passed") or not intersections.get("reported_pairs"):
            intersections["projection_rollback_rounds"] = rounds
            return current, current_provenance, intersections
        current, current_provenance, rollback = rollback_reported_self_intersections(
            current,
            proxy_points,
            faces,
            current_provenance,
            intersections,
            # Direct face rollback removes the bulk of projection collisions
            # with minimal loss of source detail.  If a transition keeps
            # folding after three certification rounds, also restore its
            # one-ring neighborhood so a new crease is not created directly
            # beside the previous intersection.
            vertex_rings=0 if round_index <= 3 else 1,
        )
        rounds.append({"round": round_index, **rollback})
        if rollback["rolled_back_vertices"] == 0:
            intersections["projection_rollback_rounds"] = rounds
            intersections["failure_reason"] = (
                "reported intersections remain entirely within the closure proxy fallback geometry"
            )
            return current, current_provenance, intersections
    final = projected_delta_self_intersection_report(
        current,
        faces,
        current_provenance,
        certified_proxy_report,
    )
    final["projection_rollback_rounds"] = rounds
    if not final.get("passed"):
        final["failure_reason"] = "self-intersections remain after bounded projection rollback"
    return current, current_provenance, final


def projected_delta_self_intersection_report(
    points: np.ndarray,
    faces: np.ndarray,
    provenance: dict[str, dict[str, np.ndarray]],
    certified_proxy_report: dict[str, Any] | None,
) -> dict[str, Any]:
    baseline_passed = bool(
        certified_proxy_report
        and certified_proxy_report.get("status") == "computed"
        and certified_proxy_report.get("passed") is True
    )
    if not baseline_passed:
        return {
            "method": "certified_proxy_plus_projected_delta_intersection",
            "status": "baseline_proxy_not_certified",
            "passed": False,
            "reported_pairs": [],
            "failure_reason": "closure proxy must be globally intersection-free before delta validation",
        }
    applied = np.asarray(provenance["point_data"]["source_projection_applied"]).astype(bool)
    focus = np.flatnonzero(np.any(applied[faces], axis=1))
    if focus.size == 0:
        return {
            "method": "certified_proxy_plus_projected_delta_intersection",
            "status": "computed",
            "passed": True,
            "intersection_pairs": 0,
            "reported_pairs": [],
            "changed_face_count": 0,
            "proof": "candidate equals the globally certified closure proxy",
        }
    # ``bounded_triangle_self_intersections`` needs a pair-de-duplication set
    # for a focused scan because the same pair may be reached from either
    # focused face.  Once most faces have changed, that set costs more memory
    # and time than simply checking the complete mesh, whose ordered traversal
    # rejects duplicate pairs without retaining them.  The complete scan is a
    # strictly stronger proof and avoids pathological focus-set growth on a
    # broadly source-projected shell.
    use_complete_scan = focus.size * 2 >= faces.shape[0]
    scan_focus = None if use_complete_scan else focus
    row = self_intersection_report(
        points,
        faces,
        focus_face_ids=scan_focus,
        max_candidate_pairs=100_000_000,
        max_reported_pairs=100_000,
    )
    return {
        **row,
        "method": "certified_proxy_plus_projected_delta_intersection",
        "changed_face_count": int(focus.size),
        "scan_scope": "all_faces" if use_complete_scan else "changed_faces",
        "complete_scan_threshold": "changed_face_count * 2 >= total_face_count",
        "baseline_proxy_certified": True,
        "proof": (
            "complete candidate mesh checked because changed faces cover at least half the mesh"
            if use_complete_scan
            else "unchanged face pairs inherit proxy certification; every pair touching a changed face is checked"
        ),
    }


def chunked_self_intersection_report(
    points: np.ndarray,
    faces: np.ndarray,
    *,
    focus_chunk_size: int = 25_000,
    max_candidate_pairs_per_chunk: int = 2_000_000,
) -> dict[str, Any]:
    tested = 0
    chunks = 0
    for start in range(0, faces.shape[0], focus_chunk_size):
        focus = np.arange(start, min(start + focus_chunk_size, faces.shape[0]), dtype=np.int64)
        row = self_intersection_report(
            points,
            faces,
            focus_face_ids=focus,
            max_candidate_pairs=max_candidate_pairs_per_chunk,
            max_reported_pairs=200,
        )
        chunks += 1
        tested += int(row.get("candidate_pairs_tested") or 0)
        if row.get("status") != "computed":
            return {
                **row,
                "method": "chunked_vtk_static_cell_locator_triangle_intersection",
                "chunks_completed": chunks - 1,
                "candidate_pairs_tested_total": tested,
            }
        if int(row.get("intersection_pairs") or 0) > 0:
            return {
                **row,
                "method": "chunked_vtk_static_cell_locator_triangle_intersection",
                "passed": False,
                "chunks_completed": chunks,
                "candidate_pairs_tested_total": tested,
                "early_exit": "an intersection is sufficient to reject the candidate",
            }
    return {
        "method": "chunked_vtk_static_cell_locator_triangle_intersection",
        "scope": "all_faces_in_bounded_focus_chunks",
        "status": "computed",
        "passed": True,
        "intersection_pairs": 0,
        "reported_pairs": [],
        "candidate_pairs_tested_total": tested,
        "chunks_completed": chunks,
        "focus_chunk_size": focus_chunk_size,
        "max_candidate_pairs_per_chunk": max_candidate_pairs_per_chunk,
    }


def roundtrip_validation(
    path: Path,
    expected_points: np.ndarray,
    expected_faces: np.ndarray,
) -> dict[str, Any]:
    try:
        raw = _raw_polydata(path)
        raw_points = np.asarray(raw.points, dtype=np.float64)
        raw_faces = triangle_faces(raw).astype(np.int64, copy=False)
        normalized = read_surface(path)
        normalized_points = np.asarray(normalized.points, dtype=np.float64)
        normalized_faces = triangle_faces(normalized).astype(np.int64, copy=False)
        raw_metrics = mesh_report(raw_points, raw_faces)
        normalized_metrics = mesh_report(normalized_points, normalized_faces)
    except Exception as exc:
        return {
            "status": "failed",
            "passed": False,
            "failure_reason": f"{type(exc).__name__}: {exc}",
        }
    raw_exact = (
        raw_points.shape == expected_points.shape
        and np.array_equal(raw_faces, expected_faces)
        and np.allclose(raw_points, expected_points, rtol=0.0, atol=1e-12)
    )
    raw_passed = watertight_metrics(raw_metrics)
    normalized_passed = watertight_metrics(normalized_metrics)
    topology_survived_clean = (
        normalized_points.shape[0] == expected_points.shape[0]
        and normalized_faces.shape == expected_faces.shape
        and normalized_passed
    )
    passed = raw_exact and raw_passed and topology_survived_clean
    return {
        "method": "raw_vtp_and_normalized_read_surface_roundtrip",
        "status": "computed",
        "passed": passed,
        "raw_geometry_exact": raw_exact,
        "raw_watertight": raw_passed,
        "normalized_watertight": normalized_passed,
        "normalized_point_count_preserved": normalized_points.shape[0] == expected_points.shape[0],
        "normalized_face_count_preserved": normalized_faces.shape == expected_faces.shape,
        "raw": topology_summary(raw_metrics),
        "normalized": topology_summary(normalized_metrics),
        "failure_reason": None if passed else "saved mesh changed topology or manifoldness during roundtrip",
    }


def update_projection_report(
    report: dict[str, Any],
    points: np.ndarray,
    faces: np.ndarray,
    provenance: dict[str, dict[str, np.ndarray]],
    self_intersection: dict[str, Any],
    post_write: dict[str, Any],
) -> dict[str, Any]:
    result = dict(report)
    applied = np.asarray(provenance["point_data"]["source_projection_applied"]).astype(bool)
    fallback = np.asarray(provenance["point_data"]["source_projection_fallback"]).astype(bool)
    rejection = np.asarray(provenance["point_data"]["source_projection_rejection_mask"])
    projection = dict(result.get("projection", {}))
    projection.update({
        "vertices": int(applied.size),
        "source_projected": int(np.count_nonzero(applied)),
        "explicit_proxy_fallback": int(np.count_nonzero(fallback)),
        "unclassified": int(np.count_nonzero(~(applied | fallback))),
        "fallback_without_reason": int(np.count_nonzero(fallback & (rejection == 0))),
        "source_projected_ratio": float(np.count_nonzero(applied) / max(applied.size, 1)),
        "rejection_counts": {name: int(np.count_nonzero(rejection & bit)) for name, bit in REJECTION_BITS.items()},
    })
    result["projection"] = projection
    result["quality"] = {**result.get("quality", {}), "projected": mesh_report(points, faces)}
    comparisons = dict(result.get("comparisons", {}))
    comparisons["projection_mapping"] = {
        "status": "computed",
        "source_projected": projection["source_projected"],
        "explicit_proxy_fallback": projection["explicit_proxy_fallback"],
        "unclassified": projection["unclassified"],
        "fallback_without_reason": projection["fallback_without_reason"],
    }
    comparisons["self_intersection"] = self_intersection
    comparisons["post_write_validation"] = post_write
    result["comparisons"] = comparisons
    result["self_intersection"] = self_intersection
    result["post_write_validation"] = post_write
    return result


def _raw_polydata(path: Path) -> pv.PolyData:
    mesh = pv.read(path)
    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine()
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    return mesh.triangulate()


def watertight_metrics(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics["topology"]["boundary_edges"] == 0
        and metrics["topology"]["non_manifold_edges"] == 0
        and metrics["topology"]["non_manifold_vertices"] == 0
        and metrics["topology"]["inconsistent_winding_edges"] == 0
        and metrics["quality"]["degenerate_faces"] == 0
        and metrics["volume"]["reliable"]
    )


def topology_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "points": metrics["points"],
        "triangles": metrics["triangles"],
        "boundary_edges": metrics["topology"]["boundary_edges"],
        "non_manifold_edges": metrics["topology"]["non_manifold_edges"],
        "non_manifold_vertices": metrics["topology"]["non_manifold_vertices"],
        "inconsistent_winding_edges": metrics["topology"]["inconsistent_winding_edges"],
        "degenerate_faces": metrics["quality"]["degenerate_faces"],
        "volume_reliable": metrics["volume"]["reliable"],
    }


def _percentiles(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"min": None, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "min": float(values.min()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }
